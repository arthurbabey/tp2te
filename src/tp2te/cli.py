"""tp2te command line: brief | run | report | eval | render.

- ``brief``: L0 from the template (and solution), once per lab.
- ``run``: ingest submissions (+ roster), privacy pre-flight, match sections,
  hard checks, then L0 (unless ``--brief``) and L1; prints a completeness table
  and an issue report. ``--no-llm`` stops after the checks.
- ``report``: prints the tables again from a run directory.
"""

import argparse
import sys
import textwrap
from datetime import datetime, timezone
from pathlib import Path

from .core import store
from .core.brief import generate_brief, serialise_solution
from .core.config import Config, load_config, load_dotenv, snapshot
from .core.ingest import IngestResult, ingest
from .core.matching import build_template, gradable_sections
from .core.models import GroupRun, HardFinding, LabBrief, LabRun, Template
from .core.notebook import parse_notebook, read_notebook, serialise_notebook
from .core.pipeline import analyse_groups, guard_names, match_and_check
from .core.prompts import versions
from .core.scrub import Scrubber, find_leaks
from .llm.base import LLMClient, LLMError, make_client
from .llm.fake import CachingClient

STATUS_MARK = {"done": "done", "partial": "part", "missing": "MISS"}


def _table(rows: list[list[str]], header: list[str]) -> str:
    widths = [max(len(str(x)) for x in col) for col in zip(header, *rows)]
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    return "\n".join([fmt.format(*header), fmt.format(*("-" * w for w in widths))]
                     + [fmt.format(*map(str, r)) for r in rows])


def preflight_leaks(res: IngestResult, template: Template | None, extra_names: list[str]) -> list[str]:
    """Privacy pre-flight: serialise every notebook as it would enter a prompt and look for leaks."""
    names = res.scrub_names + extra_names
    leaks = []
    for gid, doc in res.notebooks.items():
        leaks += [f"{gid}: {x}" for x in find_leaks(serialise_notebook(doc), names)]
    if template is not None:
        tdoc_text = "\n".join(c.source + "\n" + c.outputs_text for c in template.cells if not c.hidden)
        leaks += [f"template: {x}" for x in find_leaks(tdoc_text, names)]
    return leaks


def load_template(path: str, lab_id: str, cfg: Config, names: list[str]) -> Template:
    doc = parse_notebook(read_notebook(Path(path).read_bytes()), "TEMPLATE", scrubber=Scrubber(names),
                         output_cap=cfg.notebook.output_cap)
    return build_template(doc, lab_id, cfg.template)


def load_solution(path: str | None, template: Template, cfg: Config, names: list[str]) -> str | None:
    if not path:
        return None
    doc = parse_notebook(read_notebook(Path(path).read_bytes()), "SOLUTION", scrubber=Scrubber(names),
                         output_cap=cfg.notebook.output_cap)
    return serialise_solution(doc, template, cfg)


def make_llm(cfg: Config, runs_dir: str | Path) -> LLMClient:
    client = make_client(cfg.llm)
    if cfg.llm.cache and cfg.llm.provider != "fake":
        client = CachingClient(client, Path(runs_dir) / ".llm_cache")
    return client


# ---------- printing ----------
def print_template(t: Template) -> None:
    rows = [[s.section_id, s.title[:50], len(s.cell_ids), len(s.task_cell_ids), len(s.question_cell_ids)]
            for s in gradable_sections(t)]
    print(f"template: {len(t.sections)} sections, {len(rows)} with tasks or questions, "
          f"{sum(c.hidden for c in t.cells)} staff cells hidden")
    print(_table(rows, ["section", "title", "cells", "tasks", "questions"]))


def print_matching(run: LabRun, results: dict, low: float) -> None:
    rows = []
    for gid, (res, findings) in results.items():
        cells = run.groups[gid].notebook.cells
        lowc = [f"{m.section_id}({m.confidence:.2f})" for m in res.matches
                if m.anchors_total and m.confidence < low]
        kinds: dict[str, int] = {}
        for f in findings:
            kinds[f.kind] = kinds.get(f.kind, 0) + 1
        rows.append([gid, sum(c.origin == "unchanged" for c in cells), sum(c.origin == "modified" for c in cells),
                     sum(c.origin == "added" for c in cells), sum(c.hidden for c in cells),
                     ", ".join(lowc) or "-", " ".join(f"{k}={v}" for k, v in sorted(kinds.items())) or "-"])
    print(_table(rows, ["group", "unch", "mod", "added", "staff", "low-confidence sections", "hard findings"]))


def print_findings(gid: str, findings: list[HardFinding]) -> None:
    shown = [f for f in findings if f.kind != "out_of_order"]
    ooo = len(findings) - len(shown)
    print(f"  {gid}: " + ("no finding" if not findings else ""))
    for f in shown:
        print(f"    [{f.section_id or '-'}] {f.cell_id or '-'} {f.kind}: {f.detail}")
    if ooo:
        print(f"    out_of_order: {ooo} cell(s) run out of order")


def print_completeness(run: LabRun, gids: list[str]) -> None:
    gids = [g for g in gids if run.groups[g].completeness is not None]
    if not gids:
        print("no completeness result yet")
        return
    rows = []
    for s in gradable_sections(run.template):
        row = [s.section_id, s.title[:36]]
        for g in gids:
            sc = next((x for x in run.groups[g].completeness.sections if x.section_id == s.section_id), None)
            row.append("-" if sc is None else STATUS_MARK[sc.final_status] + ("*" if sc.hard_findings else ""))
        rows.append(row)
    rows.append(["score", ""] + [f"{run.groups[g].completeness.score:.2f}" for g in gids])
    print(_table(rows, ["section", "title"] + gids))
    print("(* = automatic findings in that section)")


def print_issues(run: LabRun, gids: list[str], width: int = 100) -> None:
    for g in gids:
        gr = run.groups[g]
        if gr.error:
            print(f"\n== {g}: FAILED: {gr.error}")
            continue
        if gr.completeness is None:
            continue
        ung = sum(not i.grounded for i in gr.issues)
        print(f"\n== {g}  completeness {gr.completeness.score:.2f}  "
              f"{len(gr.issues)} issue(s){f', {ung} UNGROUNDED' if ung else ''}, {len(gr.choices)} choice(s)")
        for s in gr.completeness.sections:
            if s.final_status != "done":
                print(f"  {s.section_id} {s.final_status}: {s.llm.reason}")
        for i in sorted(gr.issues, key=lambda i: i.grounded):   # ungrounded first, as on the review screen
            flag = "" if i.grounded else "  [UNGROUNDED]"
            print(f"  {i.issue_id} {i.severity} {i.category} {i.section_id} [{i.concept}] {i.title}{flag}")
            print(textwrap.indent(textwrap.fill(i.explanation, width - 6), "      "))
            for e in i.evidence:
                q = " ".join(e.quote.split())
                print(f"      {e.cell_id}: \"{q[:width - 16]}{'…' if len(q) > width - 16 else ''}\"")
        for c in gr.choices:
            print(f"  choice {c.section_id}: {c.description}")


def print_brief(b: LabBrief) -> None:
    for s in b.sections:
        print(f"  {s.section_id}: {', '.join(s.concepts)}")
    print(f"  {len(b.generic_questions)} generic questions")


# ---------- commands ----------
def cmd_brief(args) -> int:
    cfg = load_config(args.config, env=True)
    lab_id = args.lab_id or cfg.lab_id
    names = list(cfg.scrub.extra_names)
    template = load_template(args.template, lab_id, cfg, names)
    print_template(template)
    solution = load_solution(args.solution, template, cfg, names)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        brief = generate_brief(make_llm(cfg, args.runs_dir), template, cfg, solution_serialised=solution,
                               log_dir=out.parent, guard_names=names)
    except LLMError as e:
        print(f"L0 failed: {e}", file=sys.stderr)
        return 3
    out.write_text(brief.model_dump_json(indent=2), encoding="utf-8")
    print_brief(brief)
    print(f"brief: {out}")
    return 0


def _select(gids: list[str], arg: str | None, limit: int | None) -> list[str]:
    if arg:
        want = [g.strip() for g in arg.split(",") if g.strip()]
        unknown = [g for g in want if g not in gids]
        if unknown:
            raise SystemExit(f"unknown group(s): {', '.join(unknown)}; groups are {', '.join(gids)}")
        gids = want
    return gids[:limit] if limit else gids


def cmd_run(args) -> int:
    cfg = load_config(args.config, env=True)
    lab_id = args.lab_id or cfg.lab_id
    extra = cfg.scrub.extra_names
    res = ingest(args.submissions, args.roster, extra_names=extra, output_cap=cfg.notebook.output_cap,
                 filename_prefix=args.filename_prefix or cfg.ingest.filename_prefix)

    print(_table([[f.path, f.group_id or "-", f.status, f.detail] for f in res.files],
                 ["file", "group", "status", "detail"]))
    print()
    per_group: dict[str, int] = {}
    for s in res.students:
        per_group[s.group_id] = per_group.get(s.group_id, 0) + 1
    rows = []
    for gid in sorted(per_group):
        doc = res.notebooks.get(gid)
        if doc is None:
            rows.append([gid, per_group[gid], "-", "-", "-", "-", "-", "no notebook"])
            continue
        code = [c for c in doc.cells if c.kind == "code"]
        rows.append([gid, per_group[gid], len(doc.cells), len(code), len(doc.cells) - len(code),
                     sum(c.has_image for c in code), sum(c.error_name is not None for c in code),
                     sum(c.execution_count is None for c in code)])
    print(_table(rows, ["group", "students", "cells", "code", "md", "img", "errors", "not run"]))

    template = load_template(args.template, lab_id, cfg, res.scrub_names + extra) if args.template else None
    run = LabRun(
        lab_id=lab_id,
        created_at=datetime.now(timezone.utc),
        config={**snapshot(cfg), "prompts": versions()},
        template=template,
        students=res.students,
        groups={gid: GroupRun(group_id=gid, notebook=doc, matches=[]) for gid, doc in res.notebooks.items()},
    )
    d = Path(args.run_dir) if args.run_dir else store.create_run_dir(lab_id, args.runs_dir)
    d.mkdir(parents=True, exist_ok=True)
    path = store.save(run, d)
    print(f"\nrun file: {path}")

    leaks = preflight_leaks(res, template, extra)
    if leaks:
        print(f"\nPRIVACY PRE-FLIGHT FAILED: {len(leaks)} leak(s) in serialised notebooks", file=sys.stderr)
        for x in leaks:
            print(f"  {x}", file=sys.stderr)
        return 2
    print(f"privacy pre-flight: ok ({len(res.notebooks)} notebooks, no roster name, email or home path)")
    bad = [f for f in res.files if f.status not in ("ok", "duplicate")]
    if bad or res.groups_without_notebook:
        print(f"warnings: {len(bad)} file(s) not used, groups without notebook: "
              f"{', '.join(res.groups_without_notebook) or 'none'}")
    if template is None:
        return 0

    # ---- matching and hard checks ----
    print()
    print_template(template)
    results = match_and_check(run, cfg)
    store.save(run, d)
    print()
    print_matching(run, results, cfg.matching.low_confidence)
    gids = _select(sorted(run.groups), args.groups, args.max_groups)
    print("\nhard findings:")
    for g in gids:
        print_findings(g, results[g][1])
    if args.no_llm:
        return 0

    # ---- L0, L1 ----
    try:
        client = make_llm(cfg, args.runs_dir)
    except LLMError as e:
        print(f"\nLLM not configured ({e}); stopping after the checks. Use --no-llm to skip this step.",
              file=sys.stderr)
        return 3
    names = guard_names(run, cfg)
    solution = load_solution(args.solution, template, cfg, res.scrub_names + extra)
    if args.brief:
        run.brief = LabBrief.model_validate_json(Path(args.brief).read_text(encoding="utf-8"))
    else:
        print("\nL0 lab brief ...", flush=True)
        try:
            run.brief = generate_brief(client, template, cfg, solution_serialised=solution, log_dir=d,
                                       guard_names=names)
        except LLMError as e:
            print(f"L0 failed: {e}", file=sys.stderr)
            return 3
        (d / "brief.json").write_text(run.brief.model_dump_json(indent=2), encoding="utf-8")
        print_brief(run.brief)
    store.save(run, d)

    print(f"\nL1 analysis for {len(gids)} group(s) ...", flush=True)
    analyse_groups(run, client, cfg, d, gids, {g: results[g][1] for g in gids}, solution_serialised=solution,
                   on_done=lambda g, err: print(f"  {g}: {'FAILED ' + err if err else 'ok'}", flush=True))
    print()
    print_completeness(run, gids)
    print_issues(run, gids)
    print(f"\nrun file: {path}")
    return 1 if any(run.groups[g].error for g in gids) else 0


def cmd_report(args) -> int:
    run = store.load(args.run_dir)
    if run.template is None:
        print("run has no template: nothing matched or analysed yet")
        return 0
    gids = _select(sorted(run.groups), args.groups, None)
    print_completeness(run, gids)
    print_issues(run, gids)
    return 0


def _not_yet(milestone: str):
    def f(args) -> int:
        print(f"'{args.cmd}' is not implemented yet ({milestone}).", file=sys.stderr)
        return 2
    return f


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="tp2te")
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("brief", help="L0: lab brief and generic questions from the template")
    b.add_argument("--template", required=True, help="assignment notebook")
    b.add_argument("--solution", help="solution notebook (optional)")
    b.add_argument("--out", default="brief.json")
    b.add_argument("--config")
    b.add_argument("--lab-id")
    b.add_argument("--runs-dir", default=str(store.DEFAULT_RUNS_DIR), help="the LLM cache lives here")
    b.set_defaults(func=cmd_brief)

    r = sub.add_parser("run", help="ingest, match, check, analyse (L0 + L1)")
    r.add_argument("--submissions", required=True, help="zip file or folder of .ipynb")
    r.add_argument("--roster", help="CSV with group and student columns (optional: groups and "
                   "students are otherwise read from <prefix>_<LastName>_..._.ipynb file names)")
    r.add_argument("--filename-prefix", help="file name prefix before the last names, e.g. TP2_ISD_SA2025")
    r.add_argument("--template", help="assignment notebook; without it the run stops after ingest")
    r.add_argument("--solution", help="solution notebook (optional)")
    r.add_argument("--brief", help="brief.json from 'tp2te brief' (otherwise L0 runs here)")
    r.add_argument("--groups", help="comma-separated group ids to analyse, e.g. G01,G03")
    r.add_argument("--max-groups", type=int, help="analyse only the first N groups")
    r.add_argument("--no-llm", action="store_true", help="stop after matching and hard checks")
    r.add_argument("--config", help="config TOML (default: built-in defaults)")
    r.add_argument("--lab-id", help="overrides lab_id from the config")
    r.add_argument("--runs-dir", default=str(store.DEFAULT_RUNS_DIR))
    r.add_argument("--run-dir", help="write into this directory instead of a new one")
    r.set_defaults(func=cmd_run)

    rep = sub.add_parser("report", help="print completeness and issues of a run")
    rep.add_argument("run_dir")
    rep.add_argument("--groups")
    rep.set_defaults(func=cmd_report)

    for name, ms in (("render", "M3"), ("eval", "M5")):
        s = sub.add_parser(name, help=f"not implemented yet ({ms})")
        s.set_defaults(func=_not_yet(ms))

    args = p.parse_args(argv)
    load_dotenv()   # ./.env, without overriding the shell environment
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
