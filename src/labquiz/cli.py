"""labquiz command line: brief | run | eval | render.

M1 implements ``run`` up to ingest: parse submissions and roster, scrub,
check that no roster name reaches a serialised notebook, write the run file.
"""

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

from .core import store
from .core.config import load_config, snapshot
from .core.ingest import IngestResult, ingest
from .core.models import GroupRun, LabRun, Template
from .core.notebook import parse_notebook, read_notebook, serialise_notebook
from .core.scrub import Scrubber, find_leaks


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
        tdoc_text = "\n".join(c.source + "\n" + c.outputs_text for c in template.cells)
        leaks += [f"template: {x}" for x in find_leaks(tdoc_text, names)]
    return leaks


def cmd_run(args) -> int:
    cfg = load_config(args.config)
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

    template = None
    if args.template:
        scrubber = Scrubber(res.scrub_names + extra)
        tdoc = parse_notebook(read_notebook(Path(args.template).read_bytes()), "TEMPLATE",
                              scrubber=scrubber, output_cap=cfg.notebook.output_cap)
        template = Template(lab_id=lab_id, cells=tdoc.cells, sections=[])

    run = LabRun(
        lab_id=lab_id,
        created_at=datetime.now(timezone.utc),
        config=snapshot(cfg),
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
    return 0


def _not_yet(milestone: str):
    def f(args) -> int:
        print(f"'{args.cmd}' is not implemented yet ({milestone}).", file=sys.stderr)
        return 2
    return f


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="labquiz")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="ingest submissions and roster into a run file")
    r.add_argument("--submissions", required=True, help="zip file or folder of .ipynb")
    r.add_argument("--roster", help="CSV with group and student columns (optional: groups and "
                   "students are otherwise read from <prefix>_<LastName>_..._.ipynb file names)")
    r.add_argument("--filename-prefix", help="file name prefix before the last names, e.g. TP2_ISD_SA2025")
    r.add_argument("--template", help="assignment notebook (optional until M2)")
    r.add_argument("--config", help="config TOML (default: built-in defaults)")
    r.add_argument("--lab-id", help="overrides lab_id from the config")
    r.add_argument("--runs-dir", default=str(store.DEFAULT_RUNS_DIR))
    r.add_argument("--run-dir", help="write into this directory instead of a new one")
    r.set_defaults(func=cmd_run)

    for name, ms in (("brief", "M2"), ("render", "M3"), ("eval", "M5")):
        s = sub.add_parser(name, help=f"not implemented yet ({ms})")
        s.set_defaults(func=_not_yet(ms))

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
