"""L1: completeness judgment and issue report for one group."""

from collections.abc import Iterable
from pathlib import Path

from ..llm.base import LLMClient
from .brief import brief_for_prompt, solution_block, system_params
from .config import Config
from .matching import gradable_sections
from .models import (CompletenessResult, GroupAnalysis, GroupRun, HardFinding, LabBrief, SectionCompleteness,
                     SectionJudgment, Template)
from .notebook import serialise_notebook
from .prompts import call_llm
from .questions import ground_issue

STATUS_POINTS = {"done": 1.0, "partial": 0.5, "missing": 0.0}


def findings_text(findings: list[HardFinding]) -> str:
    if not findings:
        return "(none)"
    return "\n".join(f"- [{f.section_id or '-'}] {f.cell_id or '-'} {f.kind}: {f.detail}" for f in findings)


def build_completeness(template: Template, judgments: list[SectionJudgment],
                       findings: list[HardFinding]) -> CompletenessResult:
    """One entry per gradable section. A section the LLM skipped gets a stand-in
    judgment from the hard findings, marked as such in its reason."""
    by_id = {j.section_id: j for j in judgments}
    out = []
    for s in gradable_sections(template):
        hf = [f for f in findings if f.section_id == s.section_id]
        j = by_id.get(s.section_id)
        if j is None:
            status = "missing" if any(f.kind == "missing_section" for f in hf) else "partial"
            j = SectionJudgment(section_id=s.section_id, status=status,
                                reason="No judgment returned by the LLM; set from the automatic checks.")
        out.append(SectionCompleteness(section_id=s.section_id, hard_findings=hf, llm=j, final_status=j.status))
    score = sum(STATUS_POINTS[x.final_status] for x in out) / len(out) if out else 0.0
    return CompletenessResult(sections=out, score=round(score, 3))


def analyse_group(client: LLMClient, group: GroupRun, template: Template, brief: LabBrief, cfg: Config, *,
                  findings: list[HardFinding], solution_serialised: str | None = None,
                  log_dir: str | Path | None = None, guard_names: Iterable[str] = ()) -> GroupRun:
    """Returns an updated copy of ``group``: completeness, grounded issues, choices."""
    res = call_llm(client, call="L1", prompt="l1_analysis", schema=GroupAnalysis, log_dir=log_dir,
                   guard_names=guard_names, system_params=system_params(cfg), context={"group": group.group_id},
                   temperature=cfg.llm.temperature, max_tokens=cfg.llm.max_tokens,
                   params={"solution_block": solution_block(solution_serialised),
                           "max_issues": cfg.analysis.max_issues, "max_choices": cfg.analysis.max_choices,
                           "brief_json": brief_for_prompt(brief), "hard_findings": findings_text(findings),
                           "notebook_serialised": serialise_notebook(group.notebook, group.matches, template)})
    a = res.parsed
    issues = []
    for k, issue in enumerate(a.issues[: cfg.analysis.max_issues], 1):
        issue.issue_id = f"{group.group_id}-I{k}"
        issues.append(ground_issue(issue, group.notebook))
    choices = a.choices[: cfg.analysis.max_choices]
    return group.model_copy(update={"completeness": build_completeness(template, a.judgments, findings),
                                    "issues": issues, "choices": choices, "error": None})

