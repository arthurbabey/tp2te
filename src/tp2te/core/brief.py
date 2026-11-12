"""L0: lab brief and generic question pool, generated once from the template."""

from collections.abc import Iterable
from pathlib import Path

from ..llm.base import LLMClient
from .config import Config
from .matching import gradable_sections, match_notebook, template_doc, template_matches
from .models import LabBrief, NotebookDoc, Template
from .notebook import serialise_notebook
from .prompts import call_llm


def serialise_template(template: Template) -> str:
    return serialise_notebook(template_doc(template), template_matches(template), template)


def serialise_solution(solution: NotebookDoc, template: Template, cfg: Config) -> str:
    """Solution notebook aligned to the template's sections. Mutates ``solution``'s cells."""
    res = match_notebook(solution, template, cfg.matching, cfg.template)
    return serialise_notebook(solution, res.matches, template)


def solution_block(solution_serialised: str | None) -> str:
    if not solution_serialised:
        return ""
    return ("\nThe teacher's solution notebook follows, for reference only. Student work that differs "
            "from it can still be correct.\n<solution>\n" + solution_serialised + "\n</solution>\n")


def system_params(cfg: Config) -> dict:
    return {"teacher_language": cfg.languages.teacher, "quiz_language": cfg.languages.quiz}


def generate_brief(client: LLMClient, template: Template, cfg: Config, *, solution_serialised: str | None = None,
                   log_dir: str | Path | None = None, guard_names: Iterable[str] = ()) -> LabBrief:
    ids = [s.section_id for s in gradable_sections(template)]
    res = call_llm(client, call="L0", prompt="l0_brief", schema=LabBrief, log_dir=log_dir,
                   guard_names=guard_names, system_params=system_params(cfg),
                   temperature=cfg.llm.temperature, max_tokens=cfg.llm.max_tokens,
                   params={"lab_id": template.lab_id, "solution_block": solution_block(solution_serialised),
                           "section_ids": ", ".join(ids), "n_generic": cfg.analysis.n_generic,
                           "template_serialised": serialise_template(template)})
    brief = res.parsed
    order = {sid: k for k, sid in enumerate(ids)}
    brief.sections = sorted((s for s in brief.sections if s.section_id in order),
                            key=lambda s: order[s.section_id])
    brief.generic_questions = [q.model_copy(update={"type": "concept", "snippet": None, "evidence": [],
                                                    "source_issue_index": None})
                               for q in brief.generic_questions]
    return brief


def brief_for_prompt(brief: LabBrief) -> str:
    """Sections only: the generic questions are not context for L1/L2."""
    return brief.model_dump_json(include={"sections"}, indent=1)

