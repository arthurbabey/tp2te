"""Hard completeness checks on a matched notebook (SPEC section 6, last paragraph).

Run after ``matching.match_notebook``. Findings are hints for the L1 call and
for the teacher; the LLM may overrule them.

Answer detection is by subtraction: the template's own text is removed from the
student cell, together with any added cells that follow it (students often
answer in a new cell below), and what remains is the student's work.
"""

import re

from .config import MatchingConfig, TemplateConfig
from .matching import MatchResult, code_lines, gradable_sections, norm, norm_lines
from .models import Cell, HardFinding, NotebookDoc, Template

TODO_RE = re.compile(r"\bTODO\b|^\s*\.\.\.\s*$", re.M)
_TAG_RE = re.compile(r"<[^>]+>")
_FILLER = {"pass", "...", "…"}
_INLINE_PLACEHOLDER = re.compile(r"\*[^*\n]{1,60}:\s*\*")      # "*Réponse:*" inside a line


def _with_followers(doc: NotebookDoc, template_cell_id: str) -> tuple[list[Cell], bool]:
    """Student cells matched to a template cell, plus the added cells right after
    each of them. The bool says whether the template cell was found at all."""
    out, found = [], False
    cells = doc.cells
    for i, c in enumerate(cells):
        if c.template_cell_id != template_cell_id:
            continue
        found = True
        out.append(c)
        for f in cells[i + 1:]:
            if f.origin != "added" or f.hidden:
                break
            out.append(f)
    return out, found


def task_residual(cells: list[Cell], template_cell: Cell) -> list[str]:
    """Code lines the student wrote: not comments, not template lines, not filler."""
    tset = {norm(x) for x in code_lines(template_cell.source)}
    return [x for c in cells if c.kind == "code" for x in code_lines(c.source)
            if norm(x) not in tset and x not in _FILLER]


def answer_residual(cells: list[Cell], template_cell: Cell) -> str:
    """Text the student added to a question: template lines and placeholders removed."""
    tset = set(norm_lines(template_cell.source))
    kept = []
    for c in cells:
        for line in c.source.splitlines():
            if norm(line) in tset:
                continue
            t = _INLINE_PLACEHOLDER.sub(" ", line) if c.kind == "markdown" else line
            t = _TAG_RE.sub(" ", t)
            t = re.sub(r"[*_`#>|-]+|\.\.\.|…", " ", t)
            kept.append(" ".join(t.split()))
    return " ".join(k for k in kept if k)


def hard_checks(doc: NotebookDoc, template: Template, result: MatchResult,
                mcfg: MatchingConfig, tcfg: TemplateConfig) -> list[HardFinding]:
    tcells = {c.cell_id: c for c in template.cells}
    sec = result.cell_section
    findings: list[HardFinding] = []
    empty_task_cells: set[str] = set()
    by_section = {m.section_id: m for m in result.matches}

    for s in gradable_sections(template):
        m = by_section.get(s.section_id)
        if m is None or not m.cell_ids or m.anchors_found == 0:
            findings.append(HardFinding(kind="missing_section", section_id=s.section_id, cell_id=None,
                                        detail=f"no student cell found for section {s.section_id} ({s.title})"))
            continue
        for tid in s.task_cell_ids:
            cells, found = _with_followers(doc, tid)
            residual = task_residual(cells, tcells[tid])
            if not residual:
                empty_task_cells.update(c.cell_id for c in cells)
                findings.append(HardFinding(
                    kind="empty_cell", section_id=s.section_id, cell_id=cells[0].cell_id if cells else None,
                    detail="task cell left as in the template" if found else
                    f"task cell (template {tid}) not found in the notebook"))
                continue
            for c in cells:
                if c.kind == "code" and TODO_RE.search(c.source):
                    findings.append(HardFinding(kind="todo_left", section_id=s.section_id, cell_id=c.cell_id,
                                                detail="placeholder (TODO or ...) left in a task cell"))
        for qid in s.question_cell_ids:
            cells, found = _with_followers(doc, qid)
            answer = answer_residual(cells, tcells[qid])
            if len(answer) < mcfg.min_answer_chars:
                findings.append(HardFinding(
                    kind="unanswered_question", section_id=s.section_id,
                    cell_id=cells[0].cell_id if cells else None,
                    detail=(f"{len(answer)} characters of answer" if found else
                            f"question cell (template {qid}) not found in the notebook")))

    max_ec, max_cell = None, None
    for c in doc.cells:
        if c.kind != "code" or c.hidden:
            continue
        sid = sec.get(c.cell_id)
        if c.error_name:
            findings.append(HardFinding(kind="error_output", section_id=sid, cell_id=c.cell_id,
                                        detail=f"output contains {c.error_name}"))
        if code_lines(c.source) and c.execution_count is None and c.cell_id not in empty_task_cells:
            findings.append(HardFinding(kind="not_executed", section_id=sid, cell_id=c.cell_id,
                                        detail="code cell was not run"))
        if c.execution_count is not None:
            if max_ec is not None and c.execution_count < max_ec:
                findings.append(HardFinding(kind="out_of_order", section_id=sid, cell_id=c.cell_id,
                                            detail=f"exec {c.execution_count} after exec {max_ec} ({max_cell}): "
                                                   "outputs may not match the code shown"))
            if max_ec is None or c.execution_count > max_ec:
                max_ec, max_cell = c.execution_count, c.cell_id
    return findings
