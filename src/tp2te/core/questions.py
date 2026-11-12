"""Questions: grounding check now (M2, used on L1 issues); L2 and L3 come in M3.

Grounding (SPEC section 7): every Evidence quote must appear in the named
cell's source or output after whitespace normalisation. For a question, the
snippet must also appear in one of its evidence cells. Items that fail are kept
but marked ``grounded = False``.
"""

from .models import Cell, Evidence, Issue, NotebookDoc, QuestionDraft


def ws(text: str) -> str:
    return " ".join(text.split())


def evidence_ok(ev: Evidence, cells: dict[str, Cell]) -> bool:
    c = cells.get(ev.cell_id)
    q = ws(ev.quote)
    return bool(c is not None and not c.hidden and q and (q in ws(c.source) or q in ws(c.outputs_text)))


def _cells(doc: NotebookDoc) -> dict[str, Cell]:
    return {c.cell_id: c for c in doc.cells}


def is_grounded(evidence: list[Evidence], doc: NotebookDoc) -> bool:
    cells = _cells(doc)
    return bool(evidence) and all(evidence_ok(e, cells) for e in evidence)


def ground_issue(issue: Issue, doc: NotebookDoc) -> Issue:
    issue.grounded = is_grounded(issue.evidence, doc)
    return issue


def question_grounded(q: QuestionDraft, doc: NotebookDoc) -> bool:
    if not is_grounded(q.evidence, doc):
        return False
    if q.snippet:
        cells = _cells(doc)
        s = ws(q.snippet)
        return any(s in ws(cells[e.cell_id].source) or s in ws(cells[e.cell_id].outputs_text)
                   for e in q.evidence)
    return True
