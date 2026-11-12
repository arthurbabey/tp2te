"""Template sections and alignment of student cells to them (SPEC section 6).

No LLM, no embeddings. Steps:

1. ``build_template``: split the assignment notebook into sections at markdown
   headings (down to ``heading_level``) and at exercise numbers in bold
   ("**3.1)", "<b>1.1)</b>"), and mark task cells, question cells and staff
   cells with the markers of ``TemplateConfig``.
2. ``match_notebook``: align student cells to template cells. The nbformat cell
   id, which survives in most submitted cells, is the strongest anchor; then
   exact text, then fuzzy similarity (full ratio, and line containment in both
   directions for split cells and answers typed into a template cell). The
   alignment is a global monotone one (weighted LCS), so it is order-consistent
   by construction; a cell that matches well but out of order is kept as a
   moved anchor.
3. Every student cell is attributed to a section: matched cells to their
   template cell's section, added cells to the section of the nearest preceding
   matched cell (or the following one when there is none).
4. Merged cells (one student cell containing several template cells) are
   detected and reported; confidence per section is anchors found / anchors
   expected.
"""

import re
from dataclasses import dataclass, field

from rapidfuzz import fuzz

from .config import MatchingConfig, TemplateConfig
from .models import Cell, NotebookDoc, SectionMatch, Template, TemplateSection

_TAG_RE = re.compile(r"<[^>]+>")
_LINK_RE = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
_NUM_TITLE_RE = re.compile(r"^(\d+(?:\.\d+)*)\.?\s+(.*)$")


# ---------- text helpers ----------
def norm(text: str) -> str:
    """Lowercase, whitespace collapsed."""
    return " ".join(text.lower().split())


def norm_lines(text: str) -> list[str]:
    return [n for n in (norm(line) for line in text.splitlines()) if n]


def code_lines(src: str) -> list[str]:
    """Non-empty lines that are not comments."""
    return [s for s in (line.strip() for line in src.splitlines()) if s and not s.startswith("#")]


def clean_title(text: str, limit: int = 80) -> str:
    t = _LINK_RE.sub(r"\1", _TAG_RE.sub(" ", text))
    t = re.sub(r"[*_`#]+", "", t)
    t = " ".join(t.split()).strip(" .:-)")
    return t if len(t) <= limit else t[: limit - 1].rstrip() + "…"


def is_staff(c: Cell, cfg: TemplateConfig) -> bool:
    low = c.source.lower()
    return any(m.lower() in low for m in cfg.staff_markers)


def is_task(c: Cell, cfg: TemplateConfig) -> bool:
    if c.kind != "code":
        return False
    if c.nb_cell_id and c.nb_cell_id in cfg.extra_task_cells:
        return True
    low = c.source.lower()
    if any(m.lower() in low for m in cfg.not_task_markers):
        return False
    return not code_lines(c.source) or any(m.lower() in low for m in cfg.task_markers)


def is_question(c: Cell, cfg: TemplateConfig) -> bool:
    return c.kind == "markdown" and bool(re.search(cfg.answer_placeholder, c.source))


# ---------- 1. template sections ----------
def section_start(c: Cell, cfg: TemplateConfig) -> tuple[str | None, str, int] | None:
    """(number, title, level) when this cell opens a section."""
    if c.kind != "markdown":
        return None
    lines = [line for line in c.source.splitlines() if line.strip()]
    item = re.compile(cfg.item_pattern)
    for line in lines[: cfg.item_max_line]:
        m = item.search(line)
        if m:
            return m.group(1), clean_title(line[m.end():]), cfg.heading_level + 1
    for line in lines:
        h = _HEADING_RE.match(line.strip())
        if h and len(h.group(1)) <= cfg.heading_level:
            title = clean_title(h.group(2))
            n = _NUM_TITLE_RE.match(title)
            return (n.group(1), n.group(2), len(h.group(1))) if n else (None, title, len(h.group(1)))
    return None


def build_template(doc: NotebookDoc, lab_id: str, cfg: TemplateConfig) -> Template:
    cells = [c.model_copy(update={"hidden": is_staff(c, cfg)}) for c in doc.cells]
    sections: list[TemplateSection] = []
    used: set[str] = set()
    for c in cells:
        start = section_start(c, cfg)
        if start is None and not sections:
            start = (None, "Introduction", 1)
        if start is not None:
            num, title, level = start
            sid = f"S{num}" if num else f"S{len(sections)}"
            base, k = sid, 1
            while sid in used:
                k += 1
                sid = f"{base}-{k}"
            used.add(sid)
            sections.append(TemplateSection(section_id=sid, title=title or sid, level=level, cell_ids=[],
                                            task_cell_ids=[], question_cell_ids=[]))
        s = sections[-1]
        s.cell_ids.append(c.cell_id)
        if c.hidden:
            continue
        if is_task(c, cfg):
            s.task_cell_ids.append(c.cell_id)
        elif is_question(c, cfg):
            s.question_cell_ids.append(c.cell_id)
    return Template(lab_id=lab_id, cells=cells, sections=sections)


def gradable_sections(template: Template) -> list[TemplateSection]:
    """Sections with something to do: the ones judged for completeness."""
    return [s for s in template.sections if s.task_cell_ids or s.question_cell_ids]


def template_doc(template: Template) -> NotebookDoc:
    return NotebookDoc(group_id="TEMPLATE", cells=template.cells, content_hash="")


def template_matches(template: Template) -> list[SectionMatch]:
    hidden = {c.cell_id for c in template.cells if c.hidden}
    return [SectionMatch(section_id=s.section_id, cell_ids=[c for c in s.cell_ids if c not in hidden],
                         anchors_found=len(s.cell_ids), anchors_total=len(s.cell_ids), confidence=1.0,
                         method="anchor") for s in template.sections]


# ---------- 2-4. matching ----------
@dataclass
class MatchResult:
    matches: list[SectionMatch]
    cell_section: dict[str, str]                 # student cell id -> section id
    how: dict[str, str] = field(default_factory=dict)   # cell id -> id | exact | fuzzy | moved | merge
    notes: list[str] = field(default_factory=list)


def similarity(s: str, t: str, s_lines: list[str], t_lines: list[str]) -> float:
    """Best of: full similarity (edited cell), share of the student's lines found in
    the template cell (split cell), share of the template's lines found in the
    student cell (answer typed into a template cell)."""
    if not s or not t:
        return 0.0
    best = fuzz.ratio(s, t) / 100
    if s_lines and t_lines:
        tset, sset = set(t_lines), set(s_lines)
        best = max(best, sum(x in tset for x in s_lines) / len(s_lines),
                   sum(x in sset for x in t_lines) / len(t_lines))
    return best


ID_WEIGHT = 2.0


def match_notebook(doc: NotebookDoc, template: Template, mcfg: MatchingConfig,
                   tcfg: TemplateConfig) -> MatchResult:
    """Align ``doc`` to ``template``. Sets origin, template_cell_id and hidden on doc's cells."""
    t, s = template.cells, doc.cells
    tn = [norm(c.source) for c in t]
    sn = [norm(c.source) for c in s]
    tl = [norm_lines(c.source) for c in t]
    sl = [norm_lines(c.source) for c in s]
    t_by_nb = {c.nb_cell_id: j for j, c in enumerate(t) if c.nb_cell_id}

    # weights: id match dominates, then exact/fuzzy similarity above the threshold
    W = [[0.0] * len(t) for _ in s]
    sim = [[0.0] * len(t) for _ in s]
    for i, sc in enumerate(s):
        jid = t_by_nb.get(sc.nb_cell_id) if sc.nb_cell_id else None
        for j, tc in enumerate(t):
            if tc.kind != sc.kind:
                continue
            sim[i][j] = 1.0 if sn[i] and sn[i] == tn[j] else similarity(sn[i], tn[j], sl[i], tl[j])
            if j == jid:
                W[i][j] = ID_WEIGHT + sim[i][j]
            elif sim[i][j] >= mcfg.fuzzy_threshold:
                W[i][j] = sim[i][j]

    # global monotone alignment (weighted LCS)
    n, m = len(s), len(t)
    dp = [[0.0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        row, prev, w = dp[i], dp[i - 1], W[i - 1]
        for j in range(1, m + 1):
            best = max(prev[j], row[j - 1])
            if w[j - 1] > 0:
                best = max(best, prev[j - 1] + w[j - 1])
            row[j] = best
    match: list[int | None] = [None] * n
    i, j = n, m
    while i > 0 and j > 0:
        if W[i - 1][j - 1] > 0 and dp[i][j] == dp[i - 1][j - 1] + W[i - 1][j - 1]:
            match[i - 1] = j - 1
            i, j = i - 1, j - 1
        elif dp[i][j] == dp[i - 1][j]:
            i -= 1
        else:
            j -= 1

    used = {j for j in match if j is not None}
    how: dict[int, str] = {}
    for i, j in enumerate(match):
        if j is not None:
            how[i] = "id" if W[i][j] >= ID_WEIGHT else ("exact" if sn[i] and sn[i] == tn[j] else "fuzzy")
    # moved anchors: out-of-order cells kept when the id matches or the text matches well
    for i in range(n):
        if match[i] is not None:
            continue
        jid = t_by_nb.get(s[i].nb_cell_id) if s[i].nb_cell_id else None
        if jid is not None and jid not in used and t[jid].kind == s[i].kind:
            cand = jid
        else:
            free = [j for j in range(m) if j not in used and sim[i][j] >= mcfg.moved_anchor]
            cand = max(free, key=lambda j: (sim[i][j], -j)) if free else None
        if cand is not None:
            match[i], how[i] = cand, "moved"
            used.add(cand)

    # merges: a student cell containing whole template cells that nothing else matched
    merged_into: dict[int, list[int]] = {}
    for i in range(n):
        if match[i] is not None and sn[i] == tn[match[i]]:
            continue
        hits = [j for j in range(m) if j not in used and t[j].kind == s[i].kind
                and len(tn[j]) >= mcfg.merge_min_chars and tn[j] in sn[i]]
        if hits:
            merged_into[i] = hits
            used.update(hits)

    tsec = {}
    for sec in template.sections:
        for cid in sec.cell_ids:
            tsec[cid] = sec.section_id
    notes: list[str] = []
    found_by_sec: dict[str, set[str]] = {}
    fuzzy_secs: set[str] = set()
    sec_of: list[str | None] = [None] * n
    for i, c in enumerate(s):
        j = match[i]
        hits = merged_into.get(i, [])
        if j is None and hits:
            j_attr = hits[0]
            how[i] = "merge"
        else:
            j_attr = j
        if j is not None:
            c.template_cell_id = t[j].cell_id
            c.origin = "unchanged" if sn[i] == tn[j] else "modified"
            found_by_sec.setdefault(tsec[t[j].cell_id], set()).add(t[j].cell_id)
            if how[i] == "fuzzy":
                fuzzy_secs.add(tsec[t[j].cell_id])
        else:
            c.template_cell_id = None
            c.origin = "added"
        for h in hits:
            found_by_sec.setdefault(tsec[t[h].cell_id], set()).add(t[h].cell_id)
        if hits:
            parts = ([t[j].cell_id] if j is not None else []) + [t[h].cell_id for h in hits]
            notes.append(f"{c.cell_id} merges template cells {', '.join(parts)}")
        c.hidden = is_staff(c, tcfg) or (j is not None and t[j].hidden)
        if j_attr is not None:
            sec_of[i] = tsec[t[j_attr].cell_id]

    # added cells: nearest preceding attributed cell, else the following one
    last = None
    for i in range(n):
        if sec_of[i] is None:
            sec_of[i] = last
        last = sec_of[i] if (match[i] is not None or i in merged_into) else last
    nxt = None
    for i in reversed(range(n)):
        if match[i] is not None or i in merged_into:
            nxt = sec_of[i]
        elif sec_of[i] is None:
            sec_of[i] = nxt

    cell_section = {s[i].cell_id: sec_of[i] for i in range(n) if sec_of[i] is not None}
    hidden_t = {c.cell_id for c in t if c.hidden}
    matches = []
    for sec in template.sections:
        total = [cid for cid in sec.cell_ids if cid not in hidden_t]
        found = found_by_sec.get(sec.section_id, set()) - hidden_t
        cids = [c.cell_id for c in s if cell_section.get(c.cell_id) == sec.section_id and not c.hidden]
        conf = len(found) / len(total) if total else 1.0
        method = "unmatched" if total and not found else ("fuzzy" if sec.section_id in fuzzy_secs else "anchor")
        matches.append(SectionMatch(section_id=sec.section_id, cell_ids=cids, anchors_found=len(found),
                                    anchors_total=len(total), confidence=round(conf, 3), method=method))
    return MatchResult(matches=matches, cell_section=cell_section,
                       how={s[i].cell_id: h for i, h in how.items()}, notes=notes)
