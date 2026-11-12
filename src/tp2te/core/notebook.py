"""nbformat -> NotebookDoc, output truncation, prompt serialisation.

Notebooks are read as data only; nothing here executes a cell. Only cell
sources and text outputs are kept: notebook metadata, cell metadata and
attachments are dropped at parse time, so they never reach a prompt or the
run file.
"""

import hashlib
import json
import re
from collections.abc import Sequence

import nbformat

from .models import Cell, NotebookDoc, SectionMatch, Template
from .scrub import Scrubber

IMAGE_MARKER = "[image output]"
ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
TAG_RE = re.compile(r"<[^>]+>")


class NotebookParseError(ValueError):
    pass


def read_notebook(data: bytes | str) -> nbformat.NotebookNode:
    if isinstance(data, bytes):
        data = data.decode("utf-8-sig")
    try:
        return nbformat.reads(data, as_version=4)
    except Exception as e:  # nbformat raises several unrelated types
        raise NotebookParseError(f"{type(e).__name__}: {e}") from e


def truncate(text: str, cap: int) -> str:
    """Keep head and tail of ``text`` so the total stays near ``cap`` characters."""
    if cap <= 0 or len(text) <= cap:
        return text
    half = cap // 2
    cut = len(text) - 2 * half
    return f"{text[:half]}\n[... {cut} characters cut ...]\n{text[-half:]}"


def _joined(v) -> str:
    return "".join(v) if isinstance(v, list) else (v or "")


def _outputs(cell) -> tuple[str, bool, str | None]:
    parts: list[str] = []
    has_image = False
    error_name = None
    for o in cell.get("outputs", []):
        kind = o.get("output_type")
        if kind == "stream":
            parts.append(_joined(o.get("text")))
        elif kind in ("execute_result", "display_data"):
            data = o.get("data", {})
            if any(k.startswith("image/") for k in data):
                # The text/plain of a figure is only "<Figure size ...>"; the
                # serialiser writes IMAGE_MARKER instead.
                has_image = True
                continue
            if "text/plain" in data:
                # DataFrames: text/plain is the plain-text form of the HTML table.
                parts.append(_joined(data["text/plain"]))
            elif "text/html" in data:
                parts.append(TAG_RE.sub("", _joined(data["text/html"])))
        elif kind == "error":
            error_name = o.get("ename") or "Error"
            tb = o.get("traceback") or []
            parts.append("\n".join(tb) if tb else f"{error_name}: {o.get('evalue', '')}")
    text = "\n".join(p.rstrip("\n") for p in parts if p)
    return ANSI_RE.sub("", text), has_image, error_name


def parse_notebook(nb: nbformat.NotebookNode, group_id: str, *,
                   scrubber: Scrubber | None = None, output_cap: int = 1500) -> NotebookDoc:
    clean = scrubber.scrub if scrubber else (lambda s: s)
    cells: list[Cell] = []
    for i, c in enumerate(nb.get("cells", [])):
        is_code = c.get("cell_type") == "code"
        out_text, has_image, error_name = _outputs(c) if is_code else ("", False, None)
        cells.append(Cell(
            cell_id=f"c{i:03d}",
            idx=i,
            kind="code" if is_code else "markdown",   # raw cells are treated as markdown
            source=clean(_joined(c.get("source"))),
            outputs_text=truncate(clean(out_text), output_cap),
            has_image=has_image,
            error_name=error_name,
            execution_count=c.get("execution_count") if is_code else None,
            nb_cell_id=c.get("id"),
        ))
    return NotebookDoc(group_id=group_id, cells=cells, content_hash=content_hash(cells))


def content_hash(cells: Sequence[Cell]) -> str:
    payload = [[c.kind, c.source, c.outputs_text, c.has_image, c.error_name, c.execution_count]
               for c in cells]
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False).encode()).hexdigest()


# ---------- prompt serialisation (SPEC section 7) ----------
def _attr(v: str) -> str:
    return v.replace("&", "&amp;").replace('"', "&quot;")


def serialise_cell(c: Cell) -> str:
    attrs = f'id="{c.cell_id}" type="{c.kind}" origin="{c.origin}"'
    if c.kind == "code" and c.execution_count is not None:
        attrs += f' exec="{c.execution_count}"'
    lines = [f"<cell {attrs}>", c.source.rstrip("\n")]
    if c.outputs_text or c.has_image:
        lines.append("<output>")
        if c.outputs_text:
            lines.append(c.outputs_text.rstrip("\n"))
        if c.has_image:
            lines.append(IMAGE_MARKER)
        lines.append("</output>")
    lines.append("</cell>")
    return "\n".join(lines)


def serialise_notebook(doc: NotebookDoc, matches: Sequence[SectionMatch] | None = None,
                       template: Template | None = None) -> str:
    """Whole notebook as prompt text. With matches, cells are wrapped in sections;
    cells no match claims go into a trailing "unmatched" section so no student work is lost.
    Hidden (staff) cells are never serialised."""
    by_id = {c.cell_id: c for c in doc.cells if not c.hidden}
    if not matches:
        return "\n".join(serialise_cell(c) for c in doc.cells if not c.hidden)
    titles = {s.section_id: s.title for s in template.sections} if template else {}
    out, seen = [], set()
    for m in matches:
        out.append(f'<section id="{m.section_id}" title="{_attr(titles.get(m.section_id, ""))}" '
                   f'match_confidence="{m.confidence:.2f}">')
        for cid in m.cell_ids:
            if cid in by_id:
                out.append(serialise_cell(by_id[cid]))
                seen.add(cid)
        out.append("</section>")
    rest = [c for c in doc.cells if c.cell_id not in seen and not c.hidden]
    if rest:
        out.append('<section id="unmatched" title="" match_confidence="0.00">')
        out.extend(serialise_cell(c) for c in rest)
        out.append("</section>")
    return "\n".join(out)
