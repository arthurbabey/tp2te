import nbformat
import pytest
from nbformat.v4 import new_raw_cell

from conftest import code, error, image, make_nb, md, result, stream
from tp2te.core.models import SectionMatch, Template, TemplateSection
from tp2te.core.notebook import (IMAGE_MARKER, NotebookParseError, parse_notebook, read_notebook,
                                   serialise_notebook, truncate)


def test_truncate_keeps_head_and_tail():
    s = "H" * 1000 + "M" * 3000 + "T" * 1000
    out = truncate(s, 1500)
    assert out.startswith("H" * 750) and out.endswith("T" * 750)
    assert "characters cut" in out and len(out) < 1600
    assert truncate("short", 1500) == "short"


def test_outputs():
    nb = make_nb(
        md("# Title"),
        code("print(1)", [stream("1\n"), stream("2\n")], execution_count=3),
        code("df", [result("   a\n0  1", "<table><tr><td>1</td></tr></table>")], execution_count=4),
        code("df.hist()", [image()], execution_count=5),
        code("1/0", [error("ZeroDivisionError", "division by zero",
                           ["\x1b[0;31mZeroDivisionError\x1b[0m: division by zero"])], execution_count=6),
        code("x = 1", execution_count=None),
        new_raw_cell("raw text"),
    )
    doc = parse_notebook(nb, "G01")
    c = doc.cells
    assert [x.cell_id for x in c] == ["c000", "c001", "c002", "c003", "c004", "c005", "c006"]
    assert [x.kind for x in c] == ["markdown", "code", "code", "code", "code", "code", "markdown"]
    assert c[1].outputs_text == "1\n2" and c[1].execution_count == 3
    assert c[2].outputs_text == "   a\n0  1"                       # plain text, not HTML
    assert c[3].has_image and "<Figure" not in c[3].outputs_text
    assert c[4].error_name == "ZeroDivisionError"
    assert c[4].outputs_text == "ZeroDivisionError: division by zero"   # ANSI stripped
    assert c[5].execution_count is None
    assert all(x.nb_cell_id for x in c)                               # nbformat ids kept


def test_output_cap():
    nb = make_nb(code("print(x)", [stream("x" * 5000)]))
    assert len(parse_notebook(nb, "G01", output_cap=200).cells[0].outputs_text) < 260


def test_content_hash():
    a = parse_notebook(make_nb(code("x = 1")), "G01")
    b = parse_notebook(make_nb(code("x = 1")), "G02")
    c = parse_notebook(make_nb(code("x = 2")), "G01")
    assert a.content_hash == b.content_hash != c.content_hash


def test_read_notebook_rejects_garbage():
    with pytest.raises(NotebookParseError):
        read_notebook(b"{not json")
    assert read_notebook(nbformat.writes(make_nb(md("x"))).encode("utf-8-sig")).cells[0].source == "x"


def test_serialise_flat_matches_spec_format():
    nb = make_nb(code("X_train, X_test = split(X)", [stream("(800, 12) (200, 12)\n"), image()], execution_count=12),
                 md("Our answer"))
    text = serialise_notebook(parse_notebook(nb, "G01"))
    assert text == (
        '<cell id="c000" type="code" origin="added" exec="12">\n'
        "X_train, X_test = split(X)\n<output>\n(800, 12) (200, 12)\n" + IMAGE_MARKER + "\n</output>\n</cell>\n"
        '<cell id="c001" type="markdown" origin="added">\nOur answer\n</cell>'
    )


def test_serialise_with_sections_keeps_unmatched_cells():
    doc = parse_notebook(make_nb(md("## Split"), code("a"), md("stray")), "G01")
    tpl = Template(lab_id="TP2", cells=[], sections=[
        TemplateSection(section_id="S3", title='Train/test "split"', level=2, cell_ids=[],
                        task_cell_ids=[], question_cell_ids=[])])
    m = [SectionMatch(section_id="S3", cell_ids=["c000", "c001"], anchors_found=1, anchors_total=1,
                      confidence=0.9, method="anchor")]
    text = serialise_notebook(doc, m, tpl)
    assert text.startswith('<section id="S3" title="Train/test &quot;split&quot;" match_confidence="0.90">')
    assert '<section id="unmatched"' in text and "stray" in text
