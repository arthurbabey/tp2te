"""Hard checks: one fixture notebook per finding kind (SPEC section 13)."""

import pytest

from conftest import LAB, code, error, filled, lab_cell, lab_template, make_nb, md, stream
from tp2te.core.checks import hard_checks
from tp2te.core.config import Config
from tp2te.core.matching import build_template, match_notebook
from tp2te.core.notebook import parse_notebook

CFG = Config()
TEMPLATE = build_template(parse_notebook(lab_template(), "T"), "LAB9", CFG.template)


def notebook(**override):
    """Complete notebook, exec counts increasing; override[key] = cell, or None to delete."""
    cells, ec = [], 0
    for k, kind, *_ in LAB:
        if k in override:
            if override[k] is not None:
                cells.append(override[k])
            continue
        ec += kind == "code"
        cells.append(lab_cell(k, filled(k), ec=ec))
    return make_nb(*cells)


def findings(nb):
    doc = parse_notebook(nb, "G01")
    res = match_notebook(doc, TEMPLATE, CFG.matching, CFG.template)
    return [(f.kind, f.section_id) for f in hard_checks(doc, TEMPLATE, res, CFG.matching, CFG.template)]


def test_complete_notebook_has_no_finding():
    assert findings(notebook()) == []


def test_empty_cell():
    assert findings(notebook(t11=lab_cell("t11", ec=1))) == [("empty_cell", "S1.1")]


def test_empty_cell_with_only_filler_and_comments():
    assert findings(notebook(t11=lab_cell("t11", "# Votre code ici\n# I will do it later\npass\n", ec=1))) == \
        [("empty_cell", "S1.1")]


def test_task_answered_in_added_cell_below():
    nb = notebook(t11=lab_cell("t11", ec=1))
    nb.cells.insert(5, code("df.head()", execution_count=2))
    assert findings(nb) == []


def test_todo_left():
    cell = lab_cell("t21", "# Votre code ici\n# TODO: fix the seed\nX_train, X_test = split(X)", ec=2)
    assert findings(notebook(t21=cell)) == [("todo_left", "S2.1")]


def test_not_executed():
    assert findings(notebook(t22=lab_cell("t22", filled("t22"), ec=None))) == [("not_executed", "S2.2")]


def test_error_output():
    cell = lab_cell("t22", filled("t22"), ec=3, outputs=[error("NameError", "name 'X_train' is not defined")])
    assert findings(notebook(t22=cell)) == [("error_output", "S2.2")]


@pytest.mark.parametrize("answer", ["*Réponse:*", "*Réponse:*\n...", "*Réponse:*\n- ...\n- ok"])
def test_unanswered_question(answer):
    assert findings(notebook(a23=lab_cell("a23", answer))) == [("unanswered_question", "S2.3")]


def test_answer_inline_or_in_added_cell_counts():
    inline = lab_cell("a23", "*Réponse:* Because the model has already seen the training rows.")
    assert findings(notebook(a23=inline)) == []
    nb = notebook(a23=lab_cell("a23"))
    nb.cells.insert(14, md("Because the training score is optimistic, the model saw those rows."))
    assert findings(nb) == []


def test_unanswered_question_cell_deleted():
    assert findings(notebook(a23=None)) == [("unanswered_question", "S2.3")]


def test_missing_section():
    assert findings(notebook(q31=None)) == [("missing_section", "S3.1")]


def test_out_of_order():
    nb = notebook()
    codes = [c for c in nb.cells if c.cell_type == "code"]
    codes[2].execution_count, codes[3].execution_count = 9, 4      # t21 run after t22
    assert findings(nb) == [("out_of_order", "S2.2")]


def test_staff_cell_is_not_checked():
    staff = lab_cell("staff1", "Points obtenus: 2/5\n```python\nbroken(\n```")
    assert findings(notebook(staff1=staff)) == []


def test_stream_output_is_fine():
    cell = lab_cell("t11", filled("t11"), ec=1, outputs=[stream("   a  b\n")])
    assert findings(notebook(t11=cell)) == []
