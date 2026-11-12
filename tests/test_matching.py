"""Section matching. Scripted perturbations of a template (SPEC section 13), each
run with nbformat ids kept (the normal case) and with ids regenerated (copied
cells, other editors), which exercises the text-only path."""

import pytest

from conftest import FILLED, LAB, REPO, code, filled, lab_cell, lab_template, make_nb, md
from labquiz.core.config import Config
from labquiz.core.matching import build_template, gradable_sections, match_notebook, section_start
from labquiz.core.notebook import parse_notebook, read_notebook, serialise_notebook

CFG = Config()
SECTION = {k: s for k, _, _, s in LAB}


def template():
    return build_template(parse_notebook(lab_template(), "T"), "LAB9", CFG.template)


def run_match(cells_with_truth, keep_ids):
    """cells_with_truth: list of (nbformat cell, expected section)."""
    cells = []
    for k, (c, _) in enumerate(cells_with_truth):
        if not keep_ids:
            c["id"] = f"regen{k:03d}"
        cells.append(c)
    doc = parse_notebook(make_nb(*cells), "G01")
    res = match_notebook(doc, template(), CFG.matching, CFG.template)
    got = [res.cell_section.get(c.cell_id) for c in doc.cells]
    want = [s for _, s in cells_with_truth]
    return doc, res, got, want


def complete_notebook():
    return [(lab_cell(k, filled(k)), SECTION[k]) for k, *_ in LAB]


def by_key(cells, key):
    return next(i for i, (c, _) in enumerate(cells) if c["id"] == key)


# ---------- template ----------
def test_template_sections_tasks_questions():
    t = template()
    assert [s.section_id for s in t.sections] == ["S0", "S1", "S1.1", "S1.2", "S2", "S2.1", "S2.2", "S2.3",
                                                 "S3", "S3.1"]
    by = {s.section_id: s for s in t.sections}
    assert by["S2.2"].title.startswith("Fit a logistic regression")          # <b>2.2)</b> form
    assert by["S1"].title == "Loading data" and by["S1"].level == 2 and by["S1.1"].level == 3
    assert by["S1.1"].task_cell_ids == ["c004"] and by["S1.2"].question_cell_ids == ["c005"]
    assert by["S2.3"].question_cell_ids == ["c013"]
    assert [c.cell_id for c in t.cells if c.hidden] == ["c006"]              # staff grading cell
    assert [s.section_id for s in gradable_sections(t)] == ["S1.1", "S1.2", "S2.1", "S2.2", "S2.3", "S3.1"]


def test_section_start_ignores_numbers_further_down_and_toc_links():
    cfg = CFG.template
    deep = parse_notebook(make_nb(md("a\nb\nc\nd\ne\nf\n**4.1) late**")), "T").cells[0]
    toc = parse_notebook(make_nb(md("[- 4.1: Moyennes](#Part41)")), "T").cells[0]
    assert section_start(deep, cfg) is None and section_start(toc, cfg) is None


def test_task_markers():
    nb = make_nb(md("## 1. A"), code("# Testez votre code ici. Cette cellule n'est pas corrigée\n(df)"),
                 code("x = 1  # given"), code(""), code("ax = (df\n  # Complétez ici\n)"))
    t = build_template(parse_notebook(nb, "T"), "L", CFG.template)
    assert t.sections[0].task_cell_ids == ["c003", "c004"]


# ---------- perturbations ----------
@pytest.mark.parametrize("keep_ids", [True, False])
def test_complete_notebook(keep_ids):
    doc, res, got, want = run_match(complete_notebook(), keep_ids)
    assert got == want
    assert all(m.confidence == 1.0 for m in res.matches)
    origins = {c.source.split("\n")[0][:12]: c.origin for c in doc.cells}
    assert origins["# Lab 9"] == "unchanged" and origins["# Votre code"] == "modified"


@pytest.mark.parametrize("keep_ids", [True, False])
def test_move_a_cell(keep_ids):
    cells = complete_notebook()
    moved = cells.pop(by_key(cells, "t22"))
    cells.insert(by_key(cells, "a23") + 1, moved)            # task 2.2 written after the 2.3 answer
    _, res, got, want = run_match(cells, keep_ids)
    assert got == want
    assert "moved" in res.how.values()


@pytest.mark.parametrize("keep_ids", [True, False])
def test_split_a_cell(keep_ids):
    cells = complete_notebook()
    i = by_key(cells, "q12")
    cells[i:i + 1] = [(lab_cell("q12", "**1.2) Describe the columns of the dataset in full sentences.**"), "S1.2"),
                      (md("*Réponse:*\n" + FILLED["q12"]), "S1.2")]
    _, _, got, want = run_match(cells, keep_ids)
    assert got == want


@pytest.mark.parametrize("keep_ids", [True, False])
def test_merge_two_cells(keep_ids):
    cells = complete_notebook()
    i = by_key(cells, "q22")
    merged = LAB[10][2] + "\n\n" + LAB[12][2]                # 2.2 and 2.3 statements in one cell
    del cells[by_key(cells, "q23")]
    cells[i] = (lab_cell("q22", merged), "S2.2")
    _, res, got, want = run_match(cells, keep_ids)
    assert got == want
    assert any("merges" in n for n in res.notes)
    s23 = next(m for m in res.matches if m.section_id == "S2.3")
    assert s23.anchors_found == 2                            # the merged statement counts as found


@pytest.mark.parametrize("keep_ids", [True, False])
def test_delete_a_heading(keep_ids):
    cells = complete_notebook()
    del cells[by_key(cells, "h2")]
    _, res, got, want = run_match(cells, keep_ids)
    assert got == want
    s2 = next(m for m in res.matches if m.section_id == "S2")
    assert s2.method == "unmatched" and s2.cell_ids == []


@pytest.mark.parametrize("keep_ids", [True, False])
def test_rename_a_heading(keep_ids):
    cells = complete_notebook()
    i = by_key(cells, "h2")
    cells[i] = (lab_cell("h2", "## 2. Modelling with scikit-learn\n\nWe train a classifier on the data."), "S2")
    doc, _, got, want = run_match(cells, keep_ids)
    assert got == want
    assert doc.cells[i].origin == "modified"


@pytest.mark.parametrize("keep_ids", [True, False])
def test_add_ten_cells(keep_ids):
    cells = complete_notebook()
    i = by_key(cells, "t22") + 1
    extra = []
    for k in range(10):
        extra.append((code(f"model_{k} = make_model(depth={k})\nscores_{k} = cv(model_{k})") if k % 2 else
                      md(f"Trying variant number {k} of our pipeline, with different settings."), "S2.2"))
    cells[i:i] = extra
    doc, _, got, want = run_match(cells, keep_ids)
    assert got == want
    assert sum(c.origin == "added" for c in doc.cells) == 10


@pytest.mark.parametrize("keep_ids", [True, False])
def test_low_confidence_when_section_deleted(keep_ids):
    cells = [x for x in complete_notebook() if x[1] != "S1.1"]
    _, res, got, want = run_match(cells, keep_ids)
    assert got == want
    s11 = next(m for m in res.matches if m.section_id == "S1.1")
    assert s11.confidence == 0.0 and s11.method == "unmatched"


def test_staff_cells_hidden_and_not_serialised():
    cells = complete_notebook()
    i = by_key(cells, "staff1")
    cells[i] = (lab_cell("staff1", "<b>Corrections Partie 1: </b> Points obtenus: 4/5\n\nRemarques: info() manque"),
                "S1.2")
    doc, res, _, _ = run_match(cells, True)
    assert doc.cells[i].hidden
    text = serialise_notebook(doc, res.matches, template())
    assert "Points obtenus" not in text and "info() manque" not in text


# ---------- real template and notebooks (local only) ----------
REAL_T = REPO / "data" / "TP2_ISD_SA2025.ipynb"
REAL_S = REPO / "data" / "students"


@pytest.mark.skipif(not REAL_T.is_file(), reason="real template not present (data/ is local only)")
def test_real_template_and_notebooks():
    cfg = Config()
    t = build_template(parse_notebook(read_notebook(REAL_T.read_bytes()), "T"), "TP2", cfg.template)
    ids = [s.section_id for s in gradable_sections(t)]
    assert ids == ["S1.1", "S1.2", "S1.3", "S2.2", "S2.3", "S3.1", "S3.2", "S3.3", "S3.5", "S4.1", "S4.2",
                   "S4.3", "S4.4", "S4.5", "S4.6", "S4.8", "S4.9"]
    assert sum(c.hidden for c in t.cells) == 7
    tsec = {cid: s.section_id for s in t.sections for cid in s.cell_ids}
    t_by_nb = {c.nb_cell_id: c.cell_id for c in t.cells}
    for p in sorted(REAL_S.glob("*.ipynb")):
        doc = parse_notebook(read_notebook(p.read_bytes()), "G")
        res = match_notebook(doc, t, cfg.matching, cfg.template)
        for c in doc.cells:                     # every cell that kept a template id lands in its section
            if c.nb_cell_id in t_by_nb:
                assert res.cell_section[c.cell_id] == tsec[t_by_nb[c.nb_cell_id]], (p.name, c.cell_id)
        assert "Points obtenus" not in serialise_notebook(doc, res.matches, t), p.name
