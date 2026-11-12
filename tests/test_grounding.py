from conftest import code, make_nb, md, stream
from tp2te.core.models import Evidence, Issue, QuestionDraft
from tp2te.core.notebook import parse_notebook
from tp2te.core.questions import ground_issue, is_grounded, question_grounded

DOC = parse_notebook(make_nb(
    code("scaler.fit(X)\nX_train, X_test = train_test_split(X,  y)", [stream("accuracy:   0.99\n")]),
    md("We get a  very good\nscore."),
    md("Points obtenus: 3/5"),
), "G01")
DOC.cells[2].hidden = True


def ev(cid, q):
    return Evidence(cell_id=cid, quote=q)


def test_whitespace_differences_pass():
    assert is_grounded([ev("c000", "scaler.fit(X) X_train, X_test = train_test_split(X, y)")], DOC)
    assert is_grounded([ev("c000", "accuracy: 0.99")], DOC)                 # quote from the output
    assert is_grounded([ev("c001", "We get a very good score.")], DOC)


def test_invented_or_misplaced_quotes_fail():
    assert not is_grounded([ev("c000", "scaler.fit(X_train)")], DOC)         # repaired code
    assert not is_grounded([ev("c001", "scaler.fit(X)")], DOC)               # wrong cell
    assert not is_grounded([ev("c099", "scaler.fit(X)")], DOC)               # no such cell
    assert not is_grounded([ev("c002", "Points obtenus")], DOC)              # hidden staff cell
    assert not is_grounded([ev("c000", "  ")], DOC)
    assert not is_grounded([], DOC)                                          # no evidence at all
    assert not is_grounded([ev("c000", "scaler.fit(X)"), ev("c000", "made up")], DOC)   # all must hold


def test_ground_issue_sets_flag():
    i = Issue(section_id="S2", category="methodology", severity="major", concept="data leakage", title="t",
              explanation="e", evidence=[ev("c000", "scaler.fit(X)")])
    assert ground_issue(i, DOC).grounded


def test_question_snippet_must_come_from_evidence_cell():
    q = QuestionDraft(type="spot_issue", section_id="S2", concept="c", difficulty=2, prompt="p",
                      snippet="scaler.fit(X)", evidence=[ev("c000", "scaler.fit(X)")], expected_points=["a", "b"],
                      source_issue_index=None)
    assert question_grounded(q, DOC)
    assert not question_grounded(q.model_copy(update={"snippet": "We get a very good score."}), DOC)
