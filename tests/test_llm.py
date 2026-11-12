"""LLM layer and L0/L1, with FakeClient and a fake HTTP endpoint. No network."""

import io
import json
import re
import urllib.error

import pytest

from conftest import LAB, filled, lab_cell, lab_template, make_nb, md, nb_bytes
from labquiz import cli
from labquiz.core import store
from labquiz.core.analysis import analyse_group, build_completeness
from labquiz.core.brief import generate_brief
from labquiz.core.checks import hard_checks
from labquiz.core.config import Config
from labquiz.core.matching import build_template, match_notebook
from labquiz.core.models import (Evidence, GroupAnalysis, GroupRun, Issue, LabBrief, QuestionDraft, SectionBrief,
                                 SectionJudgment)
from labquiz.core.notebook import parse_notebook
from labquiz.core.prompts import PrivacyError, call_llm, versions
from labquiz.core.scrub import find_leaks
from labquiz.llm import openai_compat
from labquiz.llm.base import LLMError, make_client, prompt_key
from labquiz.llm.fake import CachingClient, FakeClient
from labquiz.llm.openai_compat import OpenAICompatClient

CFG = Config()
TEMPLATE = build_template(parse_notebook(lab_template(), "T"), "LAB9", CFG.template)


# ---------- OpenAI-compatible client over a fake HTTP endpoint ----------
class FakeHTTP:
    def __init__(self, replies):
        self.replies = list(replies)      # dict (JSON body) or int (HTTP error code)
        self.bodies = []

    def __call__(self, req, timeout=None):
        self.bodies.append(json.loads(req.data))
        self.headers = dict(req.header_items())
        r = self.replies.pop(0)
        if isinstance(r, int):
            raise urllib.error.HTTPError(req.full_url, r, "err", {}, io.BytesIO(b"slow down"))
        return io.BytesIO(json.dumps(r).encode())


def reply(content, pt=10, ct=5, finish="stop"):
    return {"model": "m-1", "choices": [{"message": {"content": content}, "finish_reason": finish}],
            "usage": {"prompt_tokens": pt, "completion_tokens": ct}}


GOOD = SectionJudgment(section_id="S1.1", status="done", reason="ok").model_dump_json()


@pytest.fixture
def http(monkeypatch):
    def install(*replies):
        h = FakeHTTP(replies)
        monkeypatch.setattr(openai_compat.urllib.request, "urlopen", h)
        monkeypatch.setattr(openai_compat.time, "sleep", lambda s: None)
        return h
    return install


def client(**kw):
    return OpenAICompatClient("https://llm.example/v1/", "m", "sk-test", **kw)


def test_openai_compat_json_schema(http):
    h = http(reply(GOOD))
    r = client().complete(system="sys", user="usr", schema=SectionJudgment)
    assert r.parsed.status == "done" and (r.input_tokens, r.output_tokens, r.model) == (10, 5, "m-1")
    body = h.bodies[0]
    assert body["response_format"]["type"] == "json_schema"
    assert body["response_format"]["json_schema"]["schema"]["title"] == "SectionJudgment"
    assert body["messages"][0] == {"role": "system", "content": "sys"}
    assert h.headers["Authorization"] == "Bearer sk-test"


def test_openai_compat_retries_once_on_validation_error(http):
    h = http(reply('{"section_id": "S1.1", "status": "finished"}'), reply("```json\n" + GOOD + "\n```"))
    r = client().complete(system="s", user="u", schema=SectionJudgment)
    assert r.parsed.status == "done" and r.input_tokens == 20
    assert "did not validate" in h.bodies[1]["messages"][-1]["content"]
    assert h.bodies[1]["messages"][-2]["role"] == "assistant"


def test_openai_compat_gives_up_after_retry(http):
    http(reply("not json"), reply("still not json", finish="length"))
    with pytest.raises(LLMError, match="after a retry .output cut at max_tokens"):
        client().complete(system="s", user="u", schema=SectionJudgment)


def test_openai_compat_prompt_mode_for_local_models(http):
    h = http(reply("Here you go:\n" + GOOD + "\nHope it helps"))
    r = client(structured="prompt").complete(system="s", user="u", schema=SectionJudgment)
    assert r.parsed.reason == "ok"
    assert "response_format" not in h.bodies[0] and "JSON schema" in h.bodies[0]["messages"][0]["content"]


def test_openai_compat_http_retry_and_error(http):
    http(429, reply(GOOD))
    assert client().complete(system="s", user="u", schema=SectionJudgment).parsed.status == "done"
    http(401)
    with pytest.raises(LLMError, match="HTTP 401"):
        client().complete(system="s", user="u", schema=SectionJudgment)


def test_unconfigured_client_is_refused():
    with pytest.raises(LLMError, match="base_url"):
        make_client(Config().llm)                                     # config defaults: nothing set
    with pytest.raises(LLMError, match="unknown LLM provider"):
        make_client(Config(llm={"provider": "nope"}).llm)


# ---------- FakeClient and cache ----------
def test_fake_client_replays_by_prompt_key(tmp_path):
    key = prompt_key("s", "u", SectionJudgment, "fake")
    assert FakeClient({key: GOOD}).complete(system="s", user="u", schema=SectionJudgment).parsed.reason == "ok"
    (tmp_path / f"{key}.json").write_text(json.dumps({"raw": GOOD}))
    assert FakeClient(tmp_path).complete(system="s", user="u", schema=SectionJudgment).parsed.status == "done"
    with pytest.raises(LLMError, match="no recorded response"):
        FakeClient({}).complete(system="s", user="other", schema=SectionJudgment)


def test_caching_client_records_then_replays(tmp_path):
    inner = FakeClient(responder=lambda schema, s, u: GOOD)
    c = CachingClient(inner, tmp_path / "cache")
    a = c.complete(system="s", user="u", schema=SectionJudgment)
    b = c.complete(system="s", user="u", schema=SectionJudgment)
    assert not a.cached and b.cached and a.parsed == b.parsed and len(inner.calls) == 1
    # the recording is what FakeClient replays
    assert FakeClient(tmp_path / "cache").complete(system="s", user="u", schema=SectionJudgment).parsed == a.parsed


# ---------- call_llm: privacy guard and log ----------
def test_privacy_guard_blocks_and_logs(tmp_path):
    fake = FakeClient(responder=lambda *a: GOOD)
    with pytest.raises(PrivacyError):
        call_llm(fake, call="L1", prompt="l1_analysis", schema=GroupAnalysis, log_dir=tmp_path,
                 guard_names=["Zoé Mürner-Kahl"], system_params={"teacher_language": "fr", "quiz_language": "fr"},
                 params={"solution_block": "", "max_issues": 3, "max_choices": 2, "brief_json": "{}",
                         "hard_findings": "", "notebook_serialised": "<cell>by zoe murner</cell>"})
    assert fake.calls == []                                            # nothing was sent
    assert "privacy guard" in store.read_llm_log(tmp_path)[0]["error"]


def test_prompt_versions_are_file_hashes():
    v = versions()
    assert set(v) >= {"system", "l0_brief", "l1_analysis"} and all(len(x) == 8 for x in v.values())


# ---------- L0 and L1 ----------
def ev(cid, q):
    return Evidence(cell_id=cid, quote=q)


def test_generate_brief_filters_and_normalises(tmp_path):
    out = LabBrief(
        sections=[SectionBrief(section_id=s, concepts=["c"], expected_work="w", common_pitfalls=[])
                  for s in ("S2.1", "S1.1", "S9.9")],
        generic_questions=[QuestionDraft(type="why", section_id="S2.1", concept="split", difficulty=1, prompt="p",
                                         snippet="x", evidence=[ev("c001", "x")], expected_points=["a", "b"],
                                         source_issue_index=0)])
    fake = FakeClient(responder=lambda *a: out)
    brief = generate_brief(fake, TEMPLATE, CFG, solution_serialised="<cell>sol</cell>", log_dir=tmp_path)
    assert [s.section_id for s in brief.sections] == ["S1.1", "S2.1"]          # template order, unknown dropped
    q = brief.generic_questions[0]
    assert (q.type, q.snippet, q.evidence, q.source_issue_index) == ("concept", None, [], None)
    user = fake.calls[0]["user"]
    assert "S1.1, S1.2, S2.1, S2.2, S2.3, S3.1" in user and "<solution>" in user
    assert '<section id="S2.1"' in user and "Points obtenus" not in user
    log = store.read_llm_log(tmp_path)
    assert log[0]["call"] == "L0" and log[0]["prompt"].startswith("l0_brief@")


def student_group():
    cells = [lab_cell(k, filled(k)) for k, *_ in LAB]
    cells[4] = lab_cell("t11", ec=1)                                     # task 1.1 left empty
    doc = parse_notebook(make_nb(*cells), "G01")
    res = match_notebook(doc, TEMPLATE, CFG.matching, CFG.template)
    g = GroupRun(group_id="G01", notebook=doc, matches=res.matches)
    return g, hard_checks(doc, TEMPLATE, res, CFG.matching, CFG.template)


def test_analyse_group(tmp_path):
    g, findings = student_group()
    t21 = next(c.cell_id for c in g.notebook.cells if "train_test_split" in c.source)
    out = GroupAnalysis(
        judgments=[SectionJudgment(section_id="S1.1", status="missing", reason="empty"),
                   SectionJudgment(section_id="S2.1", status="done", reason="ok"),
                   SectionJudgment(section_id="S7", status="done", reason="unknown section")],
        issues=[Issue(section_id="S2.1", category="questionable_choice", severity="minor", concept="split",
                      title="t", explanation="e", evidence=[ev(t21, "test_size=0.2,  random_state=0")]),
                Issue(section_id="S2.2", category="bug", severity="major", concept="x", title="t2",
                      explanation="e", evidence=[ev(t21, "model.fit(X_test)")])],
        choices=[])
    fake = FakeClient(responder=lambda *a: out)
    g2 = analyse_group(fake, g, TEMPLATE, LabBrief(sections=[], generic_questions=[]), CFG, findings=findings,
                       log_dir=tmp_path, guard_names=["Zoé Mürner-Kahl"])
    assert [(i.issue_id, i.grounded) for i in g2.issues] == [("G01-I1", True), ("G01-I2", False)]
    comp = {s.section_id: s for s in g2.completeness.sections}
    assert list(comp) == ["S1.1", "S1.2", "S2.1", "S2.2", "S2.3", "S3.1"]
    assert comp["S1.1"].final_status == "missing" and comp["S1.1"].hard_findings[0].kind == "empty_cell"
    assert comp["S1.2"].final_status == "partial" and "No judgment" in comp["S1.2"].llm.reason
    assert g2.completeness.score == round((0 + 1 + 4 * 0.5) / 6, 3)
    user = fake.calls[0]["user"]
    assert "empty_cell: task cell left as in the template" in user and '<section id="S1.1"' in user
    assert g.completeness is None                                       # input left untouched


def test_completeness_missing_section_fallback():
    from labquiz.core.models import HardFinding
    f = [HardFinding(kind="missing_section", section_id="S3.1", cell_id=None, detail="")]
    c = build_completeness(TEMPLATE, [], f)
    assert {s.section_id: s.final_status for s in c.sections}["S3.1"] == "missing"


# ---------- CLI end to end with a fake LLM ----------
def responder(schema, system, user):
    if schema is LabBrief:
        return LabBrief(sections=[SectionBrief(section_id="S2.1", concepts=["train/test split"], expected_work="w",
                                               common_pitfalls=["no seed"])], generic_questions=[])
    cid = re.search(r'<cell id="(c\d+)"[^>]*>\n# Votre code ici\nX_train', user).group(1)
    return GroupAnalysis(
        judgments=[SectionJudgment(section_id="S2.1", status="done", reason="Split present.")],
        issues=[Issue(section_id="S2.1", category="questionable_choice", severity="minor", concept="split",
                      title="Fixed seed", explanation="Seed fixed.", evidence=[ev(cid, "random_state=0")])],
        choices=[])


def write_submissions(d):
    d.mkdir()
    for fname, names in (("LAB9_Murner-Kahl_Dufresnoy.ipynb", "- Zoé Mürner-Kahl\n- Ilian Dufresnoy"),
                         ("LAB9_Brevart_AitBenali.ipynb", "- Ana Sofía Brévart\n- Yoan Ait Benali")):
        cells = [lab_cell(k, filled(k)) for k, *_ in LAB]
        cells.insert(1, md("**Groupe:**\n" + names))
        (d / fname).write_bytes(nb_bytes(make_nb(*cells)))


def test_cli_run_end_to_end(tmp_path, roster, monkeypatch, capsys):
    write_submissions(tmp_path / "subs")
    (tmp_path / "t.ipynb").write_bytes(nb_bytes(lab_template()))
    fake = FakeClient(responder=responder)
    monkeypatch.setattr(cli, "make_llm", lambda cfg, runs_dir: fake)
    rc = cli.main(["run", "--submissions", str(tmp_path / "subs"), "--roster", str(roster),
                   "--template", str(tmp_path / "t.ipynb"), "--lab-id", "LAB9", "--run-dir", str(tmp_path / "run")])
    out = capsys.readouterr().out
    assert rc == 0, out
    assert "privacy pre-flight: ok" in out and "L1 analysis for 2 group(s)" in out
    assert "G01-I1 minor questionable_choice S2.1 [split] Fixed seed" in out
    assert re.search(r"S2\.1\s+Split the data.*done\s+done", out)
    run = store.load(tmp_path / "run")
    assert run.brief is not None and all(g.completeness and g.issues[0].grounded for g in run.groups.values())
    assert run.config["prompts"]["l1_analysis"]
    assert [r["call"] for r in store.read_llm_log(tmp_path / "run")].count("L1") == 2
    names = [s.display_name for s in run.students]
    for c in fake.calls:                                              # privacy: nothing personal was sent
        assert find_leaks(c["system"] + c["user"], names) == []

    capsys.readouterr()
    assert cli.main(["report", str(tmp_path / "run"), "--groups", "G02"]) == 0
    rep = capsys.readouterr().out
    assert "G02-I1" in rep and "G01-I1" not in rep


def test_cli_run_no_llm_and_unconfigured(tmp_path, roster, capsys):
    write_submissions(tmp_path / "subs")
    (tmp_path / "t.ipynb").write_bytes(nb_bytes(lab_template()))
    base = ["run", "--submissions", str(tmp_path / "subs"), "--roster", str(roster),
            "--template", str(tmp_path / "t.ipynb")]
    assert cli.main(base + ["--run-dir", str(tmp_path / "a"), "--no-llm"]) == 0
    assert "hard findings:" in capsys.readouterr().out
    assert store.load(tmp_path / "a").groups["G01"].matches                  # matching saved
    assert cli.main(base + ["--run-dir", str(tmp_path / "b")]) == 3          # default config: no LLM set
    assert "LLM not configured" in capsys.readouterr().err
