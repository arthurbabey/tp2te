from datetime import datetime, timezone
from pathlib import Path

import pytest

from conftest import REPO, nb_bytes, zip_from, group_nb
from tp2te import cli
from tp2te.core import store
from tp2te.core.models import GroupRun, LabRun, NotebookDoc, Student
from tp2te.core.scrub import Scrubber


def _run() -> LabRun:
    doc = NotebookDoc(group_id="G01", cells=[], content_hash="h")
    return LabRun(lab_id="TP2", created_at=datetime(2026, 11, 11, tzinfo=timezone.utc), config={"a": 1},
                  students=[Student(student_id="G01-A", group_id="G01", display_name="Zoé")],
                  groups={"G01": GroupRun(group_id="G01", notebook=doc, matches=[])})


def test_save_load_roundtrip(tmp_path):
    d = store.create_run_dir("TP 2/x", tmp_path)
    assert d.name.startswith("TP-2-x-")
    run = _run()
    store.save(run, d)
    assert store.load(d) == run
    assert [p.name for p in d.iterdir()] == ["run.json"]          # no temp file left behind
    assert store.list_runs(tmp_path) == [d.name]


def test_create_run_dir_is_unique(tmp_path):
    assert store.create_run_dir("TP2", tmp_path) != store.create_run_dir("TP2", tmp_path)


def test_failed_save_keeps_previous_file(tmp_path, monkeypatch):
    store.save(_run(), tmp_path)
    before = (tmp_path / "run.json").read_text()
    monkeypatch.setattr(store.os, "replace", lambda *a: (_ for _ in ()).throw(OSError("disk full")))
    with pytest.raises(OSError):
        store.save(_run().model_copy(update={"lab_id": "OTHER"}), tmp_path)
    assert (tmp_path / "run.json").read_text() == before
    assert [p.name for p in tmp_path.iterdir()] == ["run.json"]


def test_llm_log(tmp_path):
    store.append_llm_log(tmp_path, {"call": "L1", "input_tokens": 10})
    store.append_llm_log(tmp_path, {"call": "L2", "raw": "é"})
    log = store.read_llm_log(tmp_path)
    assert [r["call"] for r in log] == ["L1", "L2"] and "ts" in log[0] and log[1]["raw"] == "é"


# ---------- CLI end to end ----------
def test_cli_run(submissions_dir, roster, tmp_path, capsys):
    z = tmp_path / "subs.zip"
    z.write_bytes(zip_from({p.name: p.read_bytes() for p in submissions_dir.iterdir()}))
    rc = cli.main(["run", "--submissions", str(z), "--roster", str(roster), "--lab-id", "TP2",
                   "--run-dir", str(tmp_path / "run")])
    out = capsys.readouterr().out
    assert rc == 0 and "privacy pre-flight: ok" in out
    run = store.load(tmp_path / "run")
    assert sorted(run.groups) == ["G01", "G02"] and len(run.students) == 5


def test_cli_run_fails_on_leak(submissions_dir, roster, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(Scrubber, "scrub", lambda self, text: text)   # simulate a broken scrubber
    rc = cli.main(["run", "--submissions", str(submissions_dir), "--roster", str(roster),
                   "--run-dir", str(tmp_path / "run")])
    assert rc == 2 and "PRIVACY PRE-FLIGHT FAILED" in capsys.readouterr().err


def test_core_never_imports_streamlit():
    for p in (REPO / "src" / "tp2te").rglob("*.py"):
        if "app" in p.relative_to(REPO / "src" / "tp2te").parts:
            continue
        assert "streamlit" not in p.read_text(), p
