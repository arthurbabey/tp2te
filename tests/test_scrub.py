"""Scrubber. The privacy promise: after serialisation, no roster name, email or
home-directory path appears in any prompt. Also run on the real notebooks."""

import re
import unicodedata

import pytest

from conftest import REAL_DATA, REAL_ROSTER, code, error, image, make_nb, md, result, stream
from tp2te.core.ingest import ingest
from tp2te.core.notebook import parse_notebook, serialise_notebook
from tp2te.core.scrub import Scrubber, find_leaks, fold, name_tokens

NAMES = ["Zoé Mürner-Kahl", "Ilian Dufresnoy", "Yoan Ait Benali"]


def naive_hits(text: str, names: list[str]) -> list[str]:
    """Independent of scrub.py: plain case-insensitive search, with and without accents."""
    def strip(s):
        return "".join(c for c in unicodedata.normalize("NFD", s) if not unicodedata.combining(c)).lower()
    hay = {text.lower(), strip(text)}
    hits = []
    for n in names:
        for tok in re.split(r"[\s\-]+", n):
            if len(tok) < 3:
                continue
            for t in {tok.lower(), strip(tok)}:
                if any(re.search(rf"(?<![a-z0-9]){re.escape(t)}(?![a-z0-9])", h) for h in hay):
                    hits.append(tok)
    return hits


def test_fold_is_length_preserving():
    s = "Zoé MÜRNER-Kahl Ça ß"
    assert len(fold(s)) == len(s)
    assert fold("Zoé MÜRNER") == "zoe murner"


def test_name_tokens():
    assert name_tokens("Zoé Mürner-Kahl") == ["zoe", "murner", "kahl"]
    assert name_tokens("Li Ba") == []          # too short to scrub safely


@pytest.mark.parametrize("text", [
    "- MÜRNER-KAHL zoe",
    "- Zoe Murner Kahl",
    "kahl, zoé",
    "TP2_Murner_Kahl.ipynb",
    "par Ilian DUFRESNOY et Yoan AIT BENALI",
])
def test_names_in_any_order_case_and_accents(text):
    out = Scrubber(NAMES).scrub(text)
    assert "[NAME]" in out
    assert find_leaks(out, NAMES) == []
    assert naive_hits(out, NAMES) == []


def test_does_not_scrub_inside_other_words():
    s = Scrubber(NAMES)
    assert s.scrub("zoeline kahlua benalities") == "zoeline kahlua benalities"


def test_email_and_home_paths():
    s = Scrubber(NAMES)
    out = s.scrub("mail zoe.murner@school.ch\n/Users/zmurner/lab/x.csv\n/home/idu/tp\nC:\\Users\\yoan\\tp2\\data.csv")
    assert "[EMAIL]" in out and "school.ch" not in out
    assert "/Users/[USER]/lab" in out and "/home/[USER]/tp" in out and "C:\\Users\\[USER]\\tp2" in out
    assert find_leaks(out, NAMES) == []


def test_find_leaks_reports_unscrubbed_text():
    leaks = find_leaks("by Ilian dufresnoy, ilian@x.org, /Users/ilian/", NAMES)
    kinds = " ".join(leaks)
    assert "dufresnoy" in kinds and "email" in kinds and "home path" in kinds


def test_serialised_notebook_has_no_roster_data():
    nb = make_nb(
        md("**Groupe:** Zoé Mürner-Kahl, ilian DUFRESNOY\nContact: zoe.murner@school.ch"),
        code("# auteur: Yoan Ait Benali\nprint('Ilian')",
             [stream("hello Zoé\n"), result("'/Users/zmurner/tp2'"), image()]),
        code("open('C:\\\\Users\\\\yoan\\\\x.csv')",
             [error("FileNotFoundError", "No such file: '/home/idufresnoy/x.csv'",
                    ["\x1b[31mFileNotFoundError\x1b[0m", "File /Users/zmurner/venv/lib/x.py:3"])]),
        metadata={"authors": [{"name": "Zoé Mürner-Kahl"}], "kernelspec": {"display_name": "env-METASENTINEL"}},
    )
    nb.cells[0].metadata["tags"] = ["CELLMETASENTINEL"]
    doc = parse_notebook(nb, "G01", scrubber=Scrubber(NAMES))
    text = serialise_notebook(doc)
    assert find_leaks(text, NAMES) == []
    assert naive_hits(text, NAMES) == []
    assert "METASENTINEL" not in doc.model_dump_json()   # metadata is never carried
    assert "\x1b" not in text


@pytest.mark.skipif(not REAL_ROSTER.is_file(), reason="real notebooks and roster not present (data/ is local only)")
def test_real_notebooks_preflight():
    res = ingest(REAL_DATA, REAL_ROSTER)
    names = [s.display_name for s in res.students]
    assert res.notebooks, "no real notebook was ingested"
    assert all(f.status == "ok" for f in res.files), [(f.path, f.status, f.detail) for f in res.files]
    for gid, doc in res.notebooks.items():
        text = serialise_notebook(doc)
        assert find_leaks(text, names) == [], gid
        assert naive_hits(text, names) == [], gid


@pytest.mark.skipif(not REAL_ROSTER.is_file(), reason="real notebooks and roster not present (data/ is local only)")
def test_real_notebooks_preflight_without_roster():
    """File names only give last names; first names must still be scrubbed.
    The full names of the local roster are the independent reference."""
    from tp2te.core.ingest import load_roster
    names = [r.student for r in load_roster(REAL_ROSTER)]
    res = ingest(REAL_DATA)
    assert len(res.students) == len(names)
    for gid, doc in res.notebooks.items():
        text = serialise_notebook(doc)
        assert find_leaks(text, names) == [], gid
        assert naive_hits(text, names) == [], gid
