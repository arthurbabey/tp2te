import pytest

from conftest import ROSTER_CSV, group_nb, nb_bytes, zip_from
from tp2te.core.ingest import RosterRow, build_students, group_ids, ingest, load_roster, match_group


def test_load_roster_french_semicolon_bom(tmp_path):
    p = tmp_path / "r.csv"
    p.write_text("﻿" + ROSTER_CSV, encoding="utf-8")
    rows = load_roster(p)
    assert len(rows) == 5 and rows[0] == RosterRow("Groupe 1", "Zoé Mürner-Kahl", None)


def test_load_roster_comma_with_file_column(tmp_path):
    p = tmp_path / "r.csv"
    p.write_text("group,student,file\n7,Ann Lee,a.ipynb\n7,Bo Kim,\n", encoding="utf-8")
    rows = load_roster(p)
    assert rows[0].file == "a.ipynb" and rows[1].file is None


def test_load_roster_missing_column(tmp_path):
    p = tmp_path / "r.csv"
    p.write_text("team_name;person\nx;y\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing column"):
        load_roster(p)


def test_group_ids_and_student_ids():
    rows = [RosterRow("Groupe 7", "A Aa"), RosterRow("Groupe 7", "B Bb"), RosterRow("Groupe 12", "C Cc")]
    gids = group_ids(rows)
    assert gids == {"Groupe 7": "G07", "Groupe 12": "G12"}
    assert [s.student_id for s in build_students(rows, gids)] == ["G07-A", "G07-B", "G12-A"]
    # labels without distinct numbers: numbered in roster order
    assert group_ids([RosterRow("red", "x"), RosterRow("blue", "y")]) == {"red": "G01", "blue": "G02"}


ROWS = [RosterRow("1", "Zoé Mürner-Kahl"), RosterRow("1", "Ilian Dufresnoy"),
        RosterRow("2", "Ana Sofía Brévart"), RosterRow("2", "Yoan Ait Benali"), RosterRow("3", "Léa Quintavalle")]
GIDS = group_ids(ROWS)


@pytest.mark.parametrize("path,gid", [
    ("TP2_ISD_SA2025_Murner-Kahl_Dufresnoy.ipynb", "G01"),
    ("TP2_ISD_SA2025_Brevart_AitBenali.ipynb", "G02"),          # concatenated surname, no accent
    ("Léa Quintavalle_1234567_assignsubmission_file_/TP2.ipynb", "G03"),   # Moodle folder layout
    ("Ana Sofia Brevart_99_assignsubmission_file_TP2.ipynb", "G02"),       # Moodle flat layout
])
def test_match_group(path, gid):
    assert match_group(path, ROWS, GIDS)[0] == gid


def test_match_group_unmatched_and_ambiguous():
    assert match_group("TP2_final.ipynb", ROWS, GIDS) == (None, "no roster name found in path")
    gid, detail = match_group("TP2_Dufresnoy_Quintavalle.ipynb", ROWS, GIDS)
    assert gid is None and detail.startswith("ambiguous")


def test_match_group_file_column_wins():
    rows = ROWS[:4] + [RosterRow("3", "Léa Quintavalle", file="TP2_Dufresnoy.ipynb")]
    assert match_group("x/TP2_Dufresnoy.ipynb", rows, group_ids(rows)) == ("G03", "roster file column")


def test_ingest_dir(submissions_dir, roster):
    res = ingest(submissions_dir, roster)
    assert sorted(res.notebooks) == ["G01", "G02"]
    assert res.groups_without_notebook == ["G03"]
    assert [s.student_id for s in res.students] == ["G01-A", "G01-B", "G02-A", "G02-B", "G03-A"]
    assert "[NAME]" in res.notebooks["G01"].cells[0].source


def test_ingest_zip_equals_dir(submissions_dir, roster, tmp_path):
    z = tmp_path / "subs.zip"
    z.write_bytes(zip_from({f"subs/{p.name}": p.read_bytes() for p in submissions_dir.iterdir()}))
    a, b = ingest(submissions_dir, roster), ingest(z, roster)
    assert {g: d.content_hash for g, d in a.notebooks.items()} == {g: d.content_hash for g, d in b.notebooks.items()}


def test_ingest_zip_edge_cases(roster, tmp_path):
    g1 = nb_bytes(group_nb("- Zoé"))
    z = tmp_path / "subs.zip"
    z.write_bytes(zip_from({
        "Zoé Mürner-Kahl_1_assignsubmission_file_/TP2.ipynb": g1,
        "Ilian Dufresnoy_2_assignsubmission_file_/TP2.ipynb": g1,          # same group, identical copy
        "Yoan Ait Benali_3_assignsubmission_file_/TP2.ipynb": nb_bytes(group_nb("- A")),
        "Ana Sofía Brévart_4_assignsubmission_file_/TP2.ipynb": nb_bytes(group_nb("- B")),  # different
        "Léa Quintavalle_5_assignsubmission_file_/TP2.ipynb": b"{broken",
        "__MACOSX/Zoé Mürner-Kahl_1_assignsubmission_file_/._TP2.ipynb": b"junk",
        "Zoé Mürner-Kahl_1_assignsubmission_file_/.ipynb_checkpoints/TP2-checkpoint.ipynb": g1,
        "readme.txt": b"hi",
    }, utf8_flag=False))
    res = ingest(z, roster)
    status = {f.path: f.status for f in res.files}
    assert sorted(status.values()) == ["conflict", "duplicate", "ok", "ok", "parse_error"]
    assert "Zoé Mürner-Kahl_1_assignsubmission_file_/TP2.ipynb" in status   # cp437 name repaired
    assert sorted(res.notebooks) == ["G01", "G02"] and res.groups_without_notebook == ["G03"]


def test_ingest_rejects_other_inputs(roster, tmp_path):
    f = tmp_path / "x.txt"
    f.write_text("x")
    with pytest.raises(ValueError, match="neither a folder nor a zip"):
        ingest(f, roster)


# ---------- file name pattern, no roster ----------
from tp2te.core.ingest import filename_surnames, harvest_names  # noqa: E402
from conftest import md  # noqa: E402


def test_filename_surnames_auto_prefix():
    paths = ["TP2_ISD_SA2025_Ablo_Rag.ipynb", "x/TP2_ISD_SA2025_Zem.ipynb", "TP2_ISD_SA2025_Nait-Abbou_Q (1).ipynb",
             "TP2_Renamed.ipynb"]                        # one off-pattern file does not shift the others
    s = filename_surnames(paths)
    assert s["TP2_ISD_SA2025_Ablo_Rag.ipynb"] == ["Ablo", "Rag"]
    assert s["x/TP2_ISD_SA2025_Zem.ipynb"] == ["Zem"]
    assert s["TP2_ISD_SA2025_Nait-Abbou_Q (1).ipynb"] == ["Nait-Abbou", "Q"]
    assert s["TP2_Renamed.ipynb"] == ["Renamed"]


def test_filename_surnames_single_file_and_explicit_prefix():
    assert filename_surnames(["TP2_ISD_SA2025_Zem.ipynb"]) == {"TP2_ISD_SA2025_Zem.ipynb": ["Zem"]}
    assert filename_surnames(["Lab3_Ann_Bo.ipynb"], prefix="Lab3") == {"Lab3_Ann_Bo.ipynb": ["Ann", "Bo"]}


def test_harvest_names():
    cells = [md("**Informations de groupe:**\n\n- MÜRNER-KAHL zoé\n- Yoan Ait Benali\n- ...\n"),
             md("- Ce résultat montre que Dufresnoy avait raison en 1952")]   # sentence, not a name
    found = harvest_names(cells, ["Murner-Kahl", "AitBenali", "Dufresnoy"])
    assert found == {"Murner-Kahl": "MÜRNER-KAHL zoé", "AitBenali": "Yoan Ait Benali"}


def test_ingest_without_roster(submissions_dir):
    res = ingest(submissions_dir)
    assert [(s.student_id, s.display_name) for s in res.students] == [
        ("G01-A", "Brévart Ana Sofía"), ("G01-B", "yoan ait benali"),
        ("G02-A", "MÜRNER-KAHL zoe"), ("G02-B", "Ilian Dufresnoy")]
    assert sorted(res.notebooks) == ["G01", "G02"] and res.groups_without_notebook == []
    src = res.notebooks["G01"].cells[0].source
    assert "Sof" not in src and "yoan" not in src.lower()       # first names scrubbed too
