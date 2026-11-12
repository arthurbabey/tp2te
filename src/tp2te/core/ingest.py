"""Submissions (zip or folder) + optional roster -> students and one NotebookDoc per group.

File names follow ``<prefix>_<LastName>_<LastName>....ipynb`` with the last
names of every member of the group. Without a roster this is the source of
groups and students; with one, it is how files are attributed to groups.

Roster: CSV with a header, comma or semicolon separated. Required columns
``group`` and ``student``; optional ``file`` (notebook file name for that
group). French headers ``groupe``, ``étudiant``/``nom`` are accepted.

A notebook is attributed to a group by the ``file`` column when given,
otherwise by finding the group members' name tokens in the notebook's path
(file name and folders, which covers both "TP2_Ablet_Ragoubi.ipynb" and
Moodle's "Anaelle Ablet_123456_assignsubmission_file_/TP2.ipynb").
Zip members are read in memory and never extracted.
"""

import csv
import io
import re
import unicodedata
import zipfile
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Literal

from .models import NotebookDoc, Student
from .notebook import NotebookParseError, parse_notebook, read_notebook
from .scrub import Scrubber, fold, name_tokens

MAX_NOTEBOOK_BYTES = 50 * 1024 * 1024

_HEADER_ALIASES = {
    "group": {"group", "groupe", "group_id", "groupid", "team", "equipe"},
    "student": {"student", "etudiant", "name", "nom", "display_name", "full_name", "nom complet"},
    "file": {"file", "fichier", "filename", "notebook"},
}


@dataclass
class RosterRow:
    group: str
    student: str
    file: str | None = None


@dataclass
class SubmissionFile:
    path: str            # relative path inside the zip or folder
    data: bytes


FileStatus = Literal["ok", "duplicate", "conflict", "unmatched", "ambiguous", "parse_error", "too_large"]


@dataclass
class FileReport:
    path: str
    status: FileStatus
    group_id: str | None = None
    detail: str = ""


@dataclass
class IngestResult:
    students: list[Student]
    notebooks: dict[str, NotebookDoc]
    files: list[FileReport]
    groups_without_notebook: list[str] = field(default_factory=list)
    scrub_names: list[str] = field(default_factory=list)   # local only: what the scrubber removes


# ---------- roster ----------
def _norm_header(h: str) -> str:
    return fold(unicodedata.normalize("NFC", h.strip()))


def load_roster(path: str | Path) -> list[RosterRow]:
    text = Path(path).read_text(encoding="utf-8-sig")
    try:
        dialect = csv.Sniffer().sniff(text.splitlines()[0], delimiters=",;\t")
    except csv.Error:
        dialect = csv.excel
    reader = csv.DictReader(io.StringIO(text), dialect=dialect)
    cols: dict[str, str] = {}
    for h in reader.fieldnames or []:
        for key, aliases in _HEADER_ALIASES.items():
            if _norm_header(h) in aliases and key not in cols:
                cols[key] = h
    missing = [k for k in ("group", "student") if k not in cols]
    if missing:
        raise ValueError(f"roster {path}: missing column(s) {missing}; found {reader.fieldnames}")
    rows = []
    for r in reader:
        g, s = (r.get(cols["group"]) or "").strip(), (r.get(cols["student"]) or "").strip()
        if not g and not s:
            continue
        if not g or not s:
            raise ValueError(f"roster {path}: incomplete row {r}")
        f = (r.get(cols["file"]) or "").strip() if "file" in cols else ""
        rows.append(RosterRow(group=g, student=unicodedata.normalize("NFC", s), file=f or None))
    if not rows:
        raise ValueError(f"roster {path}: no rows")
    return rows


def group_ids(rows: list[RosterRow]) -> dict[str, str]:
    """Roster group label -> "G07". Uses the number in the label when every label
    has a distinct one, otherwise numbers groups in roster order."""
    labels = list(dict.fromkeys(r.group for r in rows))
    nums = [re.search(r"\d+", lab) for lab in labels]
    if all(nums) and len({int(n.group()) for n in nums}) == len(labels):
        return {lab: f"G{int(n.group()):02d}" for lab, n in zip(labels, nums)}
    return {lab: f"G{i:02d}" for i, lab in enumerate(labels, 1)}


def build_students(rows: list[RosterRow], gids: dict[str, str]) -> list[Student]:
    students, counts = [], {}
    for r in rows:
        gid = gids[r.group]
        k = counts.get(gid, 0)
        counts[gid] = k + 1
        students.append(Student(student_id=f"{gid}-{_letter(k)}", group_id=gid, display_name=r.student))
    return students


def _letter(k: int) -> str:
    s = ""
    k += 1
    while k:
        k, rem = divmod(k - 1, 26)
        s = chr(65 + rem) + s
    return s


# ---------- submissions ----------
def _skip(path: str) -> bool:
    p = PurePosixPath(path)
    return (p.suffix.lower() != ".ipynb" or any(part.startswith(".") or part == "__MACOSX" for part in p.parts))


def _zip_name(info: zipfile.ZipInfo) -> str:
    name = info.filename
    if not info.flag_bits & 0x800:   # no UTF-8 flag: zipfile decoded as cp437
        try:
            name = name.encode("cp437").decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            pass
    return unicodedata.normalize("NFC", name)


def collect_notebooks(path: str | Path) -> tuple[list[SubmissionFile], list[FileReport]]:
    path = Path(path)
    files, reports = [], []
    if path.is_dir():
        for p in sorted(path.rglob("*.ipynb")):
            rel = unicodedata.normalize("NFC", p.relative_to(path).as_posix())
            if _skip(rel):
                continue
            if p.stat().st_size > MAX_NOTEBOOK_BYTES:
                reports.append(FileReport(rel, "too_large", detail=f"{p.stat().st_size} bytes"))
                continue
            files.append(SubmissionFile(rel, p.read_bytes()))
    elif zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as zf:
            for info in sorted(zf.infolist(), key=lambda i: i.filename):
                if info.is_dir():
                    continue
                rel = _zip_name(info)
                if _skip(rel):
                    continue
                if info.file_size > MAX_NOTEBOOK_BYTES:
                    reports.append(FileReport(rel, "too_large", detail=f"{info.file_size} bytes"))
                    continue
                files.append(SubmissionFile(rel, zf.read(info)))
    else:
        raise ValueError(f"{path} is neither a folder nor a zip file")
    return files, reports


def _path_tokens(path: str) -> list[str]:
    return re.findall(r"[^\W_]+", fold(unicodedata.normalize("NFC", path)))


def match_group(path: str, rows: list[RosterRow], gids: dict[str, str]) -> tuple[str | None, str]:
    """Return (group_id, detail). group_id is None when unmatched or ambiguous."""
    base = PurePosixPath(path).name
    for r in rows:
        if r.file and unicodedata.normalize("NFC", r.file) == base:
            return gids[r.group], "roster file column"
    ptoks = _path_tokens(path)
    pset = set(ptoks)
    scores: dict[str, int] = {}
    for r in rows:
        hit = sum(1 for t in name_tokens(r.student)
                  if t in pset or (len(t) >= 4 and any(t in p for p in ptoks)))
        scores[gids[r.group]] = scores.get(gids[r.group], 0) + hit
    ranked = sorted(scores.items(), key=lambda kv: -kv[1])
    if not ranked or ranked[0][1] == 0:
        return None, "no roster name found in path"
    if len(ranked) > 1 and ranked[1][1] == ranked[0][1]:
        tied = [g for g, s in ranked if s == ranked[0][1]]
        return None, f"ambiguous between {', '.join(tied)}"
    return ranked[0][0], f"{ranked[0][1]} name token(s) in path"


# ---------- file name pattern: <prefix>_<LastName>_<LastName>....ipynb ----------
def _stem_tokens(path: str) -> list[str]:
    stem = PurePosixPath(path).stem
    stem = re.sub(r"\s*\(\d+\)$", "", stem)          # "x (1).ipynb" browser copies
    return [t for t in re.split(r"_+", unicodedata.normalize("NFC", stem)) if t.strip()]


def filename_surnames(paths: list[str], prefix: str | None = None) -> dict[str, list[str]]:
    """Last names encoded in each file name.

    The prefix is ``prefix`` when given (e.g. "TP2_ISD_SA2025"). Otherwise it is
    the leading tokens shared by more than half of the files, so one renamed file
    does not shift everyone else; with a single file, leading tokens that hold a
    digit or are all caps (TP2, ISD, SA2025) are dropped.
    """
    toks = {p: _stem_tokens(p) for p in paths}
    out = {}
    if prefix:
        ptoks = [fold(t) for t in prefix.strip("_").split("_") if t]
    for p, ts in toks.items():
        if prefix:
            n = len(ptoks) if [fold(t) for t in ts[:len(ptoks)]] == ptoks else 0
        elif len(paths) >= 2:
            n = 0
            while n < len(ts) - 1:
                shared = sum(len(o) > n and fold(o[n]) == fold(ts[n]) for o in toks.values())   # includes self
                if shared < 2 or shared * 2 <= len(paths):
                    break
                n += 1
        else:
            n = 0
            while n < len(ts) - 1 and (re.search(r"\d", ts[n]) or ts[n].isupper()):
                n += 1
        out[p] = ts[n:]
    return out


def _compact(s: str) -> str:
    return re.sub(r"[\W_]+", "", fold(unicodedata.normalize("NFC", s)))


_BULLET_RE = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+(.+?)\s*$")


def harvest_names(cells, surnames: list[str]) -> dict[str, str]:
    """Full names written by the students, found from their last names.

    Looks at short bullet lines in markdown cells ("- Zemzemi Mohamed") that
    contain one of ``surnames``, also when written split or concatenated
    ("NaitAbbou" vs "Nait Abbou"). Returns {surname: full name as written}.
    """
    found: dict[str, str] = {}
    keys = {s: _compact(s) for s in surnames if len(_compact(s)) >= 3}
    for c in cells:
        if c.get("cell_type") != "markdown":
            continue
        src = c.get("source")
        src = "".join(src) if isinstance(src, list) else (src or "")
        for line in src.splitlines():
            m = _BULLET_RE.match(line)
            if not m:
                continue
            txt = re.sub(r"[*_`]+", "", m.group(1)).strip(" .,;")
            if not txt or len(txt) > 60 or re.search(r"\d", txt) or len(txt.split()) > 6:
                continue
            comp = _compact(txt)
            for s, k in keys.items():
                if k in comp and s not in found:
                    found[s] = txt
    return found


def ingest(submissions: str | Path, roster: str | Path | None = None, *,
           extra_names: list[str] | None = None, output_cap: int = 1500,
           filename_prefix: str | None = None) -> IngestResult:
    """Without a roster, each file is a group and its students are the last names
    in the file name; display names are the full names harvested from the notebook."""
    files, reports = collect_notebooks(submissions)
    surnames = filename_surnames([f.path for f in files], filename_prefix)

    # pass 1: read, attribute to a group
    parsed = []                       # (file, nb, gid)
    if roster is not None:
        rows = load_roster(roster)
        gids = group_ids(rows)
        students = build_students(rows, gids)
    else:
        rows, students = [], []
        keys = {f.path: tuple(sorted(_compact(s) for s in surnames[f.path])) for f in files}
        order = sorted(set(keys.values()), key=lambda k: [surnames[p] for p in keys if keys[p] == k][0])
        gid_of = {k: f"G{i:02d}" for i, k in enumerate(order, 1)}
    for f in files:
        try:
            nb = read_notebook(f.data)
        except NotebookParseError as e:
            reports.append(FileReport(f.path, "parse_error", None, str(e)))
            continue
        if roster is not None:
            gid, detail = match_group(f.path, rows, gids)
            if gid is None:
                reports.append(FileReport(f.path, "ambiguous" if detail.startswith("ambiguous") else "unmatched",
                                          detail=detail))
                continue
        else:
            gid, detail = gid_of[keys[f.path]], "last names: " + ", ".join(surnames[f.path])
        parsed.append((f, nb, gid, detail))

    # first names written in the notebooks, so they are scrubbed too
    harvested: dict[str, dict[str, str]] = {}
    for f, nb, gid, _ in parsed:
        harvested.setdefault(gid, {}).update(harvest_names(nb.cells, surnames[f.path]))
    if roster is None:
        seen = set()
        for f, nb, gid, _ in sorted(parsed, key=lambda x: x[2]):
            if gid in seen:
                continue
            seen.add(gid)
            for k, s in enumerate(surnames[f.path]):
                students.append(Student(student_id=f"{gid}-{_letter(k)}", group_id=gid,
                                        display_name=harvested[gid].get(s, s)))
    names = [s.display_name for s in students] + [n for h in harvested.values() for n in h.values()]
    scrubber = Scrubber(names + list(extra_names or []))

    # pass 2: parse and scrub
    notebooks: dict[str, NotebookDoc] = {}
    for f, nb, gid, detail in parsed:
        doc = parse_notebook(nb, gid, scrubber=scrubber, output_cap=output_cap)
        if gid in notebooks:
            same = notebooks[gid].content_hash == doc.content_hash
            reports.append(FileReport(f.path, "duplicate" if same else "conflict", gid,
                                      "identical copy, ignored" if same
                                      else "second, different notebook for this group; first one kept"))
            continue
        notebooks[gid] = doc
        reports.append(FileReport(f.path, "ok", gid, detail))

    all_gids = {s.group_id for s in students}
    return IngestResult(students=students, notebooks=dict(sorted(notebooks.items())), files=reports,
                        groups_without_notebook=sorted(all_gids - set(notebooks)),
                        scrub_names=sorted(set(names)))
