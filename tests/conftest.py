import io
import zipfile
from pathlib import Path

import nbformat
import pytest
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook, new_output

REPO = Path(__file__).resolve().parents[1]
REAL_DATA = REPO / "data" / "students"
REAL_ROSTER = REPO / "data" / "roster_example.csv"

# Fake people. Accents, compound and concatenated surnames on purpose.
ROSTER_CSV = """groupe;étudiant
Groupe 1;Zoé Mürner-Kahl
Groupe 1;Ilian Dufresnoy
Groupe 2;Ana Sofía Brévart
Groupe 2;Yoan Ait Benali
Groupe 3;Léa Quintavalle
"""


def make_nb(*cells, metadata=None) -> nbformat.NotebookNode:
    nb = new_notebook(cells=list(cells))
    nb.metadata.update(metadata or {"kernelspec": {"name": "python3", "display_name": "Python 3"}})
    return nb


def md(src, **kw):
    return new_markdown_cell(src, **kw)


def code(src, outputs=(), execution_count=1, **kw):
    return new_code_cell(src, outputs=list(outputs), execution_count=execution_count, **kw)


def stream(text):
    return new_output("stream", name="stdout", text=text)


def result(plain, html=None, n=1):
    data = {"text/plain": plain}
    if html:
        data["text/html"] = html
    return new_output("execute_result", data=data, execution_count=n)


def image():
    return new_output("display_data", data={"image/png": "iVBORw0KGgo=", "text/plain": "<Figure size 640x480 with 1 Axes>"})


def error(ename="ValueError", evalue="bad", traceback=None):
    return new_output("error", ename=ename, evalue=evalue, traceback=traceback or [f"{ename}: {evalue}"])


def nb_bytes(nb) -> bytes:
    return nbformat.writes(nb).encode()


def group_nb(names_line: str, extra: str = "") -> nbformat.NotebookNode:
    return make_nb(
        md("# TP2\n\n**Informations de groupe - Prénoms et noms complets:**\n\n" + names_line),
        md("## 1. Analyse <a name=\"Part1\"></a>"),
        code("df.head()" + extra, [result("   a  b\n0  1  2", "<table><tr><td>1</td></tr></table>")]),
    )


@pytest.fixture
def roster(tmp_path) -> Path:
    p = tmp_path / "roster.csv"
    p.write_text(ROSTER_CSV, encoding="utf-8")
    return p


@pytest.fixture
def submissions_dir(tmp_path) -> Path:
    d = tmp_path / "subs"
    d.mkdir()
    (d / "TP2_ISD_Murner-Kahl_Dufresnoy.ipynb").write_bytes(nb_bytes(group_nb("- MÜRNER-KAHL zoe\n- Ilian Dufresnoy")))
    (d / "TP2_ISD_Brevart_AitBenali.ipynb").write_bytes(nb_bytes(group_nb("- Brévart Ana Sofía\n- yoan ait benali")))
    return d


class _NoFlagZipInfo(zipfile.ZipInfo):
    """Writes UTF-8 names without the UTF-8 flag, as some zippers do."""
    def _encodeFilenameFlags(self):
        return self.filename.encode("utf-8"), self.flag_bits & ~0x800


def zip_from(entries: dict[str, bytes], utf8_flag: bool = True) -> bytes:
    buf = io.BytesIO()
    cls = zipfile.ZipInfo if utf8_flag else _NoFlagZipInfo
    with zipfile.ZipFile(buf, "w") as zf:
        for name, data in entries.items():
            zf.writestr(cls(name), data)
    return buf.getvalue()
