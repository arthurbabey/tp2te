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


@pytest.fixture(autouse=True)
def isolated_llm_env(monkeypatch):
    """Tests never see the developer's .env or shell LLM settings (no real API calls)."""
    from tp2te import cli
    from tp2te.core.config import API_KEY_ENVS, BASE_URL_ENVS, MODEL_ENVS
    for n in (*BASE_URL_ENVS, *MODEL_ENVS, *API_KEY_ENVS):
        monkeypatch.delenv(n, raising=False)
    monkeypatch.setattr(cli, "load_dotenv", lambda *a, **k: [])


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


# ---------- a small lab in the conventions of the real TP2 template ----------
# (key, kind, source, section). Keys double as nbformat ids, so "ids kept" is the
# normal case and "ids regenerated" is simulated by renaming them.
LAB = [
    ("intro", "md", "# Lab 9\n\nIntro text about the lab and how it is graded.", "S0"),
    ("h1", "md", "## 1. Loading data\n\nWe load the iris dataset and look at it.", "S1"),
    ("load", "code", "import pandas as pd\ndf = pd.read_csv('iris.csv')", "S1"),
    ("q11", "md", "**1.1) Show the first rows of the DataFrame using head().**", "S1.1"),
    ("t11", "code", "# Votre code ici\n", "S1.1"),
    ("q12", "md", "**1.2) Describe the columns of the dataset in full sentences.**\n\n*Réponse:*", "S1.2"),
    ("staff1", "md", "<b>Corrections Partie 1: </b> Points obtenus: /5\n\nRemarques de l'assistant:", "S1.2"),
    ("h2", "md", "## 2. Model\n\nWe train a classifier on the data.", "S2"),
    ("q21", "md", "**2.1) Split the data into a training and a test set with a fixed seed.**", "S2.1"),
    ("t21", "code", "# Votre code ici\n", "S2.1"),
    ("q22", "md", "<p><b>2.2)</b> Fit a logistic regression and report the accuracy on the test set.</p>", "S2.2"),
    ("t22", "code", "# Votre code ici\n", "S2.2"),
    ("q23", "md", "**2.3) Why do we evaluate on the test set and not on the training set?**", "S2.3"),
    ("a23", "md", "*Réponse:*", "S2.3"),
    ("h3", "md", "## 3. Conclusion\n\nSummarise what you learned in this lab.", "S3"),
    ("q31", "md", "**3.1) What would you change to improve the model?**\n\n*Réponse:*", "S3.1"),
]

# What a diligent group writes into each cell (appended to the template text).
FILLED = {
    "t11": "df.head()",
    "q12": "The dataset has four numeric columns measuring sepals and petals, and a species label.",
    "t21": "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)",
    "t22": "model = LogisticRegression().fit(X_train, y_train)\nprint(model.score(X_test, y_test))",
    "a23": "Because the training score is optimistic: the model has already seen those rows.",
    "q31": "We would scale the features and tune the regularisation with cross-validation.",
}


def lab_cell(key: str, source: str | None = None, *, cell_id: str | None = None, ec: int | None = 1,
             outputs=()):
    _, kind, src, _ = next(x for x in LAB if x[0] == key)
    src = src if source is None else source
    if kind == "md":
        c = md(src)
    else:
        c = code(src, outputs, execution_count=ec)
    c["id"] = cell_id or key
    return c


def lab_template() -> nbformat.NotebookNode:
    return make_nb(*(lab_cell(k) for k, *_ in LAB))


def filled(key: str) -> str:
    _, kind, src, _ = next(x for x in LAB if x[0] == key)
    return (src.rstrip("\n") + "\n" + FILLED[key]) if key in FILLED else src
