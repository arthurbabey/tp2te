"""Run directory on disk: run.json (the LabRun) and llm_log.jsonl.

Every save is atomic (write to a temp file, then rename), so a crash or a
Streamlit refresh mid-save never leaves a truncated run file.
"""

import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from .models import LabRun

RUN_FILE = "run.json"
LLM_LOG = "llm_log.jsonl"
DEFAULT_RUNS_DIR = Path("runs")


def new_run_id(lab_id: str, now: datetime | None = None) -> str:
    now = now or datetime.now(timezone.utc)
    safe = re.sub(r"[^A-Za-z0-9_-]+", "-", lab_id).strip("-") or "lab"
    return f"{safe}-{now:%Y%m%d-%H%M%S}"


def run_dir(run_id: str, runs_dir: str | Path = DEFAULT_RUNS_DIR) -> Path:
    return Path(runs_dir) / run_id


def create_run_dir(lab_id: str, runs_dir: str | Path = DEFAULT_RUNS_DIR) -> Path:
    base = run_dir(new_run_id(lab_id), runs_dir)
    d, n = base, 1
    while d.exists():
        n += 1
        d = base.with_name(f"{base.name}-{n}")
    d.mkdir(parents=True)
    return d


def _atomic_write(path: Path, text: str) -> None:
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def save(run: LabRun, directory: str | Path) -> Path:
    path = Path(directory) / RUN_FILE
    _atomic_write(path, run.model_dump_json(indent=2))
    return path


def load(directory: str | Path) -> LabRun:
    return LabRun.model_validate_json((Path(directory) / RUN_FILE).read_text(encoding="utf-8"))


def list_runs(runs_dir: str | Path = DEFAULT_RUNS_DIR) -> list[str]:
    d = Path(runs_dir)
    if not d.is_dir():
        return []
    return sorted((p.name for p in d.iterdir() if (p / RUN_FILE).is_file()), reverse=True)


def append_llm_log(directory: str | Path, record: dict) -> None:
    """One JSON line per LLM call: call type, prompt version, model, tokens, latency, raw response."""
    record = {"ts": datetime.now(timezone.utc).isoformat(), **record}
    with open(Path(directory) / LLM_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_llm_log(directory: str | Path) -> list[dict]:
    path = Path(directory) / LLM_LOG
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
