"""Orchestration: matching and checks for every group, L0 once, L1 per group in
parallel. The run file is saved after each step and after each group, so a
crash or a refresh loses nothing; identical LLM calls are replayed from the cache."""

from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from ..llm.base import LLMClient
from . import store
from .analysis import analyse_group
from .checks import hard_checks
from .config import Config
from .matching import MatchResult, match_notebook
from .models import HardFinding, LabRun


def guard_names(run: LabRun, cfg: Config) -> list[str]:
    """Names the privacy guard refuses to send. Local only."""
    return [s.display_name for s in run.students] + list(cfg.scrub.extra_names)


def match_and_check(run: LabRun, cfg: Config) -> dict[str, tuple[MatchResult, list[HardFinding]]]:
    """Deterministic, no LLM. Stores matches and cell origins in the run."""
    out = {}
    for gid, g in run.groups.items():
        res = match_notebook(g.notebook, run.template, cfg.matching, cfg.template)
        g.matches = res.matches
        out[gid] = (res, hard_checks(g.notebook, run.template, res, cfg.matching, cfg.template))
    return out


def analyse_groups(run: LabRun, client: LLMClient, cfg: Config, run_dir: str | Path, gids: Iterable[str],
                   findings: dict[str, list[HardFinding]], *, solution_serialised: str | None = None,
                   on_done: Callable[[str, str | None], None] | None = None) -> None:
    """L1 for ``gids``, ``cfg.llm.workers`` at a time. A failing group gets ``error``
    set and does not stop the others. Saves after each group."""
    if run.brief is None:
        raise ValueError("run has no lab brief; run L0 first")
    names = guard_names(run, cfg)
    gids = list(gids)
    with ThreadPoolExecutor(max_workers=max(1, cfg.llm.workers)) as ex:
        futs = {ex.submit(analyse_group, client, run.groups[gid], run.template, run.brief, cfg,
                          findings=findings[gid], solution_serialised=solution_serialised,
                          log_dir=run_dir, guard_names=names): gid for gid in gids}
        for fut in as_completed(futs):
            gid = futs[fut]
            try:
                run.groups[gid] = fut.result()
                err = None
            except Exception as e:  # one group failing must not lose the others
                err = f"{type(e).__name__}: {e}"
                run.groups[gid].error = err
            store.save(run, run_dir)
            if on_done:
                on_done(gid, err)
