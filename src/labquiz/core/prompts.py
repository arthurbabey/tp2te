"""Prompt rendering, prompt versions, and the one way to call the LLM.

Prompts are Jinja files in ``labquiz/prompts``. A prompt's version is the hash
of its file, so any edit shows up in the log and the run's config snapshot
without anyone having to bump a number.

``call_llm`` is the single entry point used by L0-L4: it renders the prompts,
refuses to send text that contains a roster name, email or home path (privacy
guard, independent of the scrubber), calls the client and appends one line to
``llm_log.jsonl``.
"""

import hashlib
from collections.abc import Iterable
from functools import lru_cache
from pathlib import Path

import jinja2

from ..llm.base import LLMClient, LLMError, LLMResult, T
from . import store
from .scrub import find_leaks

PROMPT_DIR = Path(__file__).resolve().parents[1] / "prompts"
SYSTEM = "system"

_env = jinja2.Environment(loader=jinja2.FileSystemLoader(PROMPT_DIR), undefined=jinja2.StrictUndefined,
                          autoescape=False, keep_trailing_newline=False)


class PrivacyError(LLMError):
    """A prompt about to be sent contains personal data."""


@lru_cache(maxsize=None)
def version(name: str) -> str:
    return hashlib.sha256((PROMPT_DIR / f"{name}.j2").read_bytes()).hexdigest()[:8]


def versions() -> dict[str, str]:
    return {p.stem: version(p.stem) for p in sorted(PROMPT_DIR.glob("*.j2"))}


def render(name: str, **params) -> str:
    return _env.get_template(f"{name}.j2").render(**params).strip() + "\n"


def call_llm(client: LLMClient, *, call: str, prompt: str, params: dict, system_params: dict,
             schema: type[T], log_dir: str | Path | None, guard_names: Iterable[str],
             temperature: float = 0.2, max_tokens: int = 8000, context: dict | None = None) -> LLMResult[T]:
    system = render(SYSTEM, **system_params)
    user = render(prompt, **params)
    record = {"call": call, "prompt": f"{prompt}@{version(prompt)}", "system": f"{SYSTEM}@{version(SYSTEM)}",
              "model": client.model, **(context or {})}
    leaks = find_leaks(system + "\n" + user, list(guard_names))
    if leaks:
        if log_dir:
            store.append_llm_log(log_dir, {**record, "error": f"privacy guard: {len(leaks)} leak(s), not sent"})
        raise PrivacyError(f"{call}: prompt not sent, it contains personal data: {'; '.join(leaks[:5])}")
    try:
        res = client.complete(system=system, user=user, schema=schema, temperature=temperature,
                              max_tokens=max_tokens)
    except Exception as e:
        if log_dir:
            store.append_llm_log(log_dir, {**record, "error": f"{type(e).__name__}: {e}"})
        raise
    if log_dir:
        store.append_llm_log(log_dir, {**record, "model": res.model, "input_tokens": res.input_tokens,
                                       "output_tokens": res.output_tokens, "latency_s": res.latency_s,
                                       "cached": res.cached, "prompt_chars": len(system) + len(user),
                                       "raw": res.raw})
    return res
