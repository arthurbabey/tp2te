"""Recorded responses, keyed on prompt hash (``base.prompt_key``).

``FakeClient`` replays them, for tests and for UI work without spending tokens.
``CachingClient`` wraps a real client and records into the same format, so
calling Generate twice costs nothing and a real run can be replayed offline.
One JSON file per call: ``<key>.json`` = {"raw", "model", "input_tokens", "output_tokens", "schema"}.
"""

import json
import threading
from collections.abc import Callable
from pathlib import Path

from pydantic import BaseModel

from .base import LLMClient, LLMError, LLMResult, T, parse_reply, prompt_key


class FakeClient:
    """Replays recordings from a directory or a dict {key: raw JSON text}.

    ``responder(schema, system, user) -> BaseModel | str`` answers calls that have
    no recording, so tests can build replies from the prompt they receive.
    ``calls`` records every request.
    """

    def __init__(self, recordings: str | Path | dict[str, str] | None = None, *,
                 responder: Callable[[type[BaseModel], str, str], BaseModel | str] | None = None,
                 model: str = "fake"):
        self.model = model
        self.responder = responder
        self.calls: list[dict] = []
        self._dir = Path(recordings) if isinstance(recordings, (str, Path)) else None
        self._dict = recordings if isinstance(recordings, dict) else {}

    def complete(self, *, system: str, user: str, schema: type[T],
                 temperature: float = 0.2, max_tokens: int = 4000) -> LLMResult[T]:
        self.calls.append({"system": system, "user": user, "schema": schema.__name__})
        key = prompt_key(system, user, schema, self.model)
        raw = self._dict.get(key)
        if raw is None and self._dir is not None and (self._dir / f"{key}.json").is_file():
            raw = json.loads((self._dir / f"{key}.json").read_text(encoding="utf-8"))["raw"]
        if raw is None and self.responder is not None:
            out = self.responder(schema, system, user)
            raw = out.model_dump_json() if isinstance(out, BaseModel) else out
        if raw is None:
            raise LLMError(f"FakeClient: no recorded response for {schema.__name__} call {key}")
        return LLMResult(parsed=parse_reply(raw, schema), model=self.model,
                         input_tokens=len(system + user) // 4, output_tokens=len(raw) // 4,
                         latency_s=0.0, raw=raw)


class CachingClient:
    def __init__(self, inner: LLMClient, directory: str | Path):
        self.inner = inner
        self.model = inner.model
        self.dir = Path(directory)
        self._lock = threading.Lock()

    def complete(self, *, system: str, user: str, schema: type[T],
                 temperature: float = 0.2, max_tokens: int = 4000) -> LLMResult[T]:
        key = prompt_key(system, user, schema, self.model)
        path = self.dir / f"{key}.json"
        if path.is_file():
            rec = json.loads(path.read_text(encoding="utf-8"))
            return LLMResult(parsed=parse_reply(rec["raw"], schema), model=rec["model"],
                             input_tokens=0, output_tokens=0, latency_s=0.0, raw=rec["raw"], cached=True)
        res = self.inner.complete(system=system, user=user, schema=schema,
                                  temperature=temperature, max_tokens=max_tokens)
        with self._lock:
            self.dir.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"raw": res.raw, "model": res.model, "schema": schema.__name__,
                                        "input_tokens": res.input_tokens, "output_tokens": res.output_tokens},
                                       ensure_ascii=False), encoding="utf-8")
        return res
