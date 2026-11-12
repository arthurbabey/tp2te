"""LLMClient protocol, LLMResult, JSON helpers shared by the clients, and the factory."""

import hashlib
import json
import re
from typing import Generic, Protocol, TypeVar

from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)


class LLMError(RuntimeError):
    pass


class LLMResult(BaseModel, Generic[T]):
    parsed: T
    model: str
    input_tokens: int
    output_tokens: int
    latency_s: float
    raw: str = ""                     # response text as received (for the log)
    cached: bool = False


class LLMClient(Protocol):
    model: str

    def complete(self, *, system: str, user: str, schema: type[T],
                 temperature: float = 0.2, max_tokens: int = 4000) -> LLMResult[T]: ...


def prompt_key(system: str, user: str, schema: type[BaseModel], model: str = "") -> str:
    """Key of a call for recording and replay: prompt text, schema and model."""
    h = hashlib.sha256()
    for part in (model, schema.__name__, json.dumps(schema.model_json_schema(), sort_keys=True), system, user):
        h.update(part.encode())
        h.update(b"\0")
    return h.hexdigest()[:24]


_FENCE_RE = re.compile(r"```(?:json)?\s*\n(.*?)\n```", re.S)


def extract_json(text: str) -> str:
    """JSON from a reply: a fenced block if there is one, else the outermost {...}."""
    m = _FENCE_RE.search(text)
    if m:
        return m.group(1)
    a, b = text.find("{"), text.rfind("}")
    return text[a:b + 1] if a != -1 and b > a else text


def parse_reply(text: str, schema: type[T]) -> T:
    """Validate a reply. Raises ValidationError (also for invalid JSON)."""
    return schema.model_validate_json(extract_json(text))


def schema_instruction(schema: type[BaseModel]) -> str:
    return ("Return one JSON object that validates against this JSON schema, and nothing else:\n"
            + json.dumps(schema.model_json_schema(), ensure_ascii=False))


def retry_message(err: ValidationError | ValueError) -> str:
    return (f"Your previous answer did not validate against the schema:\n{err}\n"
            "Return the corrected JSON object only.")


def make_client(cfg) -> LLMClient:
    """Factory keyed on cfg.provider (cfg is config.LLMConfig)."""
    if cfg.provider == "openai_compat":
        from .openai_compat import OpenAICompatClient
        return OpenAICompatClient.from_config(cfg)
    if cfg.provider == "fake":
        from .fake import FakeClient
        return FakeClient(cfg.base_url or None)
    raise LLMError(f"unknown LLM provider {cfg.provider!r} (openai_compat, fake)")
