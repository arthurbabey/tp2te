"""Any OpenAI-compatible chat completions endpoint: hosted APIs, Swiss/EU hosts
exposing the same interface, local servers (Ollama, vLLM). Standard library only.

Structured output (``structured``):
- ``json_schema``: response_format json_schema with the Pydantic schema;
- ``json_object``: JSON mode, schema given in the system prompt;
- ``prompt``: schema in the system prompt, JSON read from the reply (fenced or bare).
The reply is always validated with Pydantic; on a validation error the call is
retried once with the error appended.
"""

import json
import os
import time
import urllib.error
import urllib.request

from pydantic import ValidationError

from .base import LLMError, LLMResult, T, parse_reply, retry_message, schema_instruction

RETRY_STATUS = {408, 409, 429, 500, 502, 503, 504}


class OpenAICompatClient:
    def __init__(self, base_url: str, model: str, api_key: str | None = None, *,
                 structured: str = "json_schema", timeout_s: float = 300, http_retries: int = 3):
        if not base_url or base_url.endswith("..."):
            raise LLMError("llm.base_url is not set in the config")
        if not model or model == "...":
            raise LLMError("llm.model is not set in the config")
        if structured not in ("json_schema", "json_object", "prompt"):
            raise LLMError(f"llm.structured must be json_schema, json_object or prompt, not {structured!r}")
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model, self.api_key, self.structured = model, api_key, structured
        self.timeout_s, self.http_retries = timeout_s, http_retries

    @classmethod
    def from_config(cls, cfg) -> "OpenAICompatClient":
        key = os.environ.get(cfg.api_key_env) if cfg.api_key_env else None
        return cls(cfg.base_url, cfg.model, key, structured=cfg.structured, timeout_s=cfg.timeout_s)

    # ---- HTTP ----
    def _post(self, body: dict) -> dict:
        data = json.dumps(body).encode()
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        for attempt in range(self.http_retries + 1):
            req = urllib.request.Request(self.url, data=data, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(req, timeout=self.timeout_s) as r:
                    return json.loads(r.read().decode())
            except urllib.error.HTTPError as e:
                detail = e.read().decode(errors="replace")[:500]
                if e.code in RETRY_STATUS and attempt < self.http_retries:
                    time.sleep(min(2 ** attempt * 2, 30))
                    continue
                raise LLMError(f"HTTP {e.code} from {self.url}: {detail}") from e
            except (urllib.error.URLError, TimeoutError) as e:
                if attempt < self.http_retries:
                    time.sleep(min(2 ** attempt * 2, 30))
                    continue
                raise LLMError(f"cannot reach {self.url}: {e}") from e
        raise LLMError("unreachable")

    def complete(self, *, system: str, user: str, schema: type[T],
                 temperature: float = 0.2, max_tokens: int = 4000) -> LLMResult[T]:
        if self.structured != "json_schema":
            system = system + "\n\n" + schema_instruction(schema)
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        body = {"model": self.model, "temperature": temperature, "max_tokens": max_tokens}
        if self.structured == "json_schema":
            body["response_format"] = {"type": "json_schema", "json_schema": {
                "name": schema.__name__, "schema": schema.model_json_schema(), "strict": False}}
        elif self.structured == "json_object":
            body["response_format"] = {"type": "json_object"}

        t0 = time.monotonic()
        tin = tout = 0
        model = self.model
        for attempt in range(2):
            resp = self._post({**body, "messages": messages})
            usage = resp.get("usage") or {}
            tin += usage.get("prompt_tokens", 0)
            tout += usage.get("completion_tokens", 0)
            model = resp.get("model", model)
            try:
                choice = resp["choices"][0]
                text = choice["message"].get("content") or ""
            except (KeyError, IndexError, TypeError) as e:
                raise LLMError(f"unexpected response shape: {str(resp)[:300]}") from e
            try:
                parsed = parse_reply(text, schema)
                return LLMResult(parsed=parsed, model=model, input_tokens=tin, output_tokens=tout,
                                 latency_s=round(time.monotonic() - t0, 2), raw=text)
            except (ValidationError, ValueError) as e:
                if attempt == 1:
                    reason = " (output cut at max_tokens)" if choice.get("finish_reason") == "length" else ""
                    raise LLMError(f"reply did not validate as {schema.__name__} after a retry{reason}: "
                                   f"{str(e)[:500]}") from e
                messages = messages + [{"role": "assistant", "content": text},
                                       {"role": "user", "content": retry_message(e)}]
        raise LLMError("unreachable")
