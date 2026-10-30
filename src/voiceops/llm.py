"""Thin wrapper over any OpenAI-compatible chat API (Gemini, OpenAI, Groq, Ollama...)."""
from __future__ import annotations

import json
import logging
import random
import re
import time

from .config import settings

log = logging.getLogger(__name__)


class LLMError(RuntimeError):
    pass


class LLM:
    def __init__(self, model: str | None = None):
        from openai import OpenAI

        if not settings.api_key and settings.provider != "ollama":
            raise LLMError(
                f"No API key: set {settings.preset['key_env']} (provider '{settings.provider}') "
                "in your environment or .env file.")
        self.model = model or settings.llm_model
        self.client = OpenAI(api_key=settings.api_key or "ollama", base_url=settings.llm_base_url,
                             max_retries=0, timeout=90)
        self.calls = 0

    @property
    def can_embed(self) -> bool:
        return bool(settings.preset.get("embed_model"))

    def _retry(self, fn, *args, **kwargs):
        delay = 4.0
        for attempt in range(8):
            try:
                self.calls += 1
                return fn(*args, **kwargs)
            except Exception as exc:  # noqa: BLE001
                status = getattr(exc, "status_code", None)
                retryable = status in (408, 409, 429, 500, 502, 503, 504) or status is None
                limit = 7 if status is not None else 3  # connection errors: give up sooner
                if not retryable or attempt >= limit:
                    raise
                wait = delay + random.random()
                match = re.search(r"retry in ([\d.]+)s", str(exc))
                if match:
                    wait = float(match.group(1)) + 1
                log.warning("LLM call failed (%s); retrying in %.0fs", status or exc, wait)
                time.sleep(wait)
                delay = min(delay * 2, 60)

    def chat(self, messages: list[dict], tools: list[dict] | None = None,
             temperature: float | None = None, json_mode: bool = False):
        kwargs = {"model": self.model, "messages": messages,
                  "temperature": settings.temperature if temperature is None else temperature}
        if tools:
            kwargs["tools"] = tools
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        response = self._retry(self.client.chat.completions.create, **kwargs)
        return response.choices[0].message

    def complete(self, prompt: str, system: str | None = None, json_mode: bool = False) -> str:
        messages = ([{"role": "system", "content": system}] if system else []) + [
            {"role": "user", "content": prompt}]
        return (self.chat(messages, json_mode=json_mode).content or "").strip()

    def embed(self, texts: list[str]) -> list[list[float]]:
        out = []
        for i in range(0, len(texts), 64):
            response = self._retry(self.client.embeddings.create,
                                   model=settings.preset["embed_model"], input=texts[i:i + 64])
            out.extend(d.embedding for d in response.data)
        return out


def parse_json(text: str) -> dict:
    """Pull the first JSON object out of a model reply (tolerates ```json fences)."""
    if not text:
        return {}
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    while start != -1:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i + 1])
                    except json.JSONDecodeError:
                        break
        start = text.find("{", start + 1)
    return {}
