"""Runtime settings, read from environment variables (and .env if present)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # python-dotenv is optional
    pass

ROOT = Path(__file__).resolve().parents[2]

# Provider presets for any OpenAI-compatible chat API.
PROVIDERS = {
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "key_env": "GEMINI_API_KEY",
        "model": "gemini-2.5-flash",
        "embed_model": "gemini-embedding-001",
    },
    "openai": {
        "base_url": None,
        "key_env": "OPENAI_API_KEY",
        "model": "gpt-4.1-mini",
        "embed_model": "text-embedding-3-small",
    },
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "key_env": "GROQ_API_KEY",
        "model": "llama-3.3-70b-versatile",
        "embed_model": None,
    },
    "ollama": {
        "base_url": "http://localhost:11434/v1",
        "key_env": "OLLAMA_API_KEY",
        "model": "qwen2.5:7b-instruct",
        "embed_model": None,
    },
}


def _env(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    return value if value not in (None, "") else default


@dataclass
class Settings:
    provider: str = field(default_factory=lambda: _env("LLM_PROVIDER", "gemini"))
    model: str | None = field(default_factory=lambda: _env("LLM_MODEL"))
    base_url: str | None = field(default_factory=lambda: _env("LLM_BASE_URL"))
    temperature: float = field(default_factory=lambda: float(_env("LLM_TEMPERATURE", "0")))
    # Dense retrieval backend: fastembed (local, multilingual), api (provider), none (BM25 only)
    embeddings: str = field(default_factory=lambda: _env("EMBEDDINGS", "fastembed"))
    fastembed_model: str = field(default_factory=lambda: _env(
        "FASTEMBED_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"))
    top_k: int = field(default_factory=lambda: int(_env("TOP_K", "6")))
    dataset_dir: Path = field(default_factory=lambda: Path(_env(
        "DATASET_DIR", str(ROOT / "Capstone_Project_Dataset_FTTT_2026"))).resolve())
    cache_dir: Path = field(default_factory=lambda: Path(_env("CACHE_DIR", str(ROOT / ".cache"))))
    today: str = field(default_factory=lambda: _env("TODAY", "2026-10-20"))
    # Gnani voice settings
    stt_language: str = field(default_factory=lambda: _env("STT_LANGUAGE", "en-IN"))
    tts_model: str = field(default_factory=lambda: _env("TTS_MODEL", "timbre-v2.5"))
    tts_voice: str = field(default_factory=lambda: _env("TTS_VOICE", "Kaveri"))
    tts_voice_hi: str = field(default_factory=lambda: _env("TTS_VOICE_HI", "Nalini"))
    tts_voice_hinglish: str = field(default_factory=lambda: _env("TTS_VOICE_HINGLISH", "Poorvi"))

    @property
    def preset(self) -> dict:
        return PROVIDERS.get(self.provider, PROVIDERS["openai"])

    @property
    def llm_model(self) -> str:
        return self.model or self.preset["model"]

    @property
    def llm_base_url(self) -> str | None:
        return self.base_url or self.preset["base_url"]

    @property
    def api_key(self) -> str | None:
        return _env(self.preset["key_env"]) or _env("LLM_API_KEY")


settings = Settings()
