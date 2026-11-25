"""Shared chat logic for the web API and desktop app."""

import os
from dataclasses import asdict
from pathlib import Path

from answering import build_model_input, generate_answer
from retrieval import LocalIndex

ROOT = Path(__file__).resolve().parent
DEFAULT_MODELS = {
    "openrouter": "openrouter/free",
    "groq": "openai/gpt-oss-20b",
    "gemini": "gemini-3.1-flash-lite",
    "openai": "gpt-4o-mini",
}
ENV_SETTINGS = {
    f"{provider.upper()}_{suffix}" for provider in DEFAULT_MODELS for suffix in ("API_KEY", "MODEL")
}
KEY_PLACEHOLDERS = {f"your_{provider}_api_key_here" for provider in DEFAULT_MODELS}
QUESTION_MAX_LENGTH = 1500
QUESTION_ERROR = f"Enter a question from 1 to {QUESTION_MAX_LENGTH:,} characters."


class InvalidQuestion(ValueError):
    """The chat received an empty, oversized, or non-text question."""


class DeskGuard:
    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str | None = None,
        cache_path: Path = ROOT / ".cache" / "embeddings.json",
    ):
        settings = {}
        env_path = ROOT / ".env"
        if env_path.exists():
            for line in env_path.read_text(encoding="utf-8-sig").splitlines():
                key, _, value = line.partition("=")
                if key.strip() in ENV_SETTINGS:
                    settings[key.strip()] = value.strip().strip("\"'")

        def setting(name: str) -> str:
            return os.getenv(name, settings.get(name, "")).strip()

        provider = "openai"
        if api_key is None:
            api_key = ""
            for candidate in DEFAULT_MODELS:
                key = setting(f"{candidate.upper()}_API_KEY")
                if key and key not in KEY_PLACEHOLDERS:
                    api_key, provider = key, candidate
                    break
        if api_key.startswith("sk-or-"):
            provider = "openrouter"
        elif api_key.startswith("gsk_"):
            provider = "groq"

        self.provider = provider
        self.api_key = "" if api_key in KEY_PLACEHOLDERS else api_key
        self.model = model or setting(f"{provider.upper()}_MODEL") or DEFAULT_MODELS[provider]
        self.index = LocalIndex(ROOT / "articles", cache_path)
        self.article_count = len({chunk.article_id for chunk in self.index.chunks})

    @property
    def model_enabled(self) -> bool:
        return bool(self.api_key)

    def ask(self, question: object) -> dict:
        question = question.strip() if isinstance(question, str) else ""
        if not 1 <= len(question) <= QUESTION_MAX_LENGTH:
            raise InvalidQuestion(QUESTION_ERROR)

        matches = self.index.retrieve(question)
        model_input = build_model_input(question, matches, provider=self.provider)
        answer = generate_answer(model_input, matches, self.api_key, self.model, provider=self.provider)
        chunks_by_id = {match.chunk.id: match.chunk for match in matches}
        return {
            "answer": answer.text,
            "status": answer.status,
            "sources": [asdict(chunks_by_id[source_id]) for source_id in answer.source_ids],
            "retrieved": [asdict(match) for match in matches],
            "model_input": model_input,
            "model_requested": bool(self.api_key and matches),
        }
