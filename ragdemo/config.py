import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

from ragdemo.errors import DemoError

ROOT = Path(__file__).resolve().parents[1]
CHAT_MODEL = "minimax/minimax-m3.1-flash:free"
EMBEDDING_MODEL = "liquid/lfm-2.5-embedding-350m:free"
TOKENIZER_MODEL = "LiquidAI/LFM2.5-Embedding-350M"
TOKENIZER_REVISION = "f35ae2c91d687658dbf1f2b449382f0b019b9808"
TOP_K = 3
# Bump when session-held classes or pipeline behavior change during development.
PIPELINE_VERSION = 2


@dataclass(frozen=True)
class Settings:
    api_key: str = field(repr=False)
    chat_model: str = CHAT_MODEL
    embedding_model: str = EMBEDDING_MODEL
    min_cosine: float = 0.30
    safety_checks_enabled: bool = True
    max_chars: int = 200_000
    max_question_chars: int = 4_000
    max_pdf_bytes: int = 10 * 1024 * 1024
    max_pdf_pages: int = 100

    @classmethod
    def load(cls) -> "Settings":
        load_dotenv(ROOT / ".env", override=False)
        try:
            threshold = float(os.getenv("RAG_MIN_COSINE", "0.30"))
            if not 0 <= threshold <= 1:
                raise ValueError
        except ValueError:
            raise DemoError("RAG_MIN_COSINE must be a number between 0 and 1.") from None
        safety_value = os.getenv("RAG_SAFETY_CHECKS_ENABLED", "true").strip().lower()
        if safety_value not in {"true", "false", "1", "0"}:
            raise DemoError("RAG_SAFETY_CHECKS_ENABLED must be true or false.")
        return cls(
            api_key=os.getenv("OPENROUTER_API_KEY", "").strip(),
            chat_model=os.getenv("OPENROUTER_CHAT_MODEL", CHAT_MODEL).strip() or CHAT_MODEL,
            embedding_model=os.getenv("OPENROUTER_EMBEDDING_MODEL", EMBEDDING_MODEL).strip()
            or EMBEDDING_MODEL,
            min_cosine=threshold,
            safety_checks_enabled=safety_value in {"true", "1"},
        )

    def require_key(self) -> None:
        if not self.api_key:
            raise DemoError("Add OPENROUTER_API_KEY to .env, then restart the dashboard.")
