"""Application paths and environment-backed settings."""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def env_path(name: str, default: str) -> Path:
    value = os.getenv(name, default)
    path = Path(value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


DATASET_PATH = env_path("DATASET_PATH", "data/processed/case_items.json")
FRONTEND_DIR = env_path("FRONTEND_DIR", "frontend/dist/client")
HOST = os.getenv("HOST", "127.0.0.1")
PORT = int(os.getenv("PORT", "5051"))
DEBUG = os.getenv("DEBUG", "0") == "1"

AI_API_KEY = os.getenv("AI_API_KEY", "")
AI_BASE_URL = os.getenv("AI_BASE_URL", "https://api.deepseek.com")
AI_MODEL = os.getenv("AI_MODEL", "deepseek-v4-flash")
AI_TIMEOUT = int(os.getenv("AI_TIMEOUT", "60"))
AI_FIRST_TOKEN_TIMEOUT = float(os.getenv("AI_FIRST_TOKEN_TIMEOUT", "8"))
AI_FIRST_TOKEN_MAX_ATTEMPTS = min(max(int(os.getenv("AI_FIRST_TOKEN_MAX_ATTEMPTS", "2")), 1), 2)
AI_MAX_CONTEXT_CHARS = int(os.getenv("AI_MAX_CONTEXT_CHARS", "5200"))

# Browser duplex voice: streaming speech recognition through Xunfei IAT.
# Speech output is scheduled in the browser so barge-in can stop it immediately.
XF_APP_ID = os.getenv("XF_APP_ID", "")
XF_API_KEY = os.getenv("XF_API_KEY", "")
XF_API_SECRET = os.getenv("XF_API_SECRET", "")
XF_ASR_RES_ID = os.getenv("XF_ASR_RES_ID", "")
# Xunfei exposes the dialect and multilingual SLM recognisers through the
# same packet protocol.  Separate credentials remain optional because both
# capabilities can be enabled on one application, while dedicated service
# applications can override them without changing the browser contract.
XF_ASR_HOST = os.getenv("XF_ASR_HOST", "iat.cn-huabei-1.xf-yun.com")
XF_LEGACY_ASR_HOST = os.getenv("XF_LEGACY_ASR_HOST", "iat.xf-yun.com")
XF_MULTILINGUAL_APP_ID = os.getenv("XF_MULTILINGUAL_APP_ID", XF_APP_ID)
XF_MULTILINGUAL_API_KEY = os.getenv("XF_MULTILINGUAL_API_KEY", XF_API_KEY)
XF_MULTILINGUAL_API_SECRET = os.getenv("XF_MULTILINGUAL_API_SECRET", XF_API_SECRET)
XF_MULTILINGUAL_ASR_HOST = os.getenv("XF_MULTILINGUAL_ASR_HOST", XF_ASR_HOST)
XF_MULTILINGUAL_LANGUAGE_HINT = os.getenv(
    "XF_MULTILINGUAL_LANGUAGE_HINT",
    "en|ja|ko",
)

EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY", "")
EMBEDDING_BASE_URL = os.getenv("EMBEDDING_BASE_URL", "https://api.vectorengine.ai/v1")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
EMBEDDING_TIMEOUT = int(os.getenv("EMBEDDING_TIMEOUT", "60"))
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "64"))
EMBEDDING_WORKERS = int(os.getenv("EMBEDDING_WORKERS", "6"))
EMBEDDING_REQUEST_TIMEOUT = float(os.getenv("EMBEDDING_REQUEST_TIMEOUT", "5"))
EMBEDDING_MAX_RETRIES = int(os.getenv("EMBEDDING_MAX_RETRIES", "4"))
EMBEDDING_RETRY_BACKOFF = float(os.getenv("EMBEDDING_RETRY_BACKOFF", "3"))
EMBEDDING_REQUEST_DELAY = float(os.getenv("EMBEDDING_REQUEST_DELAY", "0"))
EMBEDDING_INDEX_PATH = env_path(
    "EMBEDDING_INDEX_PATH",
    "data/embeddings/case_embeddings.json",
)
EMBEDDING_TEXT_MAX_CHARS = int(os.getenv("EMBEDDING_TEXT_MAX_CHARS", "1400"))
EMBEDDING_MIN_SCORE = float(os.getenv("EMBEDDING_MIN_SCORE", "0.15"))
SEARCH_USE_EMBEDDING = os.getenv("SEARCH_USE_EMBEDDING", "0") == "1"
