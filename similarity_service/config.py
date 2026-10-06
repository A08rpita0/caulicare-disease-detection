"""Similarity-service settings, read from environment variables."""
import os
from pathlib import Path

SERVICE_DIR = Path(__file__).resolve().parent


class Config:
    # Relative paths are resolved against this folder, not the process cwd.
    EMBEDDING_MODEL_PATH = SERVICE_DIR / os.getenv("EMBEDDING_MODEL_PATH", "models/dinov2-small.onnx")
    IMAGE_INDEX_DIR = SERVICE_DIR / os.getenv("IMAGE_INDEX_DIR", "index")

    # Shared secret sent by the main app in the X-Service-Token header. When set,
    # /search rejects requests without it. Empty = open (local development only).
    SERVICE_TOKEN = os.getenv("SERVICE_TOKEN", "").strip()

    MAX_UPLOAD_MB = float(os.getenv("MAX_UPLOAD_MB", "8"))
    MAX_CONTENT_LENGTH = int(MAX_UPLOAD_MB * 1024 * 1024) + 64 * 1024  # file + multipart overhead
    ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP"}  # verified from the decoded bytes
    MAX_IMAGE_PIXELS = 40_000_000  # decompression-bomb guard
    DEFAULT_K = 5
    MAX_K = 10
    # Searches allowed to run at once (1 keeps a 512 MB instance safe).
    MAX_CONCURRENT_SEARCHES = int(os.getenv("MAX_CONCURRENT_SEARCHES", "1"))
    PRELOAD = os.getenv("PRELOAD", "true").strip().lower() in {"1", "true", "yes", "on"}


class TestConfig(Config):
    TESTING = True
    PRELOAD = False
