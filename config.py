"""Application configuration, read from environment variables (see .env.example)."""
import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")


def _bool(name: str, default: bool = False) -> bool:
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


class Config:
    ENV = os.getenv("FLASK_ENV", "production")
    DEBUG = _bool("FLASK_DEBUG", False)
    SECRET_KEY = os.getenv("SECRET_KEY") or os.urandom(32).hex()

    # --- Model -------------------------------------------------------------
    # Relative paths are resolved against the project root, not the process cwd.
    MODEL_PATH = BASE_DIR / os.getenv("MODEL_PATH", "models/model1.h5")
    # Optional: download the model at startup if MODEL_PATH is missing
    # (e.g. a GitHub Release or object-storage URL).
    MODEL_URL = os.getenv("MODEL_URL", "")
    # Load the model at boot (True) or on the first prediction (False).
    PRELOAD_MODEL = _bool("PRELOAD_MODEL", True)

    # --- Similarity service (separate service, see similarity_service/) ------
    # It runs the cauliflower-leaf validator and visual similarity search.
    # Base URL, e.g. http://localhost:8001 locally. Empty = uploads cannot be
    # verified, so /api/predict refuses them with 503 (fail closed).
    SIMILARITY_SERVICE_URL = os.getenv("SIMILARITY_SERVICE_URL", "").strip()
    # Shared secret sent as X-Service-Token; must equal the service's SERVICE_TOKEN.
    SIMILARITY_SERVICE_TOKEN = os.getenv("SIMILARITY_SERVICE_TOKEN", "").strip()
    # Seconds to wait for the service. It runs in parallel with the CNN, so a slow or
    # sleeping service only delays the result up to this limit, then is skipped.
    SIMILARITY_TIMEOUT = float(os.getenv("SIMILARITY_TIMEOUT", "30"))
    SIMILAR_RESULTS = int(os.getenv("SIMILAR_RESULTS", "5"))
    # Seconds to wait for the cauliflower-leaf check (also in the similarity service).
    # Every analysis needs it, so it waits longer, allowing a free-tier service to wake.
    LEAF_CHECK_TIMEOUT = float(os.getenv("LEAF_CHECK_TIMEOUT", "75"))

    # --- Uploads -----------------------------------------------------------
    MAX_UPLOAD_MB = float(os.getenv("MAX_UPLOAD_MB", "8"))
    MAX_CONTENT_LENGTH = int(MAX_UPLOAD_MB * 1024 * 1024)
    ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}
    ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP"}  # verified from file bytes, not the name
    MAX_IMAGE_PIXELS = 40_000_000  # decompression-bomb guard

    # --- Rate limiting (per client IP, in-memory) --------------------------
    RATE_LIMIT_PREDICT = os.getenv("RATE_LIMIT_PREDICT", "30/minute")
    RATE_LIMIT_SEARCH = os.getenv("RATE_LIMIT_SEARCH", "60/minute")

    # Number of reverse proxies in front of the app that append to X-Forwarded-For.
    # 0 = direct (header ignored). Set to the number of proxies that append to the header.
    TRUSTED_PROXY_HOPS = int(os.getenv("TRUSTED_PROXY_HOPS", "0"))
    # Header set by a trusted edge proxy that carries the real client IP and that
    # clients cannot forge, e.g. "CF-Connecting-IP" on Render (Cloudflare).
    # Empty = use the socket / ProxyFix address. Only set it when such a proxy
    # is guaranteed to be in front of the app, otherwise clients could spoof it.
    CLIENT_IP_HEADER = os.getenv("CLIENT_IP_HEADER", "").strip()

    # Predictions allowed to run at the same time (1 keeps a 512 MB instance safe).
    MAX_CONCURRENT_PREDICTIONS = int(os.getenv("MAX_CONCURRENT_PREDICTIONS", "1"))

    # Comma-separated origins allowed to call /api/* cross-origin. Empty = same-origin only.
    CORS_ORIGINS = [o.strip() for o in os.getenv("CORS_ORIGINS", "").split(",") if o.strip()]


class TestConfig(Config):
    TESTING = True
    PRELOAD_MODEL = False
    SIMILARITY_SERVICE_URL = ""
    RATE_LIMIT_PREDICT = "1000/minute"
    RATE_LIMIT_SEARCH = "1000/minute"
