# Production image for CauliCare (Flask + gunicorn + TensorFlow CPU).
# Visual similarity search runs in a separate image: similarity_service/Dockerfile.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    PORT=8080 \
    MALLOC_ARENA_MAX=2 \
    OMP_NUM_THREADS=1 \
    TF_NUM_INTRAOP_THREADS=1 \
    TF_NUM_INTEROP_THREADS=1

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

# Run as an unprivileged user.
RUN useradd --create-home --uid 1000 app
COPY --chown=app:app . .
USER app

HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
  CMD python -c "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:%s/api/health' % os.environ.get('PORT', '8080'), timeout=4)" || exit 1

EXPOSE 8080
# One worker keeps a single copy of the model in RAM; threads serve page loads while a
# prediction runs. `exec` lets gunicorn receive the platform's shutdown signal.
CMD ["sh", "-c", "exec gunicorn app:app --bind 0.0.0.0:${PORT} --workers 1 --threads 4 --timeout 300 --graceful-timeout 30 --keep-alive 5"]
