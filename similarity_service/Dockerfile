# CauliCare Similarity: DINOv2-small (ONNX Runtime) + FAISS. No TensorFlow.
# Build from the repository root:  docker build -t caulicare-similarity similarity_service
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PORT=8001 \
    MALLOC_ARENA_MAX=2 \
    OMP_NUM_THREADS=1

WORKDIR /srv
COPY requirements.txt similarity_service/requirements.txt
RUN pip install -r similarity_service/requirements.txt

# DINOv2-small (Apache-2.0): downloaded once at build time and SHA-256 verified;
# a checksum mismatch fails the build.
COPY download_model.py similarity_service/download_model.py
COPY models/preprocessor_config.json similarity_service/models/preprocessor_config.json
RUN python similarity_service/download_model.py

RUN useradd --create-home --uid 1000 app
COPY --chown=app:app . similarity_service/
USER app

HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
  CMD python -c "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:%s/health' % os.environ.get('PORT', '8001'), timeout=4)" || exit 1

EXPOSE 8001
CMD ["sh", "-c", "exec gunicorn similarity_service.app:app --bind 0.0.0.0:${PORT} --workers 1 --threads 2 --timeout 120 --graceful-timeout 30 --keep-alive 5"]
