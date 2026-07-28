# =========================================================
# Stage 1: Dependency builder
# =========================================================
FROM python:3.11-slim-bookworm AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TOKENIZERS_PARALLELISM=false \
    HF_HOME=/opt/huggingface-cache

WORKDIR /build

# Create an isolated Python environment.
RUN python -m venv /opt/venv

ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .

RUN python -m pip install --upgrade pip setuptools wheel

# Install CPU-only PyTorch.
RUN python -m pip install \
    --index-url https://download.pytorch.org/whl/cpu \
    torch

# Install application dependencies.
RUN python -m pip install \
    --no-cache-dir \
    -r requirements.txt

# Download embedding and reranker models during the build.
# This avoids downloading them when the container starts.
RUN python - <<'PY'
from sentence_transformers import CrossEncoder, SentenceTransformer

SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L6-v2"
)

print("Sentence Transformer models downloaded.")
PY


# =========================================================
# Stage 2: Production runtime
# =========================================================
FROM python:3.11-slim-bookworm AS runtime

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TOKENIZERS_PARALLELISM=false \
    HF_HOME=/opt/huggingface-cache \
    PATH="/opt/venv/bin:$PATH" \
    PORT=8000

# Runtime library required by XGBoost.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy only installed dependencies, not pip's build environment.
COPY --from=builder /opt/venv /opt/venv

# Copy the downloaded embedding models.
COPY --from=builder \
    /opt/huggingface-cache \
    /opt/huggingface-cache

# Copy runtime application files.
COPY app/ app/
COPY artifacts/ artifacts/
COPY vector_store/chroma/ vector_store/chroma/
COPY demo_cases/ demo_cases/

# Run as a non-root user.
RUN useradd \
        --create-home \
        --uid 10001 \
        appuser \
    && chown -R appuser:appuser \
        /app \
        /opt/huggingface-cache

USER appuser

EXPOSE 8000

HEALTHCHECK \
    --interval=30s \
    --timeout=10s \
    --start-period=120s \
    --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=5)" || exit 1

CMD ["uvicorn","app.main:app","--host","0.0.0.0","--port","8000"]