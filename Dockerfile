# -----------------------------------------------------------------------
# docling-service — CPU-only image.  No CUDA, no GPU.
# -----------------------------------------------------------------------
FROM python:3.11-slim-bookworm

# System libraries needed by docling's PDF/image stack
RUN apt-get update && apt-get install -y --no-install-recommends \
        # OpenCV / image processing (used by docling's layout engine)
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        # PDF rasterisation (poppler is used as a fallback renderer)
        poppler-utils \
        # Build tools for wheels that don't have a pre-built binary
        gcc \
        g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (leverages Docker layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download and cache docling model weights at image build time so the
# first request doesn't stall waiting for HuggingFace downloads.
# Set a predictable cache directory so it ends up in this layer.
ENV HF_HOME=/app/.cache/huggingface
ENV DOCLING_ARTIFACTS_PATH=/app/.cache/docling

# Build-time model warm-up: instantiate the default (auto OCR, fast tables)
# converter to trigger all model downloads.  This adds ~3–5 GB to the image
# but eliminates cold-start latency in production.
#
# To skip this step (e.g. models are volume-mounted at runtime), comment out
# the COPY + RUN block below and mount the model cache at /app/.cache instead.
COPY scripts/warmup_models.py /tmp/warmup_models.py
RUN OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python /tmp/warmup_models.py \
    && rm /tmp/warmup_models.py

# Copy application source (after deps + model download so edits don't bust
# the model-download cache layer)
COPY app/ app/

EXPOSE 8001

# Single worker — docling manages its own concurrency via the internal thread
# pool; running multiple uvicorn workers would duplicate model memory usage.
CMD ["uvicorn", "app.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8001", \
     "--workers", "1", \
     "--log-level", "info"]
