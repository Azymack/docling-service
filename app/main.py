"""
docling-service entry point.

OMP/MKL thread limits are applied HERE, at the very top of the import chain,
before numpy, torch, or onnxruntime are loaded.  If they are set after those
libraries initialise their thread pools, the env vars have no effect.
"""

from __future__ import annotations

import os

# Read from env first so docker-compose / k8s overrides are honoured.
# Fall back to 2 threads per library — a conservative default that prevents
# oversubscription when MAX_CONCURRENT_REQUESTS (default 4) threads all call
# into native math code simultaneously (4 × 2 = 8 OS threads, not 4 × <core-count>).
_omp = os.environ.get("OMP_NUM_THREADS", "2")
_mkl = os.environ.get("MKL_NUM_THREADS", "2")

os.environ["OMP_NUM_THREADS"] = _omp
os.environ["MKL_NUM_THREADS"] = _mkl
os.environ["OPENBLAS_NUM_THREADS"] = _omp
os.environ["NUMEXPR_NUM_THREADS"] = _omp

# -------------------------------------------------------------------
# All other imports follow — docling and its transitive dependencies
# (numpy, torch, onnxruntime) will now pick up the limits above.
# -------------------------------------------------------------------

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.config import Settings
from app.routes.convert import router as convert_router
from app.routes.convert_gpu import router as convert_gpu_router
from app.routes.health import router as health_router
from app.routes.metrics import router as metrics_router
from app.services.converter import DoclingConverterService, detect_gpu_device


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = Settings()
    _configure_logging(settings.LOG_LEVEL)

    logger = logging.getLogger(__name__)
    logger.info(
        "Starting docling-service: max_concurrent=%d queue_timeout_s=%.0f "
        "max_file_mb=%d omp_threads=%s mkl_threads=%s",
        settings.MAX_CONCURRENT_REQUESTS,
        settings.QUEUE_TIMEOUT_SECONDS,
        settings.MAX_FILE_SIZE_MB,
        _omp,
        _mkl,
    )

    # CPU converter — always created.  Explicitly sets device=cpu so the
    # service stays on CPU even on a host where DOCLING_DEVICE=cuda is set.
    converter = DoclingConverterService(
        settings,
        accelerator_device="cpu",
        name="cpu",
    )
    converter.initialize()

    app.state.settings = settings
    app.state.converter = converter

    # GPU converter — created only when a GPU is detected.  Initialisation
    # failure is non-fatal: the CPU service continues running and /convert-gpu
    # returns 503.
    app.state.gpu_converter = None
    gpu_device = detect_gpu_device()
    if gpu_device:
        logger.info("GPU detected: %s — initialising GPU converter.", gpu_device)
        try:
            gpu_converter = DoclingConverterService(
                settings,
                accelerator_device=gpu_device,
                max_concurrent_override=settings.MAX_CONCURRENT_GPU_REQUESTS,
                queue_timeout_override=settings.GPU_QUEUE_TIMEOUT_SECONDS,
                name="gpu",
            )
            gpu_converter.initialize()
            app.state.gpu_converter = gpu_converter
            logger.info(
                "GPU converter ready: device=%s max_concurrent=%d",
                gpu_device,
                settings.MAX_CONCURRENT_GPU_REQUESTS,
            )
        except Exception:
            logger.exception(
                "GPU converter initialisation failed — /convert-gpu will return 503."
            )
    else:
        logger.info("No GPU detected — /convert-gpu will return 503.")

    yield

    logger.info("Shutting down docling-service.")
    converter.shutdown()
    if app.state.gpu_converter is not None:
        app.state.gpu_converter.shutdown()


app = FastAPI(
    title="docling-service",
    description=(
        "PDF-to-text/markdown conversion microservice using docling. "
        "POST /convert uses CPU; POST /convert-gpu uses GPU when available. "
        "Both return identical response shapes for direct benchmarking."
    ),
    version="1.1.0",
    lifespan=lifespan,
)

app.include_router(convert_router)
app.include_router(convert_gpu_router)
app.include_router(health_router)
app.include_router(metrics_router)
