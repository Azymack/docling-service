from __future__ import annotations

import asyncio
import logging
from typing import Annotated

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile, status

from app.models.schemas import ConvertResponse, OcrMode, PageResult, TableMode
from app.services.converter import DoclingConverterService

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/convert-gpu", response_model=ConvertResponse)
async def convert_pdf_gpu(
    request: Request,
    file: Annotated[UploadFile, File(description="PDF file to convert using GPU")],
    ocr_mode: Annotated[OcrMode, Form()] = OcrMode.AUTO,
    table_mode: Annotated[TableMode, Form()] = TableMode.FAST,
) -> ConvertResponse:
    """
    GPU-accelerated PDF conversion using docling.

    Accepts the same parameters and returns the same response shape as
    POST /convert, so outputs from both endpoints can be diffed directly.

    Extra response fields versus /convert:
    - device_used: per-component device map (layout / table_structure / ocr).
      Reports "cpu (configured=cuda, CUDA not available at runtime)" if the
      host has no GPU, rather than silently claiming GPU was used.
    - gpu_peak_memory_mb: peak CUDA memory allocated during this request,
      reset before each conversion.  Null if CUDA is unavailable.

    Note on GPU utilisation: docling's GPU support for its AI models is
    work-in-progress.  Real-world measurements have found GPU utilisation
    near 0% in some configurations.  Use /health to confirm device_mapping
    and monitor gpu_peak_memory_mb to verify GPU is actually being exercised.
    """
    gpu_converter: DoclingConverterService | None = getattr(
        request.app.state, "gpu_converter", None
    )
    if gpu_converter is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "No GPU converter is available. Either no GPU was detected at "
                "startup or GPU initialisation failed. Check GET /health for "
                "details. Use POST /convert for CPU-based conversion."
            ),
        )

    settings = request.app.state.settings
    file_bytes = await file.read()
    filename = file.filename or "upload.pdf"
    file_size_mb = len(file_bytes) / (1024 * 1024)

    if file_size_mb > settings.MAX_FILE_SIZE_MB:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=(
                f"File size {file_size_mb:.1f} MB exceeds the limit of "
                f"{settings.MAX_FILE_SIZE_MB} MB."
            ),
        )

    if not file_bytes.startswith(b"%PDF"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file does not appear to be a valid PDF.",
        )

    # Snapshot device mapping before the conversion — reflects what devices
    # each component is *configured* to use plus runtime GPU availability.
    device_info = gpu_converter.probe_devices()

    logger.info(
        "GPU conversion request: filename=%s size_mb=%.2f ocr_mode=%s "
        "table_mode=%s device=%s device_mapping=%s in_flight=%d",
        filename,
        file_size_mb,
        ocr_mode.value,
        table_mode.value,
        gpu_converter.device,
        device_info,
        gpu_converter.in_flight,
    )

    try:
        result = await gpu_converter.convert(file_bytes, filename, ocr_mode, table_mode)
    except asyncio.TimeoutError:
        logger.warning(
            "GPU queue full — rejecting after timeout: filename=%s", filename
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "GPU service is at capacity. All GPU slots are busy. "
                "Try again shortly, or use POST /convert for CPU conversion."
            ),
        )
    except Exception:
        logger.exception("GPU conversion failed: filename=%s", filename)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="GPU conversion failed due to an internal error.",
        )

    logger.info(
        "GPU conversion complete: filename=%s pages=%d duration_s=%.2f gpu_peak_mb=%s",
        filename,
        result["page_count"],
        result["processing_time_seconds"],
        result.get("gpu_peak_memory_mb"),
    )

    return ConvertResponse(
        status="success",
        filename=filename,
        page_count=result["page_count"],
        ocr_mode_used=ocr_mode.value,
        processing_time_seconds=result["processing_time_seconds"],
        pages=[PageResult(**p) for p in result["pages"]],
        full_markdown=result["full_markdown"],
        structured_json=result["structured_json"],
        device_used=device_info,
        gpu_peak_memory_mb=result.get("gpu_peak_memory_mb"),
    )
