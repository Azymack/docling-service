from __future__ import annotations

import asyncio
import logging
from typing import Annotated

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile, status

from app.models.schemas import ConvertResponse, OcrMode, PageResult, TableMode
from app.services.converter import DoclingConverterService

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/convert", response_model=ConvertResponse)
async def convert_pdf(
    request: Request,
    file: Annotated[UploadFile, File(description="PDF file to convert")],
    ocr_mode: Annotated[OcrMode, Form()] = OcrMode.AUTO,
    table_mode: Annotated[TableMode, Form()] = TableMode.FAST,
) -> ConvertResponse:
    settings = request.app.state.settings
    converter: DoclingConverterService = request.app.state.converter

    file_bytes = await file.read()
    filename = file.filename or "upload.pdf"
    file_size_mb = len(file_bytes) / (1024 * 1024)

    # Size guard — before touching docling so we don't waste memory
    if file_size_mb > settings.MAX_FILE_SIZE_MB:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=(
                f"File size {file_size_mb:.1f} MB exceeds the limit of "
                f"{settings.MAX_FILE_SIZE_MB} MB."
            ),
        )

    # Magic-bytes check is more reliable than Content-Type, which some clients
    # send as application/octet-stream even for valid PDFs.
    if not file_bytes.startswith(b"%PDF"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file does not appear to be a valid PDF.",
        )

    logger.info(
        "Conversion request: filename=%s size_mb=%.2f ocr_mode=%s table_mode=%s in_flight=%d",
        filename,
        file_size_mb,
        ocr_mode.value,
        table_mode.value,
        converter.in_flight,
    )

    try:
        result = await converter.convert(file_bytes, filename, ocr_mode, table_mode)
    except asyncio.TimeoutError:
        logger.warning(
            "Queue full — rejecting after timeout: filename=%s", filename
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Service is at capacity. All conversion slots are busy. "
                "Try again shortly."
            ),
        )
    except Exception:
        logger.exception("Conversion failed: filename=%s", filename)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Conversion failed due to an internal error.",
        )

    logger.info(
        "Conversion complete: filename=%s pages=%d duration_s=%.2f",
        filename,
        result["page_count"],
        result["processing_time_seconds"],
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
    )
