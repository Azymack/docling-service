from __future__ import annotations

from fastapi import APIRouter, Request

from app.models.schemas import GpuStatus, HealthResponse
from app.services.converter import DoclingConverterService

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    converter: DoclingConverterService = request.app.state.converter
    gpu_converter: DoclingConverterService | None = getattr(
        request.app.state, "gpu_converter", None
    )

    ready = converter.is_ready

    gpu_status: GpuStatus | None = None
    if gpu_converter is not None:
        gpu_status = GpuStatus(
            device=gpu_converter.device,
            converter_ready=gpu_converter.is_ready,
            in_flight=gpu_converter.in_flight,
            max_concurrent=gpu_converter.max_concurrent,
            device_mapping=gpu_converter.probe_devices(),
        )

    return HealthResponse(
        status="ready" if ready else "starting",
        converter_ready=ready,
        in_flight=converter.in_flight,
        max_concurrent=converter.max_concurrent,
        gpu=gpu_status,
    )
