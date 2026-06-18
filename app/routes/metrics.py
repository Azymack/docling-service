from __future__ import annotations

from fastapi import APIRouter, Request

from app.models.schemas import MetricsResponse
from app.services.converter import DoclingConverterService

router = APIRouter()


@router.get("/metrics", response_model=MetricsResponse)
async def metrics(request: Request) -> MetricsResponse:
    converter: DoclingConverterService = request.app.state.converter
    m = converter.metrics
    return MetricsResponse(**m)
