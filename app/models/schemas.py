from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel


class OcrMode(str, Enum):
    AUTO = "auto"
    FORCE = "force"
    OFF = "off"


class TableMode(str, Enum):
    FAST = "fast"
    ACCURATE = "accurate"


class PageResult(BaseModel):
    page_number: int
    markdown: str
    had_ocr: bool


class ConvertResponse(BaseModel):
    status: str
    filename: str
    page_count: int
    ocr_mode_used: str
    processing_time_seconds: float
    pages: list[PageResult]
    full_markdown: str
    structured_json: dict[str, Any]
    # GPU-specific fields — None for CPU endpoint, populated for /convert-gpu.
    # Kept on the shared model so both responses are directly diffable.
    device_used: dict[str, str] | None = None
    gpu_peak_memory_mb: float | None = None


class GpuStatus(BaseModel):
    device: str
    converter_ready: bool
    in_flight: int
    max_concurrent: int
    # Per-component device map; shows "cpu (CUDA not available)" when the
    # host has no GPU so callers can tell the difference between "ran on GPU"
    # and "fell back silently to CPU".
    device_mapping: dict[str, str]


class HealthResponse(BaseModel):
    status: str
    converter_ready: bool
    in_flight: int
    max_concurrent: int
    gpu: GpuStatus | None = None


class MetricsResponse(BaseModel):
    total_requests: int
    total_failures: int
    average_processing_time_seconds: float
    current_in_flight: int
