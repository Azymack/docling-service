"""
Basic API tests for docling-service.

These tests use the real DocumentConverter (no mocking) because the whole point
of this service is docling integration.  They require model weights to be
available (either from HuggingFace cache or pre-baked into the Docker image).
They are slow on first run (~30–60 s for model load) and much faster on
subsequent runs once the models are cached on disk.
"""

from __future__ import annotations

import io

import pytest
from fastapi.testclient import TestClient

from tests.conftest import make_pdf


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------


def test_health_ready(client: TestClient) -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ready"
    assert data["converter_ready"] is True
    assert isinstance(data["in_flight"], int)
    assert isinstance(data["max_concurrent"], int)
    assert data["max_concurrent"] >= 1


# ---------------------------------------------------------------------------
# /metrics
# ---------------------------------------------------------------------------


def test_metrics_shape(client: TestClient) -> None:
    resp = client.get("/metrics")
    assert resp.status_code == 200
    data = resp.json()
    for key in ("total_requests", "total_failures", "average_processing_time_seconds", "current_in_flight"):
        assert key in data


# ---------------------------------------------------------------------------
# /convert — happy path
# ---------------------------------------------------------------------------


def test_convert_basic_response_shape(client: TestClient, sample_pdf: bytes) -> None:
    resp = client.post(
        "/convert",
        files={"file": ("test.pdf", sample_pdf, "application/pdf")},
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()

    assert data["status"] == "success"
    assert data["filename"] == "test.pdf"
    assert isinstance(data["page_count"], int)
    assert data["page_count"] >= 1
    assert data["ocr_mode_used"] == "auto"
    assert isinstance(data["processing_time_seconds"], float)
    assert data["processing_time_seconds"] >= 0

    assert isinstance(data["full_markdown"], str)
    assert isinstance(data["structured_json"], dict)

    pages = data["pages"]
    assert isinstance(pages, list)
    assert len(pages) == data["page_count"]

    for page in pages:
        assert isinstance(page["page_number"], int)
        assert isinstance(page["markdown"], str)
        assert isinstance(page["had_ocr"], bool)


def test_convert_ocr_off(client: TestClient, sample_pdf: bytes) -> None:
    resp = client.post(
        "/convert",
        files={"file": ("native.pdf", sample_pdf, "application/pdf")},
        data={"ocr_mode": "off"},
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["ocr_mode_used"] == "off"
    for page in data["pages"]:
        assert page["had_ocr"] is False


def test_convert_table_mode_accurate(client: TestClient, sample_pdf: bytes) -> None:
    resp = client.post(
        "/convert",
        files={"file": ("accurate.pdf", sample_pdf, "application/pdf")},
        data={"table_mode": "accurate"},
    )
    assert resp.status_code == 200, resp.text


def test_convert_page_count_matches_pages_list(
    client: TestClient, sample_pdf: bytes
) -> None:
    resp = client.post(
        "/convert",
        files={"file": ("pages.pdf", sample_pdf, "application/pdf")},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["pages"]) == data["page_count"]


# ---------------------------------------------------------------------------
# /convert — error cases
# ---------------------------------------------------------------------------


def test_convert_413_oversized(client: TestClient) -> None:
    """File that exceeds MAX_FILE_SIZE_MB should return 413."""
    from app.main import app

    # Temporarily drop the limit to 1 byte via patching app state
    original = app.state.settings.MAX_FILE_SIZE_MB
    app.state.settings.MAX_FILE_SIZE_MB = 0  # anything > 0 bytes will fail

    try:
        tiny_pdf = make_pdf("x")
        resp = client.post(
            "/convert",
            files={"file": ("big.pdf", tiny_pdf, "application/pdf")},
        )
        assert resp.status_code == 413
    finally:
        app.state.settings.MAX_FILE_SIZE_MB = original


def test_convert_400_not_a_pdf(client: TestClient) -> None:
    """A file that doesn't start with %PDF should return 400."""
    fake = b"This is not a PDF file at all."
    resp = client.post(
        "/convert",
        files={"file": ("fake.pdf", fake, "application/pdf")},
    )
    assert resp.status_code == 400


def test_convert_400_empty_file(client: TestClient) -> None:
    resp = client.post(
        "/convert",
        files={"file": ("empty.pdf", b"", "application/pdf")},
    )
    assert resp.status_code == 400


# ---------------------------------------------------------------------------
# /convert-gpu
# ---------------------------------------------------------------------------


def _gpu_available(client: TestClient) -> bool:
    """Return True if the GPU converter initialised successfully."""
    resp = client.get("/health")
    return resp.json().get("gpu") is not None


def test_convert_gpu_no_gpu_returns_503_or_200(
    client: TestClient, sample_pdf: bytes
) -> None:
    """
    Without a GPU the endpoint must return 503.
    With a GPU it must return 200 with the same shape as /convert plus
    device_used and gpu_peak_memory_mb fields.
    """
    resp = client.post(
        "/convert-gpu",
        files={"file": ("test_gpu.pdf", sample_pdf, "application/pdf")},
    )
    if not _gpu_available(client):
        assert resp.status_code == 503
        return

    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["status"] == "success"
    assert data["filename"] == "test_gpu.pdf"
    assert isinstance(data["page_count"], int)
    assert data["page_count"] >= 1
    assert isinstance(data["full_markdown"], str)
    assert isinstance(data["structured_json"], dict)
    assert len(data["pages"]) == data["page_count"]

    # GPU-specific fields
    assert isinstance(data["device_used"], dict)
    for key in ("layout", "table_structure", "ocr"):
        assert key in data["device_used"], f"device_used missing key: {key}"
    # gpu_peak_memory_mb may be None if CUDA memory stats are unavailable
    assert "gpu_peak_memory_mb" in data


def test_convert_gpu_same_shape_as_cpu(
    client: TestClient, sample_pdf: bytes
) -> None:
    """Both endpoints must return the same core fields so outputs are diffable."""
    if not _gpu_available(client):
        pytest.skip("No GPU converter available")

    cpu_resp = client.post(
        "/convert",
        files={"file": ("shape_test.pdf", sample_pdf, "application/pdf")},
    )
    gpu_resp = client.post(
        "/convert-gpu",
        files={"file": ("shape_test.pdf", sample_pdf, "application/pdf")},
    )
    assert cpu_resp.status_code == 200
    assert gpu_resp.status_code == 200

    cpu_data = cpu_resp.json()
    gpu_data = gpu_resp.json()

    # Core fields must be identical in structure
    for field in ("status", "filename", "page_count", "ocr_mode_used", "pages"):
        assert field in cpu_data
        assert field in gpu_data

    assert cpu_data["page_count"] == gpu_data["page_count"]


def test_health_reports_gpu_status(client: TestClient) -> None:
    """
    /health must always include a 'gpu' field.
    If no GPU: gpu is null.
    If GPU: gpu contains device, converter_ready, in_flight, max_concurrent,
            and device_mapping.
    """
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert "gpu" in data  # field must exist (may be null)

    gpu = data["gpu"]
    if gpu is not None:
        for key in ("device", "converter_ready", "in_flight", "max_concurrent", "device_mapping"):
            assert key in gpu, f"gpu health missing key: {key}"
        assert isinstance(gpu["device_mapping"], dict)
        for comp in ("layout", "table_structure", "ocr"):
            assert comp in gpu["device_mapping"]
