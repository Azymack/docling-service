"""
Shared fixtures for the docling-service test suite.

Model loading is expensive.  The 'client' fixture is session-scoped so the
FastAPI lifespan runs exactly once per pytest session, pre-warming the default
DocumentConverter and reusing it for all tests.
"""

from __future__ import annotations

import io

import pytest
from fastapi.testclient import TestClient


def make_pdf(text: str = "Insurance Policy\nCoverage: $1,000,000") -> bytes:
    """Generate a minimal single-page native-text PDF with reportlab."""
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas

    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    c.setFont("Helvetica", 12)
    y = 750
    for line in text.splitlines():
        c.drawString(72, y, line)
        y -= 18
    c.showPage()
    c.save()
    buf.seek(0)
    return buf.getvalue()


@pytest.fixture(scope="session")
def client():
    from app.main import app

    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="session")
def sample_pdf() -> bytes:
    return make_pdf()
