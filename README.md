# docling-service

FastAPI microservice that wraps [docling](https://github.com/DS4SD/docling) for
PDF-to-text/markdown conversion.  Sits upstream of a VLM-based field-extraction
step in an insurance-document pipeline.

> **Security note:** This service has no authentication.  It is designed for
> trusted internal network use only.  Do not expose it on a public interface
> without an API gateway or auth layer in front of it.

---

## Running locally

```bash
# Install dependencies (Python 3.11+)
pip install -r requirements.txt

# Start the server (models are downloaded from HuggingFace on first run)
uvicorn app.main:app --host 0.0.0.0 --port 8001 --workers 1
```

### With Docker (recommended for production)

```bash
# Build — downloads model weights into the image layer (~3–5 GB)
docker build -t docling-service .

# Run with a hard memory ceiling (adjust to your host)
docker run \
  --memory=24g --memory-swap=24g \
  -p 8001:8001 \
  -e MAX_CONCURRENT_REQUESTS=4 \
  -e OMP_NUM_THREADS=2 \
  -e MKL_NUM_THREADS=2 \
  docling-service

# Or use docker-compose (includes healthcheck and model-cache volume)
docker compose up
```

---

## API

### `POST /convert`

Accepts a PDF via `multipart/form-data`.

| Field | Type | Default | Description |
|---|---|---|---|
| `file` | file | required | PDF to convert |
| `ocr_mode` | string | `auto` | See OCR modes below |
| `table_mode` | string | `fast` | `fast` or `accurate` |

**Response 200**

```json
{
  "status": "success",
  "filename": "policy.pdf",
  "page_count": 10,
  "ocr_mode_used": "auto",
  "processing_time_seconds": 4.2,
  "pages": [
    {
      "page_number": 1,
      "markdown": "# Section heading\n\nParagraph text...",
      "had_ocr": false
    }
  ],
  "full_markdown": "...entire document...",
  "structured_json": { "...docling native document structure..." }
}
```

**Error responses**

| Code | Condition |
|---|---|
| 400 | Invalid/corrupt PDF or missing file |
| 413 | File exceeds `MAX_FILE_SIZE_MB` |
| 500 | Unexpected docling error (details logged server-side) |
| 503 | Service at capacity — all slots busy, waited > `QUEUE_TIMEOUT_SECONDS` |

### `GET /health`

Liveness + readiness check.  Reports converter status and current concurrency
load.  Use the `in_flight` / `max_concurrent` ratio to monitor memory pressure.

```json
{
  "status": "ready",
  "converter_ready": true,
  "in_flight": 2,
  "max_concurrent": 4
}
```

### `GET /metrics`

Aggregate counters.  JSON structure is forward-compatible with Prometheus label
conventions if you later add a proper exporter.

```json
{
  "total_requests": 100,
  "total_failures": 2,
  "average_processing_time_seconds": 3.8,
  "current_in_flight": 1
}
```

---

## OCR modes

| Mode | Behaviour | Best for |
|---|---|---|
| `auto` *(default)* | OCR only on pages with no extractable text layer | Mixed document sets |
| `force` | Full-page OCR on every page, ignoring any text layer | Documents known to be fully scanned (photocopies, etc.) |
| `off` | No OCR; text-extraction only | Native digital PDFs — fastest path, lowest memory |

**OCR engine:** RapidOCR (ONNX-based) is used rather than EasyOCR.  RapidOCR
uses ~200 MB vs EasyOCR's ~1.5 GB, which matters when multiple OCR-heavy
requests run concurrently.

---

## Tuning `MAX_CONCURRENT_REQUESTS` for your hardware

This is **not** purely a CPU-core-count decision — memory is the real constraint.

Rough sizing guide (baseline model footprint ~3–5 GB idle):

| Document type | Extra RAM per in-flight request |
|---|---|
| Native digital PDF (`ocr_mode=off` or `auto` with text layer) | ~200–500 MB |
| Scanned PDF (`ocr_mode=force` or `auto` without text layer) | ~1–2 GB |

**Formula:** `MAX_CONCURRENT_REQUESTS ≤ (available_RAM − baseline) / per_request_peak`

On the reference 31 GB host:
- Conservative (safe): `MAX_CONCURRENT_REQUESTS=4` with native PDFs, `2` with heavy OCR
- The service OOMed previously under concurrent load, so start at `2` and
  increase once you have steady-state memory metrics from `/health`.

Monitor `/health` in production: if `in_flight` frequently hits `max_concurrent`
and processing times are climbing, you are memory-constrained, not CPU-constrained.
Reduce the limit or add RAM before increasing it.

---

## Configuration (environment variables)

| Variable | Default | Description |
|---|---|---|
| `MAX_CONCURRENT_REQUESTS` | `4` | Max simultaneous conversions |
| `QUEUE_TIMEOUT_SECONDS` | `60` | How long to wait for a free slot before returning 503 |
| `MAX_FILE_SIZE_MB` | `50` | Hard upload size limit |
| `OMP_NUM_THREADS` | `2` | OpenMP thread cap (set before numpy/torch load) |
| `MKL_NUM_THREADS` | `2` | MKL thread cap |
| `DEFAULT_OCR_MODE` | `auto` | Fallback when `ocr_mode` is not sent |
| `DEFAULT_TABLE_MODE` | `fast` | Fallback when `table_mode` is not sent |
| `LOG_LEVEL` | `INFO` | Python logging level |

---

## Running tests

```bash
pip install -r requirements.txt
pytest tests/ -v
```

Tests use the real DocumentConverter (no mocking) — the first run is slow while
models download.  Subsequent runs use the on-disk HuggingFace cache and are much
faster.

---

## Out of scope

- GPU / CUDA
- Authentication / authorisation
- Persistent storage of uploaded files or results (stateless per-request)
- Async job queue / polling endpoints (synchronous request/response only)
