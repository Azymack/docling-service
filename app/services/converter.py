"""
DoclingConverterService — a long-lived wrapper around docling's DocumentConverter.

Design notes
------------
Architecture choice: single FastAPI app, two DoclingConverterService instances
(one CPU, one GPU).  The alternative — a separate GPU microservice — was
rejected because: (a) both endpoints must be hittable from the same benchmark
script with identical parameters, (b) docling DocumentConverter instances are
independent and don't interfere, and (c) a second process doubles operational
complexity (two ports, two healthchecks, two docker images) for no gain.
The tradeoff is that both model copies live in the same process; on an H200
this is acceptable since VRAM and RAM are not the same constraint.

Concurrency model
-----------------
* One ThreadPoolExecutor per service instance (size == max_concurrent) runs
  docling's synchronous blocking work off the asyncio event loop.  Docling's
  torch and onnxruntime calls release the GIL during inference, so genuine
  wall-clock parallelism is achieved in the thread pool.
* An asyncio.Semaphore (same bound) enforces the in-flight cap.
  asyncio.wait_for around acquire gives a hard timeout → 503 rather than
  queueing indefinitely.
* The GPU service has its own semaphore and pool sized by MAX_CONCURRENT_GPU_REQUESTS
  (default 1), independent of the CPU service's MAX_CONCURRENT_REQUESTS (default 4).

AcceleratorDevice handling
--------------------------
docling 2.103+ uses AcceleratorOptions(device=...) to select the inference
device for layout analysis, TableFormer, and OCR.  The default device is 'auto'
which picks up CUDA if available — so the CPU service MUST pass device=CPU
explicitly, otherwise it silently runs on GPU on an H200 host.

Important: AcceleratorOptions is a pydantic-settings BaseSettings subclass that
reads from DOCLING_DEVICE env var when no kwargs are given.  Explicit kwargs
passed to the constructor override the env var (confirmed in docling 2.103).

OCR engine selection
--------------------
CPU path  : RapidOCR (onnxruntime backend) — compact ONNX session, ~200 MB,
            no GPU path in the onnxruntime variant.
GPU path  : EasyOCR — PyTorch-based, uses the same CUDA context as the layout
            and TableFormer models.  In docling 2.103+ the correct approach is
            to set AcceleratorOptions.device=CUDA; EasyOcrOptions.use_gpu=None
            (the default) then auto-derives GPU usage from the accelerator
            device.  The old use_gpu=True kwarg is deprecated by docling.

Per-page markdown
-----------------
DoclingDocument.export_to_markdown() operates on the whole document; there is
no native per-page export.  We iterate the document's item tree via
iterate_items(), filter elements whose provenance places them on the target
page (items carry a list of ProvenanceItem each with a .page_no field), and
render each element using the same type-dispatch logic.  iterate_items()
yields items in reading order, preserving reading order within each page.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lazy docling imports — keep this module importable without triggering
# model loading (important for tests and startup ordering).
# ---------------------------------------------------------------------------

def _import_docling():
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import (
        PdfPipelineOptions,
        TableFormerMode,
        RapidOcrOptions,
    )
    from docling.datamodel.document import DocumentStream

    return (
        DocumentConverter,
        PdfFormatOption,
        InputFormat,
        PdfPipelineOptions,
        TableFormerMode,
        RapidOcrOptions,
        DocumentStream,
    )


def _import_docling_types():
    try:
        from docling_core.types.doc import (
            SectionHeaderItem,
            TextItem,
            TableItem,
            ListItem,
            PictureItem,
        )
    except ImportError:
        from docling.datamodel.base_models import (  # type: ignore[no-reattr]
            SectionHeaderItem,
            TextItem,
            TableItem,
            ListItem,
            PictureItem,
        )
    return SectionHeaderItem, TextItem, TableItem, ListItem, PictureItem


# ---------------------------------------------------------------------------
# GPU detection helper
# ---------------------------------------------------------------------------

def detect_gpu_device() -> str | None:
    """
    Return the best available non-CPU device string, or None.

    Returns 'cuda:N' (specific GPU index) rather than bare 'cuda' so that
    on a multi-GPU host the benchmark targets a single, predictable device.
    Returns 'mps' on Apple Silicon.  Returns None if only CPU is available.
    """
    try:
        import torch
        if torch.cuda.is_available():
            return f"cuda:{torch.cuda.current_device()}"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return None


# ---------------------------------------------------------------------------


class DoclingConverterService:

    def __init__(
        self,
        settings: Any,
        *,
        accelerator_device: str = "cpu",
        max_concurrent_override: int | None = None,
        queue_timeout_override: float | None = None,
        name: str = "cpu",
    ) -> None:
        self._settings = settings
        self._accelerator_device = accelerator_device
        self._name = name
        self._max_concurrent = (
            max_concurrent_override
            if max_concurrent_override is not None
            else settings.MAX_CONCURRENT_REQUESTS
        )
        self._queue_timeout = (
            queue_timeout_override
            if queue_timeout_override is not None
            else settings.QUEUE_TIMEOUT_SECONDS
        )

        self._converters: dict[tuple, Any] = {}
        self._build_lock = threading.Lock()

        self._semaphore: asyncio.Semaphore | None = None
        self._executor = ThreadPoolExecutor(
            max_workers=self._max_concurrent,
            thread_name_prefix=f"docling-{name}-worker",
        )

        self._total_requests = 0
        self._total_failures = 0
        self._total_time = 0.0
        self._in_flight = 0
        self._metrics_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Startup
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        """
        Called from FastAPI lifespan (inside the running event loop).
        Pre-warms the default converter and creates the asyncio semaphore.
        """
        from app.models.schemas import OcrMode, TableMode

        self._semaphore = asyncio.Semaphore(self._max_concurrent)

        default_key = (OcrMode.AUTO, TableMode.FAST)
        self._converters[default_key] = self._build_converter(OcrMode.AUTO, TableMode.FAST)
        logger.info(
            "[%s] Default DocumentConverter ready (ocr=auto, table=fast, device=%s).",
            self._name,
            self._accelerator_device,
        )

    # ------------------------------------------------------------------
    # Converter pool management
    # ------------------------------------------------------------------

    def _build_converter(self, ocr_mode: Any, table_mode: Any) -> Any:
        from app.models.schemas import OcrMode, TableMode
        from docling.datamodel.accelerator_options import AcceleratorOptions, AcceleratorDevice

        (
            DocumentConverter,
            PdfFormatOption,
            InputFormat,
            PdfPipelineOptions,
            TableFormerMode,
            RapidOcrOptions,
            _,
        ) = _import_docling()

        logger.info(
            "[%s] Building DocumentConverter: ocr_mode=%s table_mode=%s device=%s",
            self._name,
            ocr_mode,
            table_mode,
            self._accelerator_device,
        )

        opts = PdfPipelineOptions()
        opts.do_table_structure = True
        opts.table_structure_options.mode = (
            TableFormerMode.FAST if table_mode == TableMode.FAST else TableFormerMode.ACCURATE
        )

        # Explicitly set the accelerator device — never leave it as 'auto'.
        # With device='auto' (the docling default), the CPU service would
        # silently run on GPU whenever one is present.  Explicit kwargs
        # override DOCLING_DEVICE env vars in docling 2.103+ pydantic-settings.
        device_str = self._accelerator_device
        is_cuda = device_str.startswith("cuda")
        opts.accelerator_options = AcceleratorOptions(
            device=device_str,
            # Flash Attention 2 halves memory and improves throughput on
            # Ampere+ GPUs (A100, H100, H200).  Only meaningful for CUDA.
            cuda_use_flash_attention2=is_cuda,
        )

        # OCR engine selection:
        # CPU path  → RapidOCR (onnxruntime, ~200 MB, CPU-only in the
        #             rapidocr-onnxruntime package we pin in requirements.txt)
        # GPU path  → EasyOCR with use_gpu=None (docling 2.103+ preferred API:
        #             the model derives GPU usage from AcceleratorOptions.device,
        #             so use_gpu=True is no longer needed and is deprecated).
        #
        # Note: RapidOCR's onnxruntime backend could use CUDA if onnxruntime-gpu
        # were installed, but that conflicts with rapidocr-onnxruntime (CPU).
        # EasyOCR is simpler for GPU because it shares the existing CUDA context.
        if ocr_mode == OcrMode.OFF:
            opts.do_ocr = False
        elif ocr_mode == OcrMode.FORCE:
            opts.do_ocr = True
            if not is_cuda and device_str != "mps":
                opts.ocr_options = RapidOcrOptions(force_full_page_ocr=True)
            else:
                from docling.datamodel.pipeline_options import EasyOcrOptions
                opts.ocr_options = EasyOcrOptions(force_full_page_ocr=True)
        else:  # auto
            opts.do_ocr = True
            if not is_cuda and device_str != "mps":
                opts.ocr_options = RapidOcrOptions(force_full_page_ocr=False)
            else:
                from docling.datamodel.pipeline_options import EasyOcrOptions
                opts.ocr_options = EasyOcrOptions()  # use_gpu=None → follows AcceleratorOptions

        return DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)}
        )

    def _get_converter(self, ocr_mode: Any, table_mode: Any) -> Any:
        key = (ocr_mode, table_mode)
        if key in self._converters:
            return self._converters[key]
        with self._build_lock:
            if key not in self._converters:
                self._converters[key] = self._build_converter(ocr_mode, table_mode)
        return self._converters[key]

    # ------------------------------------------------------------------
    # Async public API
    # ------------------------------------------------------------------

    async def convert(
        self,
        file_bytes: bytes,
        filename: str,
        ocr_mode: Any,
        table_mode: Any,
    ) -> dict:
        """
        Acquire the concurrency semaphore (raising asyncio.TimeoutError if the
        queue is full), then run the CPU-bound conversion in the thread pool.
        """
        assert self._semaphore is not None, "initialize() was not called"

        try:
            await asyncio.wait_for(
                self._semaphore.acquire(),
                timeout=self._queue_timeout,
            )
        except asyncio.TimeoutError:
            raise

        with self._metrics_lock:
            self._in_flight += 1

        loop = asyncio.get_running_loop()
        try:
            result = await loop.run_in_executor(
                self._executor,
                self._convert_sync,
                file_bytes,
                filename,
                ocr_mode,
                table_mode,
            )
            with self._metrics_lock:
                self._total_requests += 1
                self._total_time += result["processing_time_seconds"]
            return result
        except Exception:
            with self._metrics_lock:
                self._total_requests += 1
                self._total_failures += 1
            raise
        finally:
            self._semaphore.release()
            with self._metrics_lock:
                self._in_flight -= 1

    # ------------------------------------------------------------------
    # Synchronous work (runs inside the thread pool)
    # ------------------------------------------------------------------

    def _convert_sync(
        self,
        file_bytes: bytes,
        filename: str,
        ocr_mode: Any,
        table_mode: Any,
    ) -> dict:
        (
            _,
            _,
            _,
            _,
            _,
            _,
            DocumentStream,
        ) = _import_docling()

        converter = self._get_converter(ocr_mode, table_mode)
        is_cuda = self._accelerator_device.startswith("cuda")

        logger.debug(
            "[%s] Conversion start: filename=%s ocr_mode=%s table_mode=%s in_flight=%d",
            self._name,
            filename,
            ocr_mode,
            table_mode,
            self._in_flight,
        )

        # Reset peak GPU memory counter so per-request spikes are accurate.
        if is_cuda:
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
            except ImportError:
                pass

        t0 = time.perf_counter()
        stream = DocumentStream(name=filename, stream=BytesIO(file_bytes))
        result = converter.convert(stream, raises_on_error=True)
        elapsed = time.perf_counter() - t0

        # Capture peak GPU memory after conversion completes.
        gpu_peak_mb: float | None = None
        if is_cuda:
            try:
                import torch
                if torch.cuda.is_available():
                    gpu_peak_mb = round(torch.cuda.max_memory_allocated() / (1024 ** 2), 1)
                    logger.info(
                        "[%s] GPU memory peak: filename=%s peak_mb=%.0f elapsed_s=%.2f",
                        self._name,
                        filename,
                        gpu_peak_mb,
                        elapsed,
                    )
            except ImportError:
                pass

        document = result.document
        page_count = len(document.pages)
        full_markdown = document.export_to_markdown()

        try:
            structured_json: dict = document.export_to_dict()
        except AttributeError:
            structured_json = json.loads(document.model_dump_json())

        pages = []
        for page_no in sorted(document.pages.keys()):
            md = self._export_page_markdown(document, page_no)
            had_ocr = self._detect_page_ocr(result, page_no, ocr_mode)
            pages.append(
                {
                    "page_number": page_no,
                    "markdown": md,
                    "had_ocr": had_ocr,
                }
            )

        logger.debug(
            "[%s] Conversion done: filename=%s pages=%d elapsed_s=%.2f",
            self._name,
            filename,
            page_count,
            elapsed,
        )

        return {
            "page_count": page_count,
            "full_markdown": full_markdown,
            "structured_json": structured_json,
            "pages": pages,
            "processing_time_seconds": round(elapsed, 3),
            "gpu_peak_memory_mb": gpu_peak_mb,
        }

    def _export_page_markdown(self, document: Any, page_no: int) -> str:
        """
        Render a single page's content as markdown.

        Docling's export_to_markdown() has no per-page parameter, so we filter
        the item tree by provenance page number and dispatch on item type using
        the same logic export_to_markdown() uses.  iterate_items() returns items
        in document reading order, preserving that order within the page.
        """
        SectionHeaderItem, TextItem, TableItem, ListItem, PictureItem = (
            _import_docling_types()
        )

        lines: list[str] = []
        for item, level in document.iterate_items():
            prov = getattr(item, "prov", None)
            if not prov:
                continue
            if not any(getattr(p, "page_no", None) == page_no for p in prov):
                continue

            if isinstance(item, SectionHeaderItem):
                heading_level = max(1, getattr(item, "level", 1))
                lines.append(f"{'#' * heading_level} {item.text}")
            elif isinstance(item, TableItem):
                try:
                    lines.append(item.export_to_markdown())
                except Exception:
                    lines.append("[table]")
            elif isinstance(item, ListItem):
                indent = "  " * max(0, level - 1)
                lines.append(f"{indent}- {item.text}")
            elif isinstance(item, TextItem):
                lines.append(item.text)
            elif isinstance(item, PictureItem):
                lines.append(f"![image on page {page_no}]()")

        return "\n\n".join(lines)

    def _detect_page_ocr(self, result: Any, page_no: int, ocr_mode: Any) -> bool:
        """
        Return True if OCR was applied to this page.

        For off/force modes this is deterministic.  For auto mode, we inspect
        the backend page's parsed text cells: if the PDF had no extractable text
        cells, docling would have fallen back to OCR.
        """
        from app.models.schemas import OcrMode

        if ocr_mode == OcrMode.OFF:
            return False
        if ocr_mode == OcrMode.FORCE:
            return True

        # auto: look at the low-level parsed page
        for page in getattr(result, "pages", []):
            if getattr(page, "page_no", None) == page_no:
                parsed = getattr(page, "parsed_page", None)
                if parsed is None:
                    return True
                cells = getattr(parsed, "cells", None) or []
                return len(cells) == 0

        return False

    # ------------------------------------------------------------------
    # Observability
    # ------------------------------------------------------------------

    def probe_devices(self) -> dict[str, str]:
        """
        Return which device each model component is configured (and able) to use.

        Reports configuration intent plus runtime GPU availability check, so
        callers can tell the difference between "ran on GPU" and "silently
        fell back to CPU because CUDA was not available".

        For the GPU service, EasyOCR follows AcceleratorOptions.device
        (docling 2.103+ behaviour), so layout, table_structure, and ocr
        all share the same device when using EasyOCR.
        """
        device_str = self._accelerator_device

        if device_str == "cpu":
            return {"layout": "cpu", "table_structure": "cpu", "ocr": "cpu"}

        try:
            import torch

            if device_str.startswith("cuda"):
                if not torch.cuda.is_available():
                    note = f"cpu (configured={device_str}, CUDA not available at runtime)"
                    return {"layout": note, "table_structure": note, "ocr": note}
                # EasyOCR.use_gpu=None follows AcceleratorOptions.device (docling 2.103+)
                return {
                    "layout": device_str,
                    "table_structure": device_str,
                    "ocr": device_str,
                }

            if device_str == "mps":
                mps_ok = (
                    hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
                )
                if not mps_ok:
                    note = "cpu (configured=mps, MPS not available at runtime)"
                    return {"layout": note, "table_structure": note, "ocr": note}
                return {"layout": "mps", "table_structure": "mps", "ocr": "mps"}

        except ImportError:
            note = f"cpu (torch not importable, cannot use {device_str})"
            return {"layout": note, "table_structure": note, "ocr": note}

        return {"layout": device_str, "table_structure": device_str, "ocr": device_str}

    @property
    def in_flight(self) -> int:
        with self._metrics_lock:
            return self._in_flight

    @property
    def max_concurrent(self) -> int:
        return self._max_concurrent

    @property
    def device(self) -> str:
        return self._accelerator_device

    @property
    def is_ready(self) -> bool:
        return bool(self._converters)

    @property
    def metrics(self) -> dict:
        with self._metrics_lock:
            avg = (
                round(self._total_time / self._total_requests, 3)
                if self._total_requests > 0
                else 0.0
            )
            return {
                "total_requests": self._total_requests,
                "total_failures": self._total_failures,
                "average_processing_time_seconds": avg,
                "current_in_flight": self._in_flight,
            }

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False)
