"""
Run at Docker build time to pre-download and cache docling model weights.
Keeps the model download in a dedicated image layer, separate from app code,
so code-only rebuilds don't re-download ~3–5 GB of weights.
"""

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    RapidOcrOptions,
    TableFormerMode,
)

opts = PdfPipelineOptions()
opts.do_ocr = True
opts.ocr_options = RapidOcrOptions(force_full_page_ocr=False)
opts.do_table_structure = True
opts.table_structure_options.mode = TableFormerMode.FAST

DocumentConverter(
    format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)}
)

print("Model warm-up complete.")
