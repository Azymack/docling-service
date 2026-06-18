from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Concurrency & memory safety
    MAX_CONCURRENT_REQUESTS: int = 4
    QUEUE_TIMEOUT_SECONDS: float = 60.0
    MAX_FILE_SIZE_MB: int = 50

    # Thread-oversubscription guard — set before numpy/torch/onnxruntime load.
    # Applied at process start in main.py; listed here so they appear in the
    # pydantic-settings env-var documentation.
    OMP_NUM_THREADS: int = 2
    MKL_NUM_THREADS: int = 2

    # GPU concurrency — separate from CPU because GPU memory is a different
    # constraint.  Default 1: a single H200 can be saturated by one docling
    # job; raising this only helps if requests are I/O-bound (e.g. PDF parsing
    # before GPU inference starts).
    MAX_CONCURRENT_GPU_REQUESTS: int = 1
    GPU_QUEUE_TIMEOUT_SECONDS: float = 120.0

    # Conversion defaults (can be overridden per-request)
    DEFAULT_OCR_MODE: str = "auto"
    DEFAULT_TABLE_MODE: str = "fast"

    LOG_LEVEL: str = "INFO"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}
