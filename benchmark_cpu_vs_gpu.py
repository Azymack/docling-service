#!/usr/bin/env python3
"""
benchmark_cpu_vs_gpu.py — Compare docling CPU vs GPU conversion performance.

Usage
-----
    python benchmark_cpu_vs_gpu.py --pdf-dir ./sample_pdfs [options]

Options
-------
    --pdf-dir       Directory containing PDF files to benchmark (required)
    --base-url      Service base URL (default: http://localhost:8001)
    --runs          Number of timed runs per (document × ocr_mode) cell (default: 3)
    --ocr-modes     Comma-separated list of OCR modes to test (default: auto,off,force)
    --output-dir    Directory to write markdown outputs and CSV (default: ./results)
    --timeout       Per-request timeout in seconds (default: 300)

Output
------
    results/{stem}_cpu_{ocr_mode}.md   — full_markdown from /convert
    results/{stem}_gpu_{ocr_mode}.md   — full_markdown from /convert-gpu
    results/benchmark_{timestamp}.csv  — timing data for all runs
    Printed summary table to stdout

Important caveats printed with the results
------------------------------------------
    * Numbers reflect docling running IN ISOLATION on the GPU — no vLLM or
      other workloads sharing the device.  Once docling and vLLM co-reside
      on the same H200, throughput and latency will differ.
    * Docling's GPU support is described by the project team as work-in-progress.
      Independently observed real-world GPU utilisation near 0% in some
      configurations.  Check the device_used field and gpu_peak_memory_mb in
      the response — a near-zero GPU memory spike despite device=cuda means
      something fell back to CPU.
    * These numbers cannot reliably predict production speedup.  Report what
      they actually show, including if GPU is slower or identical to CPU.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

import requests

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_BASE_URL = "http://localhost:8001"
DEFAULT_RUNS = 3
DEFAULT_OCR_MODES = ["auto", "off", "force"]
DEFAULT_OUTPUT_DIR = Path("results")
DEFAULT_TIMEOUT = 300


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def _post_convert(
    base_url: str,
    endpoint: str,
    pdf_path: Path,
    ocr_mode: str,
    timeout: int,
) -> tuple[float, dict | None, str | None]:
    """
    POST a PDF to the given endpoint.
    Returns (wall_clock_seconds, response_dict_or_None, error_message_or_None).
    """
    url = f"{base_url}/{endpoint}"
    t0 = time.perf_counter()
    try:
        with pdf_path.open("rb") as fh:
            resp = requests.post(
                url,
                files={"file": (pdf_path.name, fh, "application/pdf")},
                data={"ocr_mode": ocr_mode, "table_mode": "fast"},
                timeout=timeout,
            )
        elapsed = time.perf_counter() - t0
        if resp.status_code == 200:
            return elapsed, resp.json(), None
        return elapsed, None, f"HTTP {resp.status_code}: {resp.text[:200]}"
    except requests.exceptions.Timeout:
        elapsed = time.perf_counter() - t0
        return elapsed, None, f"Timeout after {elapsed:.0f}s"
    except requests.exceptions.ConnectionError as e:
        elapsed = time.perf_counter() - t0
        return elapsed, None, f"Connection error: {e}"


def _check_health(base_url: str) -> dict:
    try:
        resp = requests.get(f"{base_url}/health", timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"WARNING: /health check failed: {e}", file=sys.stderr)
        return {}


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------

def _median(values: list[float]) -> float:
    return statistics.median(values) if values else float("nan")


def _p95(values: list[float]) -> float:
    if not values:
        return float("nan")
    sorted_v = sorted(values)
    idx = max(0, int(len(sorted_v) * 0.95) - 1)
    return sorted_v[idx]


# ---------------------------------------------------------------------------
# Core benchmark logic
# ---------------------------------------------------------------------------

def run_benchmark(
    pdf_dir: Path,
    base_url: str,
    runs: int,
    ocr_modes: list[str],
    output_dir: Path,
    timeout: int,
) -> list[dict]:
    """
    Run the full benchmark and return a list of result rows (one per
    (pdf, ocr_mode, endpoint) combination).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    pdfs = sorted(pdf_dir.glob("*.pdf"))
    if not pdfs:
        print(f"No PDF files found in {pdf_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(pdfs)} PDF(s) in {pdf_dir}")
    print(f"Runs per cell: {runs}  |  OCR modes: {ocr_modes}  |  Base URL: {base_url}")
    print()

    health = _check_health(base_url)
    gpu_info = health.get("gpu")
    if gpu_info:
        print(f"GPU converter:  device={gpu_info.get('device')}  "
              f"ready={gpu_info.get('converter_ready')}  "
              f"device_mapping={gpu_info.get('device_mapping')}")
    else:
        print("GPU converter:  NOT available — /convert-gpu will be skipped or return 503")
    print()

    rows: list[dict] = []

    for pdf_path in pdfs:
        stem = pdf_path.stem
        print(f"{'='*60}")
        print(f"Document: {pdf_path.name}  ({pdf_path.stat().st_size / 1024:.0f} KB)")
        print(f"{'='*60}")

        for ocr_mode in ocr_modes:
            print(f"\n  OCR mode: {ocr_mode}")

            cpu_times: list[float] = []
            gpu_times: list[float] = []
            cpu_errors: list[str] = []
            gpu_errors: list[str] = []
            last_cpu_response: dict | None = None
            last_gpu_response: dict | None = None
            gpu_peak_mb_vals: list[float] = []

            for run_idx in range(runs):
                # --- CPU ---
                elapsed_cpu, resp_cpu, err_cpu = _post_convert(
                    base_url, "convert", pdf_path, ocr_mode, timeout
                )
                if err_cpu:
                    cpu_errors.append(err_cpu)
                    print(f"    run {run_idx+1} CPU  ERROR: {err_cpu}")
                else:
                    cpu_times.append(elapsed_cpu)
                    last_cpu_response = resp_cpu
                    print(
                        f"    run {run_idx+1} CPU  {elapsed_cpu:.2f}s  "
                        f"pages={resp_cpu.get('page_count')}  "
                        f"ocr_pages={sum(1 for p in resp_cpu.get('pages', []) if p.get('had_ocr'))}"
                    )

                # --- GPU ---
                elapsed_gpu, resp_gpu, err_gpu = _post_convert(
                    base_url, "convert-gpu", pdf_path, ocr_mode, timeout
                )
                if err_gpu:
                    gpu_errors.append(err_gpu)
                    print(f"    run {run_idx+1} GPU  ERROR: {err_gpu}")
                else:
                    gpu_times.append(elapsed_gpu)
                    last_gpu_response = resp_gpu
                    gpu_peak = resp_gpu.get("gpu_peak_memory_mb")
                    if gpu_peak is not None:
                        gpu_peak_mb_vals.append(gpu_peak)
                    print(
                        f"    run {run_idx+1} GPU  {elapsed_gpu:.2f}s  "
                        f"pages={resp_gpu.get('page_count')}  "
                        f"gpu_peak_mb={gpu_peak}  "
                        f"device_used={resp_gpu.get('device_used')}"
                    )

            # Save markdown outputs (last successful run only)
            if last_cpu_response:
                out_cpu = output_dir / f"{stem}_cpu_{ocr_mode}.md"
                out_cpu.write_text(last_cpu_response.get("full_markdown", ""), encoding="utf-8")

            if last_gpu_response:
                out_gpu = output_dir / f"{stem}_gpu_{ocr_mode}.md"
                out_gpu.write_text(last_gpu_response.get("full_markdown", ""), encoding="utf-8")

            # Aggregate stats for this cell
            cpu_med = _median(cpu_times)
            cpu_p95 = _p95(cpu_times)
            gpu_med = _median(gpu_times)
            gpu_p95 = _p95(gpu_times)
            speedup = (cpu_med / gpu_med) if (gpu_med > 0 and cpu_times and gpu_times) else None
            gpu_peak_med = _median(gpu_peak_mb_vals) if gpu_peak_mb_vals else None

            row = {
                "filename": pdf_path.name,
                "file_size_kb": round(pdf_path.stat().st_size / 1024, 1),
                "ocr_mode": ocr_mode,
                "cpu_runs_ok": len(cpu_times),
                "cpu_median_s": round(cpu_med, 3) if cpu_times else "",
                "cpu_p95_s": round(cpu_p95, 3) if cpu_times else "",
                "gpu_runs_ok": len(gpu_times),
                "gpu_median_s": round(gpu_med, 3) if gpu_times else "",
                "gpu_p95_s": round(gpu_p95, 3) if gpu_times else "",
                "speedup_ratio": round(speedup, 2) if speedup is not None else "",
                "gpu_peak_memory_mb_median": round(gpu_peak_med, 0) if gpu_peak_med is not None else "",
                "cpu_errors": "; ".join(cpu_errors),
                "gpu_errors": "; ".join(gpu_errors),
                # Device info from last GPU response
                "gpu_device_used_layout": (
                    (last_gpu_response or {}).get("device_used", {}) or {}
                ).get("layout", ""),
                "gpu_device_used_ocr": (
                    (last_gpu_response or {}).get("device_used", {}) or {}
                ).get("ocr", ""),
            }
            rows.append(row)

    return rows


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def _print_summary(rows: list[dict]) -> None:
    print("\n" + "=" * 100)
    print("BENCHMARK SUMMARY")
    print("=" * 100)
    print(
        f"{'File':<30} {'OCR':<6} {'CPU med':>8} {'GPU med':>8} {'Speedup':>8} "
        f"{'GPU peak MB':>12} {'GPU layout device'}"
    )
    print("-" * 100)
    for r in rows:
        speedup_str = f"{r['speedup_ratio']:.2f}x" if r["speedup_ratio"] != "" else "n/a"
        gpu_peak_str = str(r["gpu_peak_memory_mb_median"]) if r["gpu_peak_memory_mb_median"] != "" else "n/a"
        cpu_med_str = f"{r['cpu_median_s']:.2f}s" if r["cpu_median_s"] != "" else "err"
        gpu_med_str = f"{r['gpu_median_s']:.2f}s" if r["gpu_median_s"] != "" else "err"
        print(
            f"{r['filename'][:29]:<30} {r['ocr_mode']:<6} "
            f"{cpu_med_str:>8} {gpu_med_str:>8} {speedup_str:>8} "
            f"{gpu_peak_str:>12}   {r['gpu_device_used_layout']}"
        )

    # Aggregate across all rows that have valid timings for both endpoints
    valid = [r for r in rows if r["cpu_median_s"] != "" and r["gpu_median_s"] != ""]
    if valid:
        all_speedups = [r["speedup_ratio"] for r in valid if r["speedup_ratio"] != ""]
        print("-" * 100)
        if all_speedups:
            print(
                f"{'Aggregate':<30} {'all':<6} "
                f"{'':>8} {'':>8} "
                f"{_median(all_speedups):>7.2f}x {'':>12}   "
                f"(median speedup across {len(all_speedups)} cells)"
            )

    print("\n" + "=" * 100)
    print("IMPORTANT CAVEATS — READ BEFORE DRAWING CONCLUSIONS")
    print("=" * 100)
    print("""
  1. ISOLATION: These numbers reflect docling running ALONE on the GPU.
     Once vLLM (or any other CUDA workload) co-resides on the same H200,
     GPU memory bandwidth and compute will be shared.  Latency and
     throughput will differ — potentially significantly — under concurrent load.

  2. GPU SUPPORT MATURITY: Docling's GPU acceleration is documented as
     work-in-progress by the project team.  At least one published real-world
     test found GPU utilisation near 0% despite device=cuda being configured.
     Check the 'GPU layout device' column above — if it reads 'cpu (CUDA not
     available at runtime)', no GPU acceleration occurred at all.  Even when
     it reads 'cuda', verify gpu_peak_memory_mb > 0; a near-zero spike
     suggests the GPU was barely used.

  3. SINGLE-RUN NOISE: Timing from a single run is not reliable.  This script
     reports median and p95 across {runs} runs, which is still a small sample
     for benchmarking.  Treat large speedup ratios with scepticism unless
     confirmed across many runs and document sizes.

  4. INTERPRETATION: A speedup ratio < 1.0 means GPU was SLOWER.  This is
     plausible if model inference is too fast to amortise GPU memory transfer
     overhead, or if EasyOCR's GPU initialisation time dominates.  Report
     what the numbers show — do not assume GPU is the right choice.
""")


def _write_csv(rows: list[dict], output_dir: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = output_dir / f"benchmark_{timestamp}.csv"
    if not rows:
        return csv_path
    fieldnames = list(rows[0].keys())
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return csv_path


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark docling CPU vs GPU conversion endpoints.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--pdf-dir",
        required=True,
        type=Path,
        help="Directory containing .pdf files to benchmark",
    )
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help=f"Service base URL (default: {DEFAULT_BASE_URL})",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=DEFAULT_RUNS,
        help=f"Timed runs per (document × ocr_mode) cell (default: {DEFAULT_RUNS})",
    )
    parser.add_argument(
        "--ocr-modes",
        default=",".join(DEFAULT_OCR_MODES),
        help=f"Comma-separated OCR modes to test (default: {','.join(DEFAULT_OCR_MODES)})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for markdown and CSV output (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT,
        help=f"Per-request timeout in seconds (default: {DEFAULT_TIMEOUT})",
    )
    args = parser.parse_args()

    ocr_modes = [m.strip() for m in args.ocr_modes.split(",") if m.strip()]

    rows = run_benchmark(
        pdf_dir=args.pdf_dir,
        base_url=args.base_url,
        runs=args.runs,
        ocr_modes=ocr_modes,
        output_dir=args.output_dir,
        timeout=args.timeout,
    )

    _print_summary(rows)

    csv_path = _write_csv(rows, args.output_dir)
    print(f"\nCSV written to:      {csv_path}")
    print(f"Markdown outputs in: {args.output_dir}/")
    print(
        "To diff CPU vs GPU output for a single document:\n"
        "  diff results/<stem>_cpu_auto.md results/<stem>_gpu_auto.md"
    )


if __name__ == "__main__":
    main()
