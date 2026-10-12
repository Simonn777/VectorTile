#!/usr/bin/env python3
"""Generate auditable end-to-end search timings, with an independent oracle."""
from __future__ import annotations

import argparse
import datetime as dt
import importlib.metadata
import json
import os
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
from vectortile import search

ROOT = Path(__file__).resolve().parents[1]


def oracle(queries, corpus, k):
    """One query at a time, direct differences and a full stable sort."""
    distances = np.empty((len(queries), k), dtype=np.float64)
    indices = np.empty((len(queries), k), dtype=np.int64)
    ids = np.arange(len(corpus))
    for i, query in enumerate(queries):
        d = np.sum(np.square(corpus - query), axis=1)
        chosen = np.lexsort((ids, d))[:k]
        indices[i], distances[i] = chosen, d[chosen]
    return indices, distances


def cpu_label():
    if platform.system() == "Darwin":
        return subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
    return platform.processor() or platform.machine()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backends", nargs="+", default=["reference", "numpy", "native"])
    p.add_argument("--repeats", type=int, default=9)
    p.add_argument("--warmups", type=int, default=2)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--query-block", type=int, default=16)
    p.add_argument("--corpus-block", type=int, default=256)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--output", type=Path, default=ROOT / "benchmarks/results/local.json")
    args = p.parse_args()
    if args.repeats < 1 or args.warmups < 0 or args.threads < 1:
        p.error("repeats/threads must be positive and warmups nonnegative")
    cases = [(8, 128, 8, 3)] if args.quick else [(16, 1024, 32, 5), (32, 4096, 64, 10), (64, 8192, 64, 10)]
    versions = {name: importlib.metadata.version(name) for name in ["numpy", "vectortile"]}
    try:
        versions["torch"] = importlib.metadata.version("torch")
    except importlib.metadata.PackageNotFoundError:
        pass
    report = {
        "schema_version": 1,
        "recorded_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "environment": {"os": platform.system(), "os_release": platform.release(), "machine": platform.machine(),
                        "cpu": cpu_label(), "logical_cpus": os.cpu_count(), "python": platform.python_version(), "versions": versions},
        "methodology": {
            "dtype": "float64", "distance": "squared Euclidean", "data": "independent synthetic standard-normal vectors",
            "warmups": args.warmups, "repeats": args.repeats,
            "timed_scope": "whole public search call including validation, output allocation, ctypes overhead / device transfers where applicable; excludes data generation and compilation",
            "percentiles": "NumPy linear-interpolated percentiles of repeated whole-call timings; p95 is descriptive, not a confidence bound",
            "memory": "distance-workspace estimates only, not peak RSS; exclude inputs, outputs, candidate selection and runtime overhead",
            "gpu": "No CUDA result is implied by CPU measurements. CUDA rows exist only if explicitly requested and successful.",
            "baseline": "readable direct-distance NumPy reference, not a tuned BLAS/FAISS/scipy baseline",
        },
        "rows": [],
    }
    if any(b in args.backends for b in ("torch", "cuda")):
        import torch
        report["environment"]["torch_cpu_threads"] = torch.get_num_threads()
        report["environment"]["torch_interop_threads"] = torch.get_num_interop_threads()
        report["environment"]["cuda_available"] = torch.cuda.is_available()
        if "cuda" in args.backends and torch.cuda.is_available():
            report["environment"]["cuda_device"] = torch.cuda.get_device_name(0)
            report["environment"]["cuda_runtime"] = torch.version.cuda
    manifest = ROOT / "native/build/build-info.json"
    if manifest.exists():
        report["native_build"] = json.loads(manifest.read_text())
    for case_id, (q, n, d, k) in enumerate(cases):
        seed = 20261012 + case_id
        rng = np.random.default_rng(seed)
        corpus, queries = rng.standard_normal((n, d)), rng.standard_normal((q, d))
        expected_ids, expected_d = oracle(queries, corpus, k)
        for backend in args.backends:
            settings = dict(backend=backend, query_block=args.query_block, corpus_block=args.corpus_block, threads=args.threads)
            for _ in range(args.warmups):
                search(queries, corpus, k, **settings)
            elapsed = []
            for _ in range(args.repeats):
                start = time.perf_counter_ns()
                result = search(queries, corpus, k, **settings)
                elapsed.append((time.perf_counter_ns() - start) / 1e6)
                np.testing.assert_array_equal(result.indices, expected_ids)
                np.testing.assert_allclose(result.distances, expected_d, rtol=1e-12, atol=1e-12)
            row = {
                "case": f"{q}q-{n}n-{d}d-k{k}", "queries": q, "corpus": n, "dimensions": d, "k": k,
                "seed": seed, "backend": backend, "requested_threads": args.threads if backend == "native" else None,
                "query_block": args.query_block, "corpus_block": args.corpus_block,
                "raw_ms": elapsed, "p50_ms": float(np.median(elapsed)), "p95_ms": float(np.percentile(elapsed, 95)),
                "validated_all_repeats": True, "max_abs_distance_error": float(np.max(np.abs(result.distances - expected_d))),
                "full_distance_matrix_bytes": q*n*8,
                "numpy_distance_tile_buffers_bytes": min(q,args.query_block)*min(n,args.corpus_block)*8*2,
            }
            report["rows"].append(row)
            print(f"{row['case']:24} {backend:10} p50={row['p50_ms']:9.3f} ms p95={row['p95_ms']:9.3f} ms | oracle PASS", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
