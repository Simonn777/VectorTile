# Reproduce and interpret the measurements

The published [CPU JSON](../benchmarks/results/apple-m5-cpu.json) is the source
of truth for the README chart and browser measurement panel.

## Commands

From the repository root, in the environment created by the README:

```sh
python -m pip install -e '.[dev,torch,plots]'
python scripts/build_native.py
python -m pytest -q
python scripts/benchmark.py --backends reference numpy native torch \
  --repeats 9 --warmups 2 --output benchmarks/results/local.json
python scripts/render_results.py benchmarks/results/local.json
```

The renderer updates the chart and `site/benchmark.json` from that report.
To preserve the checked-in demonstration, skip the renderer or use a separate
checkout. `local.json` and the native build directory are ignored by Git.

## Protocol

- Float64 independent standard-normal queries/corpus, generated with NumPy
  `default_rng`. Seeds and exact shapes are saved for all three cases.
- Three cases: (Q,N,D,k) = (16,1024,32,5), (32,4096,64,10), (64,8192,64,10).
- Tile configuration: Bq=16, Bc=256. Native requests four workers; the C++
  runtime caps worker count by hardware concurrency and query count.
- Two warmups followed by nine timed calls per backend. Each backend is
  measured sequentially, in the recorded order. These are warm-process,
  mostly cache-warm microbenchmarks, not cold-start or service-load tests.
- `perf_counter_ns` measures the whole public call: validation, input
  conversion if needed, result allocation, wrapper overhead and execution.
  Data generation, compilation and correctness checks are outside the timer.
- Every measured result is compared with an independent direct-distance
  NumPy oracle that handles one query at a time and fully lexsorts candidates.
  Distances use `rtol=atol=1e-12`; indices match exactly on these fixtures.
- p50 and p95 use NumPy's linear-interpolated percentiles. With nine samples,
  p95 is only descriptive; it is not a statistical confidence bound.

## Recorded CPU result (milliseconds, median / p95)

| Q × N × D; k | Dense reference | Tiled NumPy | C++ (4 requested threads) | PyTorch CPU |
| --- | ---: | ---: | ---: | ---: |
| 16 × 1,024 × 32; 5 | 0.764 / 0.999 | 1.087 / 1.192 | 0.163 / 0.182 | 0.740 / 0.813 |
| 32 × 4,096 × 64; 10 | 11.640 / 11.849 | 13.928 / 21.187 | 0.838 / 0.879 | 7.373 / 7.594 |
| 64 × 8,192 × 64; 10 | 49.744 / 53.031 | 54.311 / 54.462 | 2.326 / 2.474 | 30.253 / 30.405 |

Recorded October 12, 2026 on Apple M5 (10 logical CPUs), Darwin arm64,
Python 3.9.6, NumPy 2.0.2, PyTorch 2.8.0. PyTorch used four intra-op and ten
interop threads. Native: Apple Clang 21, `-O3`, `-ffp-contract=off`, no fast math.
Exact versions, flags, native source hashes and raw timings are in the report.

## What this does and does not show

The C++ engine outperforms this readable NumPy reference on the measured
workloads. The reference allocates a Q×N×D difference tensor and fully sorts
each row. It is intentionally inspectable and **not a tuned production
baseline**. Comparisons to FAISS, scipy, optimized BLAS or hand-written CUDA
would require additional implementations and matched numerical contracts.

NumPy tiling was slower than the dense reference on the largest recorded
case. Smaller distance workspace is a different result from lower latency.

The 64 KiB NumPy distance-buffer estimate is an array-size calculation. The
4 MiB comparison is the full distance matrix alone; the dense reference also
holds a 256 MiB difference tensor on the largest shape, plus ordering and
other arrays. Neither figure is measured peak RSS. The C++ heap estimate is
640 bytes for four workers and k=10, excluding allocations, thread stacks,
validation and output arrays. See [full accounting](BACKENDS.md).

All data are synthetic and all corpora fit in RAM. This does not demonstrate
semantic retrieval quality, disk streaming, production scalability, or
application-level user outcomes. Native threads are created for each call;
that overhead matters for small inputs.

## CUDA reproduction (requires a separate suitable machine)

```sh
python -m pytest tests/test_search.py -q
python scripts/benchmark.py --backends cuda --output benchmarks/results/cuda-local.json
```

Install the appropriate official PyTorch CUDA build first. The script raises
when CUDA is absent; it does not fall back to CPU. CUDA whole-call timings
include transfer, per-tile finite checks and returning results to CPU.
The current checked-in report contains **no GPU measurements**.
