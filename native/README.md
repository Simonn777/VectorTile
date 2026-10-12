# C++17 streaming exact search

This optional backend exhaustively computes squared Euclidean distances in
float64. It demonstrates a bounded-memory selection algorithm, a checked C ABI,
and independent query parallelism. It is a CPU backend; it makes no GPU speed
claim.

```sh
python scripts/build_native.py
python -m pytest tests/test_native.py
```

The script selects `clang++`, then `c++`, or uses the `CXX` environment variable.
It builds `native/build/libvectortile.dylib` on macOS and `.so` on Linux. No
compilation occurs on import. `VECTORTILE_NATIVE_LIB` can select an existing
compatible library, including when Python is installed outside this checkout.
Rebuild before restarting Python after changing the C++ implementation: the
adapter caches loaded library handles.

## Algorithm and memory

For every query, scan all corpus rows and accumulate direct squared differences.
Keep at most `k` candidates in a max heap, whose root is the worst retained
candidate. A new point replaces it only if its `(distance, corpus_index)` key is
smaller. Sort only the retained heap at the end of each query.

- Distance work: `O(Q N D)`; selection worst case: `O(Q N log k)` (constant heap
  work for `k=1`); final sorting: `O(Q k log k)`.
- Heap workspace: `O(T k)` for `T` active workers. A candidate contains one
  float64 distance and one int64 index (16 bytes on the supported platforms).
- Required output: `O(Q k)`; inputs: `O((Q + N) D)`.
- No `Q x N` distance matrix or `N`-element candidate array is allocated.
- Python conversion may copy inputs, and Python validation creates temporary
  finite-value masks. This heap estimate is not total process RSS.

## Parallelism and numerical behavior

Workers claim query rows through an atomic counter. The actual worker count is
the minimum of requested threads, query count, hardware concurrency (with a
fallback of one), and 64. The calling thread is one of these workers. Each worker
owns one reusable heap; each query output row has exactly one writer. Separate
Python calls may run concurrently because `ctypes.CDLL` releases the GIL.

Each distance is accumulated in feature order. Compiler flags disable fused
multiply-add contraction and do not enable fast math. Results are reproducible
across thread counts for the same binary and inputs. Ties in the computed
float64 distance use increasing corpus index. This is exhaustive search, not
exact real arithmetic; floating-point rounding can affect nearly tied points.
NumPy may sum with a different reduction order, so cross-backend comparisons
use numeric tolerances and well-separated or deliberately identical vectors.

## Validation and failure handling

Both the Python adapter and native boundary validate shapes, dimensions, `k`,
thread count, and finite input values. The C ABI checks element and byte-size
products before indexing, pointer alignment, overflowing address ranges, null
nonempty buffers, and output overlap. It cannot verify that an arbitrary C
pointer actually owns the declared allocation; external C callers must supply
valid buffers of the specified lengths. Inputs must not change during a call.

Distance overflow causes failure, even if that point would not have been a
neighbor. C++ exceptions do not cross the ABI. Worker failures cancel further
work and threads are joined. Status codes are documented in
[`include/vectortile.h`](include/vectortile.h). A failed call may have partially
written output; the Python adapter discards it and raises an exception. The
optional error message is always terminated when capacity is nonzero and the
pointer is nonnull.

## Optional sanitizer check

Run the independent C++ smoke harness, instrumenting both implementation and
test executable. This avoids loading an ASan library into an uninstrumented
Python runtime:

```sh
mkdir -p native/build
clang++ -std=c++17 -O1 -g -pthread -ffp-contract=off \
  -fsanitize=address,undefined -fno-omit-frame-pointer \
  -I native/include native/src/vectortile.cpp native/tests/smoke.cpp \
  -o native/build/smoke_sanitized
native/build/smoke_sanitized
```

Use `c++` if it supports the same sanitizer flags. The build script also accepts
`--debug` and `--sanitize address,undefined` for advanced shared-library testing;
loading that library may require the compiler's sanitizer runtime to be loaded
first.
