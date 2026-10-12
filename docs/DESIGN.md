# Engineering decisions

## Why this problem?

Embedding retrieval appears in ML evaluation, nearest-neighbor baselines and
similarity search. A readable implementation often materializes every pairwise
distance, which costs Q×N storage before selecting k neighbors. VectorTile
separates distance evaluation, top-k state, and execution backend so those
costs can be inspected and tested independently.

This library assumes the corpus itself fits in memory. It is not a disk-backed
index, vector database, or approximate-nearest-neighbor service.

## Three execution strategies

1. **Dense NumPy reference**: direct broadcast differences and a full stable
   lexicographic sort. Easy to inspect, deliberately limited to small inputs.
2. **Tiled NumPy / PyTorch**: reduce each corpus tile to local top-k, then merge
   with the current best. NumPy's feature loop bounds distance workspace
   independently of D; PyTorch's broadcast tensor gives parallel feature
   reduction at the cost of a D-sized temporary dimension.
3. **Native C++**: scan vectors into a fixed-size max heap. The root is the
   worst retained candidate. A better pair replaces it. Parallelism is across
   query rows, so workers do not share a heap or write the same result row.

A candidate excluded from a tile's top k already has at least k candidates
in that same tile ordered before it. It cannot be in the global top k. This
is why tile-local reduction preserves the exhaustive result, provided all
implementations use the same `(computed distance, corpus index)` ordering.

## Numerical behavior is part of the API

All backends use direct squared differences in float64, not
`||q||² + ||x||² - 2q·x`. The latter can cancel small distances when vectors
share a large offset. Direct differences avoid that particular failure, but
still have floating-point rounding and underflow/overflow limits.

The C++ build disables FMA contraction and fast math is not enabled. NumPy
and PyTorch may reduce features in another order. Consequently a near tie can
rank differently across backends even when distances agree within tolerance.
Exact means exhaustive search, not exact real-number arithmetic. Tie behavior
is deterministic for equal computed distances within a backend.

## Honest performance comparisons

NumPy tiling makes bounded workspace explicit, but its feature and row loops
can cost more time than the dense reference. C++ reduces temporary allocation
and interpreter overhead, and can process independent queries concurrently.
Neither outcome should be assumed for every shape or k.

The PyTorch backend prioritizes a common contract over maximum throughput:
float64, stable full-tile sorting, per-tile finite checks, full input transfer
for CUDA, and CPU result copies are deliberate costs. It is a starting point
for profiling, not an optimized GPU kernel.

## What to build next

- Profile actual NVIDIA hardware and distinguish kernel time, transfers,
  validation and sorting before proposing an optimization.
- Add a tuned independent baseline (e.g. FAISS exact search) with carefully
  matched precision and tie semantics.
- Measure isolated-process peak RSS or allocator peaks alongside the existing
  analytical accounting.
- Explore prepared indexes, corpus streaming from disk, thread-pool reuse,
  and shape-dependent backend selection only with supporting workloads.

These are open directions, not completed features.
