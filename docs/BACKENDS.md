# Python backend implementation notes

The public entry point is `vectortile.search`. Every backend receives validated,
C-contiguous float64 arrays. The result contains int64 `indices` and float64
`distances` of shape `(Q, k)`, with **squared** Euclidean distances. All pairs are
examined; ties in computed distance are ordered by increasing corpus index.

“Exact” refers to exhaustive search, not exact real arithmetic. Reduction order
can differ across backends, so almost-tied distances can produce different
rankings. The implementations avoid the norm/dot-product expansion, which can
lose small distances when vectors share a large offset. Every computed distance
must be finite, including candidates outside the final top k.

## NumPy (`backend="numpy"`)

Direct squared differences are accumulated feature by feature into a distance
tile. Two reusable float64 buffers have shape
`(min(Q, query_block), min(N, corpus_block))`. Their combined size is exactly
`16 * min(Q, query_block) * min(N, corpus_block)` bytes, excluding array-object
overhead. This is the **distance-buffer size**, not total process memory.

Each tile is reduced to at most k candidates by partitioning distances at the
k-th value, retaining all lower values, and resolving boundary ties by index.
Only retained candidates are sorted. These candidates are merged with the
previous best k using the same operation. Additional workspace consists of
`O(Bq*k)` candidate/merge arrays and `O(Bc+k)` per-row partition/index arrays.
Total auxiliary workspace is `O(Bq*Bc + Bq*k + Bc + k)` and does not multiply
the distance-tile size by feature dimension D.

Here Bq and Bc denote effective tile sizes. The required inputs occupy
`8*(Q+N)*D` bytes after conversion, and final output arrays occupy `16*Q*k`
bytes. Conversion may copy the original inputs. Validation also creates
temporary finite-value masks. These costs are separate from the distance
buffers above.

## Dense reference (`backend="reference"`)

The reference independently broadcasts a `(Q, N, D)` difference tensor, squares
it in place, sums over features to a `(Q, N)` matrix, and lexicographically sorts
the full rows. Its principal temporary arrays are:

- Float64 difference/square tensor: `8*Q*N*D` bytes.
- Float64 full distance matrix: `8*Q*N` bytes.
- Int64 full ordering: `8*Q*N` bytes, plus sort implementation workspace.
- Int64 corpus IDs: `8*N` bytes, broadcast as a view.

The full difference tensor remains live during sorting. This implementation is
intentionally a small-input correctness baseline, not a memory-efficient search.

## Optional PyTorch (`backend="torch"` or `backend="cuda"`)

Install with `pip install -e '.[torch]'`. `torch` explicitly selects CPU, while
`cuda` explicitly requires `torch.cuda.is_available()`. Neither selects MPS or
falls back to another backend. PyTorch is imported only when requested.

PyTorch uses float64 broadcast differences per tile so feature reductions can
run in parallel. Its principal difference tensor requires `8*Bq*Bc*D` bytes,
and a distance tile requires `8*Bq*Bc` bytes. The difference tensor is released
before the tile is sorted. Stable sorting of each tile, followed by an
index-ordered stable merge of at most 2k candidates, defines tie behavior.
Sorting adds `O(Bq*Bc)` integer workspace and `O(Bq*k)` merge workspace.

CUDA also stores the full inputs on the GPU. The host-side validated arrays and
NumPy result arrays still exist. Results are copied back to CPU before the call
returns, so end-to-end wall-clock timing includes transfer and synchronization.
The per-tile finite-value check also synchronizes CUDA. This is a correctness
and reproducibility backend; speedups require measurement on actual hardware.

The `threads` argument configures the native backend only. NumPy and PyTorch
retain their runtime's thread configuration; the search function does not alter
process-global thread settings. Tiles configure the NumPy/PyTorch algorithms;
native streaming and the dense reference do not use them.

## Validation evidence

`tests/test_search.py` compares randomized inputs with an independent Python
scalar oracle using `math.fsum`. It checks k=N, single-row tiles, blocks smaller
than k, block-boundary ties, duplicate vectors, Q=0, strided integer inputs,
input preservation, large common offsets, malformed inputs, and difference,
square, and reduction overflow. Optional CPU/CUDA tests run only where the
corresponding runtime/hardware is available; unavailable CUDA is reported as a
skip rather than a successful GPU test.
