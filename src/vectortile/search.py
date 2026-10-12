"""Direct squared-L2 search, ordered by (distance, corpus index).

"Exact" means every query/corpus pair is examined. It does not mean exact
real arithmetic: different float64 reduction orders can disagree on nearly
equal distances. No backend uses the cancellation-prone norm/dot identity.
"""

from dataclasses import dataclass
from numbers import Integral
from typing import Tuple

import numpy as np


class BackendUnavailableError(RuntimeError):
    """An explicitly requested optional backend cannot run."""


@dataclass(frozen=True)
class SearchResult:
    """Top-k corpus indices and squared distances, both shaped (queries, k)."""

    indices: np.ndarray
    distances: np.ndarray
    backend: str


def _positive_integer(value, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer, not a boolean")
    if value < 1:
        raise ValueError(f"{name} must be positive")
    return int(value)


def _array(value, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a 2D array")
    if array.dtype.kind not in "iuf":
        raise TypeError(f"{name} must contain real numeric values (not booleans)")
    try:
        with np.errstate(over="raise", invalid="raise"):
            array = np.ascontiguousarray(array, dtype=np.float64)
    except (FloatingPointError, OverflowError) as exc:
        raise ValueError(f"{name} cannot be represented as finite float64") from exc
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def _select_topk(
    distances: np.ndarray, indices: np.ndarray, k: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Partition by value; break threshold ties by index; sort at most k items.

    ``indices`` may be one shared row or a matrix. Partitioning the distances
    alone is insufficient: it could discard the smaller index at a tie.
    """
    rows, columns = distances.shape
    count = min(k, columns)
    selected_indices = np.empty((rows, count), dtype=np.int64)
    selected_distances = np.empty((rows, count), dtype=np.float64)
    for row in range(rows):
        values = distances[row]
        ids = indices if indices.ndim == 1 else indices[row]
        if count == columns:
            positions = np.arange(columns)
        else:
            threshold = np.partition(values, count - 1)[count - 1]
            below = np.flatnonzero(values < threshold)
            tied = np.flatnonzero(values == threshold)
            needed = count - below.size
            if tied.size > needed:
                tied = tied[np.argpartition(ids[tied], needed - 1)[:needed]]
            positions = np.concatenate((below, tied))
        order = np.lexsort((ids[positions], values[positions]))
        positions = positions[order]
        selected_indices[row] = ids[positions]
        selected_distances[row] = values[positions]
    return selected_indices, selected_distances


def _numpy_search(queries, corpus, k, query_block, corpus_block):
    output_indices = np.empty((len(queries), k), dtype=np.int64)
    output_distances = np.empty((len(queries), k), dtype=np.float64)
    tile_shape = (min(query_block, len(queries)), min(corpus_block, len(corpus)))
    distance_buffer = np.empty(tile_shape, dtype=np.float64)
    scratch_buffer = np.empty_like(distance_buffer)
    try:
        with np.errstate(over="raise", invalid="raise"):
            for q_start in range(0, len(queries), query_block):
                query_tile = queries[q_start : q_start + query_block]
                best_indices = np.empty((len(query_tile), 0), dtype=np.int64)
                best_distances = np.empty((len(query_tile), 0), dtype=np.float64)
                for c_start in range(0, len(corpus), corpus_block):
                    corpus_tile = corpus[c_start : c_start + corpus_block]
                    distances = distance_buffer[: len(query_tile), : len(corpus_tile)]
                    distances.fill(0)
                    scratch = scratch_buffer[: len(query_tile), : len(corpus_tile)]
                    for feature in range(queries.shape[1]):
                        np.subtract(
                            query_tile[:, feature, None],
                            corpus_tile[None, :, feature],
                            out=scratch,
                        )
                        np.square(scratch, out=scratch)
                        np.add(distances, scratch, out=distances)
                    ids = np.arange(c_start, c_start + len(corpus_tile), dtype=np.int64)
                    local_indices, local_distances = _select_topk(distances, ids, k)
                    if best_indices.shape[1] == 0:
                        best_indices, best_distances = local_indices, local_distances
                    else:
                        best_indices, best_distances = _select_topk(
                            np.concatenate((best_distances, local_distances), axis=1),
                            np.concatenate((best_indices, local_indices), axis=1),
                            k,
                        )
                output_indices[q_start : q_start + len(query_tile)] = best_indices
                output_distances[q_start : q_start + len(query_tile)] = best_distances
    except FloatingPointError as exc:
        raise OverflowError("squared-L2 distance overflow in float64") from exc
    return output_indices, output_distances


def _reference_search(queries, corpus, k):
    """Dense independent algorithm; deliberately unsuitable for large inputs."""
    try:
        with np.errstate(over="raise", invalid="raise"):
            differences = queries[:, None, :] - corpus[None, :, :]
            np.square(differences, out=differences)
            distances = differences.sum(axis=2)
    except FloatingPointError as exc:
        raise OverflowError("squared-L2 distance overflow in float64") from exc
    ids = np.broadcast_to(np.arange(len(corpus), dtype=np.int64), distances.shape)
    order = np.lexsort((ids, distances), axis=1)[:, :k]
    return order.copy(), np.take_along_axis(distances, order, axis=1)


def _torch_search(queries, corpus, k, query_block, corpus_block, *, cuda):
    try:
        import torch
    except ImportError as exc:
        raise BackendUnavailableError(
            "PyTorch is not installed; install vectortile[torch] to use this backend"
        ) from exc
    if cuda and not torch.cuda.is_available():
        raise BackendUnavailableError("CUDA was requested but torch.cuda.is_available() is false")
    device = torch.device("cuda" if cuda else "cpu")
    q_tensor = torch.as_tensor(queries, dtype=torch.float64, device=device)
    c_tensor = torch.as_tensor(corpus, dtype=torch.float64, device=device)
    output_indices = np.empty((len(queries), k), dtype=np.int64)
    output_distances = np.empty((len(queries), k), dtype=np.float64)
    with torch.inference_mode():
        for q_start in range(0, len(queries), query_block):
            query_tile = q_tensor[q_start : q_start + query_block]
            best_indices = None
            best_distances = None
            for c_start in range(0, len(corpus), corpus_block):
                corpus_tile = c_tensor[c_start : c_start + corpus_block]
                differences = query_tile[:, None, :] - corpus_tile[None, :, :]
                differences.square_()
                distances = differences.sum(dim=2)
                del differences
                if not bool(torch.isfinite(distances).all().item()):
                    raise OverflowError("squared-L2 distance overflow in float64")
                # Columns begin in increasing corpus-index order, so a stable
                # distance sort deterministically preserves the tie order.
                order = torch.argsort(distances, dim=1, stable=True)[:, :k]
                local_distances = torch.gather(distances, 1, order)
                local_indices = order + c_start
                if best_indices is None:
                    best_indices, best_distances = local_indices, local_distances
                else:
                    merged_indices = torch.cat((best_indices, local_indices), dim=1)
                    merged_distances = torch.cat((best_distances, local_distances), dim=1)
                    by_index = torch.argsort(merged_indices, dim=1)
                    merged_indices = torch.gather(merged_indices, 1, by_index)
                    merged_distances = torch.gather(merged_distances, 1, by_index)
                    by_distance = torch.argsort(merged_distances, dim=1, stable=True)[:, :k]
                    best_indices = torch.gather(merged_indices, 1, by_distance)
                    best_distances = torch.gather(merged_distances, 1, by_distance)
                del distances, order, local_indices, local_distances
            # .cpu() makes the public API synchronous, including CUDA timing.
            output_indices[q_start : q_start + len(query_tile)] = best_indices.cpu().numpy()
            output_distances[q_start : q_start + len(query_tile)] = best_distances.cpu().numpy()
    return output_indices, output_distances


def search(
    queries,
    corpus,
    k=5,
    *,
    backend="numpy",
    query_block=16,
    corpus_block=256,
    threads=1,
) -> SearchResult:
    """Exhaustively find nearest corpus rows by squared Euclidean distance.

    Inputs must be finite, real 2D numeric arrays with equal positive feature
    dimension. The corpus is nonempty and ``1 <= k <= len(corpus)``. Empty
    query batches are valid. Inputs are converted to contiguous float64.

    Backends: ``numpy`` (default, tiled), ``reference`` (dense), ``native``
    (explicitly built C++), ``torch`` (PyTorch CPU), and ``cuda`` (PyTorch CUDA).
    Optional backends never silently fall back. ``threads`` controls only the
    native backend; other backends retain their runtime's thread configuration.
    Tile sizes control NumPy and PyTorch only. All integer settings must be
    positive integers, including when unused by the chosen backend.

    Results are ordered by distance, then by corpus index. An overflowing
    computed distance raises ``OverflowError``, even if outside the top k.
    """
    if not isinstance(backend, str) or backend not in {
        "numpy", "reference", "native", "torch", "cuda"
    }:
        raise ValueError("backend must be one of: numpy, reference, native, torch, cuda")
    k = _positive_integer(k, "k")
    query_block = _positive_integer(query_block, "query_block")
    corpus_block = _positive_integer(corpus_block, "corpus_block")
    threads = _positive_integer(threads, "threads")
    queries = _array(queries, "queries")
    corpus = _array(corpus, "corpus")
    if queries.shape[1] == 0 or queries.shape[1] != corpus.shape[1]:
        raise ValueError("queries and corpus must have the same positive feature dimension")
    if len(corpus) == 0:
        raise ValueError("corpus must contain at least one row")
    if k > len(corpus):
        raise ValueError("k must not exceed the number of corpus rows")
    if backend == "numpy":
        indices, distances = _numpy_search(queries, corpus, k, query_block, corpus_block)
    elif backend == "reference":
        indices, distances = _reference_search(queries, corpus, k)
    elif backend == "native":
        from .native import native_search

        indices, distances = native_search(queries, corpus, k, threads=threads)
    else:
        indices, distances = _torch_search(
            queries, corpus, k, query_block, corpus_block, cuda=backend == "cuda"
        )
    return SearchResult(indices=indices, distances=distances, backend=backend)
