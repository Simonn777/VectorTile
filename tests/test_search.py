"""Behavioral correctness checks against an independent scalar oracle."""

import importlib.util
import math
import sys

import numpy as np
import pytest

from vectortile import BackendUnavailableError, SearchResult, search


def scalar_oracle(queries, corpus, k):
    """Python loops + accurate fsum, independent of vectorized implementation."""
    expected_indices = []
    expected_distances = []
    for query in queries:
        ranked = sorted(
            (math.fsum((float(a) - float(b)) ** 2 for a, b in zip(query, point)), index)
            for index, point in enumerate(corpus)
        )[:k]
        expected_distances.append([pair[0] for pair in ranked])
        expected_indices.append([pair[1] for pair in ranked])
    return (
        np.asarray(expected_indices, dtype=np.int64).reshape(len(queries), k),
        np.asarray(expected_distances, dtype=np.float64).reshape(len(queries), k),
    )


BACKENDS = ["numpy", "reference"]
if importlib.util.find_spec("torch") is not None:
    BACKENDS.append("torch")


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("seed", [0, 17, 832])
@pytest.mark.parametrize("blocks", [(1, 1), (3, 4), (50, 50)])
def test_randomized_against_independent_oracle(backend, seed, blocks):
    rng = np.random.default_rng(seed)
    queries = rng.normal(size=(7, 9))
    corpus = rng.normal(size=(19, 9))
    for k in (1, 5, len(corpus)):
        result = search(
            queries, corpus, k, backend=backend,
            query_block=blocks[0], corpus_block=blocks[1],
        )
        expected_indices, expected_distances = scalar_oracle(queries, corpus, k)
        assert isinstance(result, SearchResult)
        assert result.backend == backend
        np.testing.assert_array_equal(result.indices, expected_indices)
        np.testing.assert_allclose(result.distances, expected_distances, rtol=3e-15, atol=1e-14)
        assert result.indices.dtype == np.int64
        assert result.distances.dtype == np.float64


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("corpus_block", [1, 2, 3, 5, 100])
def test_threshold_ties_across_block_boundaries(backend, corpus_block):
    corpus = np.array([[2, 0], [1, 0], [0, 1], [-1, 0], [0, -1], [0, 0], [1, 0]])
    result = search([[0, 0]], corpus, 4, backend=backend, corpus_block=corpus_block)
    np.testing.assert_array_equal(result.indices, [[5, 1, 2, 3]])
    np.testing.assert_array_equal(result.distances, [[0, 1, 1, 1]])


@pytest.mark.parametrize("backend", BACKENDS)
def test_identical_vectors_choose_lowest_indices(backend):
    result = search(np.ones((3, 4)), np.ones((23, 4)), 6, backend=backend, corpus_block=4)
    np.testing.assert_array_equal(result.indices, np.tile(np.arange(6), (3, 1)))
    np.testing.assert_array_equal(result.distances, np.zeros((3, 6)))


@pytest.mark.parametrize("backend", BACKENDS)
def test_empty_queries_are_valid(backend):
    result = search(np.empty((0, 3)), np.ones((2, 3)), 2, backend=backend)
    assert result.indices.shape == result.distances.shape == (0, 2)
    assert result.indices.dtype == np.int64
    assert result.distances.dtype == np.float64


@pytest.mark.parametrize("backend", BACKENDS)
def test_noncontiguous_integer_inputs_and_no_mutation(backend):
    original = np.arange(60, dtype=np.int32).reshape(10, 6)
    unchanged = original.copy()
    queries, corpus = original[:4, ::2], original[::2, ::2]
    expected_indices, expected_distances = scalar_oracle(queries, corpus, 3)
    result = search(queries, corpus, np.int64(3), backend=backend)
    np.testing.assert_array_equal(result.indices, expected_indices)
    np.testing.assert_array_equal(result.distances, expected_distances)
    np.testing.assert_array_equal(original, unchanged)


@pytest.mark.parametrize("backend", BACKENDS)
def test_large_common_offset_preserves_small_distances(backend):
    # A norm/dot expansion would erase the small differences by cancellation.
    offset = 2.0 ** 40
    result = search([[offset, offset]], [[offset + 2, offset], [offset + 1, offset]], 2, backend=backend)
    np.testing.assert_array_equal(result.indices, [[1, 0]])
    np.testing.assert_array_equal(result.distances, [[1, 4]])


@pytest.mark.parametrize("argument", ["queries", "corpus"])
@pytest.mark.parametrize(
    "invalid,exception",
    [
        (np.ones(3), ValueError),
        (np.ones((1, 2, 3)), ValueError),
        ([[True]], TypeError),
        ([[1 + 2j]], TypeError),
        ([["2"]], TypeError),
        (np.array([[1]], dtype=object), TypeError),
        ([[np.nan]], ValueError),
        ([[np.inf]], ValueError),
        ([[-np.inf]], ValueError),
    ],
)
def test_invalid_arrays(argument, invalid, exception):
    kwargs = {"queries": [[0]], "corpus": [[0]], "k": 1}
    kwargs[argument] = invalid
    with pytest.raises(exception):
        search(**kwargs)


@pytest.mark.parametrize(
    "queries,corpus,k",
    [
        (np.ones((2, 0)), np.ones((2, 0)), 1),
        (np.ones((2, 2)), np.ones((3, 1)), 1),
        (np.ones((2, 1)), np.ones((0, 1)), 1),
        (np.ones((2, 1)), np.ones((1, 1)), 2),
    ],
)
def test_invalid_shapes_and_k_range(queries, corpus, k):
    with pytest.raises(ValueError):
        search(queries, corpus, k)


@pytest.mark.parametrize("name", ["k", "query_block", "corpus_block", "threads"])
@pytest.mark.parametrize("value", [True, np.bool_(False), 1.0, 2.5, "2", None])
def test_integer_parameters_reject_nonintegers(name, value):
    kwargs = {"k": 1, name: value}
    with pytest.raises(TypeError):
        search([[0]], [[0]], **kwargs)


@pytest.mark.parametrize("name", ["k", "query_block", "corpus_block", "threads"])
@pytest.mark.parametrize("value", [0, -1])
def test_integer_parameters_are_positive(name, value):
    kwargs = {"k": 1, name: value}
    with pytest.raises(ValueError):
        search([[0]], [[0]], **kwargs)


@pytest.mark.parametrize("backend", ["mps", "gpu", "", None, []])
def test_unknown_backend_is_explicit_error(backend):
    with pytest.raises(ValueError, match="backend"):
        search([[0]], [[0]], 1, backend=backend)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize(
    "queries,corpus",
    [
        ([[1e308]], [[-1e308]]),  # Difference overflows.
        ([[0]], [[1e200]]),  # Square overflows.
        ([[0, 0]], [[1e154, 1e154]]),  # Sum overflows.
        ([[0]], [[0], [1e200]]),  # Overflowing item is outside top k.
    ],
)
def test_distance_overflow_is_rejected(backend, queries, corpus):
    with pytest.raises(OverflowError, match="distance overflow"):
        search(queries, corpus, 1, backend=backend, corpus_block=1)


def test_torch_missing_never_falls_back(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)
    with pytest.raises(BackendUnavailableError, match="not installed"):
        search([[0]], [[0]], 1, backend="torch")


def test_cuda_unavailable_never_falls_back():
    torch = pytest.importorskip("torch")
    if torch.cuda.is_available():
        pytest.skip("CUDA available; exercised by CUDA correctness test")
    with pytest.raises(BackendUnavailableError, match="CUDA was requested"):
        search([[0]], [[0]], 1, backend="cuda")


def test_cuda_when_available():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA hardware/runtime unavailable")
    rng = np.random.default_rng(332)
    queries, corpus = rng.normal(size=(9, 13)), rng.normal(size=(31, 13))
    expected_indices, expected_distances = scalar_oracle(queries, corpus, 7)
    result = search(queries, corpus, 7, backend="cuda", query_block=4, corpus_block=9)
    np.testing.assert_array_equal(result.indices, expected_indices)
    np.testing.assert_allclose(result.distances, expected_distances, rtol=3e-15, atol=1e-14)
    ties = search(np.ones((2, 3)), np.ones((17, 3)), 5, backend="cuda", corpus_block=3)
    np.testing.assert_array_equal(ties.indices, np.tile(np.arange(5), (2, 1)))
