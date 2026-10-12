"""Native correctness, deterministic ties, and defensive C ABI tests."""

import ctypes
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from vectortile.native import _library_path, _load_library, native_search


@pytest.fixture
def native_library():
    path = _library_path()
    if not path.is_file():
        pytest.skip("Run python scripts/build_native.py to exercise the optional native backend")
    return _load_library(str(path))


def reference(queries, corpus, k):
    distances = np.sum((queries[:, None, :] - corpus[None, :, :]) ** 2, axis=2)
    indices = np.argsort(distances, axis=1, kind="stable")[:, :k]
    return indices, np.take_along_axis(distances, indices, axis=1)


@pytest.mark.parametrize("k", [1, 5, 29])
@pytest.mark.parametrize("threads", [1, 2, 10_000])
def test_native_matches_dense(native_library, k, threads):
    rng = np.random.default_rng(901)
    queries = rng.normal(size=(11, 13))
    corpus = rng.normal(size=(29, 13))
    actual_indices, actual_distances = native_search(queries, corpus, k, threads)
    expected_indices, expected_distances = reference(queries, corpus, k)
    np.testing.assert_array_equal(actual_indices, expected_indices)
    np.testing.assert_allclose(actual_distances, expected_distances, rtol=2e-14)
    assert actual_indices.dtype == np.int64
    assert actual_distances.dtype == np.float64


@pytest.mark.parametrize("threads", [1, 7])
def test_ties_follow_corpus_index(native_library, threads):
    queries = np.zeros((15, 2))
    corpus = np.array([[1, 0], [-1, 0], [0, 0], [0, 1], [0, -1], [0, 0]], dtype=float)
    indices, distances = native_search(queries, corpus, 5, threads)
    np.testing.assert_array_equal(indices, np.tile([2, 5, 0, 1, 3], (15, 1)))
    np.testing.assert_array_equal(distances, np.tile([0, 0, 1, 1, 1], (15, 1)))


def test_all_identical_points(native_library):
    indices, distances = native_search(np.ones((9, 3)), np.ones((250, 3)), 17, 64)
    np.testing.assert_array_equal(indices, np.tile(np.arange(17), (9, 1)))
    np.testing.assert_array_equal(distances, np.zeros((9, 17)))


def test_empty_queries(native_library):
    indices, distances = native_search(np.empty((0, 4)), np.ones((5, 4)), 3, 100)
    assert indices.shape == distances.shape == (0, 3)


def test_strided_and_unaligned_inputs(native_library):
    queries = np.arange(48, dtype=float).reshape(8, 6)[::2, ::2]
    storage = bytearray(1 + 9 * 3 * 8)
    corpus = np.ndarray((9, 3), dtype=np.float64, buffer=storage, offset=1)
    corpus[:] = np.arange(27).reshape(9, 3)
    assert not queries.flags.c_contiguous
    assert not corpus.flags.aligned
    actual = native_search(queries, corpus, 4)
    expected = reference(queries, corpus, 4)
    np.testing.assert_array_equal(actual[0], expected[0])
    np.testing.assert_allclose(actual[1], expected[1])


def test_independent_concurrent_calls(native_library):
    rng = np.random.default_rng(41)
    queries, corpus = rng.normal(size=(7, 19)), rng.normal(size=(53, 19))
    expected = native_search(queries, corpus, 8, threads=1)
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(lambda _: native_search(queries, corpus, 8, 3), range(8)))
    for result in results:
        np.testing.assert_array_equal(result[0], expected[0])
        np.testing.assert_array_equal(result[1], expected[1])


@pytest.mark.parametrize("queries,corpus,error_type", [
    ([[np.nan]], [[0.0]], ValueError), ([[0.0]], [[np.inf]], ValueError),
    ([[1e308]], [[-1e308]], OverflowError), ([[1e200]], [[0.0]], OverflowError),
    ([[1e154, 1e154]], [[0.0, 0.0]], OverflowError),
])
def test_nonfinite_input_or_distance_rejected(native_library, queries, corpus, error_type):
    with pytest.raises(error_type, match="finite|overflow"):
        native_search(queries, corpus, 1)


@pytest.mark.parametrize("kwargs", [
    {"k": True}, {"k": np.bool_(True)}, {"k": 1.5}, {"k": 0}, {"k": 3},
    {"k": 2**70}, {"threads": False}, {"threads": 0}, {"threads": 1.5},
    {"threads": 2**70},
])
def test_invalid_scalar_parameters(kwargs):
    params = {"k": 1, "threads": 1}
    params.update(kwargs)
    with pytest.raises(ValueError):
        native_search([[0.0]], [[0.0], [1.0]], **params)


@pytest.mark.parametrize("queries,corpus", [
    ([0.0], [[0.0]]), ([[True]], [[0.0]]), ([[1j]], [[0.0]]),
    ([["1"]], [[0.0]]), ([[]], [[]]), ([[0.0]], np.empty((0, 1))),
    ([[0.0, 1.0]], [[0.0]]), (np.empty((0, 1)), [[np.nan]]),
])
def test_invalid_arrays(queries, corpus):
    with pytest.raises(ValueError):
        native_search(queries, corpus, 1)


def test_missing_library_has_actionable_error(monkeypatch, tmp_path):
    monkeypatch.setenv("VECTORTILE_NATIVE_LIB", str(tmp_path / "missing.so"))
    with pytest.raises(RuntimeError, match="scripts/build_native.py"):
        native_search([[0.0]], [[0.0]], 1)


def abi_call(library, *, q=1, n=1, d=1, k=1, threads=1,
             queries=None, corpus=None, indices=None, distances=None, capacity=128):
    queries = np.zeros((1, 1)) if queries is None else queries
    corpus = np.zeros((1, 1)) if corpus is None else corpus
    indices = np.zeros((1, 1), dtype=np.int64) if indices is None else indices
    distances = np.zeros((1, 1)) if distances is None else distances
    error = ctypes.create_string_buffer(capacity)
    status = library.vectortile_search(
        queries.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        corpus.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        q, n, d, k, threads,
        indices.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        distances.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), error, capacity,
    )
    return status, error.value


@pytest.mark.parametrize("shapes", [
    {"q": -1}, {"n": 0}, {"d": 0}, {"k": 2}, {"threads": 0},
    {"q": 2**62, "d": 8}, {"n": 2**62, "d": 8},
    {"q": 2**61, "n": 2, "d": 1, "k": 2},
])
def test_c_abi_rejects_invalid_extents_before_reading(native_library, shapes):
    status, message = abi_call(native_library, **shapes)
    assert status == 1
    assert message


def test_c_abi_checks_finite_inputs(native_library):
    status, message = abi_call(native_library, corpus=np.array([[np.nan]]))
    assert status == 2
    assert b"non-finite" in message


def test_c_abi_rejects_overlap(native_library):
    storage = np.zeros((1, 1))
    status, message = abi_call(native_library, queries=storage, distances=storage)
    assert status == 1
    assert b"overlap" in message


def test_c_abi_null_nonempty_buffer(native_library):
    message = ctypes.create_string_buffer(128)
    status = native_library.vectortile_search(None, None, 1, 1, 1, 1, 1, None, None, message, len(message))
    assert status == 1
    assert b"Null" in message.value


def test_c_abi_short_error_buffer_terminated(native_library):
    status, message = abi_call(native_library, q=-1, capacity=1)
    assert status == 1
    assert message == b""
