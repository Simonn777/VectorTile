"""Checked ctypes adapter for the explicitly built C++17 backend."""

from __future__ import annotations

import ctypes
import numbers
import os
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np


def _library_path() -> Path:
    override = os.environ.get("VECTORTILE_NATIVE_LIB")
    if override:
        return Path(override).expanduser().resolve()
    suffix = ".dylib" if sys.platform == "darwin" else ".so"
    return Path(__file__).resolve().parents[2] / "native" / "build" / ("libvectortile" + suffix)


@lru_cache(maxsize=4)
def _load_library(path: str):
    if not Path(path).is_file():
        raise RuntimeError(
            "Native library is not built. From the repository root run "
            "`python scripts/build_native.py`, or set VECTORTILE_NATIVE_LIB "
            "to a compatible library."
        )
    try:
        library = ctypes.CDLL(path)
        library.vectortile_abi_version.argtypes = []
        library.vectortile_abi_version.restype = ctypes.c_int
        if library.vectortile_abi_version() != 1:
            raise RuntimeError("Native library ABI version is incompatible; rebuild it.")
        library.vectortile_search.argtypes = [
            ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
            ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
            ctypes.c_int64, ctypes.POINTER(ctypes.c_int64),
            ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_char),
            ctypes.c_size_t,
        ]
        library.vectortile_search.restype = ctypes.c_int
        return library
    except (OSError, AttributeError) as exc:
        raise RuntimeError(f"Unable to load compatible native library at {path}: {exc}") from exc


def _positive_integer(value, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral):
        raise ValueError(f"{name} must be a positive integer.")
    value = int(value)
    if value < 1 or value > np.iinfo(np.int64).max:
        raise ValueError(f"{name} must be between 1 and the int64 maximum.")
    return value


def _array(value, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 2 or array.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be a two-dimensional real numeric array.")
    with np.errstate(over="ignore", invalid="ignore"):
        array = np.require(array, dtype=np.float64, requirements=["C", "A"])
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain finite float64 values.")
    return array


def native_search(queries, corpus, k, threads=1):
    """Return deterministic ``(indices, squared_distances)`` from native C++.

    Validation is repeated here so calling this adapter directly is safe.
    Input arrays must not be modified concurrently during a search.
    No compilation is performed on import or on search.
    """
    k = _positive_integer(k, "k")
    threads = _positive_integer(threads, "threads")
    queries = _array(queries, "queries")
    corpus = _array(corpus, "corpus")
    q, dimensions = queries.shape
    n, corpus_dimensions = corpus.shape
    if dimensions < 1 or dimensions != corpus_dimensions:
        raise ValueError("Inputs must have the same positive feature dimension.")
    if n < 1 or k > n:
        raise ValueError("Corpus must be nonempty and k must be at most its length.")
    byte_limit = min(np.iinfo(np.intp).max, np.iinfo(np.int64).max)
    for rows, columns, item_size in (
        (q, dimensions, 8), (n, dimensions, 8), (q, k, 8), (1, k, 16)
    ):
        if rows * columns > byte_limit // item_size:
            raise ValueError("Shape or buffer byte size exceeds native limits.")
    library = _load_library(str(_library_path()))
    indices = np.empty((q, k), dtype=np.int64)
    distances = np.empty((q, k), dtype=np.float64)
    error = ctypes.create_string_buffer(512)
    status = library.vectortile_search(
        queries.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        corpus.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        q, n, dimensions, k, threads,
        indices.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        distances.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        error, len(error),
    )
    if status:
        message = error.value.decode("utf-8", errors="replace") or f"Native status {status}."
        if status == 1:
            raise ValueError(message)
        if status == 2:
            # Inputs were already checked above, so this reports a computed
            # distance overflow, matching the public search API.
            raise OverflowError(message)
        if status == 3:
            raise MemoryError(message)
        raise RuntimeError(message)
    return indices, distances
