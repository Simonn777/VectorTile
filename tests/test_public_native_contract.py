"""Cross the public dispatch boundary, not just the ctypes adapter."""
import numpy as np
import pytest
from vectortile import search
from vectortile.native import _library_path


@pytest.mark.skipif(not _library_path().is_file(), reason="native library not built")
def test_native_overflow_outside_topk_uses_public_exception_contract():
    with pytest.raises(OverflowError, match="overflow"):
        search(np.array([[0.0]]), np.array([[0.0], [1e200]]), 1, backend="native")


@pytest.mark.skipif(not _library_path().is_file(), reason="native library not built")
def test_native_public_dispatch_preserves_ties_and_result_metadata():
    result = search([[0.0, 0.0]], [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]], 2, backend="native", threads=8)
    assert result.backend == "native"
    np.testing.assert_array_equal(result.indices, [[0, 1]])
    np.testing.assert_array_equal(result.distances, [[1.0, 1.0]])
