import numpy as np

from src.dim_reduction import compute_pca


def test_1d_input():
    data = np.random.rand(100, 50)
    result = compute_pca(data, n_components=2)
    assert result.shape == (100, 2), "Output shape should be (N, 2)"


def test_2d_input():
    data = np.random.rand(100, 10, 10)
    result = compute_pca(data, n_components=2)
    assert result.shape == (100, 2), "Output shape should be (N, 2)"


def test_n_components_3():
    data = np.random.rand(100, 50)
    result = compute_pca(data, n_components=3)
    assert result.shape == (100, 3), "Output shape should be (N, 3) when n_components=3"


def test_reproducibility():
    data = np.random.rand(100, 50)
    result1 = compute_pca(data, n_components=2)
    result2 = compute_pca(data, n_components=2)
    np.testing.assert_array_almost_equal(
        result1, result2, err_msg="Results should be the same for the same input data"
    )
