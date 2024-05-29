import pytest
import numpy as np

from pca_run import computePCA


def test_1d_input():
    data = np.random.rand(100, 50)
    result = computePCA(data, n_components=2)
    assert result.shape == (100, 2), "Output shape should be (N, 2)"

def test_2d_input():
    data = np.random.rand(100, 10, 10)
    result = computePCA(data, n_components=2)
    assert result.shape == (100, 2), "Output shape should be (N, 2)"

def test_n_components_3():
    data = np.random.rand(100, 50)
    result = computePCA(data, n_components=3)
    assert result.shape == (100, 3), "Output shape should be (N, 3) when n_components=3"

def test_reproducibility():
    data = np.random.rand(100, 50)
    result1 = computePCA(data, n_components=2)
    result2 = computePCA(data, n_components=2)
    np.testing.assert_array_almost_equal(result1, result2, err_msg="Results should be the same for the same input data")