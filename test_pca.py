import unittest
import numpy as np

from pca_run import computePCA


class TestComputePCA(unittest.TestCase):

    def test_1d_input(self):
        data = np.random.rand(100, 50)
        result = computePCA(data, n_components=2)
        self.assertEqual(result.shape, (100, 2), "Output shape should be (N, 2)")

    def test_2d_input(self):
        data = np.random.rand(100, 10, 10)
        result = computePCA(data, n_components=2)
        self.assertEqual(result.shape, (100, 2), "Output shape should be (N, 2)")

    def test_n_components_3(self):
        data = np.random.rand(100, 50)
        result = computePCA(data, n_components=3)
        self.assertEqual(result.shape, (100, 3), "Output shape should be (N, 3) when n_components=3")

    def test_reproducibility(self):
        data = np.random.rand(100, 50)
        result1 = computePCA(data, n_components=2)
        result2 = computePCA(data, n_components=2)
        np.testing.assert_array_almost_equal(result1, result2, err_msg="Results should be the same for the same input data")
