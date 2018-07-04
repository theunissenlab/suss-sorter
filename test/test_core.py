import unittest

import numpy as np
from numpy.testing import assert_array_equal

from suss.core import BaseDataset, ClusterDataset, SpikeDataset


class TestBasicDataset(unittest.TestCase):

    def setUp(self):
        self.test_times = np.array([1.0, 1.5, 0.5, 2.0])
        self.correct_idx_order = [2, 0, 1, 3]
        self.test_data = np.array([10, 15, 5, 20])
        self.dataset = BaseDataset(
            times=self.test_times,
            datapoints=(self.test_data, ("int32"))
        )

        self.test_data_2d = np.array([[10, 0], [15, 0], [5, 1], [20, 1]])
        self.dataset_2d = BaseDataset(
            times=self.test_times,
            datapoints=(self.test_data_2d, ("int32", 2))
        )

    def test_init_sorted(self):
        np.testing.assert_array_equal(
                self.dataset.datapoints,
                self.test_data[self.correct_idx_order])
        np.testing.assert_array_equal(
                self.dataset.times,
                self.test_times[self.correct_idx_order])
        np.testing.assert_array_equal(
                self.dataset.ids,
                np.array([0, 1, 2, 3]),
                "ids are not in order"
        )

    def test_2d_data(self):
        np.testing.assert_array_equal(
                self.dataset_2d.datapoints,
                self.test_data_2d[[2, 0, 1, 3]]
        )

    def test_len(self):
        self.assertEqual(len(self.dataset), 4)

    def test_centroid(self):
        self.assertEqual(self.dataset.time, 1.25)
        self.assertEqual(self.dataset.centroid, 12.5)
        self.assertEqual(self.dataset_2d.time, 1.25)
        np.testing.assert_array_equal(
                self.dataset_2d.centroid,
                np.array([12.5, 0.5])
        )

    def test_comparator(self):
        d1 = BaseDataset(
                times=np.array([1, 3, 4]),
                datapoints=(np.array([1, 2, 3]), "int32")
        )
        d2 = BaseDataset(
                times=np.array([2, 4, 4.5]),
                datapoints=(np.array([1, 2, 3]), "int32")
        )
        self.assertTrue(d1 < d2)

        d2 = BaseDataset(
                times=np.array([1, 2, 5]),
                datapoints=(np.array([1, 2, 3]), "int32")
        )
        self.assertFalse(d1 < d2, "Time comparison should use median")

    def test_select(self):
        # Create a dataset
        test_times = np.array([1.0, 2.0, 3.0, 4.0])
        test_data = np.array([1, 2, 3, 4])
        dataset = BaseDataset(
            times=test_times,
            datapoints=(test_data, ("int32"))
        )

        subset = dataset.select([True, False, True, True])

        assert_array_equal(
                subset.times,
                np.array([1.0, 3.0, 4.0])
        )
        assert_array_equal(
                subset.datapoints,
                np.array([1, 3, 4])
        )
        assert_array_equal(
                subset.ids,
                np.array([0, 2, 3])
        )

        subset2 = dataset.select([1, 2])
        assert_array_equal(
                subset2.times,
                np.array([2.0, 3.0])
        )
        assert_array_equal(
                subset2.datapoints,
                np.array([2, 3])
        )
        assert_array_equal(
                subset2.ids,
                np.array([1, 2])
        )


class TestSpikeDataset(unittest.TestCase):

    def setUp(self):
        self.test_times = np.array([0.5, 1.0, 1.5, 2.0])
        self.test_data = np.array([
            [0, 0, 0, 0, -10, -5, -2, -1, 0, 0],
            [0, 0, 0, -5, -12, -2, 2, 1, 0, 0],
            [0, 0, 0, -3, -11, -2, 1, 0, 0, 0],
            [0, 0, 0, -5, -11, -2, 1, 0, 0, 0],
        ])

    def test_init(self):
        dataset = SpikeDataset(
            times=self.test_times,
            waveforms=self.test_data
        )
        assert_array_equal(
                dataset.waveforms,
                self.test_data
        )
        assert_array_equal(
                dataset.labels,
                np.zeros(4)
        )
        self.assertEqual(dataset.data_column, "waveforms")

    def test_labels(self):
        dataset = SpikeDataset(
            self.test_times,
            self.test_data,
            labels=np.array([0, 1, 1, 1])
        )
        assert_array_equal(
                dataset.waveforms,
                self.test_data
        )
        assert_array_equal(
                dataset.labels,
                np.array([0, 1, 1, 1])
        )
