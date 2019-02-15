import numpy as np

from unittest import TestCase
from pathlib import Path

from scipy.sparse import csr_matrix

from core.loader import DataLoader


class TestDataLoader(TestCase):
    def setUp(self):
        self.data_path = Path("data/tmn_out/")

    def test_init(self):
        loader = DataLoader(self.data_path, 21)
        self.assertIsInstance(loader.seq_train, np.ndarray)
        self.assertIsInstance(loader.seq_test, np.ndarray)
        self.assertEqual(21, loader.seq_train.shape[1])

        self.assertIsInstance(loader.bow_train, csr_matrix)
        self.assertIsInstance(loader.bow_test, csr_matrix)
        self.assertEqual(loader.seq_train.shape[0], loader.bow_train.shape[0])
        self.assertEqual(loader.seq_test.shape[0], loader.bow_test.shape[0])

        self.assertIsInstance(loader.label_train, np.ndarray)
        self.assertIsInstance(loader.label_test, np.ndarray)
        self.assertEqual(loader.label_train.shape,
                         (loader.seq_train.shape[0], ))
        self.assertEqual(loader.label_test.shape, (loader.seq_test.shape[0], ))
