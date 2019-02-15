import numpy as np

import torch

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

        self.assertIsInstance(loader.label_dict, dict)

    def test_to_sparse_tensor(self):
        loader = DataLoader(self.data_path, 21)
        bow_train_tensor = loader._to_sparse_tensor(loader.bow_train)
        bow_test_tensor = loader._to_sparse_tensor(loader.bow_test)

        self.assertIsInstance(bow_train_tensor, torch.sparse.FloatTensor)
        self.assertIsInstance(bow_test_tensor, torch.sparse.FloatTensor)
        self.assertEqual(bow_train_tensor.size(1), bow_test_tensor.size(1))

    def test_create_loader(self):
        loader = DataLoader(self.data_path, 21)
        torch_loader = loader._create_loader(
            loader.seq_train, loader.bow_train, loader.label_train)
        for (seq, bow, label) in torch_loader:
            self.assertIsInstance(seq, torch.FloatTensor)
            self.assertIsInstance(bow, torch.sparse.FloatTensor)
            self.assertIsInstance(label, torch.IntTensor)

            self.assertEqual(seq.size(0), 32)
            self.assertEqual(bow.size(0), 32)
            self.assertEqual(label.size(0), 32)
            break
