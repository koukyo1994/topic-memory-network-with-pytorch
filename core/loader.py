import gc
import json
import pickle

import numpy as np

import torch
import torch.utils.data

from sklearn.model_selection import train_test_split

from keras.preprocessing.sequence import pad_sequences


class DataLoader:
    def __init__(self, data_path, maxlen):
        seq_train = pickle.load(open(data_path / "dataMsgTrain.pkl", "rb"))
        seq_test = pickle.load(open(data_path / "dataMsgTest.pkl", "rb"))

        self.seq_train = pad_sequences(seq_train, maxlen=maxlen)
        self.seq_test = pad_sequences(seq_test, maxlen=maxlen)
        del seq_train, seq_test
        gc.collect()

        self.bow_train = pickle.load(
            open(data_path / "dataMsgBowTrain.pkl", "rb"))
        self.bow_test = pickle.load(
            open(data_path / "dataMsgBowTest.pkl", "rb"))
        self.label_train = pickle.load(
            open(data_path / "dataMsgLabelTrain.pkl", "rb"))
        self.label_test = pickle.load(
            open(data_path / "dataMsglabelTest.pkl", "rb"))

        self.label_dict = json.load(open(data_path / "labelDict.json"))
        self.CATEGORY = len(self.label_dict)

        self.seq_val = None
        self.bow_val = None
        self.label_val = None
        self.splitted = False

    def _to_sparse_tensor(self, x):
        coo = x.tocoo()
        values = coo.data
        indices = np.vstack((coo.row, coo.col))
        i = torch.LongTensor(indices)
        v = torch.FloatTensor(values)
        shape = coo.shape
        return torch.sparse.FloatTensor(i, v, torch.Size(shape)).cuda()

    def _create_loader(self, seq, bow, label, batch_size=32, shuffle=True):
        seq_tensor = torch.tensor(seq, dtype=torch.long).cuda()
        bow_tensor = self._to_sparse_tensor(bow)
        label_tensor = torch.tensor(label, dtype=torch.int32).cuda()
        dataset = torch.utils.data.TensorDataset(seq_tensor, bow_tensor,
                                                 label_tensor)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle)
        return loader

    def load_raw_train(self):
        return self.seq_train, self.bow_train, self.label_train

    def load_raw_test(self):
        return self.seq_test, self.bow_test, self.label_test

    def load_train_all(self, batch_size=32):
        data_loader = self._create_loader(
            self.seq_train,
            self.bow_train,
            self.label_train,
            batch_size=batch_size,
            shuffle=True)
        return data_loader

    def load_train(self, batch_size=32):
        seq_train, seq_valid, bow_train, bow_valid, label_train, label_valid =\
            train_test_split(
                self.seq_train,
                self.bow_train,
                self.label_train,
                test_size=0.3,
                shuffle=True,
                random_state=42)
        self.seq_val = seq_valid
        self.bow_val = bow_valid
        self.label_val = label_valid
        self.splitted = True

        data_loader = self._create_loader(
            seq_train,
            bow_train,
            label_train,
            batch_size=batch_size,
            shuffle=True)
        return data_loader

    def load_valid(self, batch_size=32):
        assert self.splitted, "Call `load_train` method first"
        data_loader = self._create_loader(
            self.seq_val,
            self.bow_val,
            self.label_val,
            batch_size=batch_size,
            shuffle=False)
        return data_loader

    def load_test(self, batch_size=32):
        data_loader = self._create_loader(
            self.seq_test,
            self.bow_test,
            self.label_test,
            batch_size=batch_size,
            shuffle=False)
        return data_loader
