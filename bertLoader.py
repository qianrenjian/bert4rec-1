from collections import Counter
import itertools
import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.distributions import Categorical
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm


class BertDataset():

    def __init__(self, tr_csr, max_seq_len=200, K=5):
        super(BertDataset, self).__init__()
        self.n_users, self.n_items = tr_csr.shape
        self.indices = tr_csr.indices.astype(np.int64)
        self.indptr = tr_csr.indptr.astype(np.int64)
        #self.data = tr_csr.data
        self.user_seen_cnts = np.ediff1d(self.indptr)
        self.max_seq_len = max_seq_len
        self.K = K

        self.pads = (1 + self.n_items) * np.ones(self.max_seq_len, dtype=np.int64)
        self.zeros = np.zeros(self.max_seq_len, dtype=np.bool)
        self.sharr = np.concatenate([np.ones(self.K), np.zeros(self.max_seq_len)]).astype(np.bool)

    def __getitem__(self, i):
        user_seen = self.user_seen_cnts[i]
        while user_seen < self.K:
            i = np.random.randint(0, self.n_users)
            user_seen = self.user_seen_cnts[i]

        row = self.indices[self.indptr[i]:self.indptr[i+1]].copy()
        if user_seen > self.max_seq_len:
            row = row[np.random.choice(user_seen, self.max_seq_len)]
            user_seen = self.max_seq_len

        temp = self.sharr[:user_seen].copy()
        np.random.shuffle(temp)
        neg_t = row[temp].copy()
        row[temp] = self.n_items
        base = np.concatenate((self.pads[:self.max_seq_len - user_seen], row))
        mask = np.concatenate((self.zeros[:self.max_seq_len - user_seen], temp))
        return torch.from_numpy(base), torch.from_numpy(mask), neg_t
        """
        num_pos = temp.sum()
        num_neg = user_seen - num_pos
        a = np.concatenate((self.pads[:self.max_seq_len - num_pos], row[temp]))
        b = np.concatenate((self.pads[:self.max_seq_len - num_neg], row[~temp]))
        row[temp]
        return (torch.from_numpy(a), torch.from_numpy(b), temp)
        """


    def __len__(self,):
        return self.n_users
