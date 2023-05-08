import os
import random
import torch
from torch import nn
import numpy as np
import pandas as pd
from itertools import combinations
import _pickle as cPickle

def pkl_save(file, obj):
    with open(file, "wb") as out_file:
        cPickle.dump(obj, out_file)


def pkl_load(file):
    with open(file, "rb") as in_file:
        obj = cPickle.load(in_file)
    return obj


def _read_file_ab(data_dir):
    with open(data_dir, 'r') as f:
        lines = f.readlines()
    paragraphs = [
        line.strip().split() for line in lines]
    random.shuffle(paragraphs)
    return paragraphs


def sampling_expanding(dataset, isolates_to_sample_per_batch, n_metadata, minimum_size_of_predictors, d_mean_values, d_max_samples):
    idx_samples = np.random.choice(np.array(range(len(dataset))), size = isolates_to_sample_per_batch, replace = False)
    lst = [dataset[j] for j in list(idx_samples)]
    x, y = expand_list_of_isolates(lst, n_metadata, d_mean_values, d_max_samples, minimum_size_of_predictors)
    return x, y

def expand_list_of_isolates(lst, n_metadata, d_mean_values, d_max_samples, minimum_size_of_predictors):            
    number_of_posible_predictors = [len(s)-n_metadata-1 for s in lst]
    max_number_preditors = [d_max_samples[j] for j in number_of_posible_predictors]        
    Sizes_i = [size_predictors_per_isolate(lst[j], n_metadata, d_mean_values, max_number_preditors[j], minimum_size_of_predictors) for j in range(len(lst))]
    expanded = [expand_isolate(lst[j], Sizes_i[j], n_metadata) for j in range(len(lst))]
    expanded = [e for e in expanded if e is not None]
    expanded = [l for sub in expanded for l in sub]
    x = [l[0] for l in expanded]
    y = [l[1] for l in expanded]
    return x, y


def size_predictors_per_isolate(s, n_metadata, d_mean_values, max_samples_per_isolate, minimum_size_of_predictors):
    n_ab = len(s) - n_metadata
    n_max_predictors = n_ab - 1 
    p = d_mean_values[n_max_predictors] / n_max_predictors
    sizes = np.random.binomial(n_max_predictors, d_mean_values[n_max_predictors] / n_max_predictors, max_samples_per_isolate)
    sizes = [l  for l in sizes if l >= minimum_size_of_predictors]
    return sizes


def expand_isolate(s, sizes_i, n_metadata):
    if len(sizes_i)<1:
        r = None
    else:
        r = [subsample_sentence(s, j, n_metadata) for j in sizes_i]
    return r


def subsample_sentence(s, size_sample, n_metadata):
    x = np.sort(np.random.choice(np.array(range(len(s)-n_metadata)), size = size_sample, replace = False) + n_metadata)
    x = s[0:n_metadata] + [s[p] for p in x]
    y = [t for t in s if t not in x]
    return x, y



class AB_TextDataset(torch.utils.data.Dataset):
    def __init__(self, X, Y, vocab, ab2, max_len, pos_response, Fake_position):
        self.vocab = vocab
        self.df = pd.DataFrame(list(zip(X, Y)), columns =['x', 'y'])
        self.ab2 = ab2
        self.pos_response = pos_response
        self.max_len = max_len
        self.pos_response_list = list(self.pos_response.keys())
        self.Fake_position = Fake_position
        
    def __getitem__(self, idx):
        ex = self.df.x.loc[idx]
        ey = self.df.y.loc[idx]
        l = len(ex)
        n_y = len(ey)
        p_x = [l.split("_")[0] for l in ex if l.split("_")[0] in self.ab2]
        x_pos = [a in p_x for a in self.ab2]
        x_pos = torch.tensor(x_pos, dtype=torch.bool)
        x_resp = [1 if a in ex else 0 for a in self.pos_response_list]
        x_resp = torch.tensor(x_resp, dtype=torch.int64)
        p_e = [l.split("_")[0] for l in ey if l.split("_")[0] in self.ab2]
        y_pos = [a in p_e for a in self.ab2]
        y_pos = torch.tensor(y_pos, dtype=torch.bool)
        y_resp = [1 if a in ey else 0 for a in self.pos_response_list]
        y_resp = torch.tensor(y_resp, dtype=torch.int64)
        ex = ex + ['<pad>']*(self.max_len - l)
        ex_pad = [e == '<pad>' for e in ex]
        ex_pos = torch.arange(self.max_len) + 1
        ex_pos[torch.tensor(ex_pad)] = self.Fake_position
        ex = torch.tensor(self.vocab[ex], dtype=torch.int64)
        return ex, ex_pos, x_resp, x_pos, y_resp, y_pos, torch.tensor(l, dtype=torch.int64), torch.tensor(n_y, dtype=torch.int64)

    def __len__(self):
        return self.df.shape[0]



class Vocab:
    def __init__(self, tokens=[], min_freq=0, reserved_tokens=[]):
        if tokens and isinstance(tokens[0], list):
            tokens = [token for line in tokens for token in line]
        counter = collections.Counter(tokens)
        self.token_freqs = sorted(counter.items(), key=lambda x: x[1],
                                  reverse=True)
        self.idx_to_token = list(sorted(set(['<unk>'] + reserved_tokens + [
            token for token, freq in self.token_freqs if freq >= min_freq])))
        self.token_to_idx = {token: idx
                             for idx, token in enumerate(self.idx_to_token)}
        
    def __len__(self):
        return len(self.idx_to_token)
        
    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]
        
    def to_tokens(self, indices):
        if hasattr(indices, '__len__') and len(indices) > 1:
            return [self.idx_to_token[int(index)] for index in indices]
        return self.idx_to_token[indices]
        
    @property
    def unk(self): 
        return self.token_to_idx['<unk>']

