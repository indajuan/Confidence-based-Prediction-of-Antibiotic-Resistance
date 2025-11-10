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

def _read_file_ab_cv_pathogen(data_dir):
    with open(data_dir, 'r') as f:
        lines = f.readlines()
    random.shuffle(lines)
    paragraphs = [
        line.strip().split() for line in lines]
    cv = [p[-1] for p in paragraphs]
    pathogen = [p[0] for p in paragraphs]
    paragraphs1 = [p[:-1] for p in paragraphs]
    return paragraphs1, cv, pathogen


def sampling_expanding(dataset, isolates_to_sample_per_batch, n_metadata, minimum_size_of_predictors, d_mean_values, d_max_samples, seed = None):
    """
    dataset = (5 metadata from patient, antibiotics)
    isolates_to_sample_per_batch = how many isolates from the dataset to include
    n_metadata = how many variables in the metadata = 5 variables
    minimum_size_of_predictors = minimum number of antibiotics to train the model with (do predictions with)
    d_mean_values = dictionary, if an isolate has x tests, expected number of isolates to use as prediction when we expand
    d_max_samples = dictionary, max number of data points per isolate when expanding
    """
    if seed is None:
        seed = random.randint(1, 10**18)
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    print("seed")
    print(seed)
    print("")
    rs = dataset.sample(n=isolates_to_sample_per_batch, weights='weight', random_state = seed)
    lst = rs['sentence'].tolist()
    antib, patient_info, response = expand_list_of_isolates(lst, n_metadata, d_mean_values, d_max_samples, minimum_size_of_predictors)
    return antib, patient_info, response

def expand_list_of_isolates(lst, n_metadata, d_mean_values, d_max_samples, minimum_size_of_predictors):
    patient_data = [l[:n_metadata] for l in lst]
    antibiotics = [l[n_metadata:] for l in lst]
    num_predictors = np.array([len(s) - 1 for s in antibiotics])
    max_samples = np.array([d_max_samples[n] for n in num_predictors])    
    Sizes_i = batched_size_predictors(num_predictors, d_mean_values, max_samples, minimum_size_of_predictors)
    expanded = []
    for abx, sizes, patient in zip(antibiotics, Sizes_i, patient_data):
        if sizes:
            expanded.extend([subsample_sentence(abx, sz, patient) for sz in sizes])
    antib = [e[0] for e in expanded]
    patient_info = [e[1] for e in expanded]
    response = [e[2] for e in expanded]
    return antib, patient_info, response

def batched_size_predictors(num_predictors, d_mean_values, max_samples, min_size):
    sizes_list = []
    for i in range(len(num_predictors)):
        s = num_predictors[i]
        n_samples = max_samples[i]
        if s in d_mean_values:
            p = d_mean_values[s] / s
            samples = np.random.binomial(s, p, n_samples)
            filtered = samples[samples >= min_size]
            sizes_list.append(filtered.tolist())
        else:
            sizes_list.append([])
    return sizes_list

def subsample_sentence(s, size_sample, patient):
    x = np.sort(np.random.choice(np.arange(len(s)), size=size_sample, replace=False))
    predictors = ['<cls>'] + [patient[0]] + [s[p] for p in x]
    predictor_set = set(predictors)
    response = [t for t in s if t not in predictor_set]
    patient_info = ['<cls>'] + patient
    return predictors, patient_info, response


# def expand_list_of_isolates(lst, n_metadata, d_mean_values, d_max_samples, minimum_size_of_predictors):            
#    patient_data = [l[0:n_metadata] for l in lst]
#    antibiotics = [l[n_metadata:] for l in lst]
#    number_of_posible_predictors = [len(s) - 1  for s in antibiotics]
#    max_number_preditors = [d_max_samples[j] for j in number_of_posible_predictors]
#    Sizes_i = [size_predictors_per_isolate(number_of_posible_predictors[j],  d_mean_values, max_number_preditors[j], minimum_size_of_predictors) for j in range(len(number_of_posible_predictors))]
#    expanded = [expand_isolate(antibiotics[j], Sizes_i[j], patient_data[j]) for j in range(len(antibiotics))]
#    expanded = [e for e in expanded if e is not None]
#    expanded = [l for sub in expanded for l in sub]
#    antib = [l[0] for l in expanded]
#    patient_info = [l[1] for l in expanded]
#    response = [l[2] for l in expanded]
#    return antib, patient_info, response


# def subsample_sentence(s, size_sample, patient):
#    x = np.sort(np.random.choice(np.array(range(len(s))), size = size_sample, replace = False))
#    predictors = ['<cls>'] + [patient[0]] +[s[p] for p in x]
#    response = [t for t in s if t not in predictors]
#    patient_info = ['<cls>'] + patient
#    #y = [t for t in s if t not in x]
#    return predictors, patient_info, response


# def expand_isolate(s, sizes_i, patient):
#    if len(sizes_i)<1:
#        r = None
#    else:
#        r = [subsample_sentence(s, j, patient) for j in sizes_i]
#        #r = [l for sub in r for l in sub]
#    return r

# def size_predictors_per_isolate(s, d_mean_values, max_samples_per_isolate, minimum_size_of_predictors):
#    n_max_predictors = s 
#    sizes = np.random.binomial(n_max_predictors, d_mean_values[n_max_predictors] / n_max_predictors, max_samples_per_isolate)
#    sizes = [l  for l in sizes if l >= minimum_size_of_predictors]
#    if len(sizes)==0:
#        #sizes = [n_max_predictors] ### I need to change this to []
#        sizes = []
#    return sizes


class AB_TextDataset(torch.utils.data.Dataset):
    def __init__(self, antibiotics, patient, response, vocab, ab2, max_len_ab, max_len_patient, pos_response, Fake_position):
        self.vocab = vocab
        self.df = pd.DataFrame(list(zip(antibiotics, patient, response)), columns =["antibiotics", "patient", "response"])
        self.ab2 = ab2 # list of antiboitics
        self.pos_response = pos_response
        self.max_len_ab = max_len_ab
        self.max_len_patient = max_len_patient
        self.pos_response_list = list(self.pos_response.keys())
        self.Fake_position = Fake_position
        
    def __getitem__(self, idx):
        antibiotics = self.df.antibiotics.loc[idx]
        patient = self.df.patient.loc[idx]
        response = self.df.response.loc[idx]
        length_antibiotics = len(antibiotics)
        n_y = len(response)
        #
        # positions of the antibiotics used as predictors
        antibiotics_predictors = [l.split("_")[0] for l in antibiotics if l.split("_")[0] in self.ab2]
        #
        position_antibiotics = [a in antibiotics_predictors for a in self.ab2]
        position_antibiotics = torch.tensor(position_antibiotics, dtype=torch.bool)
        x_resp = [1 if a in antibiotics else 0 for a in self.pos_response_list]
        x_resp = torch.tensor(x_resp, dtype=torch.int64)
        #
        p_e = [l.split("_")[0] for l in response if l.split("_")[0] in self.ab2]
        #print(p_e)
        y_pos = [a in p_e for a in self.ab2]
        y_pos = torch.tensor(y_pos, dtype=torch.bool)
        y_resp = [1 if a in response else 0 for a in self.pos_response_list]
        y_resp = torch.tensor(y_resp, dtype=torch.int64)
        #
        antibiotics = antibiotics + ['<pad>']*(self.max_len_ab - length_antibiotics)
        patient = patient + ['<pad>']*(self.max_len_patient - len(patient))
        # antibiotics_pad = [e == '<pad>' for e in antibiotics]
        # patient_pad = [e == '<pad>' for e in patient]
        # ex_pos = torch.arange(self.max_len_ab) 
        # ex_pos[torch.tensor(ex_pad)] = self.Fake_position
        antibiotic_pos = torch.zeros(self.max_len_ab, dtype=torch.int64)
        patient_pos = torch.zeros(self.max_len_patient, dtype=torch.int64)
        antibiotics = torch.tensor(self.vocab[antibiotics], dtype=torch.int64)
        patient = torch.tensor(self.vocab[patient], dtype=torch.int64)
        #
        return antibiotics, patient, antibiotic_pos, patient_pos, x_resp, position_antibiotics, y_resp, y_pos, torch.tensor(length_antibiotics - 1, dtype=torch.int64), torch.tensor(n_y, dtype=torch.int64)
    
    def __len__(self):
        return self.df.shape[0]



class Vocab:
    """Vocabulary for text."""
    def __init__(self, tokens=[], min_freq=0, reserved_tokens=[]):
        """Defined in :numref:`sec_text-sequence`"""
        # Flatten a 2D list if needed
        if tokens and isinstance(tokens[0], list):
            tokens = [token for line in tokens for token in line]
        # Count token frequencies
        counter = collections.Counter(tokens)
        self.token_freqs = sorted(counter.items(), key=lambda x: x[1],
                                  reverse=True)
        # The list of unique tokens
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
    def unk(self):  # Index for the unknown token
        return self.token_to_idx['<unk>']

def sampling_expanding_magnus(dataset, isolates_to_sample_per_batch, n_metadata, minimum_size_of_predictors, d_mean_values, d_max_samples):
    """
    dataset = (cls, 4 metadata from patient, antibiotics)
    isolates_to_sample_per_batch = how many isolates from the dataset to include
    n_metadata = how many variables in the metadata = 5 (cls + 4 variables)
    minimum_size_of_predictors = minimum number of antibiotics to train the model with (do predictions with)
    d_mean_values = dictionary, if an isolate has x tests, expected number of isolates to use as prediction when we expand
    d_max_samples = dictionary, max number of data points per isolate when expanding
    """
    rs = dataset.sample(n=isolates_to_sample_per_batch, weights='weight', random_state = 42)
    lst = rs['sentence'].tolist()
    #idx_samples = np.random.choice(np.array(range(len(dataset))), size = isolates_to_sample_per_batch, replace = False)
    #lst = [dataset[j] for j in list(idx_samples)]
    x, y = expand_list_of_isolates_magnus(lst, n_metadata, d_mean_values, d_max_samples, minimum_size_of_predictors)
    return x, y

def expand_list_of_isolates_magnus(lst, n_metadata, d_mean_values, d_max_samples, minimum_size_of_predictors):            
    number_of_posible_predictors = [len(s)-n_metadata-1 for s in lst]
    #print(number_of_posible_predictors)
    max_number_preditors = [d_max_samples[j] for j in number_of_posible_predictors]        
    #print(max_number_preditors)
    Sizes_i = [size_predictors_per_isolate_magnus(lst[j], n_metadata, d_mean_values, max_number_preditors[j], minimum_size_of_predictors) for j in range(len(lst))]
    expanded = [expand_isolate_magnus(lst[j], Sizes_i[j], n_metadata) for j in range(len(lst))]
    expanded = [e for e in expanded if e is not None]
    expanded = [l for sub in expanded for l in sub]
    x = [l[0] for l in expanded]
    y = [l[1] for l in expanded]
    return x, y


def size_predictors_per_isolate_magnus(s, n_metadata, d_mean_values, max_samples_per_isolate, minimum_size_of_predictors):
    n_ab = len(s) - n_metadata
    n_max_predictors = n_ab - 1 
    p = d_mean_values[n_max_predictors] / n_max_predictors
    sizes = np.random.binomial(n_max_predictors, d_mean_values[n_max_predictors] / n_max_predictors, max_samples_per_isolate)
    sizes = [5  for l in sizes if l >= minimum_size_of_predictors] # fixed number of predictors
    return sizes


def expand_isolate_magnus(s, sizes_i, n_metadata):
    if len(sizes_i)<1:
        r = None
    else:
        r = [subsample_sentence_magnus(s, j, n_metadata) for j in sizes_i]
        #r = [l for sub in r for l in sub]
    return r


def subsample_sentence_magnus(s, size_sample, n_metadata):
    x = np.sort(np.random.choice(np.array(range(len(s)-n_metadata)), size = size_sample, replace = False) + n_metadata)
    x = [7, 8, 9, 11, 12] # fixed positions
    x = s[0:n_metadata] + [s[p] for p in x]
    y = [t for t in s if t not in x]
    return x, y

