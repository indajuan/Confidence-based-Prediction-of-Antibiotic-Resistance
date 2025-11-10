import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

import numpy as np
import yaml
import os
import time
import datetime
import sys
import pandas as pd
import random

from torch_app.architecture import Encoder, AbPred, AbPred_both
from torch_app.DIRE import pkl_load, Vocab
from torch_app.DIRE import AB_TextDataset

def read_data(input_file, metadata, antibiotics):
    rows = []  # to hold the new rows
    with open(input_file, 'r') as file:
        for i, line in enumerate(file, start=1):  # unique id
            parts = line.strip().split(",", 3)  # split into 4 parts max
            if len(parts) != 4:
                raise ValueError(f"Line {i} does not have 3 commas: '{line.strip()}'")
            
            col1, col2, col3, col4 = [p.strip() for p in parts]
            # Split into words
            pathogen = col1.split()
            patient = col2.split()
            predictors = col3.split()
            responses = col4.split()

            if len(pathogen) != 1:
                raise ValueError(f"Line {i}: pathogen column must have exactly 1 word: '{col1}'")
            
            if len(patient) > 4:
                raise ValueError(f"Line {i}: patient data column must have ≤ 4 words: '{col2}'")
            
            if not (4 <= len(predictors) <= ( antibiotics - 1) ):
                raise ValueError(
                    f"Line {i}: antibiotic predictors column must have between 4 and max {antibiotics -1} words: '{col3}'"
                )
            
            if len(responses) < 1:
                raise ValueError(f"Line {i}: responses column must have at least 1 word: '{col4}'")
            
            for response_word in responses:
                rows.append({
                    "id": i, 
                    "pathogen": pathogen,
                    "patient": ['<cls>'] + pathogen + patient + (metadata - len(pathogen) - len(patient))*['<pad>'],
                    "predictors": ['<cls>'] + pathogen + predictors + (antibiotics - len(predictors) - len(pathogen) )*['<pad>'],
                    "responses": [response_word]
                })
    df = pd.DataFrame(rows)
    return df

def str_to_bool(s):
    if isinstance(s, bool):
        return s
    s = s.lower()
    if s in ['true', '1', 'yes', 'y']:
        return True
    elif s in ['false', '0', 'no', 'n']:
        return False
    raise ValueError(f"Cannot interpret '{s}' as boolean.")

#def create_bert_model(config, PAD_id, Fake_position=None, rank=0):    
def create_transformer_model(config, PAD_id, Fake_position=None, rank=0, data = "antibiotics"):    
    if rank==0:
        print("Creating tranformer model")
    minimum_size_of_predictors = config["train"]["minimum_size_of_predictors"]
    n_antibiotics = config["model"]["pos_response"]  
    n_antibiotics = len(list(n_antibiotics.keys()))  
    n_metadata = config["train"]["n_metadata"]
    #
    if data == "antibiotics":
        max_position_embeddings = n_antibiotics - 1 + 2 # -1 at least predict one, +1 for cls, +1 for species
    else:
        if data == "patient":
            max_position_embeddings = n_metadata + 1 # 1 for cls
        else:
            raise ValueError(
                """The data type for the tranformer is not correct""")
    #
    hidden_size = config["model"]["hidden_size"]
    layer_norm_eps = config["model"]["layer_norm_eps"]
    dropout_rate_attention = config["model"]["dropout_rate_attention"]
    dropout_rate_PWFF = config["model"]["dropout_rate_PWFF"]
    num_attention_heads = config["model"]["num_attention_heads"]
    self_attention_internal_dimension = config["model"]["self_attention_internal_dimension"]
    FFN_internal_dimension = config["model"]["FFN_internal_dimension"]
    encoder_stack_depth = config["model"]["encoder_stack_depth"]
    vocab_size = config["vocab"]["vocab_size"]
    #            
    transformer_model =  Encoder(encoder_stack_depth, hidden_size, num_attention_heads, max_position_embeddings, self_attention_internal_dimension, 
        FFN_internal_dimension, layer_norm_eps, dropout_rate_attention, dropout_rate_PWFF, vocab_size, PAD_id, Fake_position)        
    transformer_params = sum(p.numel() for p in transformer_model.parameters() if p.requires_grad)
    if rank==0:
        print(f"Transformer parameters {data}: {transformer_params}")
    return transformer_model

######################################################################
######################################################################
def create_classification_model(config, rank=0, models = "antibiotics"):
    if rank==0:
        print("Creating clasification model")
    hidden_size = config["model"]["hidden_size"]
    layer_norm_eps = config["model"]["layer_norm_eps"]
    number_ab = config["model"]["number_ab"]
    number_out = config["model"]["number_out"]
    if models == "antibiotics":
        classification_model = AbPred(hidden_size, number_ab, number_out, layer_norm_eps)
    else:
        if models == "both":
            classification_model = AbPred_both(hidden_size, number_ab, number_out, layer_norm_eps)
        else:
            raise ValueError(
                """The data type for the tranformer is not correct""")
    #
    classificaiton_params = sum(p.numel() for p in classification_model.parameters() if p.requires_grad)
    if rank==0:
        print("Classification parameters: {}".format(classificaiton_params))
    return classification_model

######################################################################
######################################################################

def load_previous_model(model_folder, model_antibiotics, model_patient_info, classification_model, rank=0, load_previous=True, data = "antibiotics"):
    if rank==0:
        print("Loading previous model (?)")
    state = None
    if not load_previous and rank==0:
        print("No")
        return state
    else:
        if rank==0:
            print("Yes")
        if os.path.exists(os.path.join(model_folder, "pretrain_antibiotics.torch")):
            if rank==0:
                print("loading previous model")
            model_antibiotics.load_state_dict(torch.load(os.path.join(model_folder, "pretrain_antibiotics.torch")))
            classification_model.load_state_dict(torch.load(os.path.join(model_folder, "pretrain_classification.torch")))
            if data == "both":
                model_patient_info.load_state_dict(torch.load(os.path.join(model_folder, "pretrain_patient.torch")))
        if os.path.exists(os.path.join(model_folder, "pretrain_state.pkl")):
            if rank==0:
                print("loading previous state")
            state = pkl_load(os.path.join(model_folder, "pretrain_state.pkl"))
        else:
            if rank==0:
                print("No previous state, starting a new model")
        return state


######################################################################
######################################################################
def list_to_token(lst, vocab):
    return [vocab.idx_to_token[l] for l in lst]

def filter_x(lst):
    x = [x for x in lst if x != '<pad>' ]
    x = [x for x in x if x != '<mask>']
    x = [x for x in x if x != '<cls>' ]
    return(x)

######################################################################
######################################################################


def forward_val(df_test, config, model_antibiotics, model_patient_info, classification_model, device, rank, 
    vocab, Fake_position=None, data = "antibiotics"):
    if rank==0:
        print("predicting")    
    #
    pos_response = config["model"]["pos_response"]
    ab2 = list(pos_response.keys())    
    ab2 = [l.split("_")[0] for l in ab2]
    batch_size = config["train"]["batch_size"] 
    n_antibiotics = len(list(pos_response.keys()))  
    n_metadata = config["train"]["n_metadata"]
    size_antibiotic_tranformer = n_antibiotics -1 + 2
    size_patient_transformer = n_metadata + 1 
    #
    num_batches = np.ceil(df_test.shape[0]/512)
    length_data = df_test.shape[0]
    print("number of predictions to do:")     
    print(length_data)

    antib = df_test["predictors"].tolist()
    patient_info = df_test["patient"].tolist()
    response = df_test["responses"].tolist()
    test_dataset = AB_TextDataset(antib, patient_info, response, vocab, ab2, size_antibiotic_tranformer, size_patient_transformer, pos_response, Fake_position)        
    #
    #
    test_iter = torch.utils.data.DataLoader(test_dataset, batch_size = batch_size, shuffle=False, num_workers = 0)
    #
    #
    model_antibiotics.eval()
    classification_model.eval()
    if data == "both":
        model_patient_info.eval()
    batches_count = 0
    X = []
    P = []
    Y = []
    Y_resp = []
    Y_pred = []
    Y_score = []
    print()
    #
    for antibiotics, patient, antibiotic_pos, patient_pos, x_resp, position_antibiotics_autopred, y_resp, y_pos, length_ab, n_y in test_iter:
        ti = time.time()
        x = torch.repeat_interleave(antibiotics, torch.sum(y_pos,1), dim=0).tolist()
        x = [list_to_token(lst, vocab) for lst in x]
        x = [filter_x(l) for l in x]
        p = torch.repeat_interleave(patient, torch.sum(y_pos,1), dim=0).tolist()
        p = [list_to_token(lst, vocab) for lst in p]
        p = [filter_x(l) for l in p]
        y = torch.nonzero(y_pos)[:,1].tolist()
        y = [ab2[l] for l in y]
        y_r = y_resp[y_pos].tolist()
        antibiotics = antibiotics.to(device)
        antibiotic_pos = antibiotic_pos.to(device)
        x_resp = x_resp.to(device)
        position_antibiotics_autopred = position_antibiotics_autopred.to(device)
        y_resp = y_resp.to(device)
        y_pos = y_pos.to(device)
        targets = y_resp[y_pos].to(device)
        valid_lens = None
        antibiotic_pred = model_antibiotics(antibiotics, antibiotic_pos, valid_lens)
        #
        if data == "both":
            patient = patient.to(device)
            patient_pos = patient_pos.to(device)
            patient_pred = model_patient_info(patient, patient_pos, valid_lens)
            classification_pred = classification_model(antibiotic_pred[:,0:1,:], patient_pred[:,0:1,:], y_pos)
        else:
            classification_pred = classification_model(antibiotic_pred[:,0:1,:], y_pos)
        #
        y_hat = torch.argmax(classification_pred, dim=1)
        y_hat = y_hat.tolist()
        X = X + x
        P = P + p
        Y = Y + y
        Y_resp = Y_resp + y_r
        Y_pred = Y_pred + y_hat
        Y_score = Y_score + classification_pred.tolist() 
        if rank == 0 and batches_count%1==0:
            print("(epoch, batch): ({}/{}, {}/{}) ".format(
                1, 1, batches_count+1, num_batches), end="\r", flush=True)
        #
        batches_count += 1
        if batches_count == num_batches:
            break                
    torch.cuda.empty_cache()
    df = pd.DataFrame(list(zip(X, P, Y, Y_resp, Y_pred, Y_score)), columns = ['X', 'P' ,'Y', 'Y_resp', 'Y_pred', 'Y_score']) 
    df["id"] = df_test["id"].tolist()
    df["Antibiotic_predictors"] = df["X"].apply(lambda l: " ".join(l))
    df["Patient_data"] = df["P"].apply(lambda l: " ".join(l[1:]))  # skip first element
    # Round Y_score to 4 decimals
    df["Output_neural_networks"] = df["Y_score"].apply(lambda l: [round(v, 4) for v in l])
    # Optional: drop intermediate list columns
    df = df.drop(columns=["X", "P", "Y_score"])
    df["Y_pred"] = df["Y_pred"].map({1: "R", 0: "S"})
    df["Y_resp"] = df["Y_resp"].map({1: "R", 0: "S"})
    df = df.rename(columns={"Y_pred": "AST_prediction", "Y_resp": "AST_true", "Y": "Antibiotic"})
    cols = ["id"] + [c for c in df.columns if c != "id"]
    df = df[cols]
    return df

######################################################################
######################################################################


def main_process(model_folder, config, data_model, input_file, output_file):
    print("Start process")      
    vocab = torch.load(config["vocab"]["vocab"])
    vocab_size = len(vocab.idx_to_token)
    config["vocab"]["vocab_size"] = vocab_size    
    PAD_id = vocab.token_to_idx['<pad>']
    Fake_position = config["vocab"]["pad_position"] # pad position
    pos_response = config["model"]["pos_response"]  # 1 for resistant
    ab2 = list(pos_response.keys())                 # the list of possible antibiotics
    ab2 = [l.split("_")[0] for l in ab2]            # the list of possible antibiotics
    n_metadata = config["train"]["n_metadata"] # (so cls does count)
    n_antibiotics = config["model"]["pos_response"]  
    n_antibiotics = len(list(n_antibiotics.keys()))  
    sentence = read_data(input_file, n_metadata, n_antibiotics)
    #
    rank = 0
    data_models = data_model
    model_antibiotics =  create_transformer_model(config, PAD_id, Fake_position, rank, data = "antibiotics")
    if data_models == "both":
        model_patient_info =  create_transformer_model(config, PAD_id, Fake_position, rank, data = "patient")
        classification_model = create_classification_model(config, rank, "both")
    else:
        classification_model = create_classification_model(config, rank, "antibiotics")
        model_patient_info = None
    #
    state = load_previous_model(model_folder, model_antibiotics, model_patient_info, classification_model, rank, load_previous=True, data = data_models)
    device_id = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_antibiotics.to(device_id)
    if data_models == "both":
        model_patient_info.to(device_id)
    classification_model.to(device_id)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    df_test_f = forward_val(sentence, config, model_antibiotics, model_patient_info, classification_model, device, rank, 
    vocab, Fake_position, data_models)
    df_test_f.to_csv(output_file, index=False)

if __name__ == "__main__":
    model_folder = sys.argv[1] # model folder's name
    conf_file = sys.argv[2] # config. yaml
    data_model = sys.argv[3] # both or antibiotics
    input_file = sys.argv[4] # both or antibiotics
    output_file = sys.argv[5] # output file
    with open(os.path.join(model_folder, conf_file), "r") as file:
        config = yaml.safe_load(file)

    main_process(model_folder, config, data_model, input_file, output_file)

