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

from torch_app.architecture import Encoder, AbPred
from torch_app.DIRE import pkl_save, pkl_load, Vocab
from torch_app.DIRE import _read_file_ab, sampling_expanding, AB_TextDataset


def create_bert_model(config, PAD_id, CLS_id, MASK_id, Fake_position=None, rank=0):    
    if rank==0:
        print("Creating bert model")
    hidden_size = config["model"]["hidden_size"]
    max_position_embeddings = config["model"]["max_position_embeddings"]
    layer_norm_eps = config["model"]["layer_norm_eps"]
    dropout_rate_attention = config["model"]["dropout_rate_attention"]
    dropout_rate_PWFF = config["model"]["dropout_rate_PWFF"]
    num_attention_heads = config["model"]["num_attention_heads"]
    self_attention_internal_dimension = config["model"]["self_attention_internal_dimension"]
    FFN_internal_dimension = config["model"]["FFN_internal_dimension"]
    encoder_stack_depth = config["model"]["encoder_stack_depth"]
    vocab_size = config["vocab"]["vocab_size"]            
    bert_model =  Encoder(encoder_stack_depth, hidden_size, num_attention_heads, max_position_embeddings, self_attention_internal_dimension, 
        FFN_internal_dimension, layer_norm_eps, dropout_rate_attention, dropout_rate_PWFF, vocab_size, PAD_id, Fake_position)        
    bert_params = sum(p.numel() for p in bert_model.parameters() if p.requires_grad)
    if rank==0:
        print("Bert parameters: {}".format(bert_params))    
    return bert_model

######################################################################
######################################################################
def create_classification_model(config, rank=0):
    if rank==0:
        print("Creating clasification model")
    hidden_size = config["model"]["hidden_size"]
    vocab_size = config["vocab"]["vocab_size"]
    #aminoacids = config["vocab"]["aminoacids"]
    layer_norm_eps = config["model"]["layer_norm_eps"]
    number_ab = config["model"]["number_ab"]
    number_out = config["model"]["number_out"]
    classification_model = AbPred(hidden_size, number_ab, number_out, layer_norm_eps)
    classificaiton_params = sum(p.numel() for p in classification_model.parameters() if p.requires_grad)
    if rank==0:
        print("Classification parameters: {}".format(classificaiton_params))
    return classification_model

######################################################################
######################################################################
def load_previous_model(model_folder, bert_model, classification_model, rank=0, load_previous=True):
    if rank==0:
        print("Loading previous model (?)")
    state = None
    if not load_previous and rank==0:
        print("No")
        return state
    else:
        if rank==0:
            print("Yes")
        if os.path.exists(os.path.join(model_folder, "pretrain_bert.torch")):
            if rank==0:
                print("loading previous model")
            bert_model.load_state_dict(torch.load(os.path.join(model_folder, "pretrain_bert.torch")))
            classification_model.load_state_dict(torch.load(os.path.join(model_folder, "pretrain_classification.torch")))
        if os.path.exists(os.path.join(model_folder, "pretrain_state.pkl")):
            if rank==0:
                print("loading previous state")
            state = pkl_load(os.path.join(model_folder, "pretrain_state.pkl"))
        else:
            if rank==0:
                print("No previous state, loading pre_train")
            if os.path.exists(os.path.join(model_folder, "pretrain_bert.torch")):
                if rank==0:
                    print("Loading pre_train")
                bert_model.load_state_dict(torch.load(os.path.join(model_folder, "pretrain_bert.torch")))
            else:
                if rank==0:
                    print("No previous pre-train")
        return state


######################################################################
######################################################################
def load_previous_state(df, df_test, state, model_folder, num_epochs, config, rank=0):    
    if rank==0:
        print("Loading previous state (?)")
    num_batches_train = int(config["train"]["observations_per_epoch"]  / config["train"]["batch_size"])
    num_batches_test = int(config["test"]["observations_per_epoch"]  / config["test"]["batch_size"])
    
    start_epoch = 0
    avg_batch_time = 0
    total_time = 0
    processed_batches = 0
    min_valid_loss = 1000
    
    train_loss = torch.zeros([num_epochs, 1])
    train_acc = torch.zeros([num_epochs, 1])
    val_loss = torch.zeros([num_epochs, 1])
    val_acc = torch.zeros([num_epochs, 1])    
    
    if state is not None and rank==0:
        print("loading previous state Yes:")
        print(state["epoch"]+1)
        print()
        start_epoch = state["epoch"] + 1
        avg_batch_time = state["avg_batch_time"]
        total_time = state["total_eclisped_time"]
        processed_batches = state["processed_batches"]
        min_valid_loss = state["min_valid_loss"]
        
        train_loss_l  = torch.load(os.path.join(model_folder, "pretrain_train_loss.pt"))
        train_acc_l = torch.load(os.path.join(model_folder, "pretrain_train_acc.pt"))
        val_loss_l = torch.load(os.path.join(model_folder, "pretrain_val_loss.pt"))
        val_acc_l  = torch.load(os.path.join(model_folder, "pretrain_val_acc.pt"))
        
        train_loss[:train_loss_l.shape[0], :train_loss_l.shape[1]] = train_loss_l
        train_acc[:train_acc_l.shape[0], :train_acc_l.shape[1]] = train_acc_l
        val_loss[:val_loss_l.shape[0], :val_loss_l.shape[1]] = val_loss_l
        val_acc[:val_loss_l.shape[0], :val_loss_l.shape[1]] = val_acc_l
    else:
        if rank==0:
            print("loading previous state No")
            print()
    return train_loss, train_acc, val_loss, val_acc, start_epoch, avg_batch_time, total_time, processed_batches, min_valid_loss


######################################################################
######################################################################
def start_optimizer_loss(config, bert_model, classification_model, device, rank=0):
    if rank==0:
        print("Start optmizer and loss")
    batch_size = config["train"]["batch_size"]    
    learning_rate = config["train"]["learning_rate"]
    learning_rate_warm_up = config["train"]["learning_rate_warm_up"]
    pos_response = config["model"]["pos_response"]
    ab2 = list(pos_response.keys())
    ab2 = [l.split("_")[0] for l in ab2]
    ab2_weights = config["train"]["ab2_weights"]
    types_ab2 = config["vocab"]["types_ab2"]
    #ab2 = [l.split("_")[0] for l in types_ab2]
    print([ab2_weights[ab2[j]] for j  in range(len(ab2_weights))])
    criterion = [nn.CrossEntropyLoss(weight=torch.tensor(ab2_weights[ab2[j]], requires_grad=False).to(device), reduction='mean') for j in range(len(ab2_weights))]
    #criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(lr = learning_rate, params=list(bert_model.parameters()) + list(classification_model.parameters()))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda i: min(i / (learning_rate_warm_up / batch_size), 1.0))
    return  criterion, optimizer, scheduler  



######################################################################
######################################################################

def forward_train(df_train, config, bert_model, classification_model, optimizer, scheduler, criterion, epoch_ind, total_time, 
    processed_batches, device, num_epochs, rank, MASK_id, PAD_id, CLS_id, UNK_id, vocab_size, vocab, Fake_position=None):
    if rank==0:
        print("forward train")    
    max_position_embeddings = config["model"]["max_position_embeddings"]
    observations_per_epoch = config["train"]["observations_per_epoch"]
    isolates_to_sample_per_batch = config["train"]["isolates_to_sample_per_batch"]
    n_metadata = config["train"]["n_metadata"] # (so cls does count)
    minimum_size_of_predictors = config["train"]["minimum_size_of_predictors"]
    d_mean_values = config["train"]["d_mean_values"]
    d_max_samples = config["train"]["d_max_samples"]
    encode_response = config["model"]["encode_response"]
    types_ab2 = config["vocab"]["types_ab2"]
    #
    pos_response = config["model"]["pos_response"]
    ab2 = list(pos_response.keys())    
    ab2 = [l.split("_")[0] for l in ab2]
    batch_size = config["train"]["batch_size"]
    num_batches = int(config["train"]["observations_per_epoch"]  / config["train"]["batch_size"])
    #
    print(len(df_train)) 
    x, y = sampling_expanding(df_train, len(df_train), n_metadata, minimum_size_of_predictors, d_mean_values, d_max_samples)
    print(len(x))
    print(observations_per_epoch)
    idx_ = np.random.choice(np.array(range(len(x))), size = min(len(x), observations_per_epoch), replace = False)
    x = [x[j] for j in list(idx_)]
    y = [y[j] for j in list(idx_)]
    train_dataset = AB_TextDataset(x, y, vocab, ab2, max_position_embeddings, pos_response, Fake_position)        
    train_iter = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, shuffle=True, num_workers = 0)
    #
    #
    perform  = torch.zeros(2, len(train_iter))-1.0
    bert_model.train()
    classification_model.train()
    batches_count = 0
    num_steps_reached = False
    #
    #
    for ex, ex_pos, x_resp, x_pos, y_resp, y_pos, l, n_y in train_iter:
        ti = time.time()
        ex = ex.to(device)        
        ex_pos = ex_pos.to(device)
        x_resp = x_resp.to(device)
        x_pos = x_pos.to(device)
        y_resp = y_resp.to(device)
        y_pos = y_pos.to(device)
        targets = y_resp[y_pos].to(device)
        valid_lens = None
        optimizer.zero_grad()
        bert_pred = bert_model(ex, ex_pos, valid_lens)
        #
        classification_pred = classification_model(bert_pred[:,0:1,:], y_pos)
        start_loss = 0
        ids = torch.nonzero(y_pos)[:,1]
        for r in range(len(criterion)):            
            if sum(ids==r)>0:
                if start_loss == 0:
                    loss = criterion[r](classification_pred[ids==r], targets[ids==r])
                    start_loss = 1
                else:
                    loss = loss + criterion[r](classification_pred[ids==r], targets[ids==r])
        #
        perform[0,batches_count] = loss.item()
        acc = (torch.argmax(classification_pred, dim=1) == targets).sum().item()/len(targets)
        perform[1,batches_count] = acc
        loss.backward()
        optimizer.step()
        scheduler.step()
        batch_time = time.time() - ti
        total_time += batch_time
        avg_batch_time = total_time/(processed_batches+1)            
        #
        if rank == 0 and batches_count%1==0:
            print("(epoch, batch): ({}/{}, {}/{}), time ecplipsed: {}, estimated time left: {}, loss: {:.3f}, acc: {:.3f}        ".format(
                epoch_ind+1, num_epochs, batches_count+1, num_batches, 
                str(datetime.timedelta(seconds=int(total_time))), 
                str(datetime.timedelta(seconds=int(avg_batch_time * (num_epochs * num_batches) - total_time))), 
                perform[0,batches_count], perform[1,batches_count]), end="\r", flush=True)
        processed_batches += 1
        batches_count += 1
        if batches_count == num_batches:
            num_steps_reached = True
            break        
    if rank == 0:
        print()
    del ex, ex_pos, x_resp, x_pos, y_resp, y_pos, targets, bert_pred, classification_pred, x, y, train_dataset, train_iter    
    torch.cuda.empty_cache()
    return perform, total_time, processed_batches

######################################################################
######################################################################
def forward_test(df_test, config, bert_model, classification_model, criterion, epoch_ind, val_loss, val_acc, device, rank, 
    MASK_id, PAD_id, CLS_id, UNK_id, vocab_size, vocab, Fake_position=None, num_epochs=None):
    if rank==0:
        print("forward test")    
    #
    max_position_embeddings = config["model"]["max_position_embeddings"]
    observations_per_epoch = config["test"]["observations_per_epoch"]
    isolates_to_sample_per_batch = config["test"]["isolates_to_sample_per_batch"]
    n_metadata = config["train"]["n_metadata"] # (so cls does count)
    minimum_size_of_predictors = config["train"]["minimum_size_of_predictors"]
    d_mean_values = config["train"]["d_mean_values"]
    d_max_samples = config["train"]["d_max_samples"]
    encode_response = config["model"]["encode_response"]
    types_ab2 = config["vocab"]["types_ab2"]
    #
    pos_response = config["model"]["pos_response"]
    ab2 = list(pos_response.keys())    
    ab2 = [l.split("_")[0] for l in ab2]
    batch_size = config["test"]["batch_size"]
    num_batches = int(config["test"]["observations_per_epoch"]  / config["test"]["batch_size"])
    #
    print(len(df_test)) 
    x, y = sampling_expanding(df_test, len(df_test), n_metadata, minimum_size_of_predictors, d_mean_values, d_max_samples)
    print(len(x))
    print(observations_per_epoch)
    idx_ = np.random.choice(np.array(range(len(x))), size = min(len(x), observations_per_epoch), replace = False)
    x = [x[j] for j in list(idx_)]
    y = [y[j] for j in list(idx_)]
    train_dataset = AB_TextDataset(x, y, vocab, ab2, max_position_embeddings, pos_response, Fake_position)        
    train_iter = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, shuffle=True, num_workers = 0)
    #
    #
    perform  = torch.zeros(2, len(train_iter))-1.0
    bert_model.eval()
    classification_model.eval()
    batches_count = 0
    num_steps_reached = False
    print()
    #
    for ex, ex_pos, x_resp, x_pos, y_resp, y_pos, l, n_y in train_iter:
        ex = ex.to(device)        
        ex_pos = ex_pos.to(device)
        x_resp = x_resp.to(device)
        x_pos = x_pos.to(device)
        y_resp = y_resp.to(device)
        y_pos = y_pos.to(device)
        targets = y_resp[y_pos].to(device)
        valid_lens = None
        bert_pred = bert_model(ex, ex_pos, valid_lens)
        #
        #
        classification_pred = classification_model(bert_pred[:,0:1,:], y_pos)
        start_loss = 0
        ids = torch.nonzero(y_pos)[:,1]
        for r in range(len(criterion)):            
            if sum(ids==r)>0:
                if start_loss == 0:
                    loss = criterion[r](classification_pred[ids==r], targets[ids==r])
                    start_loss = 1
                else:
                    loss = loss + criterion[r](classification_pred[ids==r], targets[ids==r])
        #
        perform[0,batches_count] = loss.item()
        acc = (torch.argmax(classification_pred, dim=1) == targets).sum().item()/len(targets)
        perform[1,batches_count] = acc
        if rank == 0 and batches_count%1==0:
            print("(epoch, batch): ({}/{}, {}/{}), loss: {:.3f}, acc: {:.3f}        ".format(
                epoch_ind+1, num_epochs, batches_count+1, num_batches, 
                perform[0,batches_count], perform[1,batches_count]), end="\r", flush=True)

        batches_count += 1
        if batches_count == num_batches:
            num_steps_reached = True
            break                            
    del ex, ex_pos, x_resp, x_pos, y_resp, y_pos, targets, bert_pred, classification_pred, x, y, train_dataset, train_iter 
    torch.cuda.empty_cache()       
    return perform

######################################################################
######################################################################

def save_model(path, model):
    if isinstance(model,torch.nn.DataParallel):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    torch.save(state_dict, path)


######################################################################
######################################################################



def main_process(model_folder, config, is_dist=False):
    print("Start train main process")      
    vocab = torch.load(config["vocab"]["vocab"])
    vocab_size = len(vocab.idx_to_token)
    config["vocab"]["vocab_size"] = vocab_size    
    PAD_id = vocab.token_to_idx['<pad>']
    CLS_id = vocab.token_to_idx['<cls>']
    MASK_id = vocab.token_to_idx['<mask>']
    SEP_id = vocab.token_to_idx['<sep>']
    UNK_id = vocab.token_to_idx['<unk>']
    Fake_position = config["vocab"]["pad_position"]
    types_ab = config["vocab"]["types_ab"]
    types_ab2 = config["vocab"]["types_ab2"]
    encode_response = config["model"]["encode_response"]
    pos_response = config["model"]["pos_response"]
    ab2 = list(pos_response.keys())
    ab2 = [l.split("_")[0] for l in ab2]
    df_train = _read_file_ab(config["data"]["train_ds"])
    df_test = _read_file_ab(config["data"]["test_ds"])
    isolates_to_sample_per_batch = config["test"]["isolates_to_sample_per_batch"]
    n_metadata = config["train"]["n_metadata"] # (so cls does count)
    minimum_size_of_predictors = config["train"]["minimum_size_of_predictors"]
    d_mean_values = config["train"]["d_mean_values"]
    d_max_samples = config["train"]["d_max_samples"]
    batch_size = config["train"]["batch_size"]
    ab2_weights = config["train"]["ab2_weights"]
    #
    print("no parallelization")
    rank = 0
    bert_model =  create_bert_model(config, PAD_id, CLS_id, MASK_id, Fake_position, rank)
    classification_model = create_classification_model(config, rank)
    state = load_previous_model(model_folder, bert_model, classification_model, rank, load_previous=True)
    device_id = 0
    bert_model.to(device_id)
    classification_model.to(device_id)
    device=0
    #
    criterion, optimizer, scheduler = start_optimizer_loss(config, bert_model, classification_model, device, rank)
    
    if rank == 0:
        print("data train")
        print(len(df_train))

    if rank == 0:
        print("data test")
        print(len(df_test))
    
    num_epochs = config["train"]["num_epochs"]
    #
    #
    # 
    if rank==0:
        print("state None?")
        print(state is None)
    
    train_loss, train_acc, val_loss, val_acc, start_epoch, avg_batch_time, total_time, processed_batches, min_valid_loss = load_previous_state(df_train, df_test, state, model_folder, num_epochs, config, rank)
    
    
    
    for epoch_ind in range(start_epoch, num_epochs):
        perform_train, total_time, processed_batches = forward_train(df_train, config, bert_model, classification_model, optimizer, scheduler, criterion, epoch_ind, total_time, 
            processed_batches, device, num_epochs, rank, MASK_id, PAD_id, CLS_id, UNK_id, vocab_size, vocab, Fake_position)
        perform_test = forward_test(df_test, config, bert_model, classification_model, criterion, epoch_ind, val_loss, val_acc, device, rank, 
            MASK_id, PAD_id, CLS_id, UNK_id, vocab_size, vocab, Fake_position, num_epochs)
        avg_loss = torch.mean(perform_test[0,:])
        if rank == 0:
            print("test done")
            train_loss[epoch_ind,:] = torch.mean(perform_train[0,:])
            train_acc[epoch_ind,:] = torch.mean(perform_train[1,:])
            val_loss[epoch_ind,:] = torch.mean(perform_test[0,:])
            val_acc[epoch_ind,:] = torch.mean(perform_test[1,:])
            #
            print("epoch {} complete, avg validation loss: {:.4f}, avg validation acc: {:.4f}, avg train loss: {:.4f}, avg train acc: {:.4f}                   ".format(
                epoch_ind, torch.mean(perform_test[0,:]), torch.mean(perform_test[1,:]), 
                torch.mean(perform_train[0,:]),  torch.mean(perform_train[1,:])))
            if min_valid_loss > avg_loss:
                print('validation loss decreased({:.4f}--->{:.4f}) \t Saving The Model'.format(min_valid_loss, avg_loss))
                print('Saving the updated models')
                min_valid_loss = avg_loss
                # 
                isExist = os.path.exists(model_folder)
                if not isExist:
                    os.makedirs(model_folder)
                save_model(os.path.join(model_folder, "pretrain_bert.torch"), bert_model)
                save_model(os.path.join(model_folder, "pretrain_classification.torch"), classification_model)
            #
            #   
            print("Saving the state")
            isExist = os.path.exists(model_folder)
            if not isExist:
                os.makedirs(model_folder)        
            state = {"epoch": epoch_ind, "avg_batch_time": avg_batch_time, "total_eclisped_time": total_time,
                "processed_batches": processed_batches, "min_valid_loss": min_valid_loss, "total_eclisped_time": total_time}        
            pkl_save(os.path.join(model_folder, "pretrain_state.pkl"), state)
            torch.save(train_loss, os.path.join(model_folder, "pretrain_train_loss.pt"))
            torch.save(train_acc, os.path.join(model_folder, "pretrain_train_acc.pt"))
            torch.save(val_loss, os.path.join(model_folder, "pretrain_val_loss.pt"))
            torch.save(val_acc, os.path.join(model_folder, "pretrain_val_acc.pt"))

if __name__ == "__main__":
    model_folder = sys.argv[1]
    conf_file = sys.argv[2]
    with open(os.path.join(model_folder, conf_file), "r") as file:
        config = yaml.safe_load(file)

    is_dist = config["data"]["is_distributed"]
    if is_dist=="False":
        is_dist = False
    else:
        is_dist = True 

    main_process(model_folder, config, is_dist)





