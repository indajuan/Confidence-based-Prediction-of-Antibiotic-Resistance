import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


############################################################################################################
############################################################################################################
############################################################################################################

def masked_softmax(X, valid_lens=None):
    if valid_lens is None:
        return F.softmax(X, dim=-1)
    else:
        batch_pad = torch.where(torch.sum(valid_lens==0,1)>0)[0]
        position_pad = torch.sum(valid_lens==0,1)[torch.sum(valid_lens==0,1)>0]
        batch_pad = torch.repeat_interleave(batch_pad, position_pad, dim=0)
        position_pad = torch.where(valid_lens==0, torch.arange(valid_lens.shape[1]), -1).reshape(-1)[torch.where(valid_lens==0, torch.arange(valid_lens.shape[1]), -1).reshape(-1)>=0]
        batch_pad, position_pad
        X[batch_pad,:,position_pad,:] = 1e-8
        return F.softmax(X, dim=-1)


############################################################################################################
############################################################################################################
############################################################################################################

class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, max_position_embeddings, self_attention_internal_dimension, dropout_rate_attention, layer_norm_eps) -> None:
        super().__init__()

        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                """The hidden size {} (embedding dimension) must be divisible 
                by the number of attention heads {}""".format(hidden_size, num_attention_heads))

        self.num_heads = num_attention_heads
        self.seq_len = max_position_embeddings
        self.d_model = hidden_size
        self.d_k = self_attention_internal_dimension 
        self.d_v = self.d_k 

        self.Wq_all = nn.Linear(self.d_model, self.num_heads * self.d_k, bias=False)
        self.Wk_all = nn.Linear(self.d_model, self.num_heads * self.d_k, bias=False)
        self.Wv_all = nn.Linear(self.d_model, self.num_heads * self.d_v, bias=False)
        self.Wo = nn.Linear(self.num_heads * self.d_v, self.d_model, bias=False)
        
        self.dropoutQK = nn.Dropout(dropout_rate_attention)
        self.dropoutO = nn.Dropout(dropout_rate_attention)
        

        self.layer_norm = nn.LayerNorm(self.d_model, eps = layer_norm_eps)
        
  
    def forward(self, x, valid_lens=None):
        residual = x
        batch_size = x.size(0)
        querys = self.Wq_all(x).view(batch_size, self.seq_len, self.num_heads, self.d_k)
        keys   = self.Wk_all(x).view(batch_size, self.seq_len, self.num_heads, self.d_k)
        values = self.Wv_all(x).view(batch_size, self.seq_len, self.num_heads, self.d_v)
        querys, keys, values = querys.transpose(1,2), keys.transpose(1,2), values.transpose(1,2)
        self_attention = torch.matmul(querys / self.d_k ** 0.5, keys.transpose(2, 3)) 
        self_attention = self.dropoutQK(masked_softmax(self_attention, valid_lens=None))        
        output = torch.matmul(self_attention, values)
        output = output.transpose(1, 2).contiguous().view(batch_size, self.seq_len, -1)
        output = self.Wo(output)
        output = self.layer_norm(self.dropoutO(output) + residual)
        return output

############################################################################################################
############################################################################################################
############################################################################################################

class PositionWiseFeedForward(nn.Module):
    def __init__(self, hidden_size, FFN_internal_dimension, dropout_rate_PWFF, layer_norm_eps):
        super().__init__()
        d_model = hidden_size
        d_ffn = FFN_internal_dimension

        self.W_1 = nn.Linear(d_model, d_ffn) 
        self.W_2 = nn.Linear(d_ffn, d_model) 
        self.relu = nn.ReLU()
        self.layer_norm = nn.LayerNorm(d_model, eps = layer_norm_eps)
        self.dropout = nn.Dropout(dropout_rate_PWFF)

    def forward(self, x):
        residual = x
        x = self.W_2(self.relu(self.W_1(x)))
        return self.layer_norm(self.dropout(x) + residual)


############################################################################################################
############################################################################################################
############################################################################################################

class SingleLayerEncoder(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, max_position_embeddings, self_attention_internal_dimension, FFN_internal_dimension, layer_norm_eps, dropout_rate_attention, dropout_rate_PWFF):
        super().__init__()

        self.attention = MultiHeadAttention(hidden_size, num_attention_heads, max_position_embeddings, self_attention_internal_dimension, dropout_rate_attention, layer_norm_eps)
        self.ffn = PositionWiseFeedForward(hidden_size, FFN_internal_dimension, dropout_rate_PWFF, layer_norm_eps)

    def forward(self, x, valid_lens=None):
        output = self.attention(x, valid_lens)
        output = self.ffn(output)
        
        return output

############################################################################################################
############################################################################################################
############################################################################################################

class Embeddings(nn.Module):
    def __init__(self, vocab_size, hidden_size, max_position_embeddings, dropout_rate, layer_norm_eps, PAD_id, Fake_position=None) -> None:
        super().__init__()

        self.token_embeddings = nn.Embedding(vocab_size, hidden_size, padding_idx=PAD_id)
        self.position_embeddings = nn.Embedding(max_position_embeddings+1, hidden_size, padding_idx=Fake_position)
        self.layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, token_ids, position_ids):
        
        token_embeddings = self.token_embeddings(token_ids)
        position_embeddings = self.position_embeddings(position_ids)
        embeddings = token_embeddings + position_embeddings
        embeddings = self.layer_norm(self.dropout(embeddings))        
        return embeddings

############################################################################################################
############################################################################################################
############################################################################################################

class Encoder(nn.Module):
    def __init__(self, encoder_stack_depth, hidden_size, num_attention_heads, max_position_embeddings, self_attention_internal_dimension, FFN_internal_dimension, layer_norm_eps, dropout_rate_attention, dropout_rate_PWFF, vocab_size, PAD_id, Fake_position):
        super().__init__()
        self.pad_indx = PAD_id
        self.embedding_layer = Embeddings(vocab_size, hidden_size, max_position_embeddings, dropout_rate_attention, layer_norm_eps, PAD_id, Fake_position)
        self.blks = nn.Sequential()
        for i in range(encoder_stack_depth):
            self.blks.add_module(f"{i}", SingleLayerEncoder(hidden_size, num_attention_heads, max_position_embeddings, self_attention_internal_dimension, FFN_internal_dimension, layer_norm_eps, dropout_rate_attention, dropout_rate_PWFF))

    def forward(self, x, position_ids, valid_lens=None):
        x = self.embedding_layer(x, position_ids) 
        for blk in self.blks:
            x = blk(x, valid_lens)
        return x

############################################################################################################
############################################################################################################
############################################################################################################
    
class mlm_classifier(nn.Module):
    def __init__(self, hidden_size, num_outputs, layer_norm_eps) -> None:
        super().__init__()
        self.mlm = nn.Sequential(nn.Linear(hidden_size, hidden_size),
            nn.ReLU(), nn.LayerNorm(hidden_size, eps = layer_norm_eps), nn.Linear(hidden_size, num_outputs))

    def forward(self, x):
        x = self.mlm(x)
        return x


############################################################################################################
############################################################################################################
############################################################################################################



class AbPred(nn.Module):
    def __init__(self, hidden_size, number_ab=16, num_outputs=2, layer_norm_eps=0.00001, **kwargs):
        super(AbPred, self).__init__(**kwargs)
        self.nn_stack = nn.ModuleList([mlm_classifier(hidden_size, num_outputs, layer_norm_eps) for _ in range(number_ab)])
    def forward(self, X, positions_y):        
        j = 0
        for encoder_layer in self.nn_stack:
            if j==0:
                mlm_Y_hat = encoder_layer(X)
                j = 1
            else:
                mlm_Y_hat = torch.cat((mlm_Y_hat, encoder_layer(X)), 1)
        Y_hat = mlm_Y_hat[positions_y]                
        return Y_hat





