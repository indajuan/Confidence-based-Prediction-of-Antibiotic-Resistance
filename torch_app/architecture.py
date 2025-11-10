import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


############################################################################################################
############################################################################################################
############################################################################################################

def masked_softmax(X, valid_lens=None):
    """Perform softmax operation by masking elements on the last axis.
    """
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
    """ This class represents the blocks "Multi-Head Attention" + "Add & Norm" blocks as described in "Attention is all you need".
        Since we only need the encoder block, it makes sense to keep these two together.
        MultiHead(Q, K, V) = Concat(head_1, ..., head_h)W^o
        head_i = Attention(QW^Q_i, KW^K_i, VW^V_i)
        Attention(Q, K, V) = softmax(QK^T/sqrt(d_k))V
    """

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
        """Multi-headed self attention
            Args:
                x (Tensor): Input with shape (batch_size, seq_length, hidden_size)
        """
        residual = x
        
        # Perform the initial projection for all heads, then using view we seperate out
        # each head such that it gains its own dimension
        
        batch_size = x.size(0)
        querys = self.Wq_all(x).view(batch_size, self.seq_len, self.num_heads, self.d_k)
        keys   = self.Wk_all(x).view(batch_size, self.seq_len, self.num_heads, self.d_k)
        values = self.Wv_all(x).view(batch_size, self.seq_len, self.num_heads, self.d_v)

        # flip dimensions such that (batch_size, seq_len, num_heads, d_k) -> (batch_size, num_heads, seq_len, d_k)
        querys, keys, values = querys.transpose(1,2), keys.transpose(1,2), values.transpose(1,2)
        
        # (Q*K^T)/sqrt(d_k)
        self_attention = torch.matmul(querys / self.d_k ** 0.5, keys.transpose(2, 3)) 
        self_attention = self.dropoutQK(masked_softmax(self_attention, valid_lens=None))
        
        output = torch.matmul(self_attention, values)

        # reoder dimensions such that num_heads and d_v are together, 
        # then combine these two dimensions to concatenate all heads.
        output = output.transpose(1, 2).contiguous().view(batch_size, self.seq_len, -1)

        # Final linear layer, residual connection, and normalization
        output = self.Wo(output)
        output = self.layer_norm(self.dropoutO(output) + residual)

        return output

############################################################################################################
############################################################################################################
############################################################################################################

class PositionWiseFeedForward(nn.Module):
    """ A Two layer feed forward network (FFN) as described in "Attention is all you need", 
        again we combine it with the subsequent layernorm and residual connection.
        FFN(x) = max[0, xW_1 + b_1]W_2 + b_2
    """

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
    """ A single encoder layer as described in "Attention is all you need", it
        consists of the two subblocks MultiHeadAttention and PositionWiseFeedForward
    """

    def __init__(self, hidden_size, num_attention_heads, max_position_embeddings, self_attention_internal_dimension, FFN_internal_dimension, layer_norm_eps, dropout_rate_attention, dropout_rate_PWFF):
        super().__init__()

        self.attention = MultiHeadAttention(hidden_size, num_attention_heads, max_position_embeddings, self_attention_internal_dimension, dropout_rate_attention, layer_norm_eps)
        self.ffn = PositionWiseFeedForward(hidden_size, FFN_internal_dimension, dropout_rate_PWFF, layer_norm_eps)

        #self.layer_norm_attention = nn.LayerNorm(hidden_size, eps = layer_norm_eps)
        #self.layer_norm_PWFF = nn.LayerNorm(hidden_size, eps = layer_norm_eps)

    def forward(self, x, valid_lens=None):
        output = self.attention(x, valid_lens)
        output = self.ffn(output)
        
        return output

############################################################################################################
############################################################################################################
############################################################################################################

class Embeddings(nn.Module):
    """The embedding layer, constructs the embeddings from animo and position embeddings
    """

    def __init__(self, vocab_size, hidden_size, max_position_embeddings, dropout_rate, layer_norm_eps, PAD_id, Fake_position=None) -> None:
        super().__init__()

        self.token_embeddings = nn.Embedding(vocab_size, hidden_size, padding_idx=PAD_id)
        #self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size, padding_idx=Fake_position)
        self.layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, token_ids, position_ids):
#        device = token_ids.device
#        input_shape = token_ids.shape
#        seq_len = input_shape[1]
#        position_ids = torch.arange(seq_len, dtype=torch.long, device=device)
#        position_ids = position_ids.unsqueeze(0).expand(input_shape)
        
        #token_embeddings = self.token_embeddings(token_ids)
        #position_embeddings = self.position_embeddings(position_ids)
        #embeddings = token_embeddings + position_embeddings
        embeddings = self.token_embeddings(token_ids)
        embeddings = self.layer_norm(self.dropout(embeddings))        
        return embeddings

############################################################################################################
############################################################################################################
############################################################################################################

class Encoder(nn.Module):
    """ A stack of SingleLayerEncoder's with the embedding at the front
    """
    def __init__(self, encoder_stack_depth, hidden_size, num_attention_heads, max_position_embeddings, self_attention_internal_dimension, FFN_internal_dimension, layer_norm_eps, dropout_rate_attention, dropout_rate_PWFF, vocab_size, PAD_id, Fake_position):
        super().__init__()
        self.pad_indx = PAD_id
        self.embedding_layer = Embeddings(vocab_size, hidden_size, max_position_embeddings, dropout_rate_attention, layer_norm_eps, PAD_id, Fake_position)
#        self.encoder_stack = nn.ModuleList([SingleLayerEncoder(hidden_size, num_attention_heads, max_position_embeddings, self_attention_internal_dimension, FFN_internal_dimension, layer_norm_eps, dropout_rate) for _ in range(encoder_stack_depth)])
        
        self.blks = nn.Sequential()
        for i in range(encoder_stack_depth):
            self.blks.add_module(f"{i}", SingleLayerEncoder(hidden_size, num_attention_heads, max_position_embeddings, self_attention_internal_dimension, FFN_internal_dimension, layer_norm_eps, dropout_rate_attention, dropout_rate_PWFF))

    def forward(self, x, position_ids, valid_lens=None):
        """
        Args: x (Tensor): Input with shape (batch_size, seq_length)
        """
        x = self.embedding_layer(x, position_ids) 
        for blk in self.blks:
            x = blk(x, valid_lens)
        return x

############################################################################################################
############################################################################################################
############################################################################################################
    
class mlm_classifier(nn.Module):
    """ This module consists of a FFNN, its purpose is 
        to take as input the output from a single BERT output 
        (a vector of size (hidden_size)) and to classify this. It is used for the 
        pre-training when classifying a single output into a token from the vocab 
    """
    def __init__(self, hidden_size, num_outputs, layer_norm_eps) -> None:
        super().__init__()
        self.mlm = nn.Sequential(nn.Linear(hidden_size, hidden_size),
            nn.ReLU(), nn.LayerNorm(hidden_size, eps = layer_norm_eps), nn.Linear(hidden_size, num_outputs))
#        self.mlm = nn.Sequential(nn.Linear(hidden_size, num_outputs))

    def forward(self, x):
        x = self.mlm(x)
        return x


############################################################################################################
############################################################################################################
############################################################################################################



class AbPred(nn.Module):
    def __init__(self, hidden_size, number_ab, num_outputs, layer_norm_eps=0.00001, **kwargs):
        super(AbPred, self).__init__(**kwargs)
        self.nn_stack = nn.ModuleList([mlm_classifier(hidden_size, num_outputs, layer_norm_eps) for _ in range(number_ab)])
    #
    def forward(self, X, positions_y):        
        j = 0
        for encoder_layer in self.nn_stack:
            if j==0:
                mlm_Y_hat = encoder_layer(X)
                j = 1
            else:
                mlm_Y_hat = torch.cat((mlm_Y_hat, encoder_layer(X)), 1)
        Y_hat = mlm_Y_hat[positions_y]                
        return Y_hat #, X_hat




class mlm_classifier_both(nn.Module):
    """ This module consists of a FFNN, its purpose is 
        to take as input the output from a single BERT output 
        (a vector of size (hidden_size)) and to classify this. It is used for the 
        pre-training when classifying a single output into a token from the vocab 
    """
    def __init__(self, hidden_size, num_outputs, layer_norm_eps) -> None:
        super().__init__()
        self.l1 =  nn.Linear(hidden_size, hidden_size)
        self.l2 =  nn.Linear(hidden_size, hidden_size)
        self.mlm = nn.Sequential(nn.Linear(hidden_size, hidden_size),
            nn.ReLU(), nn.LayerNorm(hidden_size, eps = layer_norm_eps), nn.Linear(hidden_size, num_outputs))
#        self.mlm = nn.Sequential(nn.Linear(hidden_size, num_outputs))

    def forward(self, x, y):
        z = self.l1(x) + self.l2(y)
        z = self.mlm(z)
        return z



class AbPred_both(nn.Module):
    def __init__(self, hidden_size, number_ab, num_outputs, layer_norm_eps=0.00001, **kwargs):
        super(AbPred_both, self).__init__(**kwargs)
        self.nn_stack = nn.ModuleList([mlm_classifier_both(hidden_size, num_outputs, layer_norm_eps) for _ in range(number_ab)])
    #
    def forward(self, X, X1, positions_y):        
        j = 0
        for encoder_layer in self.nn_stack:
            if j==0:
                mlm_Y_hat = encoder_layer(X,X1)
                j = 1
            else:
                mlm_Y_hat = torch.cat((mlm_Y_hat, encoder_layer(X, X1)), 1)
        Y_hat = mlm_Y_hat[positions_y]                
        return Y_hat #, X_hat



