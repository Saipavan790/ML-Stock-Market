


import torch.nn as nn
import math
import torch.optim as opt
import torch
import numpy as np




class InputEmbedding(nn.Module):
    
    def __init__(self, d_model:int, feat_dim:int):
        super().__init__()
        self.d_model = d_model
        self.feat_dim = feat_dim
        self.proj_inp = nn.Linear(self.feat_dim, self.d_model)
    
    def forward(self, x):
        # input: (batch, seq_len) -> output: (batch, seq_len, d_model)
        
        return self.proj_inp(x)*math.sqrt(self.d_model)





class PositionalEncoding(nn.Module):

    def __init__(self, d_model:int, seq_len:int, dropout:float):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(self.seq_len, self.d_model)
        position = torch.arange(0, seq_len, dtype=float).unsqueeze(1)   ## shape - (seq_len, 1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float()*(-math.log(10000.0)/self.d_model))  ## n**(2i/d)

        pe[:, 0::2] = torch.sin(position*div_term)
        pe[:, 1::2] = torch.cos(position*div_term)

        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)  ## register as a part of model state but not trainable parameter

    def forward(self, x):

        x = x + (self.pe[:, :x.size(1), :]).requires_grad_(False)
        return self.dropout(x)





class LayerNormalization(nn.Module):

    def __init__(self, eps:float=10**-6):

        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1))  ## multiplicative
        self.beta = nn.Parameter(torch.zeros(1))  ## additive

    def forward(self, x):
        mean = x.mean(dim = -1, keepdim=True)
        std = x.std(dim = -1, keepdim=True)

        return (self.alpha*(x-mean)/(std + self.eps)) + self.beta





class FeedForwardBlock(nn.Module):

    def __init__(self, d_model:int, d_ff:int, dropout:float):
        super().__init__()
        self.d_ff = d_ff
        self.d_model = d_model
        self.W1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.W2 = nn.Linear(d_ff, d_model)
        
    def forward(self, x):
        ## (batch, seq_len, d_model) -> (batch, seq_len, d_ff) -> (batch, seq_len, d_model)

        return self.W2(self.dropout(nn.ReLU()(self.W1(x))))




class MultiHeadAttentionBlock(nn.Module):

    def __init__(self, h:int, d_model:int, dropout:float):

        super().__init__()
        self.h = h
        self.d_model = d_model
        assert d_model % h == 0, "d_model is not divisible by h"
        self.d_k = d_model // h
        self.w_q = nn.Linear(d_model, d_model)  # wq
        self.w_k = nn.Linear(d_model, d_model)  # wk
        self.w_v = nn.Linear(d_model, d_model)  # wv
        self.w_o = nn.Linear(d_model, d_model)  # wo
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout:nn.Dropout):
        d_k = query.size(-1)

        ## (batch, h, seq_len, d_k) --> (batch, h, seq_len, seq_len)
        attention_scores = (query @ key.transpose(-2, -1))/math.sqrt(d_k)
        if mask is not None:
            attention_scores.masked_fill(mask==0, -1e9)
        attention_scores = attention_scores.softmax(dim=-1)  ### apply softmax to last dimension (batch, h, seq_len, seq_len)
        if dropout is not None:
            attention_scores = dropout(attention_scores)

        return (attention_scores @ value), attention_scores

    def forward(self, q, k, v, mask):
        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)

        ## (batch, seq_len, d_model) --> (batch, seq_len, h, d_k) --> (batch, h, seq_len, d_k)
        query = query.view(query.size(0), query.size(1), self.h, self.d_k).transpose(1,2)
        key = key.view(key.size(0), key.size(1), self.h, self.d_k).transpose(1,2)
        value = value.view(value.size(0), value.size(1), self.h, self.d_k).transpose(1,2)

        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)

        ## (batch, h, seq_len, d_k) --> (batch, seq_len, h, d_k) --> (batch, seq_len, d_model)
        x = x.transpose(1,2).contiguous().view(x.shape[0], -1, self.h * self.d_k)

        ## (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        return self.w_o(x)





class ResidualConnection(nn.Module):
    def __init__(self, dropout:float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()
    
    def forward(self, x, sublayer):

        return x + self.dropout(sublayer(self.norm(x)))





class EncoderBlock(nn.Module):

    def __init__(self, self_attention_block:MultiHeadAttentionBlock, feed_forward_block:FeedForwardBlock, dropout:float):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])

    def forward(self, x, src_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        x = self.residual_connections[1](x, self.feed_forward_block)

        return x





class Encoder(nn.Module):
    def __init__(self, layers:nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()
    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)




class DecoderBlock(nn.Module):
    def __init__(self, self_attention_block:MultiHeadAttentionBlock, cross_attention_block:MultiHeadAttentionBlock, feed_forward_block:FeedForwardBlock, dropout:float):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(3)])
    def forward(self, x, encoder_output, src_mask, tgt_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask))
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
        x = self.residual_connections[2](x, self.feed_forward_block)
        return x




class Decoder(nn.Module):
    def __init__(self, layers:nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()
    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)





class ProjectionLayer(nn.Module):
    def __init__(self, d_model:int, output_dim:int):
        super().__init__()
        self.proj = nn.Linear(d_model, output_dim)
    def forward(self, x):
        ## (batch, seq_len, d_model) --> (batch, seq_len, output_dim)
        return self.proj(x)





class Transformer(nn.Module):

    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: InputEmbedding, tgt_embed: InputEmbedding, src_pos: PositionalEncoding, tgt_pos: PositionalEncoding, projection_layer: ProjectionLayer) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer

    def encode(self, src, src_mask):
        # (batch, seq_len, d_model)
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)
    
    def decode(self, encoder_output: torch.Tensor, src_mask: torch.Tensor, tgt: torch.Tensor, tgt_mask: torch.Tensor):
        # (batch, seq_len, d_model)
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)
    
    def project(self, x):
        # (batch, seq_len, d_model) --> (batch, seq_len, 1)
        return self.projection_layer(x)

    def forward(self, src, src_mask, tgt, tgt_mask):

        x = self.encode(src, src_mask)
        x = self.decode(x, src_mask, tgt, tgt_mask)

        return self.project(x)





def build_transformer(input_dim: int, output_dim: int, src_seq_len: int, tgt_seq_len: int, d_model: int=512, N: int=6, h: int=8, dropout: float=0.1, d_ff: int=2048) -> Transformer:
    # Create the embedding layers
    src_embed = InputEmbedding(d_model, input_dim)
    tgt_embed = InputEmbedding(d_model, input_dim)

    # Create the positional encoding layers
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)
    
    # Create the encoder blocks
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttentionBlock(h, d_model, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(encoder_self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)

    # Create the decoder blocks
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttentionBlock(h, d_model, dropout)
        decoder_cross_attention_block = MultiHeadAttentionBlock(h, d_model, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(decoder_self_attention_block, decoder_cross_attention_block, feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)
    
    # Create the encoder and decoder
    encoder = Encoder(nn.ModuleList(encoder_blocks))
    decoder = Decoder(nn.ModuleList(decoder_blocks))
    
    # Create the projection layer
    projection_layer = ProjectionLayer(d_model, output_dim)
    
    # Create the transformer
    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer)
    
    # Initialize the parameters
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    
    return transformer







