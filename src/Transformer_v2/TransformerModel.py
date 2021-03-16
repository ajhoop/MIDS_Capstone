#!/usr/bin/env python3
import IPython
import torch
import math

class TransformerNet(torch.nn.Module):
  def __init__(self, num_src_vocab, embedding_dim, hidden_size, nheads, n_layers, max_src_len, dropout):
    self.dim = embedding_dim
    super(TransformerNet, self).__init__()
    # embedding layers
    self.enc_embedding = torch.nn.Embedding(num_src_vocab, embedding_dim)

    # encoder/decoder layers
    enc_layer = torch.nn.TransformerEncoderLayer(embedding_dim, nheads, hidden_size, dropout)
    self.encoder = torch.nn.TransformerEncoder(enc_layer, num_layers = n_layers)

    # https://github.com/pytorch/pytorch/blob/11f1014c05b902d3eef0fe01a7c432f818c2bdfe/torch/nn/functional.py#L4110
    # src: (S, N, E)
    # src_mask: (S, S)
    # src_key_padding_mask: (N, S)

    # It is empirically important to initialize weights properly
    self.init_weights()
   

  def init_weights(self):
      initrange = 0.1
      self.enc_embedding.weight.data.uniform_(-initrange, initrange)

  def forward(self, src, src_key_padding_mask=None):
    src = self.enc_embedding(src).permute(1, 0, 2)  * math.sqrt(self.dim)

    # Pass the mask input through the encoder layers in turn
    memory = self.encoder(src, src_key_padding_mask=src_key_padding_mask)
    return memory


