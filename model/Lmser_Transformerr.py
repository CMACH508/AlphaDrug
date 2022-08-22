'''
Author: QHGG
Date: 2021-11-17 18:24:36
LastEditTime: 2022-08-22 17:01:09
LastEditors: QHGG
Description: lmser transformer
FilePath: /AlphaDrug/model/CMFTgo.py
'''
import copy
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import Module


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class MFT(nn.Module):

    def __init__(self, d_model=512, nhead=4, num_layers=4, dim_feedforward=1024, pro_voc_len=10, 
    smi_voc_len=10,proPaddingIdx=0,smiPaddingIdx=0, smiMaxLen=1, proMaxLen=1, **kwargs):
        super(MFT, self).__init__()

        self.d_model = d_model
        self.proEmbedding = nn.Embedding(pro_voc_len, d_model, proPaddingIdx)
        self.smiEmbedding = nn.Embedding(smi_voc_len, d_model, smiPaddingIdx)
        self.smiPE = PositionalEncoding(d_model, 0.1, smiMaxLen)
        self.proPE = PositionalEncoding(d_model, 0.1, proMaxLen)

        mft_layer = MFTLayer(d_model, nhead, dim_feedforward)
        self.layers = _get_clones(mft_layer, num_layers)
        self.e_norm = nn.LayerNorm(d_model)
        self.d_norm = nn.LayerNorm(d_model)

        self.linear = nn.Linear(d_model, smi_voc_len)
        self.valueLinear1 = nn.Linear(d_model, smi_voc_len)
        self.valueLinear2 = nn.Linear(smi_voc_len, 1)
        self.coordsLinear1 = nn.Linear(d_model, 32)
        self.coordsLinear2 = nn.Linear(32, 3)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, tgt, smiMask, proMask, tgt_mask):
        tgt_mask = tgt_mask.squeeze(0)
        src = self.proEmbedding(src)
        src = self.proPE(src)
        src = src.permute(1, 0, 2)

        tgt = self.smiEmbedding(tgt)
        tgt = self.smiPE(tgt)
        tgt = tgt.permute(1, 0, 2)

        src_key_padding_mask = ~(proMask.to(torch.bool))
        tgt_key_padding_mask = ~(smiMask.to(torch.bool))
        memory_key_padding_mask = ~(proMask.to(torch.bool))
        
        
        e_out = src
        d_out = tgt
        for idx, mod in enumerate(self.layers):
            e_out, d_out = mod(e_out, d_out, tgt_mask=tgt_mask, src_key_padding_mask=src_key_padding_mask,\
                tgt_key_padding_mask=tgt_key_padding_mask, memory_key_padding_mask=memory_key_padding_mask)

            e_out = self.e_norm(e_out)
            d_out = self.d_norm(d_out)
            
        out = d_out.permute(1, 0, 2)
        out1 = F.log_softmax(self.linear(out), dim=-1)
        return out1

class MFTLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward):
        super(MFTLayer, self).__init__()
        self.encoder = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward)
        self.decoder = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward)

    def forward(self, src, tgt, tgt_mask=None, src_key_padding_mask=None,
    tgt_key_padding_mask=None, memory_key_padding_mask=None):
        e_out = self.encoder(src, src_key_padding_mask=src_key_padding_mask)
        t_out = self.decoder(tgt, e_out, tgt_mask=tgt_mask,tgt_key_padding_mask=tgt_key_padding_mask,
        memory_key_padding_mask=memory_key_padding_mask)

        return e_out, t_out

class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe
        return self.dropout(x)

class t(Module):
    def __init__(self, d_model, nhead):
        super(t, self).__init__()
        print(d_model, nhead)
    
    def forward():
        pass
