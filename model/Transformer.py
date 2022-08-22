'''
Author: QHGG
Date: 2021-11-17 18:24:36
LastEditTime: 2022-03-02 21:38:20
LastEditors: QHGG
Description: original transformer
FilePath: /panas/model/Transformer.py
'''
import torch
import math
import torch.nn as nn
import torch.nn.functional as F

class MFT(nn.Module):

    def __init__(self, d_model=512, nhead=4, num_layers=4, dim_feedforward=1024, pro_voc_len=10, 
    smi_voc_len=10,proPaddingIdx=0,smiPaddingIdx=0, smiMaxLen=1, proMaxLen=1, **kwargs):
        super(MFT, self).__init__()

        self.d_model = d_model
        self.proEmbedding = nn.Embedding(pro_voc_len, d_model, proPaddingIdx)
        self.smiEmbedding = nn.Embedding(smi_voc_len, d_model, smiPaddingIdx)
        self.smiPE = PositionalEncoding(d_model, 0.1, smiMaxLen)
        self.proPE = PositionalEncoding(d_model, 0.1, proMaxLen)

        self.transformer = nn.Transformer(d_model=d_model, dim_feedforward=dim_feedforward, num_encoder_layers=num_layers, num_decoder_layers=num_layers, nhead=nhead)
        self.linear = nn.Linear(d_model, smi_voc_len)

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
        
        
        out = self.transformer(src, tgt, tgt_mask=tgt_mask,
                                        src_key_padding_mask=src_key_padding_mask, 
                                        tgt_key_padding_mask=tgt_key_padding_mask, 
                                        memory_key_padding_mask=memory_key_padding_mask)
        out = out.permute(1, 0, 2)
        out1 = F.log_softmax(self.linear(out), dim=-1)
        return out1

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
