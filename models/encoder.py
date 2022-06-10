import torch
import torch.nn as nn
import math

from models.mha_ff import MultiHeadedAttention, PositionwiseFeedForward


class Classifier(nn.Module):
    def __init__(self, hidden_size):
        super(Classifier, self).__init__()
        self.linear1 = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x, mask_cls):
        h = self.linear1(x).squeeze(-1)
        sent_scores = self.sigmoid(h) * mask_cls.float()
        return sent_scores


class PositionalEncoding(nn.Module):
    def __init__(self, dropout, dim, max_len=5000):
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp((torch.arange(0,dim,2,dtype=torch.float) * 
                            -(math.log(10000.0) / dim)))
        # column step이 진행될 수록 느리게 주기가 돈다
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        pe = pe.unsqueeze(0)
        super(PositionalEncoding, self).__init__()
        
        self.register_buffer('pe', pe) # optimizer가 update하지 않는다
        self.dropout = nn.Dropout(p=dropout)
        self.dim = dim
    
    def forward(self, emb, step=None):
        emb = emb * math.sqrt(self.dim)
        if (step): 
            # 더할때 pe는 Batch 축으로 broadcasting 한다
            # step에'만' 위치정보를 부여한다 (추론용)
            # None을 추가함으로써, (1, 1, dim) shape을 가진다
            emb = emb + self.pe[:, step][:, None, :]
        else:
            # step'까지' 위치정보를 부여한다 (훈련용)
            emb = emb + self.pe[:, :emb.size(1)]
        emb = self.dropout(emb)
        return emb
    
    def get_emb(self, emb):
        return self.pe[:, :emb.size(1)]


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, heads, d_ff, dropout):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiHeadedAttention(heads,
                                              d_model, 
                                              dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(d_model,
                                                    d_ff,
                                                    dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)
    def forward(self, iter, query, inputs, mask):
        if (iter != 0):
            input_norm = self.layer_norm(inputs)
        else:
            input_norm = inputs

        mask = mask.unsqueeze(1)
        context = self.self_attn(input_norm,
                                 input_norm,
                                 input_norm,
                                 mask=mask)
        out = self.dropout(context) + inputs
        return self.feed_forward(out)


class ExtTransformerEncoder(nn.Module):
    def __init__(self, d_model, d_ff, heads, dropout, num_inter_layers=0):
        super(ExtTransformerEncoder, self).__init__()
        self.d_model = d_model
        self.num_inter_layers = num_inter_layers
        self.pos_emb = PositionalEncoding(dropout, d_model)
        self.transformer_inter = nn.ModuleList(
            [TransformerEncoderLayer(d_model, heads, d_ff, dropout)
            for _ in range(num_inter_layers)])
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.wo = nn.Linear(d_model, 1, bias=True)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, top_vecs, mask):
        batch_size, n_sents = top_vecs.size(0), top_vecs.size(1)
        pos_emb = self.pos_emb.pe[:, :n_sents]
        x = top_vecs * mask[:, :, None].float()
        x = x + pos_emb

        for i in range(self.num_inter_layers):
            # True인 element에 masked_fill 되므로, 1 - mask로 T/F를 역전한다
            x = self.transformer_inter[i](i, x, x, 1 - mask) # (B, s_len, d)
        
        x = self.layer_norm(x)
        sent_scores = self.sigmoid(self.wo(x))
        sent_scores = sent_scores.squeeze(-1) * mask.float()
        return sent_scores