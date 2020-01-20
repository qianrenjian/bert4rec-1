import numpy as np
import torch
from torch import nn
from torch.nn import TransformerEncoderLayer
from torch.nn import TransformerEncoder


def pos_enc(max_seq_len, emb_size):
    ''' Init the sinusoid position encoding table '''

    # keep dim 0 for padding token position encoding zero vector
    position_enc = np.array([
        [pos / np.power(10000, 2 * (j // 2) / emb_size) for j in range(emb_size)]
        if pos != 0 else np.zeros(emb_size) for pos in range(max_seq_len)])
    

    position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2]) # apply sin on 0th,2nd,4th...emb_dim
    position_enc[1:, 1::2] = np.cos(position_enc[1:, 1::2]) # apply cos on 1st,3rd,5th...emb_dim
    return torch.from_numpy(position_enc).type(torch.FloatTensor)

class BertRec(nn.Module):
    def __init__(self,
                 n_items,
                 max_seq_len,
                 emb_size,
                 feedforward_size,
                 dropout,
                 use_pos_emb):
        super(BertRec, self).__init__()
        self.n_items = n_items
        self.max_seq_len = max_seq_len
        self.emb_size = emb_size
        self.feedforward_size = feedforward_size
        self.dropout = dropout
        self.use_pos_emb = use_pos_emb
        self.emb = nn.Embedding(self.n_items+2, self.emb_size, padding_idx=self.n_items + 1)
        if self.use_pos_emb:
            self.pos_emb = nn.Embedding.from_pretrained(
                pos_enc(self.n_items+2, self.emb_size), 
                freeze=True, 
                padding_idx=self.n_items+1)
        enc_layer = TransformerEncoderLayer(d_model=emb_size,
                                            nhead=1,
                                            dim_feedforward=self.feedforward_size,
                                            dropout=self.dropout,
                                            activation='gelu')
        self.encoder = TransformerEncoder(enc_layer,
                                          num_layers=3,
                                          norm=None)
        self.last_ff_layer = nn.Linear(self.emb_size, self.n_items)

        self.criterion = nn.CrossEntropyLoss()
        

    def get_emb(self, base):
        x = self.emb(base)
        if self.use_pos_emb is True:
            x += self.pos_emb(base)
        return x
        
    def gen_logit(self, base, neg):
        pad = base == (1 + self.n_items)
        x = self.get_emb(base).permute(1, 0, 2)
        ret = self.encoder(x, src_key_padding_mask=pad).permute(1, 0, 2)
        logits = self.last_ff_layer(ret[neg])
        return logits

    def train_(self, base, neg, neg_t):
        neg_t = neg_t.view(-1)
        logits = self.gen_logit(base, neg)
        loss = self.criterion(logits, neg_t)
        return loss

    def recommend(self, items, K=10):
        items = np.asarray(items)
        pads = (1 + self.n_items) * np.ones(self.max_seq_len, dtype=np.int64)
        if len(items) >= self.max_seq_len:
            items = np.random.choice(items, self.max_seq_len)
        base = np.concatenate([pads[:self.max_seq_len - len(items) - 1], items, [self.n_items]])
        neg = np.zeros_like(base, dtype=np.bool)
        neg[-1] = True
        ret = self.gen_logit(torch.from_numpy(base).unsqueeze(0), 
                             torch.from_numpy(neg).unsqueeze(0)).squeeze(0).detach().numpy()
        
        return ret

        """.topk(K)
        scores = ret.values.detach().numpy()
        indices = ret.indices.detach().numpy()
        return indices, scores
        """
