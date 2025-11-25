import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# swish激活函数
class Swish(nn.Module):
    def __init__(self, dropout=False):
        super().__init__()
        if dropout:
            self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        out = x * torch.sigmoid(x)
        if hasattr(self, 'dropout'):
            out = self.dropout(out)
        return out

def cosine_distance(x, y):
    return 1 - F.cosine_similarity(x, y, dim=-1)

class SwiGLU(nn.Module):
    """
    SwiGLU activation function as an nn.Module, allowing it to be used within nn.Sequential.
    This module splits the input tensor along the last dimension and applies the SiLU (Swish)
    activation function to the first half, then multiplies it by the second half.
    """

    def __init__(self):
        super(SwiGLU, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return F.silu(x1) * x2
    
def swiglu_correction_fn(expansion_ratio: float, d_model: int) -> int:
    # set hidden dimesion to nearest multiple of 256 after expansion ratio
    return int(((expansion_ratio * d_model) + 255) // 256 * 256)

class MLP(torch.nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int = None,
                 expansion_ratio: float = 2.0,
                 layer_norm: bool = True,
                 bias: bool = False) -> nn.Module:
        super(MLP, self).__init__()
        if out_channels is None:
            out_channels = in_channels
        hidden_channels = swiglu_correction_fn(expansion_ratio, in_channels)
        self.ffn = nn.Sequential(
            nn.LayerNorm(in_channels) if layer_norm else nn.BatchNorm1d(in_channels),
            nn.Linear(in_channels, hidden_channels * 2, bias=bias),
            SwiGLU(),
            nn.Linear(hidden_channels, out_channels, bias=bias),)
        if out_channels != in_channels:
            self.short_connect = nn.Linear(in_channels, out_channels, bias=bias)
        else:
            self.short_connect = nn.Identity()
        #self.init_weights('eye')
        self.device = next(self.parameters()).device

    def init_weights(self, mode):
        if mode == 'eye':
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    if m.weight.shape[0] == m.weight.shape[1]:  # 仅对方阵使用 eye
                        nn.init.eye_(m.weight)
                    else:
                        nn.init.xavier_uniform_(m.weight)  # 非方阵使用 Xavier
        elif mode == 'xav':
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)

    def forward(self, x):
        return self.short_connect(x) + self.ffn(x)
    
class FCHead(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 mid_channels: int = None,
                 dropout: float = 0.0,
                 bias: bool = True) -> nn.Module:
        super().__init__()
        if mid_channels is None:
            mid_channels = in_channels * 2

        module = []
        #module.append(nn.BatchNorm1d(in_channels))
        module.append(nn.Linear(in_channels, mid_channels, bias = bias))
        module.append(Swish(dropout=dropout))
        module.append(nn.Linear(mid_channels, out_channels, bias = bias))

        self.module = nn.Sequential(*module)
        self.init_weights('eye')

    def init_weights(self, mode):
        # initialize transformer
        if mode == 'eye':
            for m in self.parameters():
                if m.dim() > 1:
                    nn.init.eye_(m)
        elif mode == 'xav':
            for m in self.parameters():
                if m.dim() > 1:
                    nn.init.xavier_uniform_(m)

    def forward(self, x):
        return self.module(x)
    

class Projector(torch.nn.Module):
    def __init__(
        self, channel=512, channel_out=None, res_expansion=2.0, bias=True):
        super().__init__()
        if channel_out is None:
            channel_out = channel
        
        self.net1 = nn.Sequential(
            nn.Linear(channel, int(channel * res_expansion), bias=bias),
            nn.BatchNorm1d(int(channel * res_expansion)),
            nn.ReLU(inplace=True)
        )
        self.net2 = nn.Sequential(
            nn.Linear(int(channel * res_expansion), channel_out, bias=bias),
            nn.BatchNorm1d(channel_out)
        )
        self.init_weights('eye')
        
    def get_device(self):
        return next(self.parameters()).device
    
    def init_weights(self, mode):
        # initialize transformer
        if mode == 'eye':
            for m in self.parameters():
                if m.dim() > 1:
                    nn.init.eye_(m)
        elif mode == 'xav':
            for m in self.parameters():
                if m.dim() > 1:
                    nn.init.xavier_uniform_(m)

    def forward(self, x):
        return self.net2(self.net1(x))
    

class AttentionLayer(nn.Module):
    def __init__(self, n_embd, n_head=8, resid_pdrop=0.1, attn_pdrop=0.1):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head

        # key, query, value projections for all heads
        self.key = nn.Linear(n_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)
        self.multihead_attn = nn.MultiheadAttention(n_embd, n_head, dropout=attn_pdrop, batch_first=True)

        # output projection
        self.proj = nn.Linear(n_embd, n_embd)
        self.resid_drop = nn.Dropout(resid_pdrop)

    def forward(self, domain_embs, key_padding_mask):
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(domain_embs)
        q = self.query(domain_embs)
        v = self.value(domain_embs)
        
        att_output, att = self.multihead_attn(q, k, v, key_padding_mask=key_padding_mask)

        # output projection
        y = self.resid_drop(self.proj(att_output))

        return y, att
    
class MLPPositionEmbeddings(nn.Module):
    def __init__(self, n_embd, layer_num=2, dropout=0.5):
        super().__init__()
        
        module = []
        input_dim = 1
        for _ in range(layer_num):
            module.append(nn.Linear(input_dim, n_embd))
            module.append(nn.BatchNorm1d(n_embd))
            module.append(Swish(dropout=dropout))
            input_dim = n_embd
        module.append(nn.Linear(input_dim, n_embd))

        self.emb = nn.Sequential(*module)

    def forward(self, x):
        return self.emb(x)

class DomainAttention(nn.Module):
    def __init__(self, n_embd, domain_len, n_head=8, resid_pdrop=0.1, attn_pdrop=0.1, pos_emb='box'):
        super().__init__()
        self.pos_emb = pos_emb
        self.domain_len = domain_len
        if pos_emb == 'field' or pos_emb == 'discrete':
            self.pos_emb = nn.Parameter(torch.rand(1, self.domain_len, n_embd))
        elif pos_emb == 'mlp':
            self.pos_emb = MLPPositionEmbeddings(n_embd=n_embd)
        elif pos_emb == 'box':
            self.pos_emb = nn.Embedding(100, n_embd, padding_idx=0)
        else:
            print(f'warning: unsupported pos_emb of domain attention: {pos_emb}')

        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = AttentionLayer(n_embd, n_head, resid_pdrop, attn_pdrop)
        self.ln_2 = nn.LayerNorm(n_embd)
        self.mlp = nn.ModuleDict(dict(
            c_fc    = nn.Linear(n_embd, 4 * n_embd),
            c_proj  = nn.Linear(4 * n_embd, n_embd),
            act     = Swish(),
            dropout = nn.Dropout(resid_pdrop),
        ))
        m = self.mlp
        self.mlpf = lambda x: m.dropout(m.c_proj(m.act(m.c_fc(x)))) # MLP forward

    def forward(self, domain_embs, domain_num, domain_poss=None):
        """
        input: domain_embs: [bs, l, 256]
                domain_num: [bs]
                domain_poss: [bs, l], optional
        output: [bs, 256]
        """
        bs, l, dim = domain_embs.shape
        
        if self.pos_emb == 'discrete':
            position_embeddings = self.pos_emb # Bert embeddings
        elif self.pos_emb == 'field':
            position_embeddings = domain_poss.unsqueeze(2) * self.pos_emb # Field Embedding：为每个连续特征域单独学习一个embedding向量，然后用原始值与embedding向量相乘。
        elif self.pos_emb == 'mlp':
            position_embeddings = self.pos_emb(domain_poss.view(-1, 1)).view(bs, l, dim) # MLP embeddings
        elif self.pos_emb == 'box':
            position_embeddings = self.pos_emb((domain_poss * 100).long()) # Box Embedding
        else:
            position_embeddings = 0
        x = domain_embs + position_embeddings # each position maps to a (learnable) vector

        domain_mask = self._create_padding_mask(domain_num, l) # where there is domain, mask it
        key_padding_mask = ~domain_mask # where there is no domain, mask it
        key_padding_mask[:, 0] = False # the first position is always valid
        h, att = self.attn(self.ln_1(x), key_padding_mask) 
        #h = h + domain_embs # ec_mulpro_noseq-20240409-T20-38, 0.9036
        #h = h + x # ec_mulpro_noseq-20240410-T11-03, 0.9021
        #h = h # ec_mulpro_noseq-20240410-T18-51, 0.9023 max（0.9077 overfitting）
        h = h + self.mlpf(self.ln_2(h))
        h = h * domain_mask.unsqueeze(2)
        out = h.sum(dim=1) / (domain_num+1e-8).unsqueeze(1)
        
        #h = h.masked_fill(key_padding_mask.unsqueeze(2), float('-inf')) # 将被mask的部分置负无穷，以免影响最大池化
        #out = F.adaptive_max_pool1d(h.permute(0, 2, 1), 1).squeeze(-1)
        return out
    
    def _create_padding_mask(self, length, max_length=16):
        batch_size = length.size(0)                     # 获得batch_size
        seq_range = torch.arange(0, max_length, device=length.device).long()          # tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
        seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_length) # torch.Size([bs, l])
        seq_length_expand = length.unsqueeze(1).expand_as(seq_range_expand)
        return seq_range_expand < seq_length_expand
    
class DomainMeaning(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, domain_embs, domain_num, domain_poss=None):
        """
        input: domain_embs: [bs, l, 256]
                domain_num: [bs]
                domain_poss: [bs, l], optional
        output: [bs, 256]
        """
        bs, l, dim = domain_embs.shape
        domain_mask = self._create_padding_mask(domain_num, l) # where there is domain, mask it
        domain_embs = domain_embs * domain_mask.unsqueeze(2)

        return domain_embs.sum(dim=1) / (domain_num+1e-8).unsqueeze(1)
    
    def _create_padding_mask(self, length, max_length=16):
        batch_size = length.size(0)                     # 获得batch_size
        seq_range = torch.arange(0, max_length, device=length.device).long()          # tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
        seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_length) # torch.Size([bs, l])
        seq_length_expand = length.unsqueeze(1).expand_as(seq_range_expand)
        return seq_range_expand < seq_length_expand
    
class PositionEmbeddingLearned(nn.Module):
    """
    Absolute pos embedding, learned.
    """
    def __init__(self, embed_dim=32, max_len=3000):
        super().__init__()
        self.embed = nn.Embedding(max_len, embed_dim)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.embed.weight)

    def forward(self, data):
        pos = self.embed(data.residue_idx)
        return pos

        