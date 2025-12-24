
import torch
import torch.nn as nn
import math
from torch.nn import functional as F

heads = 6
dropout = 0.2

class MultiHeaded_SA(nn.Module):
    def __init__(self, n_heads, embed_dim, head_dim, block_size):
        super().__init__()
        self.n_heads = n_heads
        self.msa_heads = nn.ModuleList([Head(embed_dim, head_dim, block_size) for _ in range(n_heads)]) 
        self.proj = nn.Linear(n_heads * head_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        out = torch.cat([self.msa_heads[i](x) for i in range(self.n_heads)], dim = -1)
        out = self.proj(out)
        out = self.dropout(out)
        return out
    
class Head(nn.Module):
    def __init__(self, embed_dim, head_dim, block_size):
        super().__init__()
        # print(embed_dim, head_dim)
        self.head_dim = head_dim
        self.block_size = block_size
        self.Wq = nn.Linear(embed_dim, head_dim)
        self.Wk = nn.Linear(embed_dim, head_dim)
        self.Wv = nn.Linear(embed_dim, head_dim)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('mask', torch.tril(torch.ones(block_size, block_size)))
        
    def forward(self, x):
        B, T, C = x.shape
        # print(x.shape)
        q = self.Wq(x) #B x T x D
        k = self.Wk(x)
        v = self.Wv(x)
        attn_wei = q @ torch.transpose(k, 1, 2)
        attn_wei = attn_wei.masked_fill(self.mask[:T, :T] == 0, float("-inf"))
        attn_wei = attn_wei/math.sqrt(self.head_dim)
        attn_wei = torch.softmax(attn_wei, dim = -1)
        attn_wei = self.dropout(attn_wei)
        out = attn_wei @ v
        return out
    
class FFN(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim), 
            nn.ReLU(),
            nn.Linear(4 *embed_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        return self.net(x)
     

class Block(nn.Module):
    def __init__(self, n_heads, embed_dim, block_size, vocab_size):
        super().__init__()
        self.n_heads = 6
        self.msa_head = MultiHeaded_SA(n_heads, embed_dim, int(embed_dim/self.n_heads), block_size)
        self.ffn = FFN(embed_dim)
        self.layer_norm_1 = nn.LayerNorm(embed_dim)
        self.layer_norm_2 = nn.LayerNorm(embed_dim)
        
    def forward(self, x):
        x = x + self.msa_head(self.layer_norm_1(x))
        x = x + self.ffn(self.layer_norm_2(x))
        return x
     
class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, block_size, device):
        super().__init__()
        self.device = device
        self.block_size = block_size
        self.tokenembedding_table = nn.Embedding(vocab_size, embed_dim)
        # nn.embed(x, y) gives a embedding matrix or llokup table of dim x, y filled with rrandom values
        self.pos_embed_table = nn.Embedding(block_size, embed_dim)
        # self.head = Head(embed_dim, embed_dim, block_size)
        self.n_heads = heads
        # self.blocks = Block(self.n_heads, embed_dim, block_size, vocab_size)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.blocks = nn.Sequential(
            Block(self.n_heads, embed_dim, block_size, vocab_size),
            Block(self.n_heads, embed_dim, block_size, vocab_size),
            Block(self.n_heads, embed_dim, block_size, vocab_size),
            Block(self.n_heads, embed_dim, block_size, vocab_size),
            Block(self.n_heads, embed_dim, block_size, vocab_size),
            Block(self.n_heads, embed_dim, block_size, vocab_size),
            self.layer_norm
        )
        self.lm_head = nn.Linear(embed_dim, vocab_size)
        
    def forward(self, idx, targets = None):
        # idx, targets are of size b x t
        B, T = idx.shape
        tok_embed = self.tokenembedding_table(idx) # b x t x c
        pos_idx = torch.arange(T).to(device=self.device)
        pos_embed = self.pos_embed_table(pos_idx).to(self.device)
        x = tok_embed + pos_embed
        # print("x:", x.shape)
        x = self.blocks(x)
        logits = self.lm_head(x)

        
        if targets != None:
            b, t, c = logits.shape
            logits = logits.view(-1, c) 
            targets = targets.view(-1)
            loss = F.cross_entropy(logits, targets)
        else:
            loss = None
        return logits, loss

    def generate(self, idx, max_new_tokens):
        # print(idx.shape)
        for i in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            # print(idx_cond.shape)
            logits, loss = self(idx_cond)
            # print(logits.shape)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim = -1)
            idx_next = torch.multinomial(probs, num_samples = 1).to(self.device)
            idx = torch.cat((idx, idx_next), dim = 1)
        return idx