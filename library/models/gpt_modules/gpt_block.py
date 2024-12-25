from torch import nn
from library.models.gpt_modules.causal_self_attention import CausalSelfAttention
from library.models.gpt_modules.mlp import MLP

class GPTBlock(nn.Module):
    def __init__(self, embedding_dim, n_heads, dropout):
        super().__init__()
        self.ln_1 = nn.LayerNorm(embedding_dim)
        self.attn = CausalSelfAttention(embedding_dim, n_heads, dropout)
        self.ln_2 = nn.LayerNorm(embedding_dim)
        self.mlp = MLP(embedding_dim, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
