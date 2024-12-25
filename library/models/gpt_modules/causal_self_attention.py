from torch import nn
from torch.nn import functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, embedding_dim, n_heads, dropout):
        super().__init__()
        self.n_heads = n_heads
        self.embedding_dim = embedding_dim
        self.dropout = dropout

        self.c_attn = nn.Linear(embedding_dim, 3 * embedding_dim)
        self.c_proj = nn.Linear(embedding_dim, embedding_dim)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch_size, sequence_length, embedding_dim = x.size()
        queries, keys, values = self.c_attn(x).split(self.embedding_dim, dim=2)
        keys = keys.view(
            batch_size, sequence_length, self.n_heads, embedding_dim // self.n_heads
        ).transpose(1, 2)  # (B, nh, T, hs)
        queries = queries.view(
            batch_size, sequence_length, self.n_heads, embedding_dim // self.n_heads
        ).transpose(1, 2)  # (B, nh, T, hs)
        values = values.view(
            batch_size, sequence_length, self.n_heads, embedding_dim // self.n_heads
        ).transpose(1, 2)  # (B, nh, T, hs)

        y = F.scaled_dot_product_attention(
            queries,
            keys,
            values,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0,
            is_causal=True,
        )
        y = (
            y.transpose(1, 2)
            .contiguous()
            .view(batch_size, sequence_length, embedding_dim)
        )  # re-assemble all head outputs side by side

        y = self.resid_dropout(self.c_proj(y))
        return y
