from torch import nn


class MLP(nn.Module):
    def __init__(self, embedding_dim, dropout):
        super().__init__()
        self.c_fc = nn.Linear(embedding_dim, 4 * embedding_dim)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * embedding_dim, embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x
