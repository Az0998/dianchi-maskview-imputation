"""
Spatiotemporal Mask-View imputer: multi-hop GNN + temporal attention.

Spatial upgrades for station outages:
  - MixHop-style 1-hop / 2-hop graph conv
  - optional denser adjacency (built outside)
  - mask-aware neighbor aggregation (down-weight unobserved neighbors)
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MixHopGraphConv(nn.Module):
    """Aggregate 1-hop and 2-hop neighborhoods."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.lin1 = nn.Linear(in_dim, out_dim)
        self.lin2 = nn.Linear(in_dim, out_dim)
        self.lin_self = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor, adj: torch.Tensor, node_w: torch.Tensor | None = None) -> torch.Tensor:
        # x: (B,T,N,C); adj: (N,N); node_w: (B,T,N,1) reliability in [0,1]
        if node_w is not None:
            # mask-aware: A_ij' ∝ A_ij * w_j
            w = node_w.squeeze(-1)  # (B,T,N)
            adj_b = adj.unsqueeze(0).unsqueeze(0) * w.unsqueeze(2)  # (B,T,N,N)
            adj_b = adj_b / (adj_b.sum(-1, keepdim=True) + 1e-6)
            h1 = torch.einsum("btij,btjc->btic", adj_b, x)
            adj2 = torch.einsum("btij,btjk->btik", adj_b, adj_b)
            h2 = torch.einsum("btij,btjc->btic", adj2, x)
        else:
            h1 = torch.einsum("ij,btjc->btic", adj, x)
            adj2 = adj @ adj
            h2 = torch.einsum("ij,btjc->btic", adj2, x)
        return self.lin_self(x) + self.lin1(h1) + self.lin2(h2)


class TemporalAttention(nn.Module):
    def __init__(self, dim: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
        )
        self.n1 = nn.LayerNorm(dim)
        self.n2 = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.attn(x, x, x, need_weights=False)
        x = self.n1(x + h)
        x = self.n2(x + self.ff(x))
        return x


class STBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int = 4, mask_aware: bool = True):
        super().__init__()
        self.gcn = MixHopGraphConv(dim, dim)
        self.temp = TemporalAttention(dim, n_heads)
        self.norm = nn.LayerNorm(dim)
        self.mask_aware = mask_aware

    def forward(self, x: torch.Tensor, adj: torch.Tensor, obs_mask: torch.Tensor | None = None) -> torch.Tensor:
        node_w = None
        if self.mask_aware and obs_mask is not None:
            # station reliability = fraction of variables observed
            node_w = obs_mask.mean(dim=-1, keepdim=True).clamp(0.05, 1.0)
        h = F.gelu(self.gcn(x, adj, node_w))
        x = self.norm(x + h)
        b, t, n, c = x.shape
        xt = x.permute(0, 2, 1, 3).reshape(b * n, t, c)
        xt = self.temp(xt)
        x = xt.reshape(b, n, t, c).permute(0, 2, 1, 3)
        return x


class MaskViewSTImputer(nn.Module):
    def __init__(
        self,
        n_features: int,
        hidden: int = 64,
        n_layers: int = 2,
        n_heads: int = 4,
        dropout: float = 0.1,
        mask_aware: bool = True,
    ):
        super().__init__()
        self.in_proj = nn.Linear(n_features * 2, hidden)
        self.blocks = nn.ModuleList(
            [STBlock(hidden, n_heads, mask_aware=mask_aware) for _ in range(n_layers)]
        )
        self.out = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_features),
        )
        self.hidden = hidden
        self.mask_aware = mask_aware

    def encode(self, x: torch.Tensor, m: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        inp = torch.cat([x, m], dim=-1)
        h = self.in_proj(inp)
        adj = adj / (adj.sum(-1, keepdim=True) + 1e-6)
        for blk in self.blocks:
            h = blk(h, adj, m)
        return h

    def forward(self, x: torch.Tensor, m: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        pred = self.out(self.encode(x, m, adj))
        return m * x + (1.0 - m) * pred

    def impute_only(self, x: torch.Tensor, m: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        return self.out(self.encode(x, m, adj))


def positional_time_encoding(t_len: int, dim: int, device) -> torch.Tensor:
    pe = torch.zeros(t_len, dim, device=device)
    position = torch.arange(0, t_len, device=device).unsqueeze(1)
    div = torch.exp(torch.arange(0, dim, 2, device=device) * (-math.log(10000.0) / dim))
    pe[:, 0::2] = torch.sin(position * div)
    pe[:, 1::2] = torch.cos(position * div)
    return pe
