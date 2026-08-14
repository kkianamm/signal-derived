"""Components for MedTsLLM + ConvNeXt + semantic Q-Former."""
from __future__ import annotations
from typing import Any, Optional
import torch
from torch import Tensor, nn
import torch.nn.functional as F


def cfg_get(obj: Any, key: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


class AttentionPool(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.score = nn.Linear(dim, 1, bias=False)

    def forward(self, tokens: Tensor) -> Tensor:
        if tokens.ndim != 3:
            raise ValueError(f"Expected [B,N,D], got {tuple(tokens.shape)}")
        weights = torch.softmax(self.score(tokens).squeeze(-1), dim=-1)
        return torch.einsum("bn,bnd->bd", weights, tokens)


def query_diversity_loss(queries: Tensor, eps: float = 1e-8) -> Tensor:
    if queries.ndim != 3:
        raise ValueError(f"Expected [B,Q,D], got {tuple(queries.shape)}")
    q = F.normalize(queries, dim=-1, eps=eps)
    sim = torch.matmul(q, q.transpose(-1, -2))
    qn = sim.size(-1)
    if qn <= 1:
        return sim.new_zeros(())
    eye = torch.eye(qn, dtype=torch.bool, device=sim.device).unsqueeze(0)
    return sim.masked_select(~eye).square().mean()


def supervised_contrastive_alignment(
    a: Tensor,
    b: Tensor,
    labels: Optional[Tensor],
    temperature: float = 0.1,
    eps: float = 1e-8,
) -> Tensor:
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("Alignment inputs must be [B,D].")
    if a.shape != b.shape:
        raise ValueError(f"Alignment shapes differ: {a.shape} vs {b.shape}")
    if a.size(0) <= 1:
        return a.new_zeros(())
    a = F.normalize(a, dim=-1, eps=eps)
    b = F.normalize(b, dim=-1, eps=eps)
    logits = torch.matmul(a, b.t()) / float(temperature)
    if labels is None:
        targets = torch.arange(a.size(0), device=a.device)
        return 0.5 * (F.cross_entropy(logits, targets) + F.cross_entropy(logits.t(), targets))
    labels = labels.view(-1)
    positive = labels[:, None].eq(labels[None, :]).to(logits.dtype)
    positive = positive / positive.sum(dim=1, keepdim=True).clamp_min(1.0)
    loss_ab = -(positive * F.log_softmax(logits, dim=1)).sum(dim=1).mean()
    loss_ba = -(positive.t() * F.log_softmax(logits.t(), dim=1)).sum(dim=1).mean()
    return 0.5 * (loss_ab + loss_ba)


class QueryFormerLayer(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float = 0.1, ff_mult: int = 4) -> None:
        super().__init__()
        self.self_norm = nn.LayerNorm(dim)
        self.cross_norm_q = nn.LayerNorm(dim)
        self.cross_norm_mem = nn.LayerNorm(dim)
        self.ff_norm = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(dim, ff_mult * dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(ff_mult * dim, dim), nn.Dropout(dropout),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, queries: Tensor, memory: Tensor) -> tuple[Tensor, Tensor]:
        qn = self.self_norm(queries)
        self_out, _ = self.self_attn(qn, qn, qn, need_weights=False)
        queries = queries + self.drop(self_out)
        qn = self.cross_norm_q(queries)
        mn = self.cross_norm_mem(memory)
        cross_out, attn = self.cross_attn(qn, mn, mn, need_weights=True, average_attn_weights=False)
        queries = queries + self.drop(cross_out)
        queries = queries + self.ff(self.ff_norm(queries))
        return queries, attn


class SemanticQueryFormer(nn.Module):
    """K class-semantic queries + remaining generic learned queries."""
    def __init__(
        self,
        dim: int,
        total_queries: int = 32,
        semantic_queries: int = 5,
        depth: int = 4,
        heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if semantic_queries < 0 or semantic_queries > total_queries:
            raise ValueError("semantic_queries must be in [0, total_queries].")
        self.dim = int(dim)
        self.total_queries = int(total_queries)
        self.semantic_count = int(semantic_queries)
        self.generic_count = self.total_queries - self.semantic_count
        if self.generic_count:
            self.generic_queries = nn.Parameter(torch.randn(self.generic_count, self.dim) * 0.02)
        else:
            self.register_parameter("generic_queries", None)
        self.layers = nn.ModuleList([
            QueryFormerLayer(self.dim, heads, dropout) for _ in range(int(depth))
        ])
        self.final_norm = nn.LayerNorm(self.dim)

    def forward(self, memory: Tensor, semantic_queries: Optional[Tensor] = None) -> tuple[Tensor, list[Tensor]]:
        batch = memory.size(0)
        pieces = []
        if self.semantic_count:
            if semantic_queries is None:
                raise ValueError("semantic_queries are required when semantic_count > 0.")
            if semantic_queries.shape != (self.semantic_count, self.dim):
                raise ValueError(
                    f"Expected semantic queries [{self.semantic_count},{self.dim}], got {tuple(semantic_queries.shape)}."
                )
            pieces.append(semantic_queries.unsqueeze(0).expand(batch, -1, -1))
        if self.generic_queries is not None:
            pieces.append(self.generic_queries.unsqueeze(0).expand(batch, -1, -1))
        queries = torch.cat(pieces, dim=1)
        attention = []
        for layer in self.layers:
            queries, attn = layer(queries, memory)
            attention.append(attn)
        return self.final_norm(queries), attention


class GatedQueryResidual(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.delta = nn.Sequential(
            nn.Linear(dim, dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim, dim)
        )
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim), nn.GELU(), nn.Linear(dim, dim), nn.Sigmoid()
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, queries: Tensor, med_context: Tensor) -> tuple[Tensor, Tensor]:
        if queries.ndim != 3 or med_context.ndim != 2:
            raise ValueError("queries=[B,Q,D], med_context=[B,D] required.")
        ctx = med_context.unsqueeze(1).expand(-1, queries.size(1), -1)
        gate = self.gate(torch.cat([queries, ctx], dim=-1))
        delta = self.delta(med_context).unsqueeze(1)
        return self.norm(queries + gate * delta), gate
