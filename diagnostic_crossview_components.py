"""Diagnosis-conditioned cross-view routing components.

Each clinically initialized diagnostic query attends to ECG waveform tokens and
ECG-image tokens *separately*, then learns a diagnosis-specific gate that routes
between the two evidence sources. This avoids concatenating both modalities into
one undifferentiated memory.
"""
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
    """Discourage different diagnosis queries from collapsing to one vector."""
    if queries.ndim != 3:
        raise ValueError(f"Expected [B,Q,D], got {tuple(queries.shape)}")
    q = F.normalize(queries, dim=-1, eps=eps)
    sim = torch.matmul(q, q.transpose(-1, -2))
    qn = sim.size(-1)
    if qn <= 1:
        return sim.new_zeros(())
    eye = torch.eye(qn, dtype=torch.bool, device=sim.device).unsqueeze(0)
    return sim.masked_select(~eye).square().mean()


def query_specific_consistency_loss(
    signal_evidence: Tensor,
    image_evidence: Tensor,
    eps: float = 1e-8,
) -> Tensor:
    """Align the *same diagnosis query* across signal and image evidence.

    Unlike global branch alignment, this only asks corresponding disease-specific
    evidence vectors to agree semantically.
    """
    if signal_evidence.shape != image_evidence.shape:
        raise ValueError(
            f"Cross-view evidence shapes differ: {signal_evidence.shape} vs {image_evidence.shape}"
        )
    if signal_evidence.ndim != 3:
        raise ValueError("Expected cross-view evidence [B,K,D].")
    a = F.normalize(signal_evidence, dim=-1, eps=eps)
    b = F.normalize(image_evidence, dim=-1, eps=eps)
    return (1.0 - (a * b).sum(dim=-1)).mean()


class DiagnosticCrossViewRouterLayer(nn.Module):
    """One diagnosis-conditioned dual-attention + routing layer."""

    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dropout: float = 0.1,
        ff_mult: int = 4,
        feature_wise_gate: bool = True,
    ) -> None:
        super().__init__()
        self.feature_wise_gate = bool(feature_wise_gate)

        self.self_norm = nn.LayerNorm(dim)
        self.signal_q_norm = nn.LayerNorm(dim)
        self.signal_mem_norm = nn.LayerNorm(dim)
        self.image_q_norm = nn.LayerNorm(dim)
        self.image_mem_norm = nn.LayerNorm(dim)
        self.ff_norm = nn.LayerNorm(dim)

        self.self_attn = nn.MultiheadAttention(
            dim, heads, dropout=dropout, batch_first=True
        )
        self.signal_attn = nn.MultiheadAttention(
            dim, heads, dropout=dropout, batch_first=True
        )
        self.image_attn = nn.MultiheadAttention(
            dim, heads, dropout=dropout, batch_first=True
        )

        gate_out = dim if self.feature_wise_gate else 1
        self.router = nn.Sequential(
            nn.Linear(dim * 3, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, gate_out),
            nn.Sigmoid(),
        )
        self.fusion_norm = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, ff_mult * dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_mult * dim, dim),
            nn.Dropout(dropout),
        )
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        queries: Tensor,
        signal_memory: Tensor,
        image_memory: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        qn = self.self_norm(queries)
        self_out, _ = self.self_attn(qn, qn, qn, need_weights=False)
        queries = queries + self.drop(self_out)

        signal_out, signal_attn = self.signal_attn(
            self.signal_q_norm(queries),
            self.signal_mem_norm(signal_memory),
            self.signal_mem_norm(signal_memory),
            need_weights=True,
            average_attn_weights=False,
        )
        image_out, image_attn = self.image_attn(
            self.image_q_norm(queries),
            self.image_mem_norm(image_memory),
            self.image_mem_norm(image_memory),
            need_weights=True,
            average_attn_weights=False,
        )

        gate = self.router(torch.cat([queries, signal_out, image_out], dim=-1))
        routed = gate * signal_out + (1.0 - gate) * image_out
        queries = self.fusion_norm(queries + self.drop(routed))
        queries = queries + self.ff(self.ff_norm(queries))

        # [B,K] summary is easier to plot than the feature-wise gate [B,K,D].
        gate_summary = gate.mean(dim=-1) if gate.size(-1) > 1 else gate.squeeze(-1)
        return queries, signal_out, image_out, signal_attn, image_attn, gate_summary


class DiagnosticCrossViewRouter(nn.Module):
    """Clinically initialized K-query cross-view ECG fusion module.

    There are no generic Q-Former queries here by default: every query has a
    diagnostic identity and explicitly retrieves evidence from each modality.
    """

    def __init__(
        self,
        dim: int,
        diagnoses: int,
        depth: int = 3,
        heads: int = 8,
        dropout: float = 0.1,
        feature_wise_gate: bool = True,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.diagnoses = int(diagnoses)
        self.layers = nn.ModuleList(
            [
                DiagnosticCrossViewRouterLayer(
                    dim=self.dim,
                    heads=heads,
                    dropout=dropout,
                    feature_wise_gate=feature_wise_gate,
                )
                for _ in range(int(depth))
            ]
        )
        self.final_norm = nn.LayerNorm(self.dim)

    def forward(
        self,
        signal_memory: Tensor,
        image_memory: Tensor,
        semantic_queries: Tensor,
    ) -> tuple[Tensor, dict[str, list[Tensor] | Tensor]]:
        if semantic_queries.shape != (self.diagnoses, self.dim):
            raise ValueError(
                f"Expected semantic queries [{self.diagnoses},{self.dim}], "
                f"got {tuple(semantic_queries.shape)}."
            )
        batch = signal_memory.size(0)
        if image_memory.size(0) != batch:
            raise ValueError("Signal/image batch sizes must match.")

        queries = semantic_queries.unsqueeze(0).expand(batch, -1, -1)
        signal_attn_all: list[Tensor] = []
        image_attn_all: list[Tensor] = []
        gates_all: list[Tensor] = []
        signal_evidence = None
        image_evidence = None

        for layer in self.layers:
            (
                queries,
                signal_evidence,
                image_evidence,
                signal_attn,
                image_attn,
                gate,
            ) = layer(queries, signal_memory, image_memory)
            signal_attn_all.append(signal_attn)
            image_attn_all.append(image_attn)
            gates_all.append(gate)

        assert signal_evidence is not None and image_evidence is not None
        queries = self.final_norm(queries)
        diagnostics = {
            "signal_attention": signal_attn_all,
            "image_attention": image_attn_all,
            "routing_gates": gates_all,
            "signal_evidence": signal_evidence,
            "image_evidence": image_evidence,
        }
        return queries, diagnostics
