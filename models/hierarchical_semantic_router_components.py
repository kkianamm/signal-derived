"""Hierarchical, patient-adaptive diagnosis-conditioned cross-view routing.

This module adds three mechanisms on top of diagnosis-conditioned routing:
1) morphology-guided hierarchy: diagnosis queries attend to a bank of clinical
   morphology concepts before touching ECG evidence;
2) patient-adaptive query refinement: each structured diagnosis query is
   conditioned on the current signal/image global context with a bounded residual;
3) disagreement-aware cross-view routing: each diagnosis separately retrieves
   waveform and image evidence, and the routing gate explicitly sees their
   disagreement. A deterministic per-diagnosis uncertainty score is exposed.
"""
from __future__ import annotations

from typing import Any

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
        raise ValueError(f"Expected [B,K,D], got {tuple(queries.shape)}")
    q = F.normalize(queries, dim=-1, eps=eps)
    sim = q @ q.transpose(-1, -2)
    k = sim.size(-1)
    if k <= 1:
        return sim.new_zeros(())
    eye = torch.eye(k, dtype=torch.bool, device=sim.device).unsqueeze(0)
    return sim.masked_select(~eye).square().mean()


def query_specific_consistency_loss(
    signal_evidence: Tensor,
    image_evidence: Tensor,
    eps: float = 1e-8,
) -> Tensor:
    """Weakly align corresponding diagnosis-specific evidence across views."""
    if signal_evidence.shape != image_evidence.shape or signal_evidence.ndim != 3:
        raise ValueError("signal/image evidence must both be [B,K,D] with equal shape")
    a = F.normalize(signal_evidence, dim=-1, eps=eps)
    b = F.normalize(image_evidence, dim=-1, eps=eps)
    return (1.0 - (a * b).sum(dim=-1)).mean()


def semantic_anchor_loss(
    adapted_queries: Tensor,
    base_queries: Tensor,
    eps: float = 1e-8,
) -> Tensor:
    """Keep patient adaptation close enough to the clinical semantic anchor.

    adapted_queries: [B,K,D]
    base_queries: [K,D] or [B,K,D]
    """
    if adapted_queries.ndim != 3:
        raise ValueError("adapted_queries must be [B,K,D]")
    if base_queries.ndim == 2:
        base_queries = base_queries.unsqueeze(0).expand(adapted_queries.size(0), -1, -1)
    if base_queries.shape != adapted_queries.shape:
        raise ValueError("base/adapted query shapes must match")
    a = F.normalize(adapted_queries, dim=-1, eps=eps)
    b = F.normalize(base_queries, dim=-1, eps=eps)
    return (1.0 - (a * b).sum(dim=-1)).mean()


class MorphologyHierarchyRefiner(nn.Module):
    """Refine diagnosis semantics through clinical morphology concepts.

    Diagnosis queries are the queries; morphology concepts are key/value memory.
    The output keeps the diagnosis identity through a residual connection.
    """

    def __init__(self, dim: int, heads: int = 8, dropout: float = 0.1) -> None:
        super().__init__()
        self.q_norm = nn.LayerNorm(dim)
        self.m_norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.ff_norm = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * dim, dim),
            nn.Dropout(dropout),
        )
        self.out_norm = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        diagnosis_queries: Tensor,
        morphology_queries: Tensor,
        batch_size: int,
    ) -> tuple[Tensor, Tensor]:
        if diagnosis_queries.ndim != 2 or morphology_queries.ndim != 2:
            raise ValueError("diagnosis_queries and morphology_queries must be [K,D] and [M,D]")
        q = diagnosis_queries.unsqueeze(0).expand(batch_size, -1, -1)
        m = morphology_queries.unsqueeze(0).expand(batch_size, -1, -1)
        out, attn = self.attn(
            self.q_norm(q), self.m_norm(m), self.m_norm(m),
            need_weights=True, average_attn_weights=False,
        )
        q = self.out_norm(q + self.drop(out))
        q = q + self.ff(self.ff_norm(q))
        return q, attn  # attn [B,H,K,M]


class PatientAdaptiveQueryRefiner(nn.Module):
    """Bounded patient-specific refinement of each diagnosis query."""

    def __init__(self, dim: int, dropout: float = 0.1, max_delta_scale: float = 0.35) -> None:
        super().__init__()
        self.max_delta_scale = float(max_delta_scale)
        # signal global, image global, and absolute cross-view global difference
        self.context = nn.Sequential(
            nn.Linear(3 * dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(dim),
        )
        self.delta = nn.Sequential(
            nn.Linear(2 * dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )
        self.gate = nn.Sequential(
            nn.Linear(2 * dim, dim),
            nn.GELU(),
            nn.Linear(dim, 1),
            nn.Sigmoid(),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(
        self,
        structured_queries: Tensor,
        signal_global: Tensor,
        image_global: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        if structured_queries.ndim != 3:
            raise ValueError("structured_queries must be [B,K,D]")
        if signal_global.shape != image_global.shape or signal_global.ndim != 2:
            raise ValueError("signal_global/image_global must be equal [B,D]")
        ctx = self.context(torch.cat([
            signal_global,
            image_global,
            torch.abs(signal_global - image_global),
        ], dim=-1))
        ctx_k = ctx.unsqueeze(1).expand(-1, structured_queries.size(1), -1)
        joint = torch.cat([structured_queries, ctx_k], dim=-1)
        delta = torch.tanh(self.delta(joint)) * self.max_delta_scale
        gate = self.gate(joint)  # [B,K,1]
        adapted = self.norm(structured_queries + gate * delta)
        return adapted, gate.squeeze(-1), delta


class DisagreementAwareRouterLayer(nn.Module):
    """Dual-view evidence retrieval with diagnosis-specific disagreement-aware routing."""

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
        self.sig_q_norm = nn.LayerNorm(dim)
        self.sig_m_norm = nn.LayerNorm(dim)
        self.img_q_norm = nn.LayerNorm(dim)
        self.img_m_norm = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.signal_attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.image_attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        gate_out = dim if self.feature_wise_gate else 1
        # q + signal evidence + image evidence + disagreement
        self.router = nn.Sequential(
            nn.Linear(4 * dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, gate_out),
            nn.Sigmoid(),
        )
        self.fuse_norm = nn.LayerNorm(dim)
        self.ff_norm = nn.LayerNorm(dim)
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
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        qn = self.self_norm(queries)
        self_out, _ = self.self_attn(qn, qn, qn, need_weights=False)
        queries = queries + self.drop(self_out)

        sig_q = self.sig_q_norm(queries)
        sig_m = self.sig_m_norm(signal_memory)
        signal_out, signal_attn = self.signal_attn(
            sig_q, sig_m, sig_m, need_weights=True, average_attn_weights=False
        )
        img_q = self.img_q_norm(queries)
        img_m = self.img_m_norm(image_memory)
        image_out, image_attn = self.image_attn(
            img_q, img_m, img_m, need_weights=True, average_attn_weights=False
        )

        disagreement_vec = torch.abs(signal_out - image_out)
        gate = self.router(torch.cat([queries, signal_out, image_out, disagreement_vec], dim=-1))
        routed = gate * signal_out + (1.0 - gate) * image_out
        queries = self.fuse_norm(queries + self.drop(routed))
        queries = queries + self.ff(self.ff_norm(queries))

        # Deterministic uncertainty: 0 = views agree, 1 = maximally opposed.
        cosine = F.cosine_similarity(signal_out, image_out, dim=-1)
        uncertainty = ((1.0 - cosine) * 0.5).clamp(0.0, 1.0)
        gate_summary = gate.mean(dim=-1) if gate.size(-1) > 1 else gate.squeeze(-1)
        return (
            queries, signal_out, image_out, signal_attn, image_attn,
            gate_summary, uncertainty,
        )


class HierarchicalPatientAdaptiveCrossViewRouter(nn.Module):
    """Full semantic router: hierarchy -> patient adaptation -> cross-view routing."""

    def __init__(
        self,
        dim: int,
        diagnoses: int,
        depth: int = 3,
        heads: int = 8,
        dropout: float = 0.1,
        feature_wise_gate: bool = True,
        max_query_delta: float = 0.35,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.diagnoses = int(diagnoses)
        self.hierarchy = MorphologyHierarchyRefiner(dim, heads=heads, dropout=dropout)
        self.patient_refiner = PatientAdaptiveQueryRefiner(
            dim, dropout=dropout, max_delta_scale=max_query_delta
        )
        self.layers = nn.ModuleList([
            DisagreementAwareRouterLayer(
                dim=dim,
                heads=heads,
                dropout=dropout,
                feature_wise_gate=feature_wise_gate,
            )
            for _ in range(int(depth))
        ])
        self.final_norm = nn.LayerNorm(dim)

    def forward(
        self,
        signal_memory: Tensor,
        image_memory: Tensor,
        diagnosis_queries: Tensor,
        morphology_queries: Tensor,
        signal_global: Tensor,
        image_global: Tensor,
    ) -> tuple[Tensor, dict[str, Tensor | list[Tensor]]]:
        batch = signal_memory.size(0)
        if image_memory.size(0) != batch:
            raise ValueError("signal/image batch sizes must match")
        if diagnosis_queries.shape != (self.diagnoses, self.dim):
            raise ValueError(
                f"Expected diagnosis queries [{self.diagnoses},{self.dim}], got {tuple(diagnosis_queries.shape)}"
            )
        structured, morphology_attn = self.hierarchy(
            diagnosis_queries, morphology_queries, batch_size=batch
        )
        adapted, adaptation_gate, adaptation_delta = self.patient_refiner(
            structured, signal_global, image_global
        )
        queries = adapted

        signal_attn_all: list[Tensor] = []
        image_attn_all: list[Tensor] = []
        routing_gates: list[Tensor] = []
        uncertainties: list[Tensor] = []
        signal_evidence = image_evidence = None

        for layer in self.layers:
            (
                queries,
                signal_evidence,
                image_evidence,
                signal_attn,
                image_attn,
                gate,
                uncertainty,
            ) = layer(queries, signal_memory, image_memory)
            signal_attn_all.append(signal_attn)
            image_attn_all.append(image_attn)
            routing_gates.append(gate)
            uncertainties.append(uncertainty)

        assert signal_evidence is not None and image_evidence is not None
        queries = self.final_norm(queries)
        diagnostics: dict[str, Tensor | list[Tensor]] = {
            "morphology_attention": morphology_attn,
            "structured_queries": structured,
            "adapted_queries": adapted,
            "adaptation_gate": adaptation_gate,
            "adaptation_delta": adaptation_delta,
            "signal_attention": signal_attn_all,
            "image_attention": image_attn_all,
            "routing_gates": routing_gates,
            "uncertainty": uncertainties[-1],
            "uncertainty_layers": uncertainties,
            "signal_evidence": signal_evidence,
            "image_evidence": image_evidence,
        }
        return queries, diagnostics
