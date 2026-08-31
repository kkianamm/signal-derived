"""Concept-constrained, reliability-aware sparse MoE routing for ECG.

Core mechanisms
---------------
1) Diagnosis queries are refined through a morphology concept bank.
2) Patient adaptation is *concept constrained*: the patient-specific update is a
   sparse/soft mixture of morphology concepts, not an unconstrained residual.
3) Each diagnosis retrieves waveform, image, and shared cross-view evidence.
4) Learned reliability scores modulate a sparse top-k mixture-of-experts router.
5) Latent modality dropout teaches the router to move away from an unavailable
   view and provides direct supervision for the reliability estimators.
"""
from __future__ import annotations

import math
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
        raise ValueError(f"Expected [B,K,D], got {tuple(queries.shape)}")
    q = F.normalize(queries, dim=-1, eps=eps)
    sim = q @ q.transpose(-1, -2)
    k = sim.size(-1)
    if k <= 1:
        return sim.new_zeros(())
    eye = torch.eye(k, dtype=torch.bool, device=sim.device).unsqueeze(0)
    return sim.masked_select(~eye).square().mean()


def masked_query_consistency_loss(
    signal_evidence: Tensor,
    image_evidence: Tensor,
    both_available: Optional[Tensor] = None,
    eps: float = 1e-8,
) -> Tensor:
    if signal_evidence.shape != image_evidence.shape or signal_evidence.ndim != 3:
        raise ValueError("signal/image evidence must both be [B,K,D] with equal shape")
    a = F.normalize(signal_evidence, dim=-1, eps=eps)
    b = F.normalize(image_evidence, dim=-1, eps=eps)
    per = 1.0 - (a * b).sum(dim=-1)  # [B,K]
    if both_available is None:
        return per.mean()
    mask = both_available.to(device=per.device, dtype=per.dtype).view(-1, 1)
    denom = mask.sum() * per.size(1)
    if float(denom.detach().cpu()) <= 0:
        return per.new_zeros(())
    return (per * mask).sum() / denom


def semantic_anchor_loss(adapted: Tensor, base: Tensor, eps: float = 1e-8) -> Tensor:
    if base.ndim == 2:
        base = base.unsqueeze(0).expand(adapted.size(0), -1, -1)
    a = F.normalize(adapted, dim=-1, eps=eps)
    b = F.normalize(base, dim=-1, eps=eps)
    return (1.0 - (a * b).sum(dim=-1)).mean()


def normalized_entropy(weights: Tensor, dim: int = -1, eps: float = 1e-8) -> Tensor:
    """Entropy normalized to [0,1] for a categorical distribution."""
    n = weights.size(dim)
    if n <= 1:
        return weights.new_zeros(())
    ent = -(weights.clamp_min(eps) * weights.clamp_min(eps).log()).sum(dim=dim)
    return (ent / math.log(n)).mean()


def reliability_supervision_loss(
    signal_reliability: Tensor,
    image_reliability: Tensor,
    signal_available: Tensor,
    image_available: Tensor,
) -> Tensor:
    """Train reliability heads using synthetic missing-modality labels."""
    sig_t = signal_available.to(signal_reliability.dtype).view(-1, 1).expand_as(signal_reliability)
    img_t = image_available.to(image_reliability.dtype).view(-1, 1).expand_as(image_reliability)
    sig = F.binary_cross_entropy(signal_reliability.clamp(1e-5, 1 - 1e-5), sig_t)
    img = F.binary_cross_entropy(image_reliability.clamp(1e-5, 1 - 1e-5), img_t)
    return 0.5 * (sig + img)


class MorphologyHierarchyRefiner(nn.Module):
    def __init__(self, dim: int, heads: int = 8, dropout: float = 0.1) -> None:
        super().__init__()
        self.q_norm = nn.LayerNorm(dim)
        self.m_norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.ff_norm = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, 4 * dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(4 * dim, dim), nn.Dropout(dropout),
        )
        self.out_norm = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, diagnosis: Tensor, morphology: Tensor, batch_size: int) -> tuple[Tensor, Tensor]:
        q = diagnosis.unsqueeze(0).expand(batch_size, -1, -1)
        m = morphology.unsqueeze(0).expand(batch_size, -1, -1)
        out, attn = self.attn(
            self.q_norm(q), self.m_norm(m), self.m_norm(m),
            need_weights=True, average_attn_weights=False,
        )
        q = self.out_norm(q + self.drop(out))
        q = q + self.ff(self.ff_norm(q))
        return q, attn  # [B,K,D], [B,H,K,M]


class ConceptConstrainedPatientRefiner(nn.Module):
    """Patient adaptation constrained to the span of clinical morphology concepts."""

    def __init__(self, dim: int, dropout: float = 0.1, max_delta_scale: float = 0.35) -> None:
        super().__init__()
        self.dim = int(dim)
        self.max_delta_scale = float(max_delta_scale)
        self.context = nn.Sequential(
            nn.Linear(3 * dim, dim), nn.GELU(), nn.Dropout(dropout), nn.LayerNorm(dim)
        )
        self.q_proj = nn.Linear(2 * dim, dim, bias=False)
        self.m_proj = nn.Linear(dim, dim, bias=False)
        self.gate = nn.Sequential(
            nn.Linear(3 * dim, dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim, 1), nn.Sigmoid(),
        )
        self.out_norm = nn.LayerNorm(dim)

    def forward(
        self,
        structured_queries: Tensor,
        morphology_queries: Tensor,
        signal_global: Tensor,
        image_global: Tensor,
        hierarchy_prior: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        b, k, d = structured_queries.shape
        m = morphology_queries.size(0)
        ctx = self.context(torch.cat([
            signal_global, image_global, torch.abs(signal_global - image_global)
        ], dim=-1))
        ctx_k = ctx.unsqueeze(1).expand(-1, k, -1)
        q = self.q_proj(torch.cat([structured_queries, ctx_k], dim=-1))  # [B,K,D]
        c = self.m_proj(morphology_queries)  # [M,D]
        scores = torch.einsum("bkd,md->bkm", q, c) / math.sqrt(d)

        # Hierarchy attention serves as a weak clinical prior, not a hard rule.
        if hierarchy_prior is not None:
            if hierarchy_prior.ndim == 4:
                hierarchy_prior = hierarchy_prior.mean(dim=1)
            prior = hierarchy_prior.clamp_min(1e-6)
            scores = scores + prior.log()

        concept_weights = torch.softmax(scores, dim=-1)  # [B,K,M]
        concept_delta = torch.einsum("bkm,md->bkd", concept_weights, morphology_queries)
        concept_delta = torch.tanh(concept_delta) * self.max_delta_scale
        gate = self.gate(torch.cat([structured_queries, ctx_k, concept_delta], dim=-1))
        adapted = self.out_norm(structured_queries + gate * concept_delta)
        return adapted, gate.squeeze(-1), concept_delta, concept_weights


class ReliabilitySparseMoELayer(nn.Module):
    """Waveform/image/shared experts with reliability-modulated sparse routing."""

    def __init__(self, dim: int, heads: int = 8, dropout: float = 0.1, top_k: int = 2) -> None:
        super().__init__()
        self.top_k = max(1, min(int(top_k), 3))
        self.self_norm = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)

        self.q_norm = nn.LayerNorm(dim)
        self.sig_norm = nn.LayerNorm(dim)
        self.img_norm = nn.LayerNorm(dim)
        self.shared_norm = nn.LayerNorm(dim)
        self.signal_attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.image_attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.shared_attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)

        # q + evidence + disagreement -> reliability for each view
        self.signal_reliability = nn.Sequential(
            nn.Linear(3 * dim, dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim, 1), nn.Sigmoid()
        )
        self.image_reliability = nn.Sequential(
            nn.Linear(3 * dim, dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim, 1), nn.Sigmoid()
        )
        # q, sig, img, shared, disagreement -> three expert logits
        self.router = nn.Sequential(
            nn.Linear(5 * dim, 2 * dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(2 * dim, 3)
        )
        self.fuse_norm = nn.LayerNorm(dim)
        self.ff_norm = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, 4 * dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(4 * dim, dim), nn.Dropout(dropout),
        )
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        queries: Tensor,
        signal_memory: Tensor,
        image_memory: Tensor,
        signal_available: Tensor,
        image_available: Tensor,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        qn = self.self_norm(queries)
        self_out, _ = self.self_attn(qn, qn, qn, need_weights=False)
        queries = queries + self.drop(self_out)

        q = self.q_norm(queries)
        sig_m = self.sig_norm(signal_memory)
        img_m = self.img_norm(image_memory)
        shared_m = self.shared_norm(torch.cat([signal_memory, image_memory], dim=1))

        sig, sig_attn = self.signal_attn(q, sig_m, sig_m, need_weights=True, average_attn_weights=False)
        img, img_attn = self.image_attn(q, img_m, img_m, need_weights=True, average_attn_weights=False)
        shared, shared_attn = self.shared_attn(q, shared_m, shared_m, need_weights=True, average_attn_weights=False)

        diff = torch.abs(sig - img)
        rel_sig = self.signal_reliability(torch.cat([queries, sig, diff], dim=-1)).squeeze(-1)
        rel_img = self.image_reliability(torch.cat([queries, img, diff], dim=-1)).squeeze(-1)

        raw_logits = self.router(torch.cat([queries, sig, img, shared, diff], dim=-1))
        eps = 1e-6
        shared_rel = torch.sqrt((rel_sig * rel_img).clamp_min(eps))
        rel_bias = torch.stack([
            rel_sig.clamp_min(eps).log(),
            rel_img.clamp_min(eps).log(),
            shared_rel.clamp_min(eps).log(),
        ], dim=-1)
        logits = raw_logits + rel_bias

        sig_av = signal_available.view(-1, 1).expand(-1, queries.size(1))
        img_av = image_available.view(-1, 1).expand(-1, queries.size(1))
        shared_av = sig_av & img_av
        availability = torch.stack([sig_av, img_av, shared_av], dim=-1)
        logits = logits.masked_fill(~availability, -1e4)
        probs = torch.softmax(logits, dim=-1)

        # Sparse top-k routing. Masking after softmax keeps gradients through selected experts.
        topv, topi = probs.topk(self.top_k, dim=-1)
        sparse = torch.zeros_like(probs).scatter(-1, topi, topv)
        sparse = sparse / sparse.sum(dim=-1, keepdim=True).clamp_min(1e-8)

        experts = torch.stack([sig, img, shared], dim=-2)  # [B,K,3,D]
        routed = (sparse.unsqueeze(-1) * experts).sum(dim=-2)
        queries = self.fuse_norm(queries + self.drop(routed))
        queries = queries + self.ff(self.ff_norm(queries))

        cosine = F.cosine_similarity(sig, img, dim=-1)
        uncertainty = ((1.0 - cosine) * 0.5).clamp(0.0, 1.0)
        out = {
            "signal_evidence": sig,
            "image_evidence": img,
            "shared_evidence": shared,
            "signal_attention": sig_attn,
            "image_attention": img_attn,
            "shared_attention": shared_attn,
            "signal_reliability": rel_sig,
            "image_reliability": rel_img,
            "routing_probs": sparse,
            "routing_probs_dense": probs,
            "uncertainty": uncertainty,
        }
        return queries, out


class ConceptReliabilitySemanticMoERouter(nn.Module):
    def __init__(
        self,
        dim: int,
        diagnoses: int,
        depth: int = 3,
        heads: int = 8,
        dropout: float = 0.1,
        top_k: int = 2,
        max_query_delta: float = 0.35,
        modality_dropout_p: float = 0.15,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.diagnoses = int(diagnoses)
        self.modality_dropout_p = float(modality_dropout_p)
        self.hierarchy = MorphologyHierarchyRefiner(dim, heads=heads, dropout=dropout)
        self.patient_refiner = ConceptConstrainedPatientRefiner(
            dim, dropout=dropout, max_delta_scale=max_query_delta
        )
        self.layers = nn.ModuleList([
            ReliabilitySparseMoELayer(dim, heads=heads, dropout=dropout, top_k=top_k)
            for _ in range(int(depth))
        ])
        self.final_norm = nn.LayerNorm(dim)

    def _apply_latent_modality_dropout(
        self,
        signal_memory: Tensor,
        image_memory: Tensor,
        signal_global: Tensor,
        image_global: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        b = signal_memory.size(0)
        sig_av = torch.ones(b, dtype=torch.bool, device=signal_memory.device)
        img_av = torch.ones(b, dtype=torch.bool, device=image_memory.device)
        if self.training and self.modality_dropout_p > 0:
            u = torch.rand(b, device=signal_memory.device)
            half = self.modality_dropout_p * 0.5
            drop_sig = u < half
            drop_img = (u >= half) & (u < self.modality_dropout_p)
            sig_av = ~drop_sig
            img_av = ~drop_img
            signal_memory = signal_memory * sig_av[:, None, None].to(signal_memory.dtype)
            image_memory = image_memory * img_av[:, None, None].to(image_memory.dtype)
            signal_global = signal_global * sig_av[:, None].to(signal_global.dtype)
            image_global = image_global * img_av[:, None].to(image_global.dtype)
        return signal_memory, image_memory, signal_global, image_global, sig_av, img_av

    def forward(
        self,
        signal_memory: Tensor,
        image_memory: Tensor,
        diagnosis_queries: Tensor,
        morphology_queries: Tensor,
        signal_global: Tensor,
        image_global: Tensor,
    ) -> tuple[Tensor, dict[str, Tensor | list[Tensor]]]:
        b = signal_memory.size(0)
        (
            signal_memory, image_memory, signal_global, image_global,
            sig_av, img_av,
        ) = self._apply_latent_modality_dropout(
            signal_memory, image_memory, signal_global, image_global
        )

        structured, hierarchy_attn = self.hierarchy(
            diagnosis_queries, morphology_queries, batch_size=b
        )
        adapted, adaptation_gate, concept_delta, concept_weights = self.patient_refiner(
            structured, morphology_queries, signal_global, image_global,
            hierarchy_prior=hierarchy_attn,
        )
        queries = adapted

        layer_outputs: list[dict[str, Tensor]] = []
        for layer in self.layers:
            queries, info = layer(
                queries, signal_memory, image_memory,
                signal_available=sig_av, image_available=img_av,
            )
            layer_outputs.append(info)

        queries = self.final_norm(queries)
        last = layer_outputs[-1]
        diagnostics: dict[str, Tensor | list[Tensor]] = {
            "morphology_attention": hierarchy_attn,
            "structured_queries": structured,
            "adapted_queries": adapted,
            "adaptation_gate": adaptation_gate,
            "concept_delta": concept_delta,
            "concept_weights": concept_weights,
            "signal_available": sig_av,
            "image_available": img_av,
            "both_available": sig_av & img_av,
            "signal_evidence": last["signal_evidence"],
            "image_evidence": last["image_evidence"],
            "shared_evidence": last["shared_evidence"],
            "signal_reliability": last["signal_reliability"],
            "image_reliability": last["image_reliability"],
            "routing_probs": last["routing_probs"],
            "routing_probs_dense": last["routing_probs_dense"],
            "uncertainty": last["uncertainty"],
            "signal_attention": [x["signal_attention"] for x in layer_outputs],
            "image_attention": [x["image_attention"] for x in layer_outputs],
            "shared_attention": [x["shared_attention"] for x in layer_outputs],
            "routing_probs_layers": [x["routing_probs"] for x in layer_outputs],
            "signal_reliability_layers": [x["signal_reliability"] for x in layer_outputs],
            "image_reliability_layers": [x["image_reliability"] for x in layer_outputs],
        }
        return queries, diagnostics
