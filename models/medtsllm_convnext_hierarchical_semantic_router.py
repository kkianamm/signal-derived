"""MedTsLLM + ConvNeXt + hierarchical patient-adaptive semantic routing.

Proposed method:
  clinical diagnosis text -> diagnosis queries
  clinical morphology text -> morphology semantic bank
  diagnosis-to-morphology refinement -> patient-adaptive diagnosis queries
  -> separate waveform/image evidence retrieval
  -> disagreement-aware diagnosis-specific routing
  -> LLM/classification

This file intentionally lives beside the earlier semantic-QFormer and diagnostic
router models so they remain available as controlled ablation baselines.
"""
from __future__ import annotations

from collections import OrderedDict
from typing import Any, Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F

try:
    from torchvision.models import (
        convnext_tiny, convnext_small, convnext_base, convnext_large,
        ConvNeXt_Tiny_Weights, ConvNeXt_Small_Weights,
        ConvNeXt_Base_Weights, ConvNeXt_Large_Weights,
    )
except Exception as exc:
    raise ImportError("torchvision with ConvNeXt support is required") from exc

from .medtsllm import MedTsLLM
from .hierarchical_semantic_router_components import (
    AttentionPool,
    HierarchicalPatientAdaptiveCrossViewRouter,
    cfg_get,
    query_diversity_loss,
    query_specific_consistency_loss,
    semantic_anchor_loss,
)


PTBXL_CLASS_DESCRIPTIONS = (
    "a normal electrocardiogram with normal rhythm and normal cardiac morphology",
    "an electrocardiogram with morphology associated with myocardial infarction",
    "an electrocardiogram showing ST segment or T wave abnormalities",
    "an electrocardiogram showing a cardiac conduction disturbance",
    "an electrocardiogram showing ventricular or atrial hypertrophy patterns",
)

PTBXL_MORPHOLOGY_DESCRIPTIONS = (
    "normal cardiac rhythm and normal electrocardiographic morphology",
    "ST segment elevation or depression indicating abnormal ventricular repolarization",
    "T wave inversion flattening or other abnormal T wave morphology",
    "pathological Q waves or loss of normal R wave progression associated with infarction",
    "widened or otherwise abnormal QRS complexes indicating ventricular conduction disturbance",
    "abnormal atrioventricular conduction or PR interval morphology",
    "high voltage or altered depolarization morphology associated with cardiac hypertrophy",
    "abnormal ventricular repolarization morphology involving ST and T wave patterns",
)


class ConvNeXtSpatialTokenEncoder(nn.Module):
    _BUILDERS = {
        "tiny": (convnext_tiny, ConvNeXt_Tiny_Weights.DEFAULT),
        "small": (convnext_small, ConvNeXt_Small_Weights.DEFAULT),
        "base": (convnext_base, ConvNeXt_Base_Weights.DEFAULT),
        "large": (convnext_large, ConvNeXt_Large_Weights.DEFAULT),
    }

    def __init__(
        self,
        variant: str = "tiny",
        pretrained: bool = True,
        freeze: bool = True,
        unfreeze_last_n: int = 0,
    ) -> None:
        super().__init__()
        variant = str(variant).lower()
        if variant not in self._BUILDERS:
            raise ValueError(f"Unknown ConvNeXt variant {variant!r}")
        builder, weights = self._BUILDERS[variant]
        self.backbone = builder(weights=weights if pretrained else None)
        self.features = self.backbone.features
        ns = getattr(self.backbone.classifier[0], "normalized_shape", None)
        if ns is None:
            raise RuntimeError("Could not infer ConvNeXt output dimension")
        self.output_dim = int(ns[0] if isinstance(ns, (tuple, list)) else ns)
        if freeze:
            for p in self.features.parameters():
                p.requires_grad = False
        if int(unfreeze_last_n) > 0:
            for block in list(self.features.children())[-int(unfreeze_last_n):]:
                for p in block.parameters():
                    p.requires_grad = True


    def forward(self, image: Tensor) -> Tensor:
        if image.ndim != 4:
            raise ValueError(f"image must be [B,C,H,W], got {tuple(image.shape)}")
        if image.size(1) == 1:
            image = image.repeat(1, 3, 1, 1)
        elif image.size(1) != 3:
            raise ValueError("ConvNeXt expects 1-channel or 3-channel images")
        return self.features(image).flatten(2).transpose(1, 2)


class MedTsLLMConvNeXtHierarchicalSemanticRouter(MedTsLLM):
    """Hierarchical patient-adaptive diagnosis-conditioned ECG cross-view router."""

    supported_tasks = ["classification"]
    supported_modes = ["multivariate"]

    def __init__(self, config: Any, dataset: Any) -> None:
        if config.task != "classification":
            raise ValueError("This model supports classification only")
        super().__init__(config, dataset)
        section = cfg_get(self.model_config, "hierarchical_semantic_router", None)
        if section is None:
            raise ValueError(
                "Missing [models.medtsllm.hierarchical_semantic_router] in TOML config"
            )
        if not self.llm_enabled:
            raise ValueError("This model requires the LLM")

        self.q_dim = int(cfg_get(section, "q_dim", 512))
        self.semantic_count = int(cfg_get(section, "semantic_queries", self.n_classes))
        if self.semantic_count != self.n_classes:
            raise ValueError(
                f"semantic_queries must equal n_classes ({self.n_classes}), got {self.semantic_count}"
            )
        depth = int(cfg_get(section, "depth", 3))
        heads = int(cfg_get(section, "heads", 8))
        dropout = float(cfg_get(section, "dropout", 0.1))
        feature_wise_gate = bool(cfg_get(section, "feature_wise_gate", True))
        max_query_delta = float(cfg_get(section, "max_query_delta", 0.35))

        conv_cfg = cfg_get(section, "convnext", {})
        self.save_frozen_convnext = bool(cfg_get(conv_cfg, "save_frozen_backbone", False))
        self.image_encoder = ConvNeXtSpatialTokenEncoder(
            variant=str(cfg_get(conv_cfg, "variant", "tiny")),
            pretrained=bool(cfg_get(conv_cfg, "pretrained", True)),
            freeze=bool(cfg_get(conv_cfg, "freeze", True)),
            unfreeze_last_n=int(cfg_get(conv_cfg, "unfreeze_last_n", 0)),
        )

        self.med_to_q = nn.Sequential(nn.Linear(self.d_llm, self.q_dim), nn.LayerNorm(self.q_dim))
        self.image_to_q = nn.Sequential(
            nn.Linear(self.image_encoder.output_dim, self.q_dim), nn.LayerNorm(self.q_dim)
        )
        self.modality_embeddings = nn.Parameter(torch.randn(2, self.q_dim) * 0.02)

        self.diagnosis_projection = nn.Sequential(
            nn.Linear(self.d_llm, self.q_dim), nn.GELU(),
            nn.Linear(self.q_dim, self.q_dim), nn.LayerNorm(self.q_dim),
        )
        self.morphology_projection = nn.Sequential(
            nn.Linear(self.d_llm, self.q_dim), nn.GELU(),
            nn.Linear(self.q_dim, self.q_dim), nn.LayerNorm(self.q_dim),
        )
        self.diagnosis_delta = nn.Parameter(torch.zeros(self.semantic_count, self.q_dim))

        class_desc = cfg_get(section, "class_descriptions", None) or PTBXL_CLASS_DESCRIPTIONS
        morph_desc = cfg_get(section, "morphology_descriptions", None) or PTBXL_MORPHOLOGY_DESCRIPTIONS
        self.class_descriptions = tuple(str(x) for x in class_desc)
        self.morphology_descriptions = tuple(str(x) for x in morph_desc)
        if len(self.class_descriptions) != self.semantic_count:
            raise ValueError("class_descriptions length must equal semantic_queries")

        self.register_buffer("_diagnosis_text_embeddings", torch.empty(0, self.d_llm), persistent=True)
        self.register_buffer("_morphology_text_embeddings", torch.empty(0, self.d_llm), persistent=True)

        self.med_pool = AttentionPool(self.q_dim)
        self.image_pool = AttentionPool(self.q_dim)
        self.query_pool = AttentionPool(self.q_dim)
        self.router = HierarchicalPatientAdaptiveCrossViewRouter(
            dim=self.q_dim,
            diagnoses=self.semantic_count,
            depth=depth,
            heads=heads,
            dropout=dropout,
            feature_wise_gate=feature_wise_gate,
            max_query_delta=max_query_delta,
        )
        self.q_to_llm = nn.Sequential(nn.Linear(self.q_dim, self.d_llm), nn.LayerNorm(self.d_llm))
        self.llm_pool = AttentionPool(self.d_llm)

        output_dim = self.n_classes if self.n_classes > 2 else 1
        self.med_aux_head = nn.Linear(self.q_dim, output_dim)
        self.image_aux_head = nn.Linear(self.q_dim, output_dim)
        self.query_aux_head = nn.Linear(self.q_dim, output_dim)
        self.direct_query_score = nn.Linear(self.q_dim, 1)
        self.direct_logit_weight = nn.Parameter(
            torch.tensor(float(cfg_get(section, "direct_logit_weight_init", -1.4)))
        )

        self.use_biomedcoop = bool(getattr(self, "use_biomedcoop", False))
        self.standard_classifier: Optional[nn.Linear] = None
        if not self.use_biomedcoop:
            self.standard_classifier = nn.Linear(self.d_llm, output_dim)

        loss_cfg = cfg_get(section, "loss", {})
        self.loss_weights = {
            "med_ce": float(cfg_get(loss_cfg, "med_ce", 0.10)),
            "image_ce": float(cfg_get(loss_cfg, "image_ce", 0.10)),
            "query_ce": float(cfg_get(loss_cfg, "query_ce", 0.10)),
            "query_consistency": float(cfg_get(loss_cfg, "query_consistency", 0.02)),
            "query_diversity": float(cfg_get(loss_cfg, "query_diversity", 0.01)),
            "semantic_anchor": float(cfg_get(loss_cfg, "semantic_anchor", 0.02)),
            "biomedcoop": float(cfg_get(loss_cfg, "biomedcoop", 1.0)),
        }
        self._auxiliary_losses: dict[str, Tensor] = {}
        self.aux_loss: Optional[Tensor] = None

        # Analysis outputs for figures/ablation.
        self.last_morphology_attention: Optional[Tensor] = None
        self.last_adaptation_gate: Optional[Tensor] = None
        self.last_routing_gates: Optional[list[Tensor]] = None
        self.last_uncertainty: Optional[Tensor] = None
        self.last_signal_attention: Optional[list[Tensor]] = None
        self.last_image_attention: Optional[list[Tensor]] = None

    def _restore_batch(self, tokens: Tensor, batch_size: int) -> Tensor:
        if tokens.shape[0] == batch_size:
            return tokens
        if tokens.shape[0] % batch_size:
            raise RuntimeError(
                f"MedTsLLM token batch cannot be restored: tokens={tokens.shape[0]}, batch={batch_size}"
            )
        channels = tokens.shape[0] // batch_size
        return tokens.view(batch_size, channels, tokens.shape[1], tokens.shape[2]).mean(1)

    def _get_image(self, inputs: dict[str, Any]) -> Tensor:
        for key in ("x_image", "image", "images"):
            value = inputs.get(key)
            if value is not None:
                return value
        raise KeyError("No image tensor found; expected inputs['x_image'] (or image/images)")

    def _encode_prompts(
        self,
        inputs: dict[str, Any],
        dtype: torch.dtype,
    ):
        x_enc = inputs["x_enc"]
        batch = x_enc.size(0)

        prompts = self.build_prompt(inputs)

        if not prompts or not prompts[0]:
            prompt_tokens = torch.zeros(
                batch,
                0,
                self.d_llm,
                device=x_enc.device,
                dtype=dtype,
            )
            prompt_mask = torch.ones(
                batch,
                0,
                device=x_enc.device,
                dtype=torch.long,
            )
            return prompt_tokens, prompt_mask

        encoded = [
            [self.encode_part(part) for part in prompt]
            for prompt in prompts
        ]

        encoded = [
            torch.cat(parts, dim=1)
            for parts in encoded
        ]

        max_len = max(item.size(1) for item in encoded)

        padded = [
            self.pad_sequence(item, max_len)
            for item in encoded
        ]

        prompt_tokens = torch.cat(
            [item[0] for item in padded],
            dim=0,
        )

        prompt_mask = torch.cat(
            [item[1] for item in padded],
            dim=0,
        )

        prompt_tokens = prompt_tokens.to(
            device=x_enc.device,
            dtype=dtype,
        )

        prompt_mask = prompt_mask.to(
            device=x_enc.device,
        )

        return prompt_tokens, prompt_mask


    @torch.no_grad()
    def _build_text_banks(self, device: torch.device) -> None:
        if self._diagnosis_text_embeddings.numel() == 0:
            vecs = []
            for text in self.class_descriptions:
                emb = self.encode_part(text)
                if emb.ndim != 3:
                    raise RuntimeError(
                        f"encode_part must return [1,L,D], got {tuple(emb.shape)}"
                    )
                vecs.append(emb.mean(dim=1).squeeze(0))
            self._diagnosis_text_embeddings = (
                torch.stack(vecs).detach().to(device=device)
            )

        if self._morphology_text_embeddings.numel() == 0:
            vecs = []
            for text in self.morphology_descriptions:
                emb = self.encode_part(text)
                if emb.ndim != 3:
                    raise RuntimeError(
                        f"encode_part must return [1,L,D], got {tuple(emb.shape)}"
                    )
                vecs.append(emb.mean(dim=1).squeeze(0))
            self._morphology_text_embeddings = (
                torch.stack(vecs).detach().to(device=device)
            )

    def _semantic_banks(
        self,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[Tensor, Tensor]:
        self._build_text_banks(device)

        diagnosis = self.diagnosis_projection(
            self._diagnosis_text_embeddings.to(
                device=device,
                dtype=dtype,
            )
        )
        diagnosis = diagnosis + self.diagnosis_delta.to(
            device=device,
            dtype=dtype,
        )

        morphology = self.morphology_projection(
            self._morphology_text_embeddings.to(
                device=device,
                dtype=dtype,
            )
        )

        return diagnosis, morphology

    def _run_llm(
        self,
        prompt_tokens: Tensor,
        prompt_mask: Tensor,
        soft_queries: Tensor,
    ) -> Tensor:
        soft_queries = soft_queries.to(dtype=prompt_tokens.dtype)

        if self.llm.config.is_encoder_decoder:
            out = self.llm(
                inputs_embeds=prompt_tokens,
                attention_mask=prompt_mask,
                decoder_inputs_embeds=soft_queries,
            ).last_hidden_state

            return out[:, -soft_queries.size(1):].to(
                soft_queries.dtype
            )

        llm_input = torch.cat(
            [prompt_tokens, soft_queries],
            dim=1,
        )

        query_mask = torch.ones(
            soft_queries.size(0),
            soft_queries.size(1),
            device=soft_queries.device,
            dtype=torch.long,
        )

        attention_mask = torch.cat(
            [prompt_mask, query_mask],
            dim=1,
        )

        out = self.llm(
            inputs_embeds=llm_input,
            attention_mask=attention_mask,
        ).last_hidden_state

        return out[:, -soft_queries.size(1):].to(
            soft_queries.dtype
        )

    def _branch_loss(self, logits: Tensor, labels: Tensor) -> Tensor:
        if self.n_classes > 2:
            return F.cross_entropy(logits, labels.long())
        return F.binary_cross_entropy_with_logits(
            logits.squeeze(-1),
            labels.to(logits.dtype),
        )

    def _set_auxiliary_losses(
        self,
        med_repr: Tensor,
        image_repr: Tensor,
        query_repr: Tensor,
        queries: Tensor,
        adapted_queries: Tensor,
        structured_queries: Tensor,
        signal_evidence: Tensor,
        image_evidence: Tensor,
        labels: Optional[Tensor],
    ) -> None:
        zero = query_repr.new_zeros(())

        bc_aux = zero
        if self.use_biomedcoop and hasattr(self, "bc_head"):
            current = getattr(self.bc_head, "aux_loss", None)
            if current is not None:
                bc_aux = current

        raw = {
            "med_ce": zero,
            "image_ce": zero,
            "query_ce": zero,
            "query_consistency": query_specific_consistency_loss(
                signal_evidence,
                image_evidence,
            ),
            "query_diversity": query_diversity_loss(queries),
            "semantic_anchor": semantic_anchor_loss(
                adapted_queries,
                structured_queries,
            ),
            "biomedcoop": bc_aux,
        }

        if labels is not None:
            raw["med_ce"] = self._branch_loss(
                self.med_aux_head(med_repr),
                labels,
            )
            raw["image_ce"] = self._branch_loss(
                self.image_aux_head(image_repr),
                labels,
            )
            raw["query_ce"] = self._branch_loss(
                self.query_aux_head(query_repr),
                labels,
            )

        weighted = {
            name: value * self.loss_weights[name]
            for name, value in raw.items()
        }

        weighted["total"] = torch.stack(
            tuple(weighted.values())
        ).sum()

        self._auxiliary_losses = weighted
        self.aux_loss = weighted["total"]

    def forward(self, inputs: dict[str, Any]) -> Tensor:
        x_enc: Tensor = inputs["x_enc"]
        if x_enc.ndim == 2:
            x_enc = x_enc.unsqueeze(-1)
        if x_enc.ndim != 3:
            raise ValueError(f"x_enc must be [B,T,C], got {tuple(x_enc.shape)}")
        if x_enc.size(-1) != self.n_features:
            raise ValueError(f"Expected {self.n_features} channels, got {x_enc.size(-1)}")
        if self.device is None:
            self.device = x_enc.device
        batch = x_enc.size(0)

        # Waveform branch.
        med_tokens = self._restore_batch(self.encode_ts(x_enc), batch)
        med_tokens = self.med_to_q(med_tokens)
        med_tokens = med_tokens + self.modality_embeddings[0].view(1, 1, -1)

        # ECG-image branch.
        image = self._get_image(inputs).to(device=x_enc.device)
        image_tokens = self.image_to_q(self.image_encoder(image))
        image_tokens = image_tokens + self.modality_embeddings[1].view(1, 1, -1)

        med_repr = self.med_pool(med_tokens)
        image_repr = self.image_pool(image_tokens)
        diagnosis_queries, morphology_queries = self._semantic_banks(
            med_tokens.device, med_tokens.dtype
        )

        queries, diagnostics = self.router(
            signal_memory=med_tokens,
            image_memory=image_tokens,
            diagnosis_queries=diagnosis_queries,
            morphology_queries=morphology_queries,
            signal_global=med_repr,
            image_global=image_repr,
        )
        query_repr = self.query_pool(queries)

        # LLM contextualization of diagnosis-specific fused evidence.
        soft_queries = self.q_to_llm(queries)
        prompt_tokens, prompt_mask = self._encode_prompts(
            inputs,
            dtype=soft_queries.dtype,
        )

        llm_query_tokens = self._run_llm(
            prompt_tokens,
            prompt_mask,
            soft_queries,
        )
        sample_repr = self.llm_pool(llm_query_tokens)

        labels = inputs.get("labels") if self.training else None
        if self.use_biomedcoop:
            if self._bc_prototypes is None:
                self._build_class_prototypes()
            prototypes = self._bc_prototypes.to(sample_repr.device)
            base_logits = self.bc_head(sample_repr, prototypes, labels=labels)
            if self.n_classes <= 2:
                base_logits = base_logits[:, 1] - base_logits[:, 0]
        else:
            if self.standard_classifier is None:
                raise RuntimeError("Standard classifier was not initialized")
            base_logits = self.standard_classifier(sample_repr)
            if self.n_classes <= 2:
                base_logits = base_logits.squeeze(-1)

        if self.n_classes > 2:
            query_logits = self.direct_query_score(queries).squeeze(-1)
            mix = torch.sigmoid(self.direct_logit_weight)
            logits = (1.0 - mix) * base_logits + mix * query_logits
        else:
            logits = base_logits

        signal_evidence = diagnostics["signal_evidence"]
        image_evidence = diagnostics["image_evidence"]
        adapted_queries = diagnostics["adapted_queries"]
        structured_queries = diagnostics["structured_queries"]
        assert isinstance(signal_evidence, Tensor)
        assert isinstance(image_evidence, Tensor)
        assert isinstance(adapted_queries, Tensor)
        assert isinstance(structured_queries, Tensor)
        self._set_auxiliary_losses(
            med_repr, image_repr, query_repr, queries,
            adapted_queries, structured_queries,
            signal_evidence, image_evidence, labels,
        )

        self.last_morphology_attention = diagnostics["morphology_attention"].detach()  # type: ignore[union-attr]
        self.last_adaptation_gate = diagnostics["adaptation_gate"].detach()  # type: ignore[union-attr]
        self.last_routing_gates = [x.detach() for x in diagnostics["routing_gates"]]  # type: ignore[arg-type]
        self.last_uncertainty = diagnostics["uncertainty"].detach()  # type: ignore[union-attr]
        self.last_signal_attention = [x.detach() for x in diagnostics["signal_attention"]]  # type: ignore[arg-type]
        self.last_image_attention = [x.detach() for x in diagnostics["image_attention"]]  # type: ignore[arg-type]
        return logits

    def predict(self, inputs: dict[str, Any]) -> Tensor:
        return self.forward(inputs)

    def get_auxiliary_losses(self) -> dict[str, Tensor]:
        return self._auxiliary_losses

    def train(self, mode: bool = True) -> "MedTsLLMConvNeXtHierarchicalSemanticRouter":
        super().train(mode)
        if self.llm_enabled and not bool(getattr(self, "lora_enabled", False)):
            self.llm.eval()
        self.image_encoder.train(mode)
        return self

    def state_dict(self) -> OrderedDict[str, Tensor]:
        state = nn.Module.state_dict(self)
        trainable = {name for name, p in self.named_parameters() if p.requires_grad}
        for key in list(state.keys()):
            if key == "word_embeddings":
                del state[key]
            elif key.startswith("llm.") and key not in trainable:
                del state[key]
            elif (
                (key.startswith("image_encoder.backbone.") or key.startswith("image_encoder.features."))
                and not self.save_frozen_convnext
                and key not in trainable
            ):
                del state[key]
        return OrderedDict(state)
