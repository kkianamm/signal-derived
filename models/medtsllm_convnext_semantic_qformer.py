"""MedTsLLM + ConvNeXt spatial tokens + semantic Q-Former.

The model reuses the ECG image representation from the current signal-derived
pipeline. Put the already-generated image tensor in inputs['x_image'] (or
'image'/'images') with shape [B,1,H,W] or [B,3,H,W].
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
    raise ImportError("torchvision with ConvNeXt support is required.") from exc

from .medtsllm import MedTsLLM
from .semantic_qformer_components import (
    AttentionPool, GatedQueryResidual, SemanticQueryFormer, cfg_get,
    query_diversity_loss, supervised_contrastive_alignment,
)


PTBXL_CLASS_DESCRIPTIONS = (
    "a normal electrocardiogram with normal rhythm and normal cardiac morphology",
    "an electrocardiogram with morphology associated with myocardial infarction",
    "an electrocardiogram showing ST segment or T wave abnormalities",
    "an electrocardiogram showing a cardiac conduction disturbance",
    "an electrocardiogram showing ventricular or atrial hypertrophy patterns",
)


class ConvNeXtSpatialTokenEncoder(nn.Module):
    """Return ConvNeXt feature-map locations BEFORE global pooling."""
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
            raise ValueError(f"Unknown ConvNeXt variant {variant!r}.")
        builder, default_weights = self._BUILDERS[variant]
        self.backbone = builder(weights=default_weights if pretrained else None)
        self.features = self.backbone.features
        last_norm = self.backbone.classifier[0]
        ns = getattr(last_norm, "normalized_shape", None)
        if ns is None:
            raise RuntimeError("Could not infer ConvNeXt output dimension.")
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
            raise ValueError(f"ConvNeXt image must be [B,C,H,W], got {tuple(image.shape)}.")
        if image.size(1) == 1:
            image = image.repeat(1, 3, 1, 1)
        elif image.size(1) != 3:
            raise ValueError("ConvNeXt expects 1-channel or 3-channel images.")
        fmap = self.features(image)
        return fmap.flatten(2).transpose(1, 2)  # [B,H*W,C]


class MedTsLLMConvNeXtSemanticQFormer(MedTsLLM):
    """Proposed PTB-XL architecture.

    MedTsLLM tokens ---- projection --\\
                                      +--> concatenated memory -> semantic QF
    ConvNeXt spatial --- projection --/                           |
                                                               gated residual
                                                                    |
                                                              Q -> LLM -> head
    """
    supported_tasks = ["classification"]
    supported_modes = ["multivariate"]

    def __init__(self, config: Any, dataset: Any) -> None:
        if config.task != "classification":
            raise ValueError("This model supports classification only.")
        super().__init__(config, dataset)

        section = cfg_get(self.model_config, "convnext_semantic_qformer", None)
        if section is None:
            raise ValueError(
                "Missing [models.medtsllm.convnext_semantic_qformer] in the TOML config."
            )
        if not self.llm_enabled:
            raise ValueError("The semantic Q-Former model requires the LLM.")

        self.q_dim = int(cfg_get(section, "q_dim", 512))
        self.q_queries = int(cfg_get(section, "q_queries", 32))
        self.semantic_count = int(cfg_get(section, "semantic_queries", self.n_classes))
        q_depth = int(cfg_get(section, "q_depth", 4))
        q_heads = int(cfg_get(section, "q_heads", 8))
        dropout = float(cfg_get(section, "dropout", 0.1))

        if self.semantic_count != self.n_classes:
            raise ValueError(
                f"semantic_queries should equal n_classes ({self.n_classes}); got {self.semantic_count}."
            )
        if self.q_queries < self.semantic_count:
            raise ValueError("q_queries must be >= semantic_queries.")

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

        self.class_query_projection = nn.Sequential(
            nn.Linear(self.d_llm, self.q_dim), nn.GELU(),
            nn.Linear(self.q_dim, self.q_dim), nn.LayerNorm(self.q_dim),
        )
        self.semantic_query_delta = nn.Parameter(torch.zeros(self.semantic_count, self.q_dim))
        self.register_buffer(
            "_semantic_text_embeddings", torch.empty(0, self.d_llm), persistent=True
        )

        descriptions = cfg_get(section, "class_descriptions", None)
        if descriptions is None:
            descriptions = PTBXL_CLASS_DESCRIPTIONS
        self.class_descriptions = tuple(str(x) for x in descriptions)
        if len(self.class_descriptions) != self.semantic_count:
            raise ValueError("class_descriptions length must equal semantic_queries.")

        self.qformer = SemanticQueryFormer(
            dim=self.q_dim,
            total_queries=self.q_queries,
            semantic_queries=self.semantic_count,
            depth=q_depth,
            heads=q_heads,
            dropout=dropout,
        )
        self.med_pool = AttentionPool(self.q_dim)
        self.image_pool = AttentionPool(self.q_dim)
        self.query_pool = AttentionPool(self.q_dim)
        self.query_residual = GatedQueryResidual(self.q_dim, dropout)
        self.q_to_llm = nn.Sequential(nn.Linear(self.q_dim, self.d_llm), nn.LayerNorm(self.d_llm))
        self.llm_pool = AttentionPool(self.d_llm)

        output_dim = self.n_classes if self.n_classes > 2 else 1
        self.med_aux_head = nn.Linear(self.q_dim, output_dim)
        self.image_aux_head = nn.Linear(self.q_dim, output_dim)
        self.query_aux_head = nn.Linear(self.q_dim, output_dim)

        # Older/alternate MedTsLLM branches may not define BioMedCoOp.
        self.use_biomedcoop = bool(getattr(self, "use_biomedcoop", False))
        self.standard_classifier: Optional[nn.Linear] = None
        if not self.use_biomedcoop:
            self.standard_classifier = nn.Linear(self.d_llm, output_dim)

        loss_cfg = cfg_get(section, "loss", {})
        self.loss_weights = {
            "med_ce": float(cfg_get(loss_cfg, "med_ce", 0.10)),
            "image_ce": float(cfg_get(loss_cfg, "image_ce", 0.10)),
            "query_ce": float(cfg_get(loss_cfg, "query_ce", 0.10)),
            "alignment": float(cfg_get(loss_cfg, "alignment", 0.05)),
            "query_diversity": float(cfg_get(loss_cfg, "query_diversity", 0.01)),
            "biomedcoop": float(cfg_get(loss_cfg, "biomedcoop", 1.0)),
        }
        self.alignment_temperature = float(cfg_get(loss_cfg, "alignment_temperature", 0.10))
        self._auxiliary_losses: dict[str, Tensor] = {}
        self.aux_loss: Optional[Tensor] = None
        self.last_cross_attention: Optional[list[Tensor]] = None
        self.last_query_gate: Optional[Tensor] = None

    def _restore_batch(self, tokens: Tensor, batch_size: int) -> Tensor:
        if tokens.shape[0] == batch_size:
            return tokens
        if tokens.shape[0] % batch_size:
            raise RuntimeError(
                f"MedTsLLM token batch cannot be restored: tokens={tokens.shape[0]}, batch={batch_size}."
            )
        channels = tokens.shape[0] // batch_size
        return tokens.view(batch_size, channels, tokens.shape[1], tokens.shape[2]).mean(1)

    def _get_image(self, inputs: dict[str, Any]) -> Tensor:
        for key in ("x_image", "image", "images"):
            value = inputs.get(key)
            if value is not None:
                return value
        raise KeyError(
            "No image tensor found. Reuse the image generated by your current ConvNeXt pipeline "
            "and place it in inputs['x_image'] (or 'image'/'images')."
        )

    def _encode_prompts(self, inputs: dict[str, Any], dtype: torch.dtype) -> Tensor:
        x_enc = inputs["x_enc"]
        batch = x_enc.size(0)
        prompts = self.build_prompt(inputs)
        if not prompts or not prompts[0]:
            return torch.zeros(batch, 0, self.d_llm, device=x_enc.device, dtype=dtype)
        encoded = [[self.encode_part(part) for part in prompt] for prompt in prompts]
        encoded = [torch.cat(parts, dim=1) for parts in encoded]
        max_len = max(item.size(1) for item in encoded)
        encoded = [self.pad_sequence(item, max_len) for item in encoded]
        return torch.cat(encoded, dim=0).to(device=x_enc.device, dtype=dtype)

    @torch.no_grad()
    def _build_semantic_text_embeddings(self, device: torch.device) -> None:
        if self._semantic_text_embeddings.numel() != 0:
            return
        vectors = []
        for description in self.class_descriptions:
            token_embeddings = self.encode_part(description)
            if token_embeddings.ndim != 3:
                raise RuntimeError(
                    f"encode_part(description) must return [1,L,d_llm], got {tuple(token_embeddings.shape)}."
                )
            vectors.append(token_embeddings.mean(dim=1).squeeze(0))
        self._semantic_text_embeddings = torch.stack(vectors, dim=0).detach().to(device=device)

    def _semantic_queries(self, device: torch.device, dtype: torch.dtype) -> Tensor:
        self._build_semantic_text_embeddings(device)
        base = self._semantic_text_embeddings.to(device=device, dtype=dtype)
        return self.class_query_projection(base) + self.semantic_query_delta.to(dtype=dtype)

    def _run_llm(self, prompt_tokens: Tensor, soft_queries: Tensor) -> Tensor:
        soft_queries = soft_queries.to(dtype=prompt_tokens.dtype)
        if self.llm.config.is_encoder_decoder:
            output = self.llm(
                inputs_embeds=prompt_tokens,
                decoder_inputs_embeds=soft_queries,
            ).last_hidden_state
            return output[:, -soft_queries.size(1):].to(soft_queries.dtype)
        llm_input = torch.cat([prompt_tokens, soft_queries], dim=1)
        output = self.llm(inputs_embeds=llm_input).last_hidden_state
        return output[:, -soft_queries.size(1):].to(soft_queries.dtype)

    def _branch_loss(self, logits: Tensor, labels: Tensor) -> Tensor:
        if self.n_classes > 2:
            return F.cross_entropy(logits, labels.long())
        return F.binary_cross_entropy_with_logits(logits.squeeze(-1), labels.to(logits.dtype))

    def _set_auxiliary_losses(
        self,
        med_repr: Tensor,
        image_repr: Tensor,
        query_repr: Tensor,
        queries: Tensor,
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
            "alignment": supervised_contrastive_alignment(
                med_repr, image_repr, labels, self.alignment_temperature
            ),
            "query_diversity": query_diversity_loss(queries),
            "biomedcoop": bc_aux,
        }
        if labels is not None:
            raw["med_ce"] = self._branch_loss(self.med_aux_head(med_repr), labels)
            raw["image_ce"] = self._branch_loss(self.image_aux_head(image_repr), labels)
            raw["query_ce"] = self._branch_loss(self.query_aux_head(query_repr), labels)
        weighted = {name: value * self.loss_weights[name] for name, value in raw.items()}
        weighted["total"] = torch.stack(tuple(weighted.values())).sum()
        self._auxiliary_losses = weighted
        self.aux_loss = weighted["total"]

    def forward(self, inputs: dict[str, Any]) -> Tensor:
        x_enc: Tensor = inputs["x_enc"]
        if x_enc.ndim == 2:
            x_enc = x_enc.unsqueeze(-1)
        if x_enc.ndim != 3:
            raise ValueError(f"x_enc must be [B,T,C], got {tuple(x_enc.shape)}.")
        if x_enc.size(-1) != self.n_features:
            raise ValueError(f"Expected {self.n_features} channels, got {x_enc.size(-1)}.")
        if self.device is None:
            self.device = x_enc.device
        batch = x_enc.size(0)

        # MedTsLLM waveform tokens.
        med_tokens = self._restore_batch(self.encode_ts(x_enc), batch)
        med_tokens = self.med_to_q(med_tokens)
        med_tokens = med_tokens + self.modality_embeddings[0].view(1, 1, -1)

        # ConvNeXt PRE-POOLING spatial tokens.
        image = self._get_image(inputs).to(device=x_enc.device)
        image_tokens = self.image_encoder(image)
        image_tokens = self.image_to_q(image_tokens)
        image_tokens = image_tokens + self.modality_embeddings[1].view(1, 1, -1)

        # No token-count interpolation: let cross-attention select evidence.
        memory = torch.cat([med_tokens, image_tokens], dim=1)

        semantic_queries = self._semantic_queries(memory.device, memory.dtype)
        queries, cross_attention = self.qformer(memory, semantic_queries=semantic_queries)

        med_repr = self.med_pool(med_tokens)
        image_repr = self.image_pool(image_tokens)
        queries, query_gate = self.query_residual(queries, med_repr)
        query_repr = self.query_pool(queries)

        soft_queries = self.q_to_llm(queries)
        prompt_tokens = self._encode_prompts(inputs, dtype=soft_queries.dtype)
        llm_query_tokens = self._run_llm(prompt_tokens, soft_queries)
        sample_repr = self.llm_pool(llm_query_tokens)

        labels = inputs.get("labels") if self.training else None
        if self.use_biomedcoop:
            if self._bc_prototypes is None:
                self._build_class_prototypes()
            prototypes = self._bc_prototypes.to(sample_repr.device)
            proto_logits = self.bc_head(sample_repr, prototypes, labels=labels)
            logits = proto_logits if self.n_classes > 2 else proto_logits[:, 1] - proto_logits[:, 0]
        else:
            if self.standard_classifier is None:
                raise RuntimeError("Standard classifier was not initialized.")
            logits = self.standard_classifier(sample_repr)
            if self.n_classes <= 2:
                logits = logits.squeeze(-1)

        self._set_auxiliary_losses(med_repr, image_repr, query_repr, queries, labels)
        self.last_cross_attention = [item.detach() for item in cross_attention]
        self.last_query_gate = query_gate.detach()
        return logits

    def predict(self, inputs: dict[str, Any]) -> Tensor:
        return self.forward(inputs)

    def get_auxiliary_losses(self) -> dict[str, Tensor]:
        return self._auxiliary_losses

    def train(self, mode: bool = True) -> "MedTsLLMConvNeXtSemanticQFormer":
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
