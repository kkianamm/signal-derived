from pathlib import Path

import torch
import torch.nn as nn
import timm

from .medtsllm import MedTsLLM


# ================================================================
# Lightweight BLIP-2-inspired Q-Former block
# ================================================================

class QFormerBlock(nn.Module):
    """
    Query tokens:
        self-attention
        -> cross-attention to frozen image features
        -> feed-forward network
    """

    def __init__(
        self,
        dim=512,
        num_heads=8,
        mlp_ratio=4.0,
        dropout=0.1,
    ):
        super().__init__()

        self.norm_self = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.norm_cross_q = nn.LayerNorm(dim)
        self.norm_cross_kv = nn.LayerNorm(dim)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        hidden_dim = int(dim * mlp_ratio)

        self.norm_ffn = nn.LayerNorm(dim)

        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, queries, visual_tokens):

        # --------------------------------------------------------
        # 1. Query self-attention
        # --------------------------------------------------------
        q = self.norm_self(queries)

        attn_out, _ = self.self_attn(
            q,
            q,
            q,
            need_weights=False,
        )

        queries = queries + attn_out

        # --------------------------------------------------------
        # 2. Cross-attention:
        #    queries attend to ConvNeXt spatial features
        # --------------------------------------------------------
        q = self.norm_cross_q(queries)
        kv = self.norm_cross_kv(visual_tokens)

        cross_out, _ = self.cross_attn(
            q,
            kv,
            kv,
            need_weights=False,
        )

        queries = queries + cross_out

        # --------------------------------------------------------
        # 3. FFN
        # --------------------------------------------------------
        queries = queries + self.ffn(
            self.norm_ffn(queries)
        )

        return queries


class ECGQFormer(nn.Module):
    """
    Converts ConvNeXt spatial feature tokens into a fixed number
    of learned visual query tokens.
    """

    def __init__(
        self,
        visual_dim,
        q_dim=512,
        num_queries=16,
        depth=4,
        num_heads=8,
        dropout=0.1,
    ):
        super().__init__()

        self.num_queries = num_queries
        self.q_dim = q_dim

        # ConvNeXt channel dimension -> Q-Former dimension
        self.visual_projection = nn.Linear(
            visual_dim,
            q_dim,
        )

        # Learnable BLIP-2-style query embeddings
        self.query_tokens = nn.Parameter(
            torch.randn(
                1,
                num_queries,
                q_dim,
            ) * 0.02
        )

        self.blocks = nn.ModuleList([
            QFormerBlock(
                dim=q_dim,
                num_heads=num_heads,
                dropout=dropout,
            )
            for _ in range(depth)
        ])

        self.final_norm = nn.LayerNorm(q_dim)

    def forward(self, visual_tokens):

        # visual_tokens:
        # [B, Nv, visual_dim]

        visual_tokens = self.visual_projection(
            visual_tokens
        )

        B = visual_tokens.size(0)

        queries = self.query_tokens.expand(
            B,
            -1,
            -1,
        )

        for block in self.blocks:
            queries = block(
                queries,
                visual_tokens,
            )

        return self.final_norm(queries)


# ================================================================
# MedTsLLM + ConvNeXt + Q-Former
# ================================================================

class MedTsLLMQFormer(MedTsLLM):

    supported_tasks = ["classification"]

    def __init__(self, config, dataset):

        # --------------------------------------------------------
        # ORIGINAL MedTsLLM
        # --------------------------------------------------------
        super().__init__(config, dataset)

        if self.task != "classification":
            raise ValueError(
                "MedTsLLMQFormer currently supports "
                "classification only."
            )

        qcfg = config.models.qformer

        self.qformer_num_queries = int(
            qcfg.get("num_queries", 16)
        )

        self.qformer_dim = int(
            qcfg.get("dim", 512)
        )

        self.qformer_depth = int(
            qcfg.get("depth", 4)
        )

        self.qformer_heads = int(
            qcfg.get("heads", 8)
        )

        self.qformer_dropout = float(
            qcfg.get("dropout", 0.1)
        )

        # ========================================================
        # ConvNeXt
        # ========================================================

        backbone = qcfg.get(
            "backbone",
            "convnext_tiny",
        )

        self.image_encoder = timm.create_model(
            backbone,
            pretrained=False,
            num_classes=self.n_classes,
        )

        repo_root = Path(__file__).resolve().parent.parent

        checkpoint_path = Path(
            qcfg.checkpoint
        )

        if not checkpoint_path.is_absolute():
            checkpoint_path = (
                repo_root / checkpoint_path
            )

        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"ConvNeXt checkpoint not found: "
                f"{checkpoint_path}"
            )

        print(
            "[QFormer] Loading image checkpoint:",
            checkpoint_path,
        )

        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
        )

        if "model_state_dict" not in checkpoint:
            raise KeyError(
                "Expected model_state_dict in "
                "ConvNeXt checkpoint."
            )

        self.image_encoder.load_state_dict(
            checkpoint["model_state_dict"],
            strict=True,
        )

        # ConvNeXt-tiny -> typically 768 channels.
        # We obtain it programmatically.
        self.visual_dim = int(
            self.image_encoder.num_features
        )

        # ========================================================
        # Freeze image encoder
        # ========================================================

        for param in self.image_encoder.parameters():
            param.requires_grad = False

        self.image_encoder.eval()

        # ========================================================
        # Q-Former
        # ========================================================

        self.qformer = ECGQFormer(
            visual_dim=self.visual_dim,
            q_dim=self.qformer_dim,
            num_queries=self.qformer_num_queries,
            depth=self.qformer_depth,
            num_heads=self.qformer_heads,
            dropout=self.qformer_dropout,
        )

        # ========================================================
        # BLIP-2-style projection:
        # Q-Former dimension -> MedTsLLM LLM dimension
        # ========================================================

        self.visual_to_llm = nn.Sequential(
            nn.Linear(
                self.qformer_dim,
                self.d_llm,
            ),
            nn.LayerNorm(self.d_llm),
        )

        print(
            f"[QFormer] visual_dim={self.visual_dim}"
        )

        print(
            f"[QFormer] q_dim={self.qformer_dim}"
        )

        print(
            f"[QFormer] queries="
            f"{self.qformer_num_queries}"
        )

        print(
            f"[QFormer] depth="
            f"{self.qformer_depth}"
        )

        print(
            f"[QFormer] heads="
            f"{self.qformer_heads}"
        )

        q_params = sum(
            p.numel()
            for p in self.qformer.parameters()
            if p.requires_grad
        )

        proj_params = sum(
            p.numel()
            for p in self.visual_to_llm.parameters()
            if p.requires_grad
        )

        image_params = sum(
            p.numel()
            for p in self.image_encoder.parameters()
        )

        print(
            f"[QFormer] frozen ConvNeXt params: "
            f"{image_params:,}"
        )

        print(
            f"[QFormer] trainable Q-Former params: "
            f"{q_params:,}"
        )

        print(
            f"[QFormer] trainable visual projection params: "
            f"{proj_params:,}"
        )

    # ============================================================
    # Keep frozen image encoder in eval mode
    # ============================================================

    def train(self, mode=True):

        super().train(mode)

        if hasattr(self, "image_encoder"):
            self.image_encoder.eval()

        return self

    # ============================================================
    # ConvNeXt -> spatial tokens
    # ============================================================

    def encode_image_tokens(self, image):

        # image:
        # [B, 3, H, W]

        with torch.no_grad():

            features = self.image_encoder.forward_features(
                image
            )

        # timm ConvNeXt generally gives:
        #
        # [B, C, H, W]
        #
        # but keep this robust to channel-last variants.

        if features.ndim == 4:

            if features.shape[1] == self.visual_dim:

                # B,C,H,W -> B,H,W,C
                features = features.permute(
                    0, 2, 3, 1
                ).contiguous()

            elif features.shape[-1] != self.visual_dim:

                raise RuntimeError(
                    "Unexpected ConvNeXt feature shape: "
                    f"{features.shape}, "
                    f"expected channel dim "
                    f"{self.visual_dim}"
                )

            # B,H,W,C -> B,N,C
            features = features.reshape(
                features.size(0),
                -1,
                self.visual_dim,
            )

        elif features.ndim == 3:

            # already B,N,C
            if features.shape[-1] != self.visual_dim:
                raise RuntimeError(
                    "Unexpected visual token dimension: "
                    f"{features.shape}"
                )

        else:
            raise RuntimeError(
                "Unexpected ConvNeXt forward_features "
                f"shape: {features.shape}"
            )

        # --------------------------------------------------------
        # Query the visual features
        # --------------------------------------------------------

        query_features = self.qformer(
            features
        )

        # B,M,q_dim -> B,M,d_llm
        visual_tokens = self.visual_to_llm(
            query_features
        )

        return visual_tokens

    # ============================================================
    # MedTsLLM prediction with Q-Former visual tokens
    # ============================================================

    def predict(self, inputs):

        if "image" not in inputs:
            raise KeyError(
                "MedTsLLMQFormer expected "
                "inputs['image']."
            )

        x_raw = inputs["x_enc"]

        bs, seq_len, n_features = x_raw.size()

        if self.device is None:
            self.device = x_raw.device

        # ========================================================
        # 1. Build ORIGINAL MedTsLLM prompt
        # ========================================================

        prompts = self.build_prompt(inputs)

        if len(prompts[0]) > 0:

            prompt_enc = [
                [
                    self.encode_part(p)
                    for p in prompt
                ]
                for prompt in prompts
            ]

            prompt_enc = [
                torch.cat(enc, dim=1)
                for enc in prompt_enc
            ]

            max_len = max(
                enc.size(1)
                for enc in prompt_enc
            )

            padded = [
                self.pad_sequence(enc, max_len)
                for enc in prompt_enc
            ]

            prompt_enc = torch.cat(
                [p[0] for p in padded],
                dim=0,
            )

            prompt_mask = torch.cat(
                [p[1] for p in padded],
                dim=0,
            )

        else:

            prompt_enc = torch.zeros(
                (
                    bs,
                    0,
                    self.d_llm,
                ),
                device=x_raw.device,
                dtype=x_raw.dtype,
            )

            prompt_mask = torch.ones(
                (
                    bs,
                    0,
                ),
                device=x_raw.device,
                dtype=torch.long,
            )

        # ========================================================
        # 2. ORIGINAL MedTsLLM ECG encoding
        # ========================================================

        x_enc = self.encode_ts(
            x_raw
        )

        # ========================================================
        # 3. Image -> ConvNeXt -> Q-Former -> visual soft tokens
        # ========================================================

        visual_tokens = self.encode_image_tokens(
            inputs["image"]
        )

        # Match LLM embedding dtype/device
        visual_tokens = visual_tokens.to(
            device=x_enc.device,
            dtype=x_enc.dtype,
        )

        visual_mask = torch.ones(
            visual_tokens.size(0),
            visual_tokens.size(1),
            device=x_enc.device,
            dtype=torch.long,
        )

        # ========================================================
        # Special handling for independent modes
        # ========================================================

        if (
            self.covariate_mode == "independent"
            or self.covariate_mode == "merge-end"
        ):

            prompt_enc = prompt_enc.repeat_interleave(
                n_features,
                dim=0,
            )

            prompt_mask = prompt_mask.repeat_interleave(
                n_features,
                dim=0,
            )

            visual_tokens = visual_tokens.repeat_interleave(
                n_features,
                dim=0,
            )

            visual_mask = visual_mask.repeat_interleave(
                n_features,
                dim=0,
            )

        # ========================================================
        # 4. Feed visual soft prompts into LLM
        # ========================================================

        if self.llm.config.is_encoder_decoder:

            # ----------------------------------------------------
            # Flan-T5 style:
            #
            # encoder:
            #   text prompt + Q-Former visual tokens
            #
            # decoder:
            #   ECG tokens
            # ----------------------------------------------------

            encoder_embeds = torch.cat(
                [
                    prompt_enc,
                    visual_tokens,
                ],
                dim=1,
            )

            encoder_mask = torch.cat(
                [
                    prompt_mask,
                    visual_mask,
                ],
                dim=1,
            )

            dec_out = self.llm(
                inputs_embeds=encoder_embeds,
                attention_mask=encoder_mask,
                decoder_inputs_embeds=x_enc,
            ).last_hidden_state

        else:

            # ----------------------------------------------------
            # Decoder-only style:
            #
            # text -> visual queries -> ECG tokens
            # ----------------------------------------------------

            enc = torch.cat(
                [
                    prompt_enc,
                    visual_tokens,
                    x_enc,
                ],
                dim=1,
            )

            ts_mask = torch.ones(
                x_enc.size(0),
                x_enc.size(1),
                device=x_enc.device,
                dtype=torch.long,
            )

            combined_mask = torch.cat(
                [
                    prompt_mask,
                    visual_mask,
                    ts_mask,
                ],
                dim=1,
            )

            dec_out = self.llm(
                inputs_embeds=enc,
                attention_mask=combined_mask,
            ).last_hidden_state

            # Keep ECG positions only, just like original
            # MedTsLLM.
            dec_out = dec_out[
                :,
                -self.n_patches:,
                :
            ]

        # Encoder-decoder output already corresponds
        # to ECG decoder tokens.

        dec_out = dec_out.to(
            x_enc.dtype
        )

        # ========================================================
        # 5. Keep ORIGINAL MedTsLLM classification head
        # ========================================================

        if (
            self.task == "classification"
            and self.use_biomedcoop
        ):

            return self._biomedcoop_classify(
                dec_out,
                bs,
                n_features,
                inputs.get("labels"),
            )

        if self.embedding_downsample_mode == "truncate":

            dec_out = dec_out[
                :,
                :,
                :self.d_ff,
            ]

        elif self.embedding_downsample_mode == "linear":

            dec_out = self.embedding_downsample_layer(
                dec_out
            )

        elif self.embedding_downsample_mode == "average":

            dec_out = dec_out.reshape(
                bs,
                self.n_patches,
                self.d_ff,
                -1,
            )

            dec_out = dec_out.mean(
                dim=-1
            )

        else:

            raise ValueError(
                "Unknown embedding_downsample_mode: "
                f"{self.embedding_downsample_mode}"
            )

        # B,d_ff,n_patches
        dec_out = dec_out.permute(
            0,
            2,
            1,
        ).contiguous()

        # ORIGINAL MedTsLLM output head
        dec_out = self.output_projection(
            dec_out
        )

        if self.covariate_mode == "independent":

            dec_out = dec_out.view(
                bs,
                self.n_features,
                self.n_outputs_per_step,
            ).mean(dim=1)

        elif self.covariate_mode == "merge-end":

            dec_out = dec_out.view(
                bs,
                self.n_features
                * self.n_outputs_per_step,
            )

            dec_out = self.feature_weighting(
                dec_out
            )

        else:

            dec_out = dec_out.view(
                bs,
                self.n_outputs_per_step,
            )

        if self.n_outputs_per_step == 1:
            dec_out = dec_out.squeeze(-1)

        return dec_out
