import torch
import torch.nn as nn

from .medtsllm_image_fusion import MedTsLLMImageFusion


# ================================================================
# Q-FORMER BLOCK
# ================================================================

class QFormerBlock(nn.Module):
    """
    Lightweight BLIP-2-inspired Q-Former block.

    1. Query self-attention
    2. Query -> visual-token cross-attention
    3. Feed-forward network

    The visual encoder is frozen.
    Only the Q-Former is trainable.
    """

    def __init__(
        self,
        dim=512,
        num_heads=8,
        mlp_ratio=4.0,
        dropout=0.1,
    ):
        super().__init__()

        # --------------------------------------------------------
        # Query self-attention
        # --------------------------------------------------------
        self.self_norm = nn.LayerNorm(dim)

        self.self_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # --------------------------------------------------------
        # Query -> image cross-attention
        # --------------------------------------------------------
        self.cross_q_norm = nn.LayerNorm(dim)
        self.cross_kv_norm = nn.LayerNorm(dim)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # --------------------------------------------------------
        # Feed-forward network
        # --------------------------------------------------------
        hidden_dim = int(dim * mlp_ratio)

        self.ffn_norm = nn.LayerNorm(dim)

        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, queries, visual_tokens):

        # ========================================================
        # 1. Query self-attention
        # ========================================================

        q = self.self_norm(queries)

        self_out, _ = self.self_attn(
            q,
            q,
            q,
            need_weights=False,
        )

        queries = queries + self_out

        # ========================================================
        # 2. Cross-attention
        #
        # Queries ask:
        # "Which parts of the ECG image are useful?"
        # ========================================================

        q = self.cross_q_norm(queries)
        kv = self.cross_kv_norm(visual_tokens)

        cross_out, _ = self.cross_attn(
            query=q,
            key=kv,
            value=kv,
            need_weights=False,
        )

        queries = queries + cross_out

        # ========================================================
        # 3. FFN
        # ========================================================

        queries = queries + self.ffn(
            self.ffn_norm(queries)
        )

        return queries


# ================================================================
# ECG Q-FORMER
# ================================================================

class ECGQFormer(nn.Module):
    """
    Input:
        ConvNeXt spatial visual tokens
        [B, Nv, visual_dim]

    Output:
        learned query tokens
        [B, M, q_dim]
    """

    def __init__(
        self,
        visual_dim,
        q_dim=512,
        num_queries=16,
        depth=2,
        num_heads=8,
        mlp_ratio=4.0,
        dropout=0.1,
    ):
        super().__init__()

        self.visual_dim = visual_dim
        self.q_dim = q_dim
        self.num_queries = num_queries

        # --------------------------------------------------------
        # ConvNeXt feature dimension -> Q-Former dimension
        # --------------------------------------------------------

        self.visual_projection = nn.Sequential(
            nn.LayerNorm(visual_dim),
            nn.Linear(
                visual_dim,
                q_dim,
            ),
        )

        # --------------------------------------------------------
        # Learned queries
        # --------------------------------------------------------

        self.query_tokens = nn.Parameter(
            torch.randn(
                1,
                num_queries,
                q_dim,
            ) * 0.02
        )

        # --------------------------------------------------------
        # Q-Former transformer blocks
        # --------------------------------------------------------

        self.blocks = nn.ModuleList([
            QFormerBlock(
                dim=q_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
            )
            for _ in range(depth)
        ])

        self.final_norm = nn.LayerNorm(q_dim)

    def forward(self, visual_tokens):

        # --------------------------------------------------------
        # Project visual tokens
        # --------------------------------------------------------

        visual_tokens = self.visual_projection(
            visual_tokens
        )

        B = visual_tokens.size(0)

        # --------------------------------------------------------
        # Same learned query bank for every sample
        # --------------------------------------------------------

        queries = self.query_tokens.expand(
            B,
            -1,
            -1,
        )

        # --------------------------------------------------------
        # Query visual ECG representation
        # --------------------------------------------------------

        for block in self.blocks:
            queries = block(
                queries,
                visual_tokens,
            )

        queries = self.final_norm(
            queries
        )

        return queries


# ================================================================
# MEDTSLLM + CONVNEXT + Q-FORMER LATE FUSION
# ================================================================

class MedTsLLMQFormerFusion(MedTsLLMImageFusion):
    """
    Stable Q-Former fusion model.

    SIGNAL PATH:
        ECG
          -> original MedTsLLM
          -> signal logits/features

    IMAGE PATH:
        ECG image
          -> frozen ConvNeXt
          -> spatial visual tokens
          -> lightweight Q-Former
          -> pooled query representation

    VISUAL REFINEMENT:
        original ConvNeXt pooled representation
          +
        gated Q-Former residual

    FINAL:
        MedTsLLM signal representation
          +
        refined visual representation
          -> same fusion head as the successful
             MedTsLLM + ConvNeXt experiment.

    IMPORTANT:
        Q-Former is NOT inserted directly into the LLM.
    """

    supported_tasks = ["classification"]

    def __init__(self, config, dataset):

        # ========================================================
        # Build the SUCCESSFUL MedTsLLM + ConvNeXt model first
        #
        # This gives us:
        #   self.image_encoder
        #   self.image_projection
        #   self.signal_norm
        #   self.fusion_head
        #
        # ConvNeXt checkpoint is also loaded here.
        # ========================================================

        super().__init__(
            config,
            dataset,
        )

        if self.task != "classification":
            raise ValueError(
                "MedTsLLMQFormerFusion supports "
                "classification only."
            )

        qcfg = config.models.qformer_fusion

        self.q_dim = int(
            qcfg.get("dim", 512)
        )

        self.num_queries = int(
            qcfg.get("num_queries", 16)
        )

        self.q_depth = int(
            qcfg.get("depth", 2)
        )

        self.q_heads = int(
            qcfg.get("heads", 8)
        )

        self.q_dropout = float(
            qcfg.get("dropout", 0.1)
        )

        self.q_mlp_ratio = float(
            qcfg.get("mlp_ratio", 4.0)
        )

        self.use_global_residual = bool(
            qcfg.get(
                "use_global_residual",
                True,
            )
        )

        # ========================================================
        # Q-Former
        # ========================================================

        self.qformer = ECGQFormer(
            visual_dim=self.image_dim,
            q_dim=self.q_dim,
            num_queries=self.num_queries,
            depth=self.q_depth,
            num_heads=self.q_heads,
            mlp_ratio=self.q_mlp_ratio,
            dropout=self.q_dropout,
        )

        # ========================================================
        # Q-Former output -> ConvNeXt feature space
        #
        # Parent image_projection expects image_dim.
        # ========================================================

        self.q_to_image = nn.Sequential(
            nn.LayerNorm(self.q_dim),
            nn.Linear(
                self.q_dim,
                self.image_dim,
            ),
        )

        # ========================================================
        # GATED RESIDUAL
        #
        # Direct Q-Former replacement starts from random queries.
        #
        # Instead:
        #
        # refined =
        #     pretrained_global_feature
        #     +
        #     gate * qformer_feature
        #
        # gate starts very small:
        # sigmoid(-3) ~= 0.047
        #
        # This lets the model start close to your already
        # successful ConvNeXt representation instead of destroying
        # it at initialization.
        # ========================================================

        gate_init = float(
            qcfg.get(
                "gate_init",
                -3.0,
            )
        )

        self.q_gate_logit = nn.Parameter(
            torch.tensor(
                gate_init,
                dtype=torch.float32,
            )
        )

        # ========================================================
        # Logging
        # ========================================================

        q_params = sum(
            p.numel()
            for p in self.qformer.parameters()
            if p.requires_grad
        )

        q_projection_params = sum(
            p.numel()
            for p in self.q_to_image.parameters()
            if p.requires_grad
        )

        initial_gate = torch.sigmoid(
            self.q_gate_logit.detach()
        ).item()

        print(
            "\n========== Q-FORMER FUSION =========="
        )

        print(
            f"ConvNeXt visual dim: {self.image_dim}"
        )

        print(
            f"Q-Former dim: {self.q_dim}"
        )

        print(
            f"Number of queries: {self.num_queries}"
        )

        print(
            f"Q-Former depth: {self.q_depth}"
        )

        print(
            f"Q-Former heads: {self.q_heads}"
        )

        print(
            f"Trainable Q-Former params: "
            f"{q_params:,}"
        )

        print(
            f"Trainable Q projection params: "
            f"{q_projection_params:,}"
        )

        print(
            f"Use global residual: "
            f"{self.use_global_residual}"
        )

        print(
            f"Initial Q-Former gate: "
            f"{initial_gate:.6f}"
        )

        print(
            "=====================================\n"
        )

    # ============================================================
    # CONVNEXT SPATIAL TOKENS
    # ============================================================

    def _spatial_to_tokens(
        self,
        feature_map,
    ):
        """
        Convert ConvNeXt feature map:

            [B, C, H, W]

        into:

            [B, H*W, C]

        Also supports channel-last ConvNeXt implementations.
        """

        if feature_map.ndim != 4:
            raise RuntimeError(
                "Expected 4-D ConvNeXt feature map, "
                f"got {feature_map.shape}"
            )

        # --------------------------------------------------------
        # NCHW
        # --------------------------------------------------------

        if feature_map.shape[1] == self.image_dim:

            tokens = (
                feature_map
                .flatten(2)
                .transpose(1, 2)
                .contiguous()
            )

        # --------------------------------------------------------
        # NHWC
        # --------------------------------------------------------

        elif feature_map.shape[-1] == self.image_dim:

            B, H, W, C = feature_map.shape

            tokens = feature_map.reshape(
                B,
                H * W,
                C,
            )

        else:

            raise RuntimeError(
                "Could not identify ConvNeXt channel "
                f"dimension. feature_map={feature_map.shape}, "
                f"expected image_dim={self.image_dim}"
            )

        return tokens

    # ============================================================
    # OVERRIDE IMAGE REPRESENTATION
    #
    # Parent MedTsLLMImageFusion.predict() automatically calls
    # self.encode_image().
    #
    # Therefore we do NOT need to rewrite the MedTsLLM signal
    # pathway or fusion classifier.
    # ============================================================

    def encode_image(
        self,
        image,
    ):

        # ========================================================
        # 1. Frozen ConvNeXt spatial features
        # ========================================================

        with torch.no_grad():

            feature_map = (
                self.image_encoder
                .forward_features(image)
            )

            # This is the original ConvNeXt pooled representation
            # used by the successful simple-fusion model.
            global_feature = (
                self.image_encoder
                .forward_head(
                    feature_map,
                    pre_logits=True,
                )
            )

        # global_feature:
        # [B, image_dim]

        if global_feature.ndim != 2:

            global_feature = (
                global_feature.flatten(1)
            )

        # ========================================================
        # 2. Spatial map -> visual tokens
        # ========================================================

        visual_tokens = self._spatial_to_tokens(
            feature_map
        )

        # Example for ConvNeXt-tiny at 224x224:
        #
        # [B, 768, 7, 7]
        #
        # becomes:
        #
        # [B, 49, 768]

        # ========================================================
        # 3. Q-Former
        # ========================================================

        query_tokens = self.qformer(
            visual_tokens
        )

        # [B, M, q_dim]

        # ========================================================
        # 4. Pool query outputs
        # ========================================================

        query_feature = query_tokens.mean(
            dim=1
        )

        # [B, q_dim]

        # ========================================================
        # 5. Put Q-Former feature into ConvNeXt feature space
        # ========================================================

        query_feature = self.q_to_image(
            query_feature
        )

        # [B, image_dim]

        # ========================================================
        # 6. Query-guided residual visual refinement
        # ========================================================

        if self.use_global_residual:

            gate = torch.sigmoid(
                self.q_gate_logit
            )

            refined_feature = (
                global_feature
                +
                gate * query_feature
            )

        else:

            # Pure Q-Former ablation:
            # remove original global pooled ConvNeXt feature.
            refined_feature = query_feature

        # ========================================================
        # 7. SAME image projection used by simple fusion model
        # ========================================================

        image_feature = self.image_projection(
            refined_feature
        )

        return image_feature
