from pathlib import Path

import torch
import torch.nn as nn
import timm

from .medtsllm import MedTsLLM


class MedTsLLMImageFusion(MedTsLLM):
    """
    Simple MedTsLLM + ConvNeXt fusion baseline.

    Signal branch:
        ECG -> original MedTsLLM -> signal logits

    Image branch:
        rendered ECG -> pretrained ConvNeXt -> pooled feature

    Fusion:
        [MedTsLLM logits ; projected image feature]
            -> MLP
            -> final class logits

    ConvNeXt is frozen.

    Qwen is NOT used.
    Q-Former is NOT used.
    """

    supported_tasks = ["classification"]

    def __init__(self, config, dataset):
        # Build the ORIGINAL MedTsLLM first.
        super().__init__(config, dataset)

        if self.task != "classification":
            raise ValueError(
                "MedTsLLMImageFusion supports classification only."
            )

        fusion_cfg = config.models.image_fusion

        self.fusion_dim = int(
            fusion_cfg.get("fusion_dim", 512)
        )

        fusion_dropout = float(
            fusion_cfg.get("dropout", 0.1)
        )

        # =========================================================
        # IMAGE ENCODER
        # =========================================================

        self.image_encoder = timm.create_model(
            fusion_cfg.backbone,
            pretrained=False,
            num_classes=self.n_classes,
        )

        repo_root = Path(__file__).resolve().parent.parent
        checkpoint_path = Path(fusion_cfg.checkpoint)

        if not checkpoint_path.is_absolute():
            checkpoint_path = repo_root / checkpoint_path

        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"ConvNeXt checkpoint not found: "
                f"{checkpoint_path}"
            )

        print(
            f"[ImageFusion] loading ConvNeXt checkpoint: "
            f"{checkpoint_path}"
        )

        ckpt = torch.load(
            checkpoint_path,
            map_location="cpu",
        )

        if "model_state_dict" not in ckpt:
            raise KeyError(
                "Expected 'model_state_dict' in image "
                "classifier checkpoint."
            )

        # The image classifier checkpoint was trained with
        # num_classes=5, so load BEFORE removing classifier.
        self.image_encoder.load_state_dict(
            ckpt["model_state_dict"],
            strict=True,
        )

        checkpoint_classes = ckpt.get("class_names", None)

        if checkpoint_classes is not None:
            if len(checkpoint_classes) != self.n_classes:
                raise RuntimeError(
                    "ConvNeXt checkpoint class count does not "
                    "match PTB-XL class count."
                )

            print(
                f"[ImageFusion] ConvNeXt classes: "
                f"{checkpoint_classes}"
            )

        # =========================================================
        # REMOVE CONVNEXT CLASSIFIER
        #
        # After reset_classifier(0), timm returns the pooled
        # representation instead of 5-class logits.
        # =========================================================

        self.image_encoder.reset_classifier(0)

        self.image_dim = int(
            self.image_encoder.num_features
        )

        # =========================================================
        # FREEZE IMAGE ENCODER
        # =========================================================

        for param in self.image_encoder.parameters():
            param.requires_grad = False

        self.image_encoder.eval()

        print(
            f"[ImageFusion] image feature dimension = "
            f"{self.image_dim}"
        )

        # =========================================================
        # IMAGE PROJECTION
        # =========================================================

        self.image_projection = nn.Sequential(
            nn.LayerNorm(self.image_dim),
            nn.Linear(
                self.image_dim,
                self.fusion_dim,
            ),
            nn.GELU(),
            nn.Dropout(fusion_dropout),
        )

        # =========================================================
        # SIGNAL LOGIT NORMALIZATION
        #
        # We deliberately keep the original MedTsLLM prediction
        # pathway unchanged and use its K-dimensional logits.
        # =========================================================

        self.signal_norm = nn.LayerNorm(
            self.n_classes
        )

        # =========================================================
        # SIMPLE FUSION HEAD
        #
        # [K MedTsLLM logits + D image features] -> K classes
        # =========================================================

        self.fusion_head = nn.Sequential(
            nn.Linear(
                self.n_classes + self.fusion_dim,
                self.fusion_dim,
            ),
            nn.GELU(),
            nn.Dropout(fusion_dropout),

            nn.Linear(
                self.fusion_dim,
                self.n_classes,
            ),
        )

        trainable = sum(
            p.numel()
            for p in self.parameters()
            if p.requires_grad
        )

        image_params = sum(
            p.numel()
            for p in self.image_encoder.parameters()
        )

        print(
            f"[ImageFusion] ConvNeXt params frozen: "
            f"{image_params:,}"
        )

        print(
            f"[ImageFusion] total trainable params: "
            f"{trainable:,}"
        )

    def train(self, mode=True):
        """
        Keep ConvNeXt in evaluation mode even when the full
        MedTsLLM fusion model is put into train mode.
        """
        super().train(mode)

        if hasattr(self, "image_encoder"):
            self.image_encoder.eval()

        return self

    def encode_image(self, image):
        """
        image:
            [B, 3, H, W]

        returns:
            [B, fusion_dim]
        """

        # ConvNeXt remains completely frozen.
        with torch.no_grad():
            image_features = self.image_encoder(image)

        if image_features.ndim != 2:
            image_features = image_features.flatten(1)

        return self.image_projection(
            image_features
        )

    def predict(self, inputs):
        if "image" not in inputs:
            raise KeyError(
                "MedTsLLMImageFusion expected inputs['image']."
            )

        # =========================================================
        # ORIGINAL MedTsLLM prediction
        #
        # This calls MedTsLLM.predict(), NOT this method again.
        # Extra key `image` is simply ignored by the parent.
        # =========================================================

        signal_logits = super().predict(inputs)

        if signal_logits.ndim != 2:
            raise RuntimeError(
                "Expected MedTsLLM classification logits "
                f"[B, K], got {signal_logits.shape}"
            )

        # =========================================================
        # IMAGE FEATURE
        # =========================================================

        image = inputs["image"]

        image_features = self.encode_image(image)

        # Be safe under mixed precision.
        image_features = image_features.to(
            dtype=signal_logits.dtype
        )

        # =========================================================
        # FUSION
        # =========================================================

        signal_features = self.signal_norm(
            signal_logits
        )

        fused = torch.cat(
            [
                signal_features,
                image_features,
            ],
            dim=-1,
        )

        logits = self.fusion_head(fused)

        return logits
