from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torchvision import transforms as T

import timm
from timm.data import resolve_data_config

from .ptbxl import PTBXLClassificationDataset


class PTBXLImageFusionDataset(PTBXLClassificationDataset):
    """
    PTB-XL dataset that returns BOTH:
        x_enc  : [T, 12] ECG signal
        image  : [3, H, W] rendered ECG image
        labels : scalar class index

    The image manifest was generated from the same
    PTBXLClassificationDataset ordering, so row idx in
    manifest.csv corresponds to sample idx here.
    """

    def __init__(self, config, split):
        super().__init__(config, split)

        fusion_cfg = config.models.image_fusion

        repo_root = Path(__file__).resolve().parent.parent
        image_root = Path(fusion_cfg.image_root)

        if not image_root.is_absolute():
            image_root = repo_root / image_root

        manifest_path = image_root / split / "manifest.csv"

        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Image manifest not found: {manifest_path}"
            )

        self.image_manifest = pd.read_csv(manifest_path)

        # ---------------------------------------------------------
        # Safety check 1: same number of samples
        # ---------------------------------------------------------
        if len(self.image_manifest) != len(self):
            raise RuntimeError(
                f"{split}: image manifest contains "
                f"{len(self.image_manifest)} samples, "
                f"but PTB-XL dataset contains {len(self)}"
            )

        # ---------------------------------------------------------
        # Safety check 2: image labels must exactly match
        # signal labels in the same order
        # ---------------------------------------------------------
        image_labels = torch.tensor(
            self.image_manifest["label"].values,
            dtype=torch.long,
        )

        if not torch.equal(image_labels, self.labels.cpu()):
            mismatch = torch.where(
                image_labels != self.labels.cpu()
            )[0]

            raise RuntimeError(
                f"{split}: image/signal label mismatch. "
                f"First mismatching indices: "
                f"{mismatch[:10].tolist()}"
            )

        # ---------------------------------------------------------
        # Get the normalization expected by this timm backbone.
        # No download: pretrained=False.
        # ---------------------------------------------------------
        probe = timm.create_model(
            fusion_cfg.backbone,
            pretrained=False,
            num_classes=self.n_classes,
        )

        data_cfg = resolve_data_config({}, model=probe)

        mean = data_cfg["mean"]
        std = data_cfg["std"]

        del probe

        img_size = int(fusion_cfg.get("img_size", 224))

        # IMPORTANT:
        # ConvNeXt is frozen in the first fusion experiment.
        # Therefore use deterministic evaluation preprocessing
        # for train/val/test.
        self.image_transform = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])

        self.repo_root = repo_root

        print(
            f"[PTBXLImageFusionDataset] split={split} "
            f"samples={len(self)} "
            f"manifest={manifest_path}"
        )

    def __getitem__(self, idx):
        # Existing PTB-XL signal + demographics + label
        out = super().__getitem__(idx)

        row = self.image_manifest.iloc[idx]
        image_path = Path(str(row["path"]))

        if not image_path.is_absolute():
            image_path = self.repo_root / image_path

        if not image_path.exists():
            raise FileNotFoundError(
                f"Image not found at index {idx}: {image_path}"
            )

        image = Image.open(image_path).convert("RGB")
        image = self.image_transform(image)

        out["image"] = image

        return out


ptbxl_image_fusion_datasets = {
    "classification": PTBXLImageFusionDataset,
}
