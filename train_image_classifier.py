"""
train_image_classifier.py

H100-oriented training/eval for classifying signal-derived images produced by
signal_to_image.py. This is the "use the signal-derived image just for
classification" branch -- a standalone vision pathway that does NOT go
through Qwen3.5 or the MedTsLLM/BiomedCoOp text path. It exists to (a) check
how much diagnostic signal the image encoding itself carries, and (b) serve
as one component you can later late-fuse with the BiomedCoOpHead logits in
kkianamm/medtsllm4 (not done here -- this script only trains the image branch).

Input: manifest CSV(s) with columns `path,label[,label_name]`, exactly what
`precompute_dataset_images()` in signal_to_image.py writes.

Usage
-----
    python train_image_classifier.py \
        --train-manifest data/ptbxl_images/train/manifest.csv \
        --val-manifest   data/ptbxl_images/val/manifest.csv \
        --test-manifest  data/ptbxl_images/test/manifest.csv \
        --backbone convnext_tiny --img-size 224 --epochs 30 \
        --batch-size 256 --out-dir runs/ptbxl_image_clf

    # evaluate a checkpoint only
    python train_image_classifier.py --mode eval --resume runs/ptbxl_image_clf/best.pt \
        --test-manifest data/ptbxl_images/test/manifest.csv

Requires: torch, torchvision, timm, scikit-learn, pandas, pillow
    pip install timm scikit-learn pandas --break-system-packages

H100 notes
----------
  * bf16 autocast end-to-end (no GradScaler needed, unlike fp16)
  * TF32 matmuls enabled for any residual fp32 ops
  * channels_last memory format (faster convs on Hopper/Ampere tensor cores)
  * torch.compile (default on) with a clean fallback if compilation fails
  * pinned-memory, persistent, prefetching DataLoader workers so the H100
    isn't starved waiting on PNG decode/resize

Metrics (accuracy / macro-F1 / macro-precision / macro-recall, "binary"
averaging if exactly 2 classes) are computed with the exact same sklearn
calls as kkianamm/medtsllm4's tasks/classification.py, so numbers are
directly comparable to the main MedTsLLM + BiomedCoOp pipeline.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

try:
    import timm
except ImportError as e:
    raise ImportError("This script needs timm: pip install timm --break-system-packages") from e


# ============================================================================
# Data
# ============================================================================
class SignalImageDataset(Dataset):
    """Reads a manifest CSV (path,label[,label_name]) written by
    signal_to_image.precompute_dataset_images()."""

    def __init__(self, manifest_csv: str, transform=None):
        self.df = pd.read_csv(manifest_csv)
        if "path" not in self.df.columns or "label" not in self.df.columns:
            raise ValueError(f"{manifest_csv} must have columns path,label")
        self.transform = transform

        if "label_name" in self.df.columns:
            self.class_names = (
                self.df[["label", "label_name"]].drop_duplicates()
                .sort_values("label")["label_name"].tolist()
            )
        else:
            self.class_names = [str(i) for i in range(int(self.df["label"].max()) + 1)]

    @property
    def n_classes(self):
        return len(self.class_names)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row["path"]).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, int(row["label"])


def build_transforms(img_size: int, mean, std, train: bool, random_erasing_p: float = 0.0):
    """No flips / rotations: a waveform plot's time axis (left-right) and
    amplitude axis (up-down) are physically meaningful, unlike a natural
    photo, so those augmentations would corrupt the signal rather than
    regularize the model. Mild crop + brightness/contrast jitter only."""
    from torchvision import transforms as T

    if train:
        tf = [
            T.Resize((int(img_size * 1.12), int(img_size * 1.12))),
            T.RandomCrop(img_size),
            T.ColorJitter(brightness=0.1, contrast=0.1),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ]
        if random_erasing_p > 0:
            tf.append(T.RandomErasing(p=random_erasing_p, scale=(0.02, 0.08)))
    else:
        tf = [
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ]
    return T.Compose(tf)


# ============================================================================
# Config
# ============================================================================
@dataclass
class TrainConfig:
    train_manifest: Optional[str] = None
    val_manifest: Optional[str] = None
    test_manifest: Optional[str] = None
    out_dir: str = "runs/image_clf"

    backbone: str = "convnext_tiny"
    img_size: int = 224
    pretrained: bool = True

    epochs: int = 30
    batch_size: int = 256
    lr: float = 3e-4
    weight_decay: float = 0.05
    warmup_epochs: int = 2
    label_smoothing: float = 0.1
    grad_clip: float = 1.0
    class_weighted_loss: bool = False
    random_erasing_p: float = 0.0

    num_workers: int = 8
    device: str = "cuda"
    compile: bool = True
    channels_last: bool = True
    seed: int = 0

    eval_metric: str = "accuracy"  # matches configs/datasets/ptbxl.toml convention


# ============================================================================
# Metrics (mirrors tasks/classification.py in kkianamm/medtsllm4 exactly)
# ============================================================================
def compute_metrics(pred_int, target_int, n_classes: int, prefix: str) -> dict:
    avg = "binary" if n_classes == 2 else "macro"
    return {
        f"{prefix}/accuracy": accuracy_score(target_int, pred_int),
        f"{prefix}/f1": f1_score(target_int, pred_int, average=avg, zero_division=0),
        f"{prefix}/precision": precision_score(target_int, pred_int, average=avg, zero_division=0),
        f"{prefix}/recall": recall_score(target_int, pred_int, average=avg, zero_division=0),
    }


# ============================================================================
# Train / eval
# ============================================================================
def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def maybe_compile(model, cfg: TrainConfig):
    if not cfg.compile:
        return model
    try:
        return torch.compile(model)
    except Exception as e:
        print(f"[warn] torch.compile failed ({e}); continuing uncompiled.")
        return model


def build_model(cfg: TrainConfig, n_classes: int):
    model = timm.create_model(cfg.backbone, pretrained=cfg.pretrained, num_classes=n_classes)
    data_cfg = timm.data.resolve_data_config({}, model=model)
    return model, data_cfg


def build_optimizer_and_schedule(model, cfg: TrainConfig, steps_per_epoch: int):
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    total_steps = max(1, cfg.epochs * steps_per_epoch)
    warmup_steps = max(1, cfg.warmup_epochs * steps_per_epoch)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    return optimizer, scheduler


@torch.no_grad()
def predict(model, loader, device, channels_last):
    model.eval()
    all_logits, all_targets = [], []
    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        if channels_last:
            imgs = imgs.to(memory_format=torch.channels_last)
        with torch.autocast(device_type="cuda" if device.startswith("cuda") else "cpu",
                             dtype=torch.bfloat16, enabled=True):
            logits = model(imgs)
        all_logits.append(logits.float().cpu())
        all_targets.append(labels)
    return torch.cat(all_logits), torch.cat(all_targets)


def evaluate(model, loader, device, channels_last, n_classes, prefix):
    logits, targets = predict(model, loader, device, channels_last)
    preds = logits.argmax(dim=1).numpy()
    scores = compute_metrics(preds, targets.numpy(), n_classes, prefix)
    return scores, logits, targets


def train(cfg: TrainConfig):
    set_seed(cfg.seed)
    os.makedirs(cfg.out_dir, exist_ok=True)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    device = cfg.device if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("[warn] CUDA not available; running on CPU (only sane for a smoke test).")

    # --- model first, so we know the right normalization stats for the data ---
    train_ds_probe = SignalImageDataset(cfg.train_manifest)
    n_classes = train_ds_probe.n_classes
    model, data_cfg = build_model(cfg, n_classes)
    model.to(device)
    if cfg.channels_last:
        model.to(memory_format=torch.channels_last)

    mean, std = data_cfg["mean"], data_cfg["std"]
    train_tf = build_transforms(cfg.img_size, mean, std, train=True,
                                 random_erasing_p=cfg.random_erasing_p)
    eval_tf = build_transforms(cfg.img_size, mean, std, train=False)

    train_ds = SignalImageDataset(cfg.train_manifest, transform=train_tf)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                               num_workers=cfg.num_workers, pin_memory=True,
                               persistent_workers=cfg.num_workers > 0,
                               prefetch_factor=4 if cfg.num_workers > 0 else None,
                               drop_last=True)

    val_loader = None
    if cfg.val_manifest:
        val_ds = SignalImageDataset(cfg.val_manifest, transform=eval_tf)
        val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                                 num_workers=cfg.num_workers, pin_memory=True,
                                 persistent_workers=cfg.num_workers > 0)

    test_loader = None
    if cfg.test_manifest:
        test_ds = SignalImageDataset(cfg.test_manifest, transform=eval_tf)
        test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False,
                                  num_workers=cfg.num_workers, pin_memory=True,
                                  persistent_workers=cfg.num_workers > 0)

    class_weight = None
    if cfg.class_weighted_loss:
        counts = train_ds.df["label"].value_counts().sort_index().values.astype(np.float64)
        weights = counts.sum() / (len(counts) * counts)
        class_weight = torch.tensor(weights, dtype=torch.float32, device=device)
        print(f"[info] class weights: {weights}")

    loss_fn = nn.CrossEntropyLoss(weight=class_weight, label_smoothing=cfg.label_smoothing)
    optimizer, scheduler = build_optimizer_and_schedule(model, cfg, steps_per_epoch=len(train_loader))
    compiled_model = maybe_compile(model, cfg)

    history = []
    best_metric = -float("inf")
    best_path = os.path.join(cfg.out_dir, "best.pt")

    with open(os.path.join(cfg.out_dir, "config.json"), "w") as f:
        json.dump({**asdict(cfg), "class_names": train_ds.class_names}, f, indent=2)

    for epoch in range(cfg.epochs):
        compiled_model.train()
        t0 = time.time()
        running_loss, n_batches = 0.0, 0
        train_preds, train_targets = [], []

        for imgs, labels in train_loader:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if cfg.channels_last:
                imgs = imgs.to(memory_format=torch.channels_last)

            with torch.autocast(device_type="cuda" if device.startswith("cuda") else "cpu",
                                 dtype=torch.bfloat16, enabled=True):
                logits = compiled_model(imgs)
                loss = loss_fn(logits, labels)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()
            n_batches += 1
            with torch.no_grad():
                train_preds.append(logits.float().argmax(dim=1).cpu())
                train_targets.append(labels.cpu())

        tp = torch.cat(train_preds).numpy()
        tt = torch.cat(train_targets).numpy()
        epoch_record = {"epoch": epoch + 1, "train/loss": running_loss / max(1, n_batches),
                         **compute_metrics(tp, tt, n_classes, "train")}

        if val_loader is not None:
            val_scores, _, _ = evaluate(compiled_model, val_loader, device, cfg.channels_last,
                                         n_classes, "val")
            epoch_record.update(val_scores)
        if test_loader is not None:
            test_scores, _, _ = evaluate(compiled_model, test_loader, device, cfg.channels_last,
                                          n_classes, "test")
            epoch_record.update(test_scores)

        dt = time.time() - t0
        print(f"epoch {epoch+1}/{cfg.epochs}  loss={epoch_record['train/loss']:.4f}  "
              f"train_acc={epoch_record['train/accuracy']:.4f}  "
              + (f"val_acc={epoch_record.get('val/accuracy', float('nan')):.4f}  " if val_loader else "")
              + f"({dt:.1f}s)")
        history.append(epoch_record)

        metric_key = f"val/{cfg.eval_metric}" if val_loader is not None else f"train/{cfg.eval_metric}"
        current = epoch_record.get(metric_key, epoch_record["train/accuracy"])
        if current > best_metric:
            best_metric = current
            torch.save({
                "model_state_dict": model.state_dict(),  # uncompiled module's state dict
                "epoch": epoch + 1,
                "metric": current,
                "class_names": train_ds.class_names,
                "cfg": asdict(cfg),
            }, best_path)
            print(f"  -> new best ({metric_key}={current:.4f}), saved to {best_path}")

        with open(os.path.join(cfg.out_dir, "history.json"), "w") as f:
            json.dump(history, f, indent=2)

    if test_loader is not None:
        print(f"[info] reloading best checkpoint ({best_path}) for final test evaluation")
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        final_scores, _, _ = evaluate(model, test_loader, device, cfg.channels_last, n_classes, "test")
        print("[final test]", final_scores)
        with open(os.path.join(cfg.out_dir, "final_test_metrics.json"), "w") as f:
            json.dump(final_scores, f, indent=2)

    return history


def eval_only(cfg: TrainConfig, resume_path: str):
    device = cfg.device if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(resume_path, map_location=device)
    class_names = ckpt["class_names"]
    n_classes = len(class_names)

    model, data_cfg = build_model(cfg, n_classes)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    if cfg.channels_last:
        model.to(memory_format=torch.channels_last)

    eval_tf = build_transforms(cfg.img_size, data_cfg["mean"], data_cfg["std"], train=False)
    test_ds = SignalImageDataset(cfg.test_manifest, transform=eval_tf)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False,
                              num_workers=cfg.num_workers, pin_memory=True)

    scores, logits, targets = evaluate(model, test_loader, device, cfg.channels_last,
                                        n_classes, "test")
    print(scores)
    return scores, logits, targets


# ============================================================================
# CLI
# ============================================================================
def parse_args() -> TrainConfig:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=["train", "eval"], default="train")
    p.add_argument("--train-manifest", dest="train_manifest")
    p.add_argument("--val-manifest", dest="val_manifest")
    p.add_argument("--test-manifest", dest="test_manifest")
    p.add_argument("--out-dir", default="runs/image_clf")
    p.add_argument("--resume", default=None, help="checkpoint path (required for --mode eval)")

    p.add_argument("--backbone", default="convnext_tiny")
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--no-pretrained", action="store_true")

    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--warmup-epochs", type=int, default=2)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--class-weighted-loss", action="store_true")
    p.add_argument("--random-erasing-p", type=float, default=0.0)

    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--device", default="cuda")
    p.add_argument("--no-compile", action="store_true")
    p.add_argument("--no-channels-last", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval-metric", default="accuracy",
                    choices=["accuracy", "f1", "precision", "recall"])

    a = p.parse_args()
    cfg = TrainConfig(
        train_manifest=a.train_manifest, val_manifest=a.val_manifest, test_manifest=a.test_manifest,
        out_dir=a.out_dir, backbone=a.backbone, img_size=a.img_size, pretrained=not a.no_pretrained,
        epochs=a.epochs, batch_size=a.batch_size, lr=a.lr, weight_decay=a.weight_decay,
        warmup_epochs=a.warmup_epochs, label_smoothing=a.label_smoothing, grad_clip=a.grad_clip,
        class_weighted_loss=a.class_weighted_loss, random_erasing_p=a.random_erasing_p,
        num_workers=a.num_workers, device=a.device, compile=not a.no_compile,
        channels_last=not a.no_channels_last, seed=a.seed, eval_metric=a.eval_metric,
    )
    return cfg, a.mode, a.resume


if __name__ == "__main__":
    cfg, mode, resume = parse_args()
    if mode == "train":
        if not cfg.train_manifest:
            raise ValueError("--train-manifest is required for --mode train")
        train(cfg)
    else:
        if not resume or not cfg.test_manifest:
            raise ValueError("--mode eval requires --resume <ckpt> and --test-manifest")
        eval_only(cfg, resume)
