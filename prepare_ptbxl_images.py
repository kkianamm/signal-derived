"""
prepare_ptbxl_images.py

Goes straight from kkianamm/medtsllm4's PTBXLClassificationDataset to rendered
images + manifest.csv for train/val/test -- no .npy file involved. This is
the direct fix for:

    FileNotFoundError: [Errno 2] No such file or directory: 'train_signals.npy'

`train_signals.npy` in the earlier example command was a placeholder for
"however you get your signals into a numpy array" -- it isn't a file your
repo produces anywhere. Your actual data comes out of PTBXLClassificationDataset
(wfdb + ptbxl_database.csv), so this script loads it exactly the way train.py
does (same toml.load + dict_to_object) and calls signal_to_image.py's
precompute_dataset_images() directly on the result.

Before running, make sure (these are medtsllm4/PTB-XL prerequisites, not
anything new this script adds):
  1. `pip install wfdb` in the medtsllm4 environment (datasets/ptbxl.py imports
     it locally inside get_data()).
  2. The actual PTB-XL files are downloaded to <repo-root>/data/ptbxl/, i.e.
     <repo-root>/data/ptbxl/ptbxl_database.csv, scp_statements.csv,
     records100/... -- see the docstring at the top of datasets/ptbxl.py.
     Get them from https://physionet.org/content/ptb-xl/ if you haven't yet.

Usage
-----
    python prepare_ptbxl_images.py \
        --repo-root /path/to/medtsllm4 \
        --out-dir data/ptbxl_images \
        --method waveform_grid

(--repo-root is the folder that *contains* datasets/ptbxl.py -- adjust
--config if your toml lives somewhere other than configs/datasets/ptbxl.toml
inside it.) Writes data/ptbxl_images/{train,val,test}/*.png + manifest.csv,
ready for train_image_classifier.py's --train-manifest/--val-manifest/--test-manifest.
"""

from __future__ import annotations

import argparse
import os
import sys

import toml


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repo-root", required=True,
                    help="path to your medtsllm4 checkout (must contain datasets/ptbxl.py)")
    p.add_argument("--config", default=None,
                    help="path to the toml config (default: <repo-root>/configs/datasets/ptbxl.toml)")
    p.add_argument("--out-dir", default="data/ptbxl_images")
    p.add_argument("--method", default="waveform_grid",
                    choices=["waveform_grid", "spectrogram", "scalogram", "gaf", "recurrence_plot"])
    p.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--keep-standardized", action="store_true",
                    help="skip undoing the StandardScaler (default: invert it, so "
                         "waveform_grid gets real mV units -- see signal_to_image.py docstring "
                         "for why that matters for the Qwen3.5 captioning step)")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    repo_root = os.path.abspath(args.repo_root)
    config_path = args.config or os.path.join(repo_root, "configs", "datasets", "ptbxl.toml")

    # Make both `import datasets.ptbxl` (their repo) and `import signal_to_image`
    # (sitting next to this file) resolve, regardless of cwd.
    sys.path.insert(0, repo_root)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    try:
        from datasets.ptbxl import PTBXLClassificationDataset, SUPERCLASS_ORDER
    except ModuleNotFoundError as e:
        raise SystemExit(
            f"Could not `import datasets.ptbxl` with --repo-root={repo_root!r} ({e}).\n"
            f"--repo-root must be the medtsllm4 checkout that directly contains the "
            f"datasets/ folder (the one with ptbxl.py inside it)."
        )
    try:
        from utils import dict_to_object  # the exact wrapper train.py uses
    except ModuleNotFoundError as e:
        raise SystemExit(f"Could not `import utils` from --repo-root={repo_root!r} ({e}).")

    from signal_to_image import precompute_dataset_images, PTBXL_LEAD_NAMES

    if not os.path.exists(config_path):
        raise SystemExit(f"Config not found: {config_path!r}. Pass --config explicitly if it lives elsewhere.")
    config = dict_to_object(toml.load(config_path))
    if config.task != "classification":
        raise SystemExit(f"{config_path} has task={config.task!r}; this script needs task=\"classification\".")

    print(f"[prepare_ptbxl_images] repo_root={repo_root}")
    print(f"[prepare_ptbxl_images] config={config_path}")
    print(f"[prepare_ptbxl_images] method={args.method}  out_dir={args.out_dir}")

    manifests = {}
    for split in args.splits:
        print(f"\n[prepare_ptbxl_images] loading split={split!r} via PTBXLClassificationDataset "
              f"(reads the raw wfdb records -- can take a few minutes the first time)...")
        ds = PTBXLClassificationDataset(config, split=split)
        print(f"  -> {len(ds)} records, {ds.n_classes} classes")

        signals = ds.records.numpy()   # [N, T, 12], StandardScaler-normalized by ClassificationDataset
        labels = ds.labels.numpy()     # [N]

        if not args.keep_standardized:
            if ds.normalizer is None:
                print("  [warn] config.data.normalize is false -- nothing to invert, using records as-is.")
            else:
                n, t, c = signals.shape
                signals = ds.normalizer.inverse_transform(signals.reshape(-1, c)).reshape(n, t, c)
                print("  -> inverted StandardScaler back to raw mV for the waveform image")

        split_out = os.path.join(args.out_dir, split)
        method_kwargs = {"lead_names": PTBXL_LEAD_NAMES}
        if args.method in ("waveform_grid", "spectrogram", "scalogram"):
            method_kwargs["fs"] = PTBXLClassificationDataset.sampling_rate

        precompute_dataset_images(
            signals, labels, split_out, method=args.method,
            class_names=SUPERCLASS_ORDER, num_workers=args.num_workers,
            overwrite=args.overwrite, **method_kwargs,
        )
        manifests[split] = os.path.join(split_out, "manifest.csv")

    print("\n[prepare_ptbxl_images] done. Feed these into train_image_classifier.py:\n")
    print("  python train_image_classifier.py \\")
    for split, flag in [("train", "--train-manifest"), ("val", "--val-manifest"), ("test", "--test-manifest")]:
        if split in manifests:
            print(f"    {flag} {manifests[split]} \\")
    print("    --backbone convnext_tiny --img-size 224 --epochs 30 --batch-size 256")


if __name__ == "__main__":
    main()
