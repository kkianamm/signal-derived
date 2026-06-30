"""
datasets/ptbxl_qwen.py

Drop this file into your medtsllm4 datasets/ folder. It subclasses
PTBXLClassificationDataset and overrides ONLY get_data(): calls the original
(unchanged) loader, then merges the per-sample Qwen3.5 descriptions
(generate_descriptions.py's output) onto the same per-record "descriptions"
field that already flows into medtsllm.py as clip_prompts. No changes to
medtsllm.py, ptbxl.py, or BiomedCoOpHead needed.

Wiring (2-line patch to datasets/__init__.py):

    from .ptbxl_qwen import ptbxl_qwen_datasets
    dataset_lookup["PTB-XL-Qwen"] = ptbxl_qwen_datasets

Then point a config at it (copy configs/datasets/ptbxl.toml ->
configs/datasets/ptbxl_qwen.toml, change one line):

    [data]
    dataset = "PTB-XL-Qwen"   # was "PTB-XL"

so "PTB-XL" vs "PTB-XL-Qwen" is an A/B you flip with one config value,
without touching the existing, already-working PTB-XL path.

CRITICAL correctness note
--------------------------
generate_descriptions.py's output is keyed by image filename, and those
filenames are "{idx:07d}.png" where idx is the position
PTBXLClassificationDataset.get_data() emitted that record at (see
signal_to_image.py's precompute_dataset_images). This file matches by
parsing idx back out of the filename -- NOT CSV row order -- and additionally
cross-checks the label recorded in manifest.csv (sibling file, same idx)
against the label get_data() just produced for that index. A label mismatch
means the images/descriptions were generated from a different run (different
filter/order/split) than the one currently training, and is a hard error
rather than a warning, because silently pairing the wrong description with
the wrong ECG would invalidate results without anyone noticing.
"""

import os
import re
import warnings

import pandas as pd

from .ptbxl import PTBXLClassificationDataset


def _index_from_path(path: str) -> int:
    name = os.path.basename(str(path))
    m = re.match(r"(\d+)\.png$", name)
    if not m:
        raise ValueError(
            f"Can't parse an index out of image filename {name!r}; expected the "
            f"precompute_dataset_images() convention '{{idx:07d}}.png'."
        )
    return int(m.group(1))


class PTBXLQwenClassificationDataset(PTBXLClassificationDataset):
    # data/ptbxl_images/<split>/{manifest,descriptions}.csv by default;
    # override on the instance/class if yours live elsewhere.
    qwen_descriptions_dir = "data/ptbxl_images"
    merge_mode = "append"  # "append": keep patient info, add Qwen text after it. "replace": Qwen text only.

    def get_data(self, split=None):
        split = split or self.split
        data = super().get_data(split)  # unchanged: {"data", "labels", "descriptions"}
        n = len(data["labels"])

        split_dir = os.path.join(self.qwen_descriptions_dir, split)
        desc_path = os.path.join(split_dir, "descriptions.csv")
        manifest_path = os.path.join(split_dir, "manifest.csv")

        if not os.path.exists(desc_path):
            raise FileNotFoundError(
                f"No Qwen descriptions at {desc_path!r}. Run prepare_ptbxl_images.py then "
                f"generate_descriptions.py for split={split!r} first, or set "
                f"qwen_descriptions_dir to wherever descriptions.csv actually lives."
            )

        by_idx = {}
        for _, row in pd.read_csv(desc_path).iterrows():
            text = row["description"]
            if isinstance(text, str) and text.strip():
                by_idx[_index_from_path(row["path"])] = text.strip()

        manifest_labels = None
        if os.path.exists(manifest_path):
            manifest_labels = {
                _index_from_path(row["path"]): int(row["label"])
                for _, row in pd.read_csv(manifest_path).iterrows()
            }
        else:
            warnings.warn(
                f"No manifest.csv next to {desc_path!r}; skipping the label cross-check that "
                f"would catch a stale/mismatched description merge. Recommended to keep "
                f"manifest.csv alongside descriptions.csv."
            )

        mismatches = []
        missing = 0
        merged = []
        for idx in range(n):
            cur_label = int(data["labels"][idx])
            if manifest_labels is not None and idx in manifest_labels and manifest_labels[idx] != cur_label:
                mismatches.append(idx)

            base = data["descriptions"][idx] if data.get("descriptions") is not None else ""
            qwen_text = by_idx.get(idx)
            if qwen_text is None:
                missing += 1
                merged.append(base)
            elif self.merge_mode == "replace":
                merged.append(qwen_text)
            else:
                merged.append(f"{base} ECG findings: {qwen_text}")

        if mismatches:
            raise RuntimeError(
                f"PTBXLQwenClassificationDataset[{split}]: {len(mismatches)}/{n} records have a "
                f"DIFFERENT label in {manifest_path} than get_data() just produced for the same "
                f"index (first few: {mismatches[:5]}). The images/descriptions were very likely "
                f"generated from a different PTB-XL split/filter/order than this run is using -- "
                f"re-run prepare_ptbxl_images.py against the exact config you're training with "
                f"before trusting this merge."
            )
        if missing:
            warnings.warn(
                f"PTBXLQwenClassificationDataset[{split}]: {missing}/{n} records had no Qwen "
                f"description in {desc_path} (generation incomplete?) -- those fell back to "
                f"patient-info-only text."
            )
        else:
            print(f"[PTBXLQwenClassificationDataset] {split}: merged {n}/{n} Qwen descriptions "
                  f"(mode={self.merge_mode}).")

        data["descriptions"] = merged
        return data


ptbxl_qwen_datasets = {
    "classification": PTBXLQwenClassificationDataset,
}
