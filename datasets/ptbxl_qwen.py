"""
datasets/ptbxl_qwen.py

Subclass of PTBXLClassificationDataset that merges per-sample Qwen ECG reports
onto the "descriptions" field (which flows into medtsllm.py as clip_prompts).
No changes to medtsllm.py, ptbxl.py, or BiomedCoOpHead needed.

WHY ACCURACY DROPPED WITH THE RAW MERGE, AND WHAT THIS FILE NOW DOES
-------------------------------------------------------------------
The Qwen text is a 6-section clinical report (Rhythm/Rate, Axis, Intervals,
QRS/R-wave progression, ST-T, Impression) produced by a vision-language model
reading a *rendered image* of the ECG. That model is not a trained ECG reader,
so the report -- especially the final "Impression" -- is an UNRELIABLE,
label-relevant free-text cue. Feeding it in as clip_prompts hurt for two
compounding reasons:

  1. Misleading conclusions. The "Impression" commits to a diagnosis that is
     often wrong. The classifier learns to trust that text (it correlates with
     the VLM's *guessed* label, only weakly with the *true* label), so at test
     time it is pulled toward the VLM's error distribution -> below the
     patient-info-only baseline.
  2. Modality over-reliance. Rich text is easier to fit than the reprogrammed
     signal, so training leans on the text and under-trains the (reliable)
     signal pathway. With an unreliable text modality, the fused model
     underperforms signal-only. This is worse on the FLAN-T5 encoder-decoder
     path, where the report is the ENCODER input and the signal decoder
     cross-attends to it, so noisy text directly shapes the patch features the
     head reads.

Three mitigations, all controlled from the [datasets.PTB-XL-Qwen] config
section (defaults chosen to recover and hopefully beat baseline):

  * drop_sections  - remove the least reliable sections before use.
                     Default drops "Impression" (the diagnostic conclusion).
                     Set keep_sections to instead whitelist sections.
  * max_chars      - cap the Qwen text so it cannot swamp the signal/prompt.
  * desc_dropout   - TRAIN-ONLY stochastic modality dropout: with this
                     probability a sample falls back to patient-info-only text,
                     forcing the signal pathway to stay strong. Eval always
                     uses the full (filtered) text.

Set drop_sections=[], max_chars=0, desc_dropout=0.0 to reproduce the old
raw-merge behavior.

Wiring (datasets/__init__.py):
    from .ptbxl_qwen import ptbxl_qwen_datasets
    dataset_lookup["PTB-XL-Qwen"] = ptbxl_qwen_datasets

CORRECTNESS NOTE (unchanged): descriptions are matched to records by the idx
parsed from the image filename "{idx:07d}.png", and the label in manifest.csv
is cross-checked against the label get_data() produced for that idx. A mismatch
is a hard error. KEEP manifest.csv alongside descriptions.csv -- without it the
label cross-check is skipped, and a stale/misaligned merge (which pairs the
wrong report with the wrong ECG and reads as pure noise) can silently tank
accuracy. If you saw a drop, verify the "merged n/n" line printed and that no
"No manifest.csv" warning appeared.
"""

import os
import re
import random
import warnings

import pandas as pd

from .ptbxl import PTBXLClassificationDataset


STRUCTURED_SECTION_LABELS = [
    "Rhythm/Rate",
    "Axis",
    "Intervals",
    "QRS/R-wave progression",
    "ST-T",
    "Impression",
]


def _index_from_path(path: str) -> int:
    name = os.path.basename(str(path))
    m = re.match(r"(\d+)\.png$", name)
    if not m:
        raise ValueError(
            f"Can't parse an index out of image filename {name!r}; expected the "
            f"precompute_dataset_images() convention '{{idx:07d}}.png'."
        )
    return int(m.group(1))


def _split_sections(text: str):
    """Parse a structured report into an ordered list of (label, body).

    Returns None if the text is not in the labeled-section format (so the
    caller can fall back to using the raw text unchanged).
    """
    pattern = re.compile(
        r"^\s*\*{0,2}(" + "|".join(re.escape(l) for l in STRUCTURED_SECTION_LABELS)
        + r")\*{0,2}\s*:\s*(.*)$",
        re.IGNORECASE,
    )
    lines = str(text).splitlines()
    sections, cur_label, cur_body = [], None, []
    for line in lines:
        m = pattern.match(line)
        if m:
            if cur_label is not None:
                sections.append((cur_label, " ".join(cur_body).strip()))
            canonical = next(
                l for l in STRUCTURED_SECTION_LABELS if l.lower() == m.group(1).lower()
            )
            cur_label, cur_body = canonical, [m.group(2).strip()]
        elif cur_label is not None:
            cur_body.append(line.strip())
    if cur_label is not None:
        sections.append((cur_label, " ".join(cur_body).strip()))
    return sections or None


class PTBXLQwenClassificationDataset(PTBXLClassificationDataset):
    # Defaults; overridable via [datasets.PTB-XL-Qwen] in the config.
    qwen_descriptions_dir = "data/ptbxl_images"
    merge_mode = "append"            # "append": patient info + Qwen text; "replace": Qwen only
    drop_sections = ["Impression"]   # least-reliable sections to remove
    keep_sections = None             # if set (list), keep ONLY these (overrides drop_sections)
    max_chars = 600                  # cap Qwen text length (0 = no cap)
    desc_dropout = 0.5               # train-only prob of falling back to patient-info-only
    dropout_seed = 0

    def __init__(self, config, split):
        super().__init__(config, split)
        # Pull overrides from [datasets.PTB-XL-Qwen] if present.
        dc = getattr(self, "dataset_config", {}) or {}
        get = dc.get if hasattr(dc, "get") else (lambda k, d=None: getattr(dc, k, d))
        self.qwen_descriptions_dir = get("qwen_descriptions_dir", self.qwen_descriptions_dir)
        self.merge_mode = get("merge_mode", self.merge_mode)
        self.drop_sections = list(get("drop_sections", self.drop_sections) or [])
        ks = get("keep_sections", self.keep_sections)
        self.keep_sections = list(ks) if ks else None
        self.max_chars = int(get("max_chars", self.max_chars))
        self.desc_dropout = float(get("desc_dropout", self.desc_dropout))
        self.dropout_seed = int(get("dropout_seed", self.dropout_seed))
        self._rng = random.Random(self.dropout_seed)

    # --- text transforms -----------------------------------------------------
    def _filter_text(self, text: str) -> str:
        sections = _split_sections(text)
        if sections is None:
            out = str(text).strip()                      # unstructured: use as-is
        else:
            if self.keep_sections is not None:
                keep = {s.lower() for s in self.keep_sections}
                chosen = [(l, b) for l, b in sections if l.lower() in keep]
            else:
                drop = {s.lower() for s in self.drop_sections}
                chosen = [(l, b) for l, b in sections if l.lower() not in drop]
            out = " ".join(f"{l}: {b}" for l, b in chosen if b).strip()
        if self.max_chars and len(out) > self.max_chars:
            out = out[: self.max_chars].rsplit(" ", 1)[0].rstrip(" ,;.") + "…"
        return out

    # --- loading -------------------------------------------------------------
    def get_data(self, split=None):
        split = split or self.split
        data = super().get_data(split)  # {"data","labels","descriptions"} (descriptions = patient info)
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
                f"would catch a stale/mismatched description merge. Keep manifest.csv alongside "
                f"descriptions.csv -- a silent misalignment reads as pure noise and drops accuracy."
            )

        mismatches, missing = [], 0
        base_list, full_list = [], []
        for idx in range(n):
            cur_label = int(data["labels"][idx])
            if manifest_labels is not None and idx in manifest_labels and manifest_labels[idx] != cur_label:
                mismatches.append(idx)

            base = data["descriptions"][idx] if data.get("descriptions") is not None else ""
            base_list.append(base)

            qwen_text = by_idx.get(idx)
            if qwen_text is not None:
                qwen_text = self._filter_text(qwen_text)
            if not qwen_text:
                missing += 1 if by_idx.get(idx) is None else 0
                full_list.append(base)
            elif self.merge_mode == "replace":
                full_list.append(qwen_text)
            else:
                full_list.append(f"{base} ECG findings: {qwen_text}")

        if mismatches:
            raise RuntimeError(
                f"PTBXLQwenClassificationDataset[{split}]: {len(mismatches)}/{n} records have a "
                f"DIFFERENT label in {manifest_path} than get_data() just produced for the same "
                f"index (first few: {mismatches[:5]}). The images/descriptions were generated from "
                f"a different PTB-XL split/filter/order than this run -- re-run "
                f"prepare_ptbxl_images.py against the exact config you're training with."
            )
        if missing:
            warnings.warn(
                f"PTBXLQwenClassificationDataset[{split}]: {missing}/{n} records had no Qwen "
                f"description -- those fell back to patient-info-only text."
            )
        else:
            print(f"[PTBXLQwenClassificationDataset] {split}: merged {n}/{n} Qwen descriptions "
                  f"(mode={self.merge_mode}, drop={self.drop_sections}, keep={self.keep_sections}, "
                  f"max_chars={self.max_chars}, desc_dropout={self.desc_dropout if split=='train' else 0.0}).")

        # Stash aligned base/full lists for stochastic dropout in __getitem__,
        # but only for THIS instance's split (get_data('train') is also called
        # to fit the normalizer on val/test instances -- don't let that clobber).
        if split == self.split:
            self._desc_base = base_list
            self._desc_full = full_list

        data["descriptions"] = full_list
        return data

    # --- access (train-only stochastic modality dropout) ---------------------
    def __getitem__(self, idx):
        out = {"x_enc": self.records[idx], "labels": self.labels[idx]}
        if self.record_descriptions is not None:
            desc = self.record_descriptions[idx]
            if (self.split == "train" and self.desc_dropout > 0.0
                    and getattr(self, "_desc_base", None) is not None
                    and self._rng.random() < self.desc_dropout):
                desc = self._desc_base[idx]     # drop Qwen text -> patient info only
            out["descriptions"] = desc
        return out


ptbxl_qwen_datasets = {
    "classification": PTBXLQwenClassificationDataset,
}
