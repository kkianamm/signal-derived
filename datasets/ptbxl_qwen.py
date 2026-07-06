"""
datasets/ptbxl_qwen.py

Drop-in PTB-XL dataset wrapper that merges per-sample Qwen/VLM ECG reports into
PTB-XL's existing `descriptions` field, so the reports flow through MedTsLLM's
normal text/prompt pathway without modifying medtsllm.py.

Recommended use for the classification experiment:

    [data]
    dataset = "PTB-XL-Qwen"

    [datasets.PTB-XL-Qwen]
    qwen_descriptions_dir = "data/ptbxl_images"
    merge_mode = "append"              # append patient info + filtered ECG report
    drop_sections = ["Impression"]     # remove noisy diagnostic conclusion
    max_chars = 600                    # prevent text from swamping the signal path
    desc_dropout = 0.5                 # train-only deterministic modality dropout

Why this wrapper exists
-----------------------
The Qwen report is generated from a rendered ECG image. It can contain useful
low-level findings, but the final diagnostic "Impression" is often a noisy
label-relevant shortcut. Feeding the raw report can hurt because the classifier
may over-trust the text modality and under-train the reliable signal pathway.

This wrapper therefore supports:
  * section filtering: drop_sections / keep_sections
  * max length capping: max_chars
  * train-only description dropout: desc_dropout
  * manifest label cross-checking to catch stale or misaligned descriptions

Important correctness notes
---------------------------
Descriptions are aligned by the image index parsed from filenames such as
"0000123.png". If manifest.csv is present, labels are cross-checked against the
current PTB-XL dataset labels. Keep manifest.csv next to descriptions.csv.

If you later add ecg_id/patient_id/strat_fold columns to manifest.csv, this
file will preserve compatibility, but the current hard alignment check is still
index + label based because the base PTBXLClassificationDataset may not expose
record IDs directly.
"""

from __future__ import annotations

import hashlib
import os
import re
import warnings
from typing import Any, Iterable, Optional

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


_SECTION_RE = re.compile(
    r"^\s*\*{0,2}(" + "|".join(re.escape(l) for l in STRUCTURED_SECTION_LABELS)
    + r")\*{0,2}\s*:\s*(.*)$",
    re.IGNORECASE,
)


_INDEX_RE = re.compile(r"(\d+)\.png$", re.IGNORECASE)


def _cfg_get(obj: Any, key: str, default: Any = None) -> Any:
    """Read `key` from either a dict-like or attribute-like config object."""
    if obj is None:
        return default
    if hasattr(obj, "get"):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _as_list(value: Any, default: Optional[list[str]] = None) -> Optional[list[str]]:
    """Normalize TOML/list/string config values to a Python list or None."""
    if value is None:
        return default
    if isinstance(value, str):
        value = [value]
    try:
        out = [str(v) for v in value]
    except TypeError:
        out = [str(value)]
    return out


def _index_from_path(path: str) -> int:
    """Parse integer index from the generated image filename, e.g. 0000123.png."""
    name = os.path.basename(str(path))
    m = _INDEX_RE.match(name)
    if not m:
        raise ValueError(
            f"Cannot parse an index from image filename {name!r}; expected the "
            f"precompute_dataset_images() convention '{{idx:07d}}.png'."
        )
    return int(m.group(1))


def _index_from_row(row: pd.Series) -> int:
    """Prefer an explicit manifest idx column if present, otherwise parse path."""
    if "idx" in row and pd.notna(row["idx"]):
        return int(row["idx"])
    return _index_from_path(row["path"])


def _split_sections(text: str) -> Optional[list[tuple[str, str]]]:
    """Parse a structured ECG report into ordered (section_label, body) pairs.

    Returns None for unstructured paragraph-style text. This lets the caller
    use paragraph text unchanged instead of accidentally deleting everything.
    """
    sections: list[tuple[str, str]] = []
    cur_label: Optional[str] = None
    cur_body: list[str] = []

    for raw_line in str(text).splitlines():
        line = raw_line.strip()
        if not line:
            continue

        m = _SECTION_RE.match(line)
        if m:
            if cur_label is not None:
                sections.append((cur_label, " ".join(cur_body).strip()))
            canonical = next(
                label for label in STRUCTURED_SECTION_LABELS
                if label.lower() == m.group(1).lower()
            )
            cur_label = canonical
            cur_body = [m.group(2).strip()]
        elif cur_label is not None:
            cur_body.append(line)

    if cur_label is not None:
        sections.append((cur_label, " ".join(cur_body).strip()))

    return sections or None


def _compact_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


class PTBXLQwenClassificationDataset(PTBXLClassificationDataset):
    """PTB-XL classification dataset augmented with Qwen/VLM ECG reports."""

    # Defaults; overridable via [datasets.PTB-XL-Qwen].
    qwen_descriptions_dir = "data/ptbxl_images"
    merge_mode = "append"             # "append" or "replace"
    drop_sections = ["Impression"]    # default: remove noisy diagnostic conclusion
    keep_sections = None              # if set, overrides drop_sections
    max_chars = 600                   # 0 disables truncation
    desc_dropout = 0.5                # train-only probability of patient-info-only text
    dropout_seed = 0

    def __init__(self, config, split):
        # Store the raw config before parent init. Some parent dataset classes call
        # self.get_data() inside super().__init__(), and Python will dispatch to
        # this subclass's get_data(). Therefore Qwen defaults/config must already
        # be available before super().__init__() runs.
        self._raw_config = config
        self._desc_base = None
        self._desc_full = None

        # Initialize instance attributes from class defaults, then refresh from
        # config if possible before parent init.
        self.qwen_descriptions_dir = type(self).qwen_descriptions_dir
        self.merge_mode = type(self).merge_mode
        self.drop_sections = list(type(self).drop_sections)
        self.keep_sections = type(self).keep_sections
        self.max_chars = type(self).max_chars
        self.desc_dropout = type(self).desc_dropout
        self.dropout_seed = type(self).dropout_seed
        self._refresh_qwen_config()

        super().__init__(config, split)

        # Refresh once more after parent init, because the parent may define
        # self.dataset_config during initialization.
        self._refresh_qwen_config()

    # ------------------------------------------------------------------
    # Config
    # ------------------------------------------------------------------
    def _get_qwen_dataset_config(self) -> Any:
        """Find the [datasets.PTB-XL-Qwen] config robustly.

        Prefer self.dataset_config if the parent class exposes it. Fall back to
        the raw top-level config saved before parent init.
        """
        dc = getattr(self, "dataset_config", None)
        if dc is not None:
            return dc

        raw = getattr(self, "_raw_config", None)
        datasets_cfg = _cfg_get(raw, "datasets", None)
        if datasets_cfg is None:
            return None

        data_cfg = _cfg_get(raw, "data", None)
        dataset_name = _cfg_get(data_cfg, "dataset", "PTB-XL-Qwen")
        return _cfg_get(datasets_cfg, dataset_name, None)

    def _refresh_qwen_config(self) -> None:
        """Load Qwen-specific settings from config, validating all values."""
        dc = self._get_qwen_dataset_config()

        self.qwen_descriptions_dir = str(
            _cfg_get(dc, "qwen_descriptions_dir", self.qwen_descriptions_dir)
        )
        self.merge_mode = str(_cfg_get(dc, "merge_mode", self.merge_mode)).lower()
        if self.merge_mode not in {"append", "replace"}:
            raise ValueError(
                f"merge_mode must be 'append' or 'replace', got {self.merge_mode!r}."
            )

        self.drop_sections = _as_list(
            _cfg_get(dc, "drop_sections", self.drop_sections), default=[]
        ) or []

        keep_sections = _as_list(_cfg_get(dc, "keep_sections", self.keep_sections), default=None)
        # Treat [] as not set, so drop_sections remains active.
        self.keep_sections = keep_sections if keep_sections else None

        self.max_chars = int(_cfg_get(dc, "max_chars", self.max_chars))
        if self.max_chars < 0:
            raise ValueError(f"max_chars must be >= 0, got {self.max_chars}.")

        self.desc_dropout = float(_cfg_get(dc, "desc_dropout", self.desc_dropout))
        if not 0.0 <= self.desc_dropout <= 1.0:
            raise ValueError(f"desc_dropout must be in [0, 1], got {self.desc_dropout}.")

        self.dropout_seed = int(_cfg_get(dc, "dropout_seed", self.dropout_seed))

    # ------------------------------------------------------------------
    # Text transforms
    # ------------------------------------------------------------------
    def _filter_text(self, text: str) -> str:
        """Filter/drop sections and cap Qwen text length."""
        text = _compact_whitespace(text)
        if not text:
            return ""

        sections = _split_sections(text)
        if sections is None:
            out = text  # paragraph-style fallback: keep unchanged except max_chars
        else:
            if self.keep_sections is not None:
                keep = {s.lower() for s in self.keep_sections}
                chosen = [(label, body) for label, body in sections if label.lower() in keep]
            else:
                drop = {s.lower() for s in self.drop_sections}
                chosen = [(label, body) for label, body in sections if label.lower() not in drop]

            out = _compact_whitespace(
                " ".join(f"{label}: {body}" for label, body in chosen if body)
            )

        if self.max_chars and len(out) > self.max_chars:
            out = out[: self.max_chars].rsplit(" ", 1)[0].rstrip(" ,;.") + "…"
        return out

    def _merge_description(self, base: str, qwen_text: str) -> str:
        base = _compact_whitespace(base)
        qwen_text = _compact_whitespace(qwen_text)
        if not qwen_text:
            return base
        if self.merge_mode == "replace":
            return qwen_text
        if base:
            return f"{base} ECG findings: {qwen_text}"
        return f"ECG findings: {qwen_text}"

    # ------------------------------------------------------------------
    # Loading and alignment
    # ------------------------------------------------------------------
    def _load_qwen_descriptions(self, desc_path: str) -> tuple[dict[int, str], int]:
        df = pd.read_csv(desc_path)
        required = {"path", "description"}
        missing_cols = required - set(df.columns)
        if missing_cols:
            raise ValueError(f"{desc_path!r} is missing required columns: {sorted(missing_cols)}")

        by_idx: dict[int, str] = {}
        empty_rows = 0
        duplicate_rows = 0

        for _, row in df.iterrows():
            idx = _index_from_row(row)
            raw_text = row["description"]
            if not isinstance(raw_text, str) or not raw_text.strip():
                empty_rows += 1
                continue
            if idx in by_idx:
                duplicate_rows += 1
                warnings.warn(
                    f"Duplicate Qwen description for idx={idx} in {desc_path!r}; "
                    f"keeping the last non-empty row."
                )
            by_idx[idx] = raw_text.strip()

        if empty_rows:
            warnings.warn(f"{desc_path!r}: skipped {empty_rows} empty description rows.")
        if duplicate_rows:
            warnings.warn(f"{desc_path!r}: saw {duplicate_rows} duplicate non-empty rows.")

        return by_idx, len(df)

    def _load_manifest_labels(self, manifest_path: str) -> Optional[dict[int, int]]:
        if not os.path.exists(manifest_path):
            warnings.warn(
                f"No manifest.csv next to {manifest_path!r}; skipping the label cross-check. "
                f"Keep manifest.csv beside descriptions.csv to catch stale/mismatched merges."
            )
            return None

        df = pd.read_csv(manifest_path)
        required = {"path", "label"}
        missing_cols = required - set(df.columns)
        if missing_cols:
            raise ValueError(f"{manifest_path!r} is missing required columns: {sorted(missing_cols)}")

        return {_index_from_row(row): int(row["label"]) for _, row in df.iterrows()}

    def get_data(self, split=None):
        split = split or getattr(self, "split", None)
        if split is None:
            raise ValueError("split must be provided before PTBXLQwenClassificationDataset.get_data().")

        data = super().get_data(split)

        # Critical: refresh here because this method can be called during parent
        # initialization, and also because val/test instances may call get_data('train')
        # internally to fit or reuse normalization statistics.
        self._refresh_qwen_config()

        n = len(data["labels"])
        split_dir = os.path.join(self.qwen_descriptions_dir, split)
        desc_path = os.path.join(split_dir, "descriptions.csv")
        manifest_path = os.path.join(split_dir, "manifest.csv")

        if not os.path.exists(desc_path):
            raise FileNotFoundError(
                f"No Qwen descriptions found at {desc_path!r}. Run prepare_ptbxl_images.py "
                f"and generate_descriptions.py for split={split!r}, or set "
                f"[datasets.PTB-XL-Qwen].qwen_descriptions_dir correctly."
            )

        by_idx, raw_desc_rows = self._load_qwen_descriptions(desc_path)
        manifest_labels = self._load_manifest_labels(manifest_path)

        base_descriptions = data.get("descriptions")
        if base_descriptions is None:
            base_descriptions = [""] * n

        label_mismatches: list[int] = []
        missing_rows = 0
        empty_after_filter = 0
        base_list: list[str] = []
        full_list: list[str] = []

        for idx in range(n):
            cur_label = int(data["labels"][idx])
            if manifest_labels is not None:
                manifest_label = manifest_labels.get(idx)
                if manifest_label is None:
                    missing_rows += 1
                elif manifest_label != cur_label:
                    label_mismatches.append(idx)

            base = str(base_descriptions[idx]) if idx < len(base_descriptions) else ""
            base_list.append(base)

            qwen_raw = by_idx.get(idx)
            if qwen_raw is None:
                full_list.append(base)
                continue

            qwen_filtered = self._filter_text(qwen_raw)
            if not qwen_filtered:
                empty_after_filter += 1
                full_list.append(base)
                continue

            full_list.append(self._merge_description(base, qwen_filtered))

        if label_mismatches:
            raise RuntimeError(
                f"PTBXLQwenClassificationDataset[{split}]: {len(label_mismatches)}/{n} records "
                f"have labels in {manifest_path!r} that differ from the labels produced by the "
                f"current PTB-XL config for the same index. First mismatches: "
                f"{label_mismatches[:10]}. Re-run prepare_ptbxl_images.py and "
                f"generate_descriptions.py with the exact same config used for training."
            )

        if missing_rows:
            warnings.warn(
                f"PTBXLQwenClassificationDataset[{split}]: {missing_rows}/{n} current dataset "
                f"indices were absent from manifest.csv; those records cannot be label-checked."
            )

        used = sum(1 for idx in range(n) if by_idx.get(idx) and full_list[idx] != base_list[idx])
        if used < n:
            warnings.warn(
                f"PTBXLQwenClassificationDataset[{split}]: used {used}/{n} Qwen descriptions "
                f"({n - used} fell back to patient-info-only text; empty_after_filter="
                f"{empty_after_filter})."
            )
        else:
            print(
                f"[PTBXLQwenClassificationDataset] {split}: merged {used}/{n} Qwen descriptions "
                f"from {raw_desc_rows} rows "
                f"(mode={self.merge_mode}, drop={self.drop_sections}, keep={self.keep_sections}, "
                f"max_chars={self.max_chars}, desc_dropout={self.desc_dropout if split == 'train' else 0.0})."
            )

        # Keep aligned text lists only for this instance's actual split. Some
        # parent implementations call get_data('train') from val/test instances
        # for normalizer fitting; do not let that clobber val/test text caches.
        if split == getattr(self, "split", split):
            self._desc_base = base_list
            self._desc_full = full_list

        data["descriptions"] = full_list
        return data

    # ------------------------------------------------------------------
    # Train-only description dropout
    # ------------------------------------------------------------------
    def _drop_description_for_idx(self, idx: int) -> bool:
        """Deterministic, DataLoader-worker-safe description dropout.

        Using random.Random inside __getitem__ is not reproducible under multiple
        workers and can change simply because the dataloader order changes. This
        hash-based dropout gives each sample a stable Bernoulli decision for a
        given seed. Evaluation never drops descriptions.
        """
        if self.split != "train" or self.desc_dropout <= 0.0:
            return False
        if self.desc_dropout >= 1.0:
            return True
        key = f"{self.dropout_seed}:{self.split}:{int(idx)}".encode("utf-8")
        digest = hashlib.blake2b(key, digest_size=8).digest()
        u = int.from_bytes(digest, byteorder="big", signed=False) / float(2**64)
        return u < self.desc_dropout

    def __getitem__(self, idx):
        # Prefer the parent implementation so future keys/masks added by the
        # original dataset are preserved. Then override only the description if
        # train-only modality dropout says to fall back to patient info.
        out = super().__getitem__(idx)

        if (
            self.record_descriptions is not None
            and getattr(self, "_desc_base", None) is not None
            and self._drop_description_for_idx(idx)
        ):
            out["descriptions"] = self._desc_base[idx]

        return out


ptbxl_qwen_datasets = {
    "classification": PTBXLQwenClassificationDataset,
}
