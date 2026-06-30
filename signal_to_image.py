"""
signal_to_image.py

Convert multi-channel physiological time-series windows (shape [T, C] — e.g. a
[512, 12] PTB-XL 12-lead ECG window from kkianamm/medtsllm4's
`datasets/ptbxl.py`) into RGB images.

This is the "third model" stage in the new pipeline:

    raw signal --(this file)--> image --(Qwen3.5-VL)--> per-sample expert text
    raw signal --(this file)--> image --(train_image_classifier.py)--> class logits

Five encodings are provided:

  waveform_grid    Stacked clinical-style line trace (pink/red ECG-paper grid
                    optional). The ONLY encoding that looks like something a
                    vision-language model has actually seen in training (real
                    ECG/EEG strips), so it is the one to feed to Qwen3.5 for
                    free-text per-sample descriptions. Default method.
  spectrogram       Per-channel STFT magnitude (dB), tiled into a grid.
  scalogram         Per-channel continuous wavelet transform (Morlet), tiled.
  gaf               Per-channel Gramian Angular Summation/Difference Field, tiled.
  recurrence_plot   Per-channel recurrence plot, tiled.

spectrogram / scalogram / gaf / recurrence_plot are abstract texture encodings.
They are well-suited as *additional* input to a CNN/ViT classifier (they often
help time-series classification, c.f. Wang & Oates 2015) but a VLM cannot read
clinical meaning out of them the way it can a waveform plot — don't send them
to Qwen expecting a clinically grounded description.

Quick start
-----------
    from signal_to_image import signal_to_image, precompute_dataset_images

    img = signal_to_image(signal, method="waveform_grid", fs=100.0)   # PIL.Image
    img.save("sample_0.png")

    # whole-dataset precompute (parallel, writes manifest.csv)
    manifest = precompute_dataset_images(
        signals, labels, out_dir="data/ptbxl_images/train", method="waveform_grid",
    )

Integration with kkianamm/medtsllm4
------------------------------------
`PTBXLClassificationDataset` (datasets/ptbxl.py) already gives you exactly the
[N, T, 12] array this module wants:

    from datasets.ptbxl import PTBXLClassificationDataset, SUPERCLASS_ORDER
    ds = PTBXLClassificationDataset(config, split="train")

    # ds.records is STANDARDIZED (StandardScaler, see ClassificationDataset.
    # normalize_records). That's fine for spectrogram/scalogram/gaf/recurrence,
    # but for waveform_grid you usually want real mV units so Qwen's reading of
    # amplitude (e.g. ST elevation) is clinically meaningful. Undo it with the
    # *same* fitted scaler:
    raw = ds.normalizer.inverse_transform(
        ds.records.numpy().reshape(-1, ds.records.shape[-1])
    ).reshape(ds.records.shape)

    manifest = precompute_dataset_images(
        raw, ds.labels.numpy(), out_dir="data/ptbxl_images/train",
        method="waveform_grid", fs=PTBXLClassificationDataset.sampling_rate,
        lead_names=PTBXL_LEAD_NAMES, class_names=SUPERCLASS_ORDER,
    )

The per-sample text Qwen3.5 generates from these images is a natural drop-in
for the "descriptions" field PTB-XL already returns per record (currently just
`Patient information: {"age": ..., "sex": ...}`, consumed as `clip_prompts` in
models/medtsllm.py) — concatenate or replace it there to feed the new
description into the main MedTsLLM prompt. That wiring is NOT done by this
file; this file only produces the images.
"""

from __future__ import annotations

import io
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.signal import fftconvolve, resample, stft

import matplotlib
matplotlib.use("Agg")  # headless rendering, must precede pyplot import
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.collections import LineCollection


__all__ = [
    "PTBXL_LEAD_NAMES",
    "signal_to_image",
    "waveform_grid_image",
    "spectrogram_image",
    "scalogram_image",
    "gaf_image",
    "recurrence_plot_image",
    "precompute_dataset_images",
]

PTBXL_LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF",
                     "V1", "V2", "V3", "V4", "V5", "V6"]


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _as_tc(signal) -> np.ndarray:
    """Validate/cast to a float64 [T, C] array (accepts [T] too -> [T, 1])."""
    x = np.asarray(signal, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    if x.ndim != 2:
        raise ValueError(f"Expected signal of shape [T, C], got {x.shape}")
    return x


def _minmax_to(x: np.ndarray, lo=-1.0, hi=1.0, eps=1e-8) -> np.ndarray:
    x_min, x_max = np.nanmin(x), np.nanmax(x)
    x = (x - x_min) / (x_max - x_min + eps)
    return x * (hi - lo) + lo


def _colorize(x01: np.ndarray, cmap_name: str = "magma") -> np.ndarray:
    """[H, W] in [0, 1] -> [H, W, 3] uint8 via a matplotlib colormap (no Figure)."""
    cmap = cm.get_cmap(cmap_name)
    rgba = cmap(np.clip(x01, 0.0, 1.0))
    return (rgba[..., :3] * 255).astype(np.uint8)


def _tile_channels(panels: Sequence[np.ndarray], labels: Optional[Sequence[str]],
                    tile_cols: int, pad: int = 4,
                    bg=(255, 255, 255)) -> Image.Image:
    """Arrange a list of [H, W, 3] uint8 arrays into a labeled grid montage."""
    n = len(panels)
    cols = max(1, min(tile_cols, n))
    rows = int(np.ceil(n / cols))
    h, w = panels[0].shape[:2]

    label_h = 16 if labels is not None else 0
    cell_h, cell_w = h + label_h, w
    canvas = Image.new("RGB", (cols * cell_w + (cols + 1) * pad,
                                rows * cell_h + (rows + 1) * pad), bg)
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    for i, panel in enumerate(panels):
        r, c = divmod(i, cols)
        x0 = pad + c * (cell_w + pad)
        y0 = pad + r * (cell_h + pad)
        if labels is not None:
            draw.text((x0, y0), str(labels[i]), fill=(0, 0, 0), font=font)
        canvas.paste(Image.fromarray(panel), (x0, y0 + label_h))
    return canvas


def _downsample_len(x: np.ndarray, target_len: int) -> np.ndarray:
    if len(x) == target_len:
        return x
    return resample(x, target_len)


# --------------------------------------------------------------------------
# 1) waveform_grid -- the encoding to feed to Qwen3.5
# --------------------------------------------------------------------------
def waveform_grid_image(signal, fs: float = 100.0, lead_names: Optional[Sequence[str]] = None,
                         style: str = "clinical", layout: str = "stack",
                         row_height: float = 0.9, width_in: float = 11.0, dpi: int = 160,
                         title: Optional[str] = None, amplitude_unit: str = "mV",
                         vert_grid_major: Optional[float] = 0.5) -> Image.Image:
    """Stacked multi-lead line plot, optionally styled like ECG paper.

    Parameters
    ----------
    signal : [T, C] array. Pass the RAW (non-standardized) signal if you have
        it -- amplitude is informative for a clinical reading; see the module
        docstring for how to invert kkianamm/medtsllm4's StandardScaler.
    style : "clinical" (pink/red grid, like ECG paper) or "plain" (white bg).
    layout : "stack" (one row per channel, works for any C) or "grid" (compact
        rows x cols panel layout; only really makes sense for C == 12).
    vert_grid_major : fixed amplitude spacing (in `amplitude_unit`) between
        major horizontal gridlines, e.g. 0.5 mV like real ECG paper (minor
        lines are drawn at 1/5 of this). Pass None to fall back to a grid
        that auto-scales to each channel's own min/max -- use that for
        already-standardized (z-scored) input, where a fixed mV spacing is
        meaningless. The time axis is always real seconds with a fixed
        0.2 s / 0.04 s major/minor grid, regardless of this setting.
    """
    x = _as_tc(signal)
    T, C = x.shape
    t = np.arange(T) / float(fs)
    if lead_names is None:
        lead_names = PTBXL_LEAD_NAMES[:C] if C == 12 else [f"Ch{i+1}" for i in range(C)]

    grid_color = "#f6a3a3" if style == "clinical" else "#dddddd"
    trace_color = "#000000"
    bg_color = "#fffdfb" if style == "clinical" else "#ffffff"

    if layout == "grid" and C > 1:
        ncols = 4 if C >= 12 else int(np.ceil(np.sqrt(C)))
        nrows = int(np.ceil(C / ncols))
    else:
        ncols, nrows = 1, C

    fig, axes = plt.subplots(nrows, ncols, figsize=(width_in, row_height * nrows),
                              dpi=dpi, squeeze=False, facecolor=bg_color)
    for i in range(nrows * ncols):
        r, c = divmod(i, ncols)
        ax = axes[r][c]
        ax.set_facecolor(bg_color)
        if i >= C:
            ax.axis("off")
            continue
        ax.plot(t, x[:, i], color=trace_color, linewidth=0.9, zorder=3)
        is_bottom = (layout == "stack" and i == C - 1) or \
                    (layout == "grid" and (r == nrows - 1 or i + ncols >= C))

        if style == "clinical":
            if vert_grid_major is not None:
                y0 = vert_grid_major * np.floor(x[:, i].min() / vert_grid_major) - vert_grid_major
                y1 = vert_grid_major * np.ceil(x[:, i].max() / vert_grid_major) + vert_grid_major
            else:
                pad = 0.1 * (x[:, i].max() - x[:, i].min() + 1e-8)
                y0, y1 = x[:, i].min() - pad, x[:, i].max() + pad
            ax.set_ylim(y0, y1)
            ax.set_xlim(0, t[-1])

            # Fine ECG-paper grid drawn as two LineCollections (fast: avoids
            # creating a matplotlib Tick+Text object per line, which is what
            # made set_xticks(..., minor=True) with ~100+ lines/axis slow).
            minor_x = np.arange(0, t[-1] + 1e-9, 0.04)
            major_x = np.arange(0, t[-1] + 1e-9, 0.2)
            v_minor = LineCollection([[(xv, y0), (xv, y1)] for xv in minor_x],
                                      colors=grid_color, linewidths=0.4, alpha=0.7, zorder=0)
            v_major = LineCollection([[(xv, y0), (xv, y1)] for xv in major_x],
                                      colors=grid_color, linewidths=0.8, alpha=0.9, zorder=0)
            ax.add_collection(v_minor)
            ax.add_collection(v_major)

            if vert_grid_major is not None:
                minor_y = np.arange(y0, y1 + 1e-9, vert_grid_major / 5)
                major_y = np.arange(y0, y1 + 1e-9, vert_grid_major)
            else:
                minor_y = np.linspace(y0, y1, 5)
                major_y = minor_y[::2]
            h_minor = LineCollection([[(0, yv), (t[-1], yv)] for yv in minor_y],
                                      colors=grid_color, linewidths=0.4, alpha=0.7, zorder=0)
            h_major = LineCollection([[(0, yv), (t[-1], yv)] for yv in major_y],
                                      colors=grid_color, linewidths=0.8, alpha=0.9, zorder=0)
            ax.add_collection(h_minor)
            ax.add_collection(h_major)

        # Real ticks are kept sparse (cheap) and only carry labels on the
        # bottom-most row; every other row is unticked.
        ax.set_yticks([])
        if is_bottom:
            ax.set_xticks(np.arange(0, t[-1] + 1e-9, 1.0))
            ax.set_xlabel("time (s)", fontsize=8)
        else:
            ax.set_xticks([])
        ax.set_ylabel(lead_names[i], rotation=0, ha="right", va="center", fontsize=8)
        ax.tick_params(labelsize=7)
        for spine in ax.spines.values():
            spine.set_visible(style != "clinical")

    if title:
        fig.suptitle(title, fontsize=10)
    fig.text(0.99, 0.01, amplitude_unit, fontsize=7, ha="right", color="#666666")
    fig.tight_layout(rect=(0, 0, 1, 0.97) if title else None)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=bg_color)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


# --------------------------------------------------------------------------
# 2) spectrogram
# --------------------------------------------------------------------------
def spectrogram_image(signal, fs: float = 100.0, nperseg: int = 64, noverlap: int = 48,
                       cmap: str = "magma", tile_cols: int = 4, panel_size: int = 160,
                       lead_names: Optional[Sequence[str]] = None) -> Image.Image:
    x = _as_tc(signal)
    T, C = x.shape
    nperseg = min(nperseg, T)
    noverlap = min(noverlap, nperseg - 1)
    panels = []
    for c in range(C):
        f, tt, Zxx = stft(x[:, c], fs=fs, nperseg=nperseg, noverlap=noverlap)
        mag_db = 20 * np.log10(np.abs(Zxx) + 1e-8)
        mag01 = _minmax_to(mag_db, 0.0, 1.0)
        img = _colorize(np.flipud(mag01), cmap)
        panels.append(np.array(Image.fromarray(img).resize((panel_size, panel_size))))
    labels = lead_names if lead_names is not None else (
        PTBXL_LEAD_NAMES[:C] if C == 12 else [f"Ch{i+1}" for i in range(C)])
    return _tile_channels(panels, labels, tile_cols)


# --------------------------------------------------------------------------
# 3) scalogram (custom Morlet CWT -- no dependency on scipy.signal.cwt, which
#    is deprecated/removed in recent SciPy; built on the stable fftconvolve).
# --------------------------------------------------------------------------
def _morlet_wavelet(scale: float, w0: float = 6.0) -> np.ndarray:
    length = max(int(10 * scale), 8)
    t = np.arange(-length, length + 1, dtype=np.float64) / scale
    norm = (np.pi ** -0.25) / np.sqrt(scale)
    return norm * np.exp(1j * w0 * t) * np.exp(-0.5 * t ** 2)


def _cwt_morlet(x: np.ndarray, scales: np.ndarray, w0: float = 6.0) -> np.ndarray:
    out = np.empty((len(scales), len(x)), dtype=np.complex128)
    for i, s in enumerate(scales):
        out[i] = fftconvolve(x, _morlet_wavelet(s, w0), mode="same")
    return out


def scalogram_image(signal, fs: float = 100.0, n_scales: int = 64, w0: float = 6.0,
                     cmap: str = "magma", tile_cols: int = 4, panel_size: int = 160,
                     lead_names: Optional[Sequence[str]] = None) -> Image.Image:
    x = _as_tc(signal)
    T, C = x.shape
    scales = np.geomspace(1, max(T / 8, 2), n_scales)
    panels = []
    for c in range(C):
        coeffs = _cwt_morlet(x[:, c], scales, w0)
        mag01 = _minmax_to(np.abs(coeffs), 0.0, 1.0)
        img = _colorize(np.flipud(mag01), cmap)
        panels.append(np.array(Image.fromarray(img).resize((panel_size, panel_size))))
    labels = lead_names if lead_names is not None else (
        PTBXL_LEAD_NAMES[:C] if C == 12 else [f"Ch{i+1}" for i in range(C)])
    return _tile_channels(panels, labels, tile_cols)


# --------------------------------------------------------------------------
# 4) Gramian Angular Field
# --------------------------------------------------------------------------
def _gaf(x: np.ndarray, image_size: int, method: str = "summation") -> np.ndarray:
    x = _downsample_len(x, image_size)
    x = _minmax_to(x, -1.0, 1.0)
    x = np.clip(x, -1.0, 1.0)
    phi = np.arccos(x)
    sin_phi = np.sin(phi)
    if method == "summation":
        gaf = np.cos(phi[:, None] + phi[None, :])
    else:  # difference
        gaf = sin_phi[:, None] * x[None, :] - x[:, None] * sin_phi[None, :]
    return gaf


def gaf_image(signal, image_size: int = 64, method: str = "summation",
              cmap: str = "viridis", tile_cols: int = 4, panel_size: int = 160,
              lead_names: Optional[Sequence[str]] = None) -> Image.Image:
    x = _as_tc(signal)
    T, C = x.shape
    panels = []
    for c in range(C):
        g01 = _minmax_to(_gaf(x[:, c], image_size, method), 0.0, 1.0)
        img = _colorize(g01, cmap)
        panels.append(np.array(Image.fromarray(img).resize((panel_size, panel_size), Image.NEAREST)))
    labels = lead_names if lead_names is not None else (
        PTBXL_LEAD_NAMES[:C] if C == 12 else [f"Ch{i+1}" for i in range(C)])
    return _tile_channels(panels, labels, tile_cols)


# --------------------------------------------------------------------------
# 5) recurrence plot
# --------------------------------------------------------------------------
def recurrence_plot_image(signal, image_size: int = 64, binary: bool = False,
                           epsilon: Optional[float] = None, cmap: str = "viridis",
                           tile_cols: int = 4, panel_size: int = 160,
                           lead_names: Optional[Sequence[str]] = None) -> Image.Image:
    x = _as_tc(signal)
    T, C = x.shape
    panels = []
    for c in range(C):
        xc = _downsample_len(x[:, c], image_size)
        d = np.abs(xc[:, None] - xc[None, :])
        if binary:
            eps = epsilon if epsilon is not None else 0.1 * (d.max() + 1e-8)
            field01 = (d < eps).astype(np.float64)
        else:
            field01 = _minmax_to(np.exp(-d / (d.std() + 1e-8)), 0.0, 1.0)
        img = _colorize(field01, cmap)
        panels.append(np.array(Image.fromarray(img).resize((panel_size, panel_size), Image.NEAREST)))
    labels = lead_names if lead_names is not None else (
        PTBXL_LEAD_NAMES[:C] if C == 12 else [f"Ch{i+1}" for i in range(C)])
    return _tile_channels(panels, labels, tile_cols)


# --------------------------------------------------------------------------
# dispatcher
# --------------------------------------------------------------------------
_METHODS = {
    "waveform_grid": waveform_grid_image,
    "spectrogram": spectrogram_image,
    "scalogram": scalogram_image,
    "gaf": gaf_image,
    "recurrence_plot": recurrence_plot_image,
}


def signal_to_image(signal, method: str = "waveform_grid", **kwargs) -> Image.Image:
    """Dispatch to one of the encodings above. signal is [T, C]."""
    if method not in _METHODS:
        raise ValueError(f"Unknown method {method!r}; choose from {list(_METHODS)}")
    return _METHODS[method](signal, **kwargs)


# --------------------------------------------------------------------------
# whole-dataset precompute (parallel) -- writes PNGs + manifest.csv
# --------------------------------------------------------------------------
def _render_and_save(args):
    idx, signal, method, kwargs, out_dir = args
    img = signal_to_image(signal, method=method, **kwargs)
    path = os.path.join(out_dir, f"{idx:07d}.png")
    img.save(path)
    return idx, path


def precompute_dataset_images(signals, labels, out_dir: str, method: str = "waveform_grid",
                               class_names: Optional[Sequence[str]] = None,
                               num_workers: Optional[int] = None,
                               overwrite: bool = False, **method_kwargs) -> "list[dict]":
    """Render every sample in `signals` [N, T, C] to `out_dir`/{idx}.png in
    parallel and write `out_dir`/manifest.csv with columns (path,label[,label_name]).

    Returns the manifest as a list of dicts (also what gets written to CSV).
    This manifest is exactly what `train_image_classifier.py`'s
    `--train-manifest` / `--val-manifest` / `--test-manifest` expect.
    """
    import csv

    os.makedirs(out_dir, exist_ok=True)
    signals = np.asarray(signals)
    labels = np.asarray(labels)
    if len(signals) != len(labels):
        raise ValueError(f"signals ({len(signals)}) and labels ({len(labels)}) length mismatch")

    manifest_path = os.path.join(out_dir, "manifest.csv")
    todo = []
    rows = {}
    for idx in range(len(signals)):
        path = os.path.join(out_dir, f"{idx:07d}.png")
        if (not overwrite) and os.path.exists(path):
            rows[idx] = path
            continue
        todo.append((idx, signals[idx], method, method_kwargs, out_dir))

    if todo:
        with ProcessPoolExecutor(max_workers=num_workers) as ex:
            futures = [ex.submit(_render_and_save, t) for t in todo]
            done = 0
            for fut in as_completed(futures):
                idx, path = fut.result()
                rows[idx] = path
                done += 1
                if done % 500 == 0 or done == len(todo):
                    print(f"[precompute_dataset_images] rendered {done}/{len(todo)}")

    manifest = []
    with open(manifest_path, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["path", "label"] + (["label_name"] if class_names is not None else [])
        writer.writerow(header)
        for idx in range(len(signals)):
            label = int(labels[idx])
            row = {"path": rows[idx], "label": label}
            out_row = [rows[idx], label]
            if class_names is not None:
                row["label_name"] = class_names[label]
                out_row.append(class_names[label])
            writer.writerow(out_row)
            manifest.append(row)

    print(f"[precompute_dataset_images] wrote {len(manifest)} images + manifest to {manifest_path}")
    return manifest


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Render signal windows (.npy [N,T,C]) to images.")
    p.add_argument("--signals", required=True, help=".npy file, shape [N, T, C]")
    p.add_argument("--labels", required=True, help=".npy file, shape [N]")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--method", default="waveform_grid", choices=list(_METHODS))
    p.add_argument("--fs", type=float, default=100.0)
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--class-names", nargs="*", default=None)
    args = p.parse_args()

    sig = np.load(args.signals)
    lab = np.load(args.labels)
    kwargs = {"fs": args.fs} if args.method in ("waveform_grid", "spectrogram", "scalogram") else {}
    precompute_dataset_images(sig, lab, args.out_dir, method=args.method,
                               class_names=args.class_names, num_workers=args.num_workers,
                               **kwargs)
