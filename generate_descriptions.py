"""
generate_descriptions.py

Runs Qwen3.5 over the waveform_grid images (from signal_to_image.py /
prepare_ptbxl_images.py) to produce ONE free-text, per-sample expert
description per image -- grounded in what that specific tracing shows, not
a generic per-class textbook sentence (that's what BIOMEDCOOP_TS_TEMPLATES
already gives you; the point here is the opposite of that).

Input:  a manifest.csv (path,label[,label_name]) as written by
        precompute_dataset_images() / prepare_ptbxl_images.py.
Output: descriptions.csv with columns (path, description), resumable and
        incrementally flushed -- safe to Ctrl-C and rerun, it skips rows
        already done. Join it back onto the dataset by the index embedded
        in the filename (0000042.png -> 42), not by row order; see
        datasets/ptbxl_qwen.py for that join.

The image is the ONLY thing in the prompt -- never the label. Qwen sees
exactly what a real reader would see (the strip), nothing else, so the same
procedure is valid to run on train/val/test alike with no leakage.

Usage
-----
    python generate_descriptions.py \
        --manifest data/ptbxl_images/train/manifest.csv \
        --out      data/ptbxl_images/train/descriptions.csv \
        --model Qwen/Qwen3.5-9B --batch-size 8

Requires: transformers >= 4.57 from a build with Qwen3.5 support (added
2026-02-09 -- if `Qwen3_5ForConditionalGeneration` / AutoModelForImageTextToText
doesn't recognize the checkpoint, `pip install -U transformers`), torch,
accelerate, pillow, pandas, tqdm.

Known rough edge (as of this writing): some users report garbled output on
image inputs traced to a missing generation_config.json on the Hub repo --
see https://huggingface.co/Qwen/Qwen3.5-9B/discussions. If you see repeated
junk tokens instead of prose, that's the first thing to check (pin/upgrade
transformers, or pass --do-sample with a explicit --temperature so generation
doesn't rely on a bundled generation_config).
"""

from __future__ import annotations

import argparse
import csv
import os

import pandas as pd
from tqdm import tqdm


PROMPT_TEXT = (
    "You are an expert cardiologist reading a 12-lead ECG strip rendered on "
    "standard ECG paper (pink grid, 25 mm/s, 10 mm/mV). Describe only what is "
    "visible in THIS image. In 3-6 sentences, cover: heart rate and rhythm "
    "regularity; PR/QRS/QT timing if discernible; ST-segment and T-wave "
    "appearance; and any notable axis or morphology findings, noting which "
    "leads they appear in. End with a one-sentence impression. Be concrete "
    "and specific to what this particular tracing shows -- avoid generic "
    "textbook statements that could describe any ECG."
)


# ============================================================================
# Model
# ============================================================================
def load_model(model_id: str, dtype: str = "bfloat16", flash_attention: bool = False):
    import torch
    from transformers import AutoProcessor, AutoModelForImageTextToText

    processor = AutoProcessor.from_pretrained(model_id)
    # required for correct batched generation: all sequences must end at the
    # same column so a single slice index works for every row in the batch.
    processor.tokenizer.padding_side = "left"

    kwargs = dict(dtype=getattr(torch, dtype), device_map="auto")
    if flash_attention:
        kwargs["attn_implementation"] = "flash_attention_2"
    model = AutoModelForImageTextToText.from_pretrained(model_id, **kwargs)
    model.eval()
    return processor, model


def run_batch(processor, model, image_paths, max_new_tokens, do_sample, temperature):
    import torch

    messages_batch = [
        [{"role": "user", "content": [{"type": "image", "image": p},
                                       {"type": "text", "text": PROMPT_TEXT}]}]
        for p in image_paths
    ]
    inputs = processor.apply_chat_template(
        messages_batch, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt", padding=True,
    ).to(model.device)

    gen_kwargs = dict(max_new_tokens=max_new_tokens, do_sample=do_sample)
    if do_sample:
        gen_kwargs["temperature"] = temperature

    with torch.inference_mode():
        out = model.generate(**inputs, **gen_kwargs)

    gen_only = out[:, inputs["input_ids"].shape[-1]:]
    texts = processor.batch_decode(gen_only, skip_special_tokens=True)
    return [t.strip() for t in texts]


# ============================================================================
# Orchestration (resumable, incrementally flushed, batch-then-fallback)
# ============================================================================
def already_done(out_path: str) -> set:
    if not os.path.exists(out_path):
        return set()
    try:
        return set(pd.read_csv(out_path)["path"])
    except Exception:
        return set()


def generate_for_manifest(manifest_csv, out_csv, generate_fn, batch_size=8, limit=None):
    """generate_fn(paths: list[str]) -> list[str]  (one description per path).

    Kept as an injected function so the IO/resume/batching logic can be
    exercised without a real model -- see the smoke test in the repo notes.
    """
    df = pd.read_csv(manifest_csv)
    if limit:
        df = df.iloc[:limit]

    done = already_done(out_csv)
    todo = df[~df["path"].isin(done)]
    print(f"[generate_descriptions] {len(done)} already done, {len(todo)} remaining "
          f"(of {len(df)} total)")

    mode = "a" if done else "w"
    with open(out_csv, mode, newline="") as f:
        writer = csv.writer(f)
        if mode == "w":
            writer.writerow(["path", "description"])

        for start in tqdm(range(0, len(todo), batch_size)):
            batch_paths = todo["path"].iloc[start:start + batch_size].tolist()
            try:
                texts = generate_fn(batch_paths)
            except Exception as e:
                print(f"[warn] batch of {len(batch_paths)} failed ({e}); retrying one-by-one")
                texts = []
                for p in batch_paths:
                    try:
                        texts.append(generate_fn([p])[0])
                    except Exception as e2:
                        print(f"[warn] failed on {p}: {e2}")
                        texts.append("")
            for p, t in zip(batch_paths, texts):
                writer.writerow([p, t])
            f.flush()

    print(f"[generate_descriptions] wrote {out_csv}")


# ============================================================================
# CLI
# ============================================================================
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--model", default="Qwen/Qwen3.5-9B")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-new-tokens", type=int, default=220)
    p.add_argument("--do-sample", action="store_true")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--flash-attention", action="store_true")
    p.add_argument("--limit", type=int, default=None, help="smoke-test on the first N rows")
    args = p.parse_args()

    processor, model = load_model(args.model, dtype=args.dtype, flash_attention=args.flash_attention)

    def generate_fn(paths):
        return run_batch(processor, model, paths, args.max_new_tokens, args.do_sample, args.temperature)

    generate_for_manifest(args.manifest, args.out, generate_fn,
                           batch_size=args.batch_size, limit=args.limit)


if __name__ == "__main__":
    main()
