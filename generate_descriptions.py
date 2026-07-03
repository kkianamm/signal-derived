"""
generate_descriptions.py

Runs Qwen3.5 over waveform_grid ECG images to produce ONE clean, final,
medical ECG description per image.

This edited version prevents outputs like:
"The user wants...", step-by-step reasoning, headings, bullets, and markdown.
It also removes <think>...</think> blocks and other reasoning-style text before
writing descriptions.csv.
"""

from __future__ import annotations

import argparse
import csv
import os
import re

import pandas as pd
from tqdm import tqdm


SYSTEM_PROMPT = """You are a clinical ECG description generator for a medical machine-learning dataset.

Your task is to write ONLY the final ECG medical description.
Do not mention the user, request, task, image, prompt, or model.
Do not explain your reasoning.
Do not show step-by-step analysis.
Do not mention counting boxes, large squares, calculations, or visual inspection steps.
Do not use headings, bullet points, numbered lists, or markdown.
Do not use phrases such as: "The user wants", "I will", "Let's", "Look at", "It looks like", or "we can see".

Write 2 to 5 concise clinical sentences in paragraph form.
Use cautious language when a finding is uncertain.
Do not invent exact measurements unless they are clearly visible.
"""


PROMPT_TEXT = """Write one concise clinical ECG description for this 12-lead ECG.

Rules:
- Output only the final description.
- Do not describe how you estimate heart rate.
- Do not mention counting boxes or large squares.
- Do not use phrases such as "wait", "look closer", "it looks like", "let's", or "rate estimation".
- Do not provide calculations.
- Do not use headings, bullets, markdown, or numbered lists.
- Use 2 to 4 short clinical sentences.
- Use cautious language if uncertain.

The description should mention only final visible findings such as rhythm regularity, approximate rate category, QRS morphology, ST-T changes if visible, and overall impression.
"""


BAD_LINE_PATTERNS = [
    r"^\s*the user wants\b.*$",
    r"^\s*the user is asking\b.*$",
    r"^\s*i will\b.*$",
    r"^\s*let'?s\b.*$",
    r"^\s*look at\b.*$",
    r"^\s*we need to\b.*$",
    r"^\s*count the\b.*$",
    r"^\s*rate calculation\b.*$",
    r"^\s*from the start\b.*$",
    r"^\s*it looks like\b.*$",
    r"^\s*maybe\b.*$",
    r".*\bcount(?:ing)? the\b.*",
    r".*\blarge squares?\b.*",
    r".*\blarge boxes?\b.*",
    r".*\brate estimation\b.*",
    r".*\bwait\b.*",
    r".*\blet'?s\b.*",
    r".*\blook closer\b.*",
    r".*\btrace them\b.*",
    r".*\b300\s*/\b.*",
    ]


SECTION_HEADING_PATTERN = re.compile(
    r"^\s*(?:\d+\s*[\.)]\s*)?\*{0,2}"
    r"(?:heart rate and rhythm|rhythm|intervals|axis|st[- ]?segment|t[- ]?waves?|qrs|qt|impression|conclusion|findings)"
    r"\*{0,2}\s*:?\s*$",
    flags=re.IGNORECASE,
)


def clean_description(text: str) -> str:
    """Clean Qwen output so the CSV contains only the final ECG description."""
    if text is None:
        return ""

    text = str(text).strip()

    # Remove Qwen/thinking-style hidden reasoning if emitted.
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.IGNORECASE | re.DOTALL)

    # Remove common assistant wrappers.
    text = re.sub(r"^\s*(?:sure|certainly|here is|here's)\b.*?:\s*", "", text, flags=re.IGNORECASE | re.DOTALL)

    # Remove markdown formatting but keep clinical text.
    text = text.replace("**", "")
    text = re.sub(r"^\s*#+\s*", "", text, flags=re.MULTILINE)

    cleaned_lines = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue

        # Remove bullets / numbering prefixes.
        line = re.sub(r"^\s*[-*•]+\s*", "", line)
        line = re.sub(r"^\s*\d+\s*[\.)]\s*", "", line)

        if SECTION_HEADING_PATTERN.match(line):
            continue

        if any(re.match(pattern, line, flags=re.IGNORECASE) for pattern in BAD_LINE_PATTERNS):
            continue

        cleaned_lines.append(line)

    text = " ".join(cleaned_lines)

    # Normalize spaces.
    text = re.sub(r"\s+", " ", text).strip()

    # Remove remaining obvious preamble before the first clinical sentence.
    preamble_markers = [
        "Heart rate",
        "The ECG",
        "This ECG",
        "This tracing",
        "Sinus",
        "Regular",
        "Atrial",
        "Ventricular",
        "There is",
        "No acute",
    ]
    lower_text = text.lower()
    candidate_positions = [lower_text.find(marker.lower()) for marker in preamble_markers if lower_text.find(marker.lower()) >= 0]
    if candidate_positions:
        first = min(candidate_positions)
        if first > 0:
            text = text[first:].strip()

     # Keep output compact for captions.
     sentences = re.split(r"(?<=[.!?])\s+", text)
     sentences = [s.strip() for s in sentences if s.strip()]
            bad_sentence_patterns = [
            r"\bcount(?:ing)? the\b",
            r"\blarge squares?\b",
            r"\blarge boxes?\b",
            r"\brate estimation\b",
            r"\bwait\b",
            r"\blet'?s\b",
            r"\blook closer\b",
            r"\btrace them\b",
            r"\b300\s*/\b",
            r"\bcalculate\b",
        ]
        
            sentences = [
                s for s in sentences
                if not any(re.search(p, s, flags=re.IGNORECASE) for p in bad_sentence_patterns)
            ]
            if len(sentences) > 5:
                text = " ".join(sentences[:5])

            return text.strip()


# ============================================================================
# Model
# ============================================================================

def load_model(model_id: str, dtype: str = "bfloat16", flash_attention: bool = False):
    import torch
    from transformers import AutoProcessor, AutoModelForImageTextToText

    processor = AutoProcessor.from_pretrained(model_id)

    # Required for correct batched generation: all sequences must end at the
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
        [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": p},
                    {"type": "text", "text": PROMPT_TEXT},
                ],
            },
        ]
        for p in image_paths
    ]

    inputs = processor.apply_chat_template(
        messages_batch,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        padding=True,
    ).to(model.device)

    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        repetition_penalty=1.05,
    )
    if do_sample:
        gen_kwargs["temperature"] = temperature

    with torch.inference_mode():
        out = model.generate(**inputs, **gen_kwargs)

    gen_only = out[:, inputs["input_ids"].shape[-1]:]
    texts = processor.batch_decode(gen_only, skip_special_tokens=True)
    return [clean_description(t) for t in texts]


# ============================================================================
# Orchestration: resumable, incrementally flushed, batch-then-fallback
# ============================================================================

def already_done(out_path: str) -> set:
    if not os.path.exists(out_path):
        return set()
    try:
        return set(pd.read_csv(out_path)["path"])
    except Exception:
        return set()


def generate_for_manifest(manifest_csv, out_csv, generate_fn, batch_size=8, limit=None):
    """generate_fn(paths: list[str]) -> list[str], one description per path."""
    df = pd.read_csv(manifest_csv)
    if limit:
        df = df.iloc[:limit]

    done = already_done(out_csv)
    todo = df[~df["path"].isin(done)]

    print(
        f"[generate_descriptions] {len(done)} already done, {len(todo)} remaining "
        f"(of {len(df)} total)"
    )

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
    p.add_argument("--max-new-tokens", type=int, default=160)
    p.add_argument("--do-sample", action="store_true")
    p.add_argument("--temperature", type=float, default=0.3)
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--flash-attention", action="store_true")
    p.add_argument("--limit", type=int, default=None, help="smoke-test on the first N rows")

    args = p.parse_args()

    processor, model = load_model(args.model, dtype=args.dtype, flash_attention=args.flash_attention)

    def generate_fn(paths):
        return run_batch(
            processor,
            model,
            paths,
            args.max_new_tokens,
            args.do_sample,
            args.temperature,
        )

    generate_for_manifest(
        args.manifest,
        args.out,
        generate_fn,
        batch_size=args.batch_size,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
