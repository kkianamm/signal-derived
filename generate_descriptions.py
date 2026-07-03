"""
generate_descriptions.py

Runs Qwen3.5 over waveform_grid ECG images to produce ONE clean, final,
medical ECG description per image.

Two bugs caused the broken, mid-reasoning-looking outputs in older CSVs:

1. Qwen3.5 is a thinking model by default. Its chat template puts the
   *opening* <think> tag in the prompt itself, so it never shows up in the
   decoded generation -- only the *closing* </think> does, and only if the
   model actually reaches it. With max_new_tokens=160, it almost never did,
   so descriptions.csv filled up with raw, unfinished reasoning text that
   has no <think>/</think> tags in it at all.
2. The old cleanup regex `<think>.*?</think>` only fires when BOTH tags are
   present in the decoded text, which per #1 they never are for a truncated
   generation -- so it silently did nothing on exactly the rows that needed it.

Fix: enable_thinking=False is now passed to apply_chat_template so the model
answers directly, extract_final_answer() handles the "only </think> can
appear" case correctly, and any row that still comes back empty (thinking
got cut off anyway, e.g. on an older transformers/template version) is
automatically retried once with a much larger token budget instead of being
written as blank or garbage.

--style paragraph (default) keeps the original compact 2-5 sentence caption.
--style structured produces a six-section report (Rhythm/Rate, Axis,
Intervals, QRS/R-wave progression, ST-T, Impression).
"""

from __future__ import annotations

import argparse
import csv
import os
import re

import pandas as pd
from tqdm import tqdm


PARAGRAPH_SYSTEM_PROMPT = """You are a clinical ECG description generator for a medical machine-learning dataset.

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


PARAGRAPH_PROMPT_TEXT = """Write one concise clinical ECG description for this 12-lead ECG.

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


STRUCTURED_SECTION_LABELS = [
    "Rhythm/Rate",
    "Axis",
    "Intervals",
    "QRS/R-wave progression",
    "ST-T",
    "Impression",
]


STRUCTURED_SYSTEM_PROMPT = """You are a clinical ECG description generator for a medical machine-learning dataset.

Your task is to write ONLY the final ECG report.
Do not mention the user, request, task, image, prompt, or model.
Do not explain your reasoning or show step-by-step analysis.
Do not mention counting boxes, large squares, calculations, or visual inspection steps.
Do not use phrases such as: "The user wants", "I will", "Let's", "Look at", "It looks like", or "we can see".

Write the report as exactly six labeled sections, in this order, each starting on its
own line with the exact label below followed by a colon, then 1 to 3 concise clinical
sentences:

Rhythm/Rate:
Axis:
Intervals:
QRS/R-wave progression:
ST-T:
Impression:

Use cautious, hedged language ("appears", "no definite", "without clear") whenever a
finding is not clearly visible. Do not invent exact numeric values (bpm, ms, degrees)
unless they are clearly estimable from the image; otherwise describe the finding
qualitatively instead of fabricating a number.
"""


STRUCTURED_PROMPT_TEXT = """Write a structured clinical ECG report for this 12-lead ECG image, covering:

- Rhythm/Rate: regularity, whether P waves are visible before each QRS, and an
  approximate rate category or range if it is clearly estimable.
- Axis: frontal QRS axis impression from limb lead polarity (leads I, II, III).
- Intervals: PR interval, QRS duration, and QT/QTc, described qualitatively
  (e.g. "within normal limits", "appears prolonged") unless a number is clearly readable.
- QRS/R-wave progression: QRS morphology, bundle branch block pattern if present,
  precordial R-wave progression and transition zone, and any pathologic Q waves.
- ST-T: ST-segment elevation or depression, and T-wave appearance.
- Impression: a one to two sentence overall summary.

Output only the six labeled sections, in the exact order and format given above --
no extra commentary before or after."""


STYLE_PROMPTS = {
    "paragraph": (PARAGRAPH_SYSTEM_PROMPT, PARAGRAPH_PROMPT_TEXT),
    "structured": (STRUCTURED_SYSTEM_PROMPT, STRUCTURED_PROMPT_TEXT),
}


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

BAD_SENTENCE_PATTERNS = [
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

_STRUCTURED_LABEL_PATTERN = re.compile(
    r"^\s*\*{0,2}(" + "|".join(re.escape(lbl) for lbl in STRUCTURED_SECTION_LABELS) + r")\*{0,2}\s*:\s*(.*)$",
    flags=re.IGNORECASE,
)


def extract_final_answer(text, thinking_was_open: bool = True) -> str:
    """Return only what comes after Qwen's thinking phase, if any.

    Qwen's chat template puts the OPENING <think> tag in the *prompt*
    (as part of add_generation_prompt when thinking is enabled), so it
    never appears in the decoded *generated* tokens -- only the closing
    </think> does, and only if the model actually reached it before
    max_new_tokens ran out.

    That means "no tags at all in the decoded text" is ambiguous on its
    own -- it's what you see BOTH when enable_thinking=False genuinely
    produced a direct answer, AND when a think block was opened by the
    prompt but generation was cut off before the model ever closed it.
    `thinking_was_open` resolves the ambiguity: run_batch() determines it
    by inspecting the actual prompt tokens (ground truth), not by
    guessing from the completion text.

    - </think> present            -> keep only what follows it.
    - <think> present (no close)  -> belt-and-suspenders strip, in case a
                                      matched pair did survive decoding.
    - no tags, thinking_was_open  -> still mid-thought; nothing to
                                      recover, return "" (triggers retry)
                                      rather than passing raw reasoning
                                      through as if it were a description.
    - no tags, thinking not open  -> this genuinely is the direct answer.
    """
    if text is None:
        return ""
    text = str(text).strip()
    if "</think>" in text:
        return text.split("</think>", 1)[1].strip()
    if "<think>" in text.lower():
        stripped = re.sub(r"<think>.*?</think>", "", text, flags=re.IGNORECASE | re.DOTALL).strip()
        return stripped
    if thinking_was_open:
        return ""
    return text


def clean_paragraph_description(text, thinking_was_open: bool = True) -> str:
    """Clean Qwen output so the CSV contains only the final 2-5 sentence description."""
    text = extract_final_answer(text, thinking_was_open=thinking_was_open)
    if not text:
        return ""

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

    sentences = [
        s for s in sentences
        if not any(re.search(p, s, flags=re.IGNORECASE) for p in BAD_SENTENCE_PATTERNS)
    ]
    if len(sentences) > 5:
        text = " ".join(sentences[:5])

    return text.strip()


def clean_structured_description(text, thinking_was_open: bool = True) -> str:
    """Clean Qwen output for --style structured, KEEPING the six section labels.

    Unlike clean_paragraph_description, this must not strip lines like
    "Axis:" or cap the total sentence count -- a full six-section report
    is expected to run well past 5 sentences.
    """
    text = extract_final_answer(text, thinking_was_open=thinking_was_open)
    if not text:
        return ""

    text = re.sub(r"^\s*(?:sure|certainly|here is|here's)\b.*?:\s*", "", text, flags=re.IGNORECASE | re.DOTALL)
    text = text.replace("**", "")

    seen_labels = set()
    lines_out = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        line = re.sub(r"^\s*[-*•]+\s*", "", line)

        m = _STRUCTURED_LABEL_PATTERN.match(line)
        if m:
            label, rest = m.group(1), m.group(2).strip()
            canonical = next(l for l in STRUCTURED_SECTION_LABELS if l.lower() == label.lower())
            seen_labels.add(canonical)
            lines_out.append(f"{canonical}:" + (f" {rest}" if rest else ""))
            continue

        if any(re.match(p, line, flags=re.IGNORECASE) for p in BAD_LINE_PATTERNS):
            continue
        if any(re.search(p, line, flags=re.IGNORECASE) for p in BAD_SENTENCE_PATTERNS):
            continue

        lines_out.append(line)

    if len(seen_labels) < len(STRUCTURED_SECTION_LABELS):
        missing = [l for l in STRUCTURED_SECTION_LABELS if l not in seen_labels]
        print(f"[warn] structured output missing section(s): {missing}")

    return "\n".join(lines_out).strip()


STYLE_CLEANERS = {
    "paragraph": clean_paragraph_description,
    "structured": clean_structured_description,
}


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


def run_batch(processor, model, image_paths, max_new_tokens, do_sample, temperature, style="paragraph"):
    import torch

    system_prompt, prompt_text = STYLE_PROMPTS[style]
    cleaner = STYLE_CLEANERS[style]

    messages_batch = [
        [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": p},
                    {"type": "text", "text": prompt_text},
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
        enable_thinking=False,  # this is a direct-description task, not multi-step
                                 # reasoning; also sidesteps the truncated-thinking bug.
    ).to(model.device)

    # Ground truth check: did the prompt itself end with an *unclosed*
    # <think> tag? That's what enable_thinking=True inserts before
    # generation starts. We check the real tokenized prompt rather than
    # guessing from the completion, because "no tags in the completion"
    # looks identical whether enable_thinking=False worked (direct
    # answer) or it silently didn't (truncated mid-thought, tags never
    # generated). Left padding means the true prompt content is always
    # at the *end* of each row regardless of batch padding.
    prompt_tail = processor.tokenizer.decode(inputs["input_ids"][0][-12:], skip_special_tokens=False)
    thinking_was_open = "<think>" in prompt_tail and "</think>" not in prompt_tail
    if thinking_was_open and not run_batch._warned_thinking_still_open:
        print(
            "[warn] enable_thinking=False does not appear to be honored by this "
            "model/processor version -- the prompt still opens a <think> block. "
            "Falling back to detecting truncated reasoning and retrying those rows; "
            "consider checking processor.chat_template for the correct kwarg name, "
            "or upgrading transformers."
        )
        run_batch._warned_thinking_still_open = True

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
    return [cleaner(t, thinking_was_open=thinking_was_open) for t in texts]


run_batch._warned_thinking_still_open = False


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


def generate_for_manifest(manifest_csv, out_csv, generate_fn, retry_fn=None, batch_size=8, limit=None):
    """generate_fn(paths: list[str]) -> list[str], one description per path.

    retry_fn, if given, is called on a single path whenever its description
    comes back empty -- i.e. extract_final_answer() determined the model was
    still inside its thinking phase when generation stopped -- with a larger
    token budget, before that row is given up on.
    """
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

            if retry_fn is not None:
                for i, (p, t) in enumerate(zip(batch_paths, texts)):
                    if t:
                        continue
                    try:
                        retried = retry_fn([p])[0]
                    except Exception as e2:
                        print(f"[warn] retry failed on {p}: {e2}")
                        retried = ""
                    if retried:
                        texts[i] = retried

            for p, t in zip(batch_paths, texts):
                if not t:
                    print(
                        f"[warn] empty description for {p} (thinking likely still cut off "
                        f"even after retry) -- consider raising --retry-max-new-tokens"
                    )
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
    p.add_argument("--style", choices=["paragraph", "structured"], default="paragraph",
                    help="paragraph: compact 2-5 sentence caption (default, backward compatible). "
                         "structured: six labeled sections (Rhythm/Rate, Axis, Intervals, "
                         "QRS/R-wave progression, ST-T, Impression). Use a separate --out path "
                         "from any existing paragraph-style CSV so the two styles don't mix.")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-new-tokens", type=int, default=160)
    p.add_argument("--retry-max-new-tokens", type=int, default=None,
                    help="Token budget for a single-image retry when a description comes back "
                         "empty, i.e. thinking got cut off before the answer. Defaults to "
                         "max(384, 3 * --max-new-tokens).")
    p.add_argument("--do-sample", action="store_true")
    p.add_argument("--temperature", type=float, default=0.3)
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--flash-attention", action="store_true")
    p.add_argument("--limit", type=int, default=None, help="smoke-test on the first N rows")

    args = p.parse_args()

    if args.style == "structured" and args.max_new_tokens < 320:
        print(
            f"[warn] --style structured writes six sections (often 12-15 sentences) but "
            f"--max-new-tokens={args.max_new_tokens} was tuned for the short paragraph style. "
            f"Rows will likely come back empty on the first pass and rely on the retry budget "
            f"below -- consider passing --max-new-tokens 400 or higher explicitly."
        )

    retry_budget = args.retry_max_new_tokens or max(384, args.max_new_tokens * 3)

    processor, model = load_model(args.model, dtype=args.dtype, flash_attention=args.flash_attention)

    def generate_fn(paths):
        return run_batch(
            processor,
            model,
            paths,
            args.max_new_tokens,
            args.do_sample,
            args.temperature,
            style=args.style,
        )

    def retry_fn(paths):
        return run_batch(
            processor,
            model,
            paths,
            retry_budget,
            args.do_sample,
            args.temperature,
            style=args.style,
        )

    generate_for_manifest(
        args.manifest,
        args.out,
        generate_fn,
        retry_fn=retry_fn,
        batch_size=args.batch_size,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
