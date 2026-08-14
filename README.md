# MedTsLLM + ConvNeXt + Semantic Q-Former overlay

## Architecture implemented

- MedTsLLM waveform tokens
- ConvNeXt **pre-pooling spatial tokens**
- 512-D projections + modality embeddings
- concatenated multimodal memory (no image-token interpolation)
- 5 PTB-XL class-semantic queries + 27 generic learned queries
- 4-layer Q-Former
- supervised MedTsLLM/ConvNeXt alignment
- query-diversity loss
- gated MedTsLLM residual into Q-Former output
- Q-Former -> LLM projection
- BioMedCoOp semantic prototype head when enabled; otherwise linear classifier
- auxiliary MedTsLLM, ConvNeXt, and query classifiers

## Important compatibility point

The available uploaded sources contain the earlier `TriMedTsLLM` implementation,
but not the Python file that currently generates the ConvNeXt image in
`signal-derived`. This overlay therefore deliberately **does not invent a new
ECG-to-image transformation**.

Reuse the same image tensor from your existing working ConvNeXt run and expose it
as one of:

- `inputs["x_image"]`
- `inputs["image"]`
- `inputs["images"]`

Shape must be `[B,1,H,W]` or `[B,3,H,W]`.

If your current ConvNeXt class builds the image inside `forward`, move/reuse that
exact image-building code before calling this model and pass the result as
`x_image`.

## Install

```bash
cd ~/Kiana2/signal-derived
cp /path/to/overlay/models/semantic_qformer_components.py models/
cp /path/to/overlay/models/medtsllm_convnext_semantic_qformer.py models/
```

Patch `models/__init__.py` using `models_init_patch.txt`.

## Make the experiment config

Start from the exact config that already produced your ConvNeXt result:

```bash
cd ~/Kiana2/signal-derived
cp configs/datasets/ptbxl_image_fusion.toml \
   configs/datasets/ptbxl_semantic_qformer.toml
```

Merge `configs/ptbxl_semantic_qformer_section.toml` into the copied config.
Set the model lookup name used by your training framework to:

```text
medtsllm_convnext_semantic_qformer
```

Do not change PTB-XL splits, input preprocessing, optimizer, batch size, LLM, or
ECG-image representation in this first experiment.

## Verify image key

```bash
grep -RniE "x_image|images|ConvNeXt|convnext" datasets models
```

Connect the existing image tensor to `inputs["x_image"]` if necessary.

## Syntax check

```bash
python -m py_compile \
  models/semantic_qformer_components.py \
  models/medtsllm_convnext_semantic_qformer.py
```

## Run

After creating `configs/datasets/ptbxl_semantic_qformer.toml`:

```bash
bash run_semantic_qformer.sh
```

## Fair ablation

Keep 32 total queries for the first run. This lets you compare your existing
standard 32-query Q-Former against 5 semantic + 27 generic queries without also
changing token budget.

Recommended initial loss weights:

- MedTsLLM auxiliary CE: 0.10
- ConvNeXt auxiliary CE: 0.10
- query auxiliary CE: 0.10
- alignment: 0.05
- query diversity: 0.01

Use one checkpoint-selection criterion across all models. If you change to
validation macro-F1, apply the same selection to all baseline/fusion runs.
