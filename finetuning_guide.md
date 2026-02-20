# DeepECG Fine-Tuning Guide

## Checkpoint Options: `base_ssl.pt` vs `wcr_afib_5y.pt`

### `base_ssl.pt` — SSL backbone only

Trained with a self-supervised contrastive objective (wav2vec-style) on ECG data. Contains **no classification head** — only the transformer backbone plus SSL-specific projectors that are discarded at fine-tune time:

- `final_proj`: `[256, 768]`
- `project_q`: `[256, 256]`

When fine-tuning from `base_ssl.pt`:
- A fresh `Linear(768, 1)` head is randomly initialized (xavier uniform)
- The backbone starts from general ECG representations, not AFIB-specific ones
- This is the standard SSL → task transfer approach

### `wcr_afib_5y.pt` — full AFIB-5Y classifier

Already fine-tuned on an AFIB-5Y prediction task from an external dataset (likely UK Biobank / WCR cohort). Contains the backbone **plus** a trained classification head. Fine-tuning from this:
- Both backbone and head carry AFIB-specific knowledge from the outset
- Often outperforms `base_ssl.pt` when the downstream task and population match
- Risk: original training distribution may differ from your data → verify performance

---

## Model Architecture

Both checkpoints use the same `ecg_transformer_classifier` architecture:

| Component | Shape | Notes |
|---|---|---|
| Input | (batch, 12, 2500) | 12-lead ECG, 2500 timepoints @ 500 Hz |
| Conv feature extractor | `[(256, 2, 2)] × 4` | Reduces time dimension |
| Post-extract projection | `Linear(256, 768)` | Projects to transformer dim |
| Transformer encoder | 12 layers, 768 dim, 12 heads, FFN 3072 | Core backbone |
| Mean pooling | (batch, seq_len, 768) → (batch, 768) | No weights |
| Classification head (`proj`) | `Linear(768, 1)` — weight `[1, 768]`, bias `[1]` | Output: raw logit |

**Structural differences between checkpoints:**

- `base_ssl.pt` (215 keys): backbone keys prefixed `encoder.layers.*` + SSL projectors (`final_proj`, `project_q`); no `proj` head
- `wcr_afib_5y.pt` (209 keys): backbone keys prefixed `encoder.encoder.layers.*` + `proj.weight`/`proj.bias`; no SSL projectors

The key prefix difference (`encoder.` vs `encoder.encoder.`) reflects that the fine-tuning wrapper (`ECGTransformerFinetuningModel`) adds one level of nesting around the backbone.

---

## Resetting the Classification Head

When fine-tuning from `wcr_afib_5y.pt`, resetting the head can be useful when:
- Your dataset's class balance/prevalence differs substantially from the WCR training data
- You want AFIB-specific backbone representations but a fresh decision boundary for your cohort
- You suspect the pre-trained head is overfit to a different population

Script to create a modified checkpoint with a reset head:

```python
# Run with: conda run -n task_env python reset_head.py
# (from the repo root)
import torch
import torch.nn as nn

ckpt_path = "ECG-algos/DeepECG_Docker/weights/wcr_afib_5y.pt"
out_path  = "ECG-algos/DeepECG_Docker/weights/wcr_afib_5y_reset_head.pt"

ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

w = ckpt["model"]["proj.weight"]  # shape [1, 768]
b = ckpt["model"]["proj.bias"]    # shape [1]

nn.init.xavier_uniform_(w)
nn.init.constant_(b, 0.0)

ckpt["model"]["proj.weight"] = w
ckpt["model"]["proj.bias"]   = b

torch.save(ckpt, out_path)
print("Saved to", out_path)
```

Then point `model.model_path` to `wcr_afib_5y_reset_head.pt` in the fine-tuning command.

---

## Strategy Comparison

| Starting point | Head init | When to use |
|---|---|---|
| `base_ssl.pt` (current) | Random (xavier uniform) | Conservative baseline; well-studied SSL transfer |
| `wcr_afib_5y.pt` | Pre-trained AFIB head | Best if source/target distributions are similar |
| `wcr_afib_5y.pt` + reset head | Random (xavier uniform) | AFIB backbone knowledge, fresh head for your data |
