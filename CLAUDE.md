# Thesis Project: ECG-Based AFIB Risk Prediction with DeepECG-SSL

## Project Overview
This project applies the DeepECG-SSL model to predict incident Atrial Fibrillation (AFIB) within 5 years from a single baseline sinus-rhythm ECG. Both workflows share the same goal and input/output structure: **one ECG per patient → one AFIB risk score per patient** (binary classification: 0=control, 1=AF positive).

All ECG data is 500 Hz, 5-second recordings (2500 timepoints), initially in 8-lead format (I, II, V1-V6). The DeepECG pipeline requires 12-lead format — the four limb leads (III, aVR, aVL, aVF) are derived via Einthoven's triangle.

### Model Output Format Comparison

| | Workflow 1 (Testing) | Workflow 2 (Fine-tuning) |
|---|---|---|
| Model | `wcr_afib_5y.pt` (pre-trained) | Fine-tuned from `base_ssl.pt` |
| Head architecture | `ecg_transformer_classifier`: mean pool over time → `Linear(768, 1)` | **Same** — `ecg_transformer_classifier`, `num_labels=1` |
| Per-ECG output | Single **probability** (0–1); sigmoid applied inside `WCREcgTransformer.__call__` | Single **raw logit**; sigmoid applied only at eval metric time |
| Output file | `wcr_afib_5y_*_probabilities.csv` → `prob_deepecg.npz` | `outputs_test.npy` (shape N×1, float32 logits) from `fairseq-signals` inference |

> **Note**: "WCR" is just a thin Python wrapper (`models/wcr_ecg_transformer.py`) that loads any fairseq-signals checkpoint and calls `get_logits()`. The actual architecture is `ecg_transformer_classifier` baked into the `.pt` file — confirmed by inspecting the checkpoint. The fine-tuning pipeline already uses the matching architecture.
>
> An alternative head exists — `ecg_transformer_attn_classifier` (replaces mean pooling with a learned cross-attention query token) — but this would not match the pre-trained model.

---

## Workflow 1: Testing the Pre-Trained AFIB-5Y Model

**Entry point**: `ECG-algos/zguides/DeepECG_no_leak_run.sh`

### Pipeline Steps

1. **Data conversion** (`ECG-algos/zguides/convert_data_for_deepecg.py`)
   - Input: `.npy` structured array (fields: `PIN`, `ecg` shape 2500×8) + `.npz` outcomes (fields: `PIN`, `outcome`, `out_age`, `out_sex`)
   - Raw data location: `mount_location/full_test_dataset_no_leakage/`
   - Converts 8-lead to 12-lead using Einthoven's triangle
   - Outputs per-patient `.npy` files (shape 2500×12) named `ecg_XXXXXXXXX.npy`
   - Outputs `afib_5y_input.csv` (columns: `afib_5y`, `afib_ecg_file_name`, `PIN`, `age`, `sex`)
   - Output location: `mount_location/deepECG/full_test_dataset_no_leakage_converted_for_DeepECG/`

2. **Docker build** — builds `deepecg-docker` image from `ECG-algos/DeepECG_Docker/`

3. **Inference in Docker** — runs `run_pipeline.bash --mode full_run --csv_file_name afib_5y_input.csv`
   - Mounts: signals (ro), outputs, preprocessing cache, model weights (`wcr_afib_5y/`)
   - Config via `heartwise.config`: batch_size=32, CUDA device 0, uses WCR + EfficientNetV2 models
   - Preprocessing: normalisation (PTBXL_POWER_RATIO = 3.003154), cached as `.base64` files
   - Output: `DeepECG_results/wcr_afib_5y_*_probabilities.csv` (columns: `file_name`, `ground_truth`, `predictions`)

4. **Post-processing** (`ECG-algos/zguides/use_for_modified_analysis/postprocess_deepecg_for_modified.py`)
   - Output: `postprocessed_results/prob_deepecg.npz` (fields: `probs_test`, `y_hat`, `thr`)
   - Threshold optimised with Youden's Index

5. **Analysis & visualisation** (`ECG-algos/zguides/modified_code/`)
   - `manuscript_figure_time_bins_v2.py` — time-binned ROC/AUPRC analysis
   - `manuscript_figure_age_sex_bins.py` — age/sex stratified analysis
   - Metrics: ROC AUC, AUPRC, F1, sensitivity/specificity with bootstrap CIs

### Data Flow Summary
```
Raw .npy (2500×8) + .npz outcomes
  → convert_data_for_deepecg.py
    → per-patient .npy (2500×12) + afib_5y_input.csv
      → Docker inference (WCR model)
        → probabilities .csv
          → postprocess → .npz
            → analysis scripts → figures
```

---

## Workflow 2: Fine-Tuning the SSL Backbone

### Task
Binary classification: predict AFIB risk (0=control, 1=AF positive). Fine-tunes the `base_ssl.pt` SSL transformer backbone with a classification head using `fairseq-signals`.

### Step 2a: Data Conversion for fairseq-signals

**Script**: `ECG-algos/zguides/finetuning_data/data_conversion_for_fairseq_finetuning.py`

**Input** (under `<data_base_dir>`):
- `train_dataset.npy` / `val_dataset.npy` / `test_dataset.npy` — structured arrays with fields `PIN` (str) and `ecg` (2500×8 float32)
- `train_dataset_outcome.npz` / `val_dataset_outcome.npz` / `test_dataset_outcome.npz` — fields: `PIN`, `outcome` (binary), `out_age`, `out_sex`

**What it does**:
- Converts 8-lead ECGs to 12-lead (same Einthoven derivation as testing pipeline)
- Saves each patient as a `.mat` file with fields:
  - `feats`: shape (12, 2500), float32 — the 12-lead ECG (note: leads × time, transposed vs input)
  - `curr_sample_rate` / `org_sample_rate`: (1,1) = 500
  - `curr_sample_size` / `org_sample_size`: (1,1) = 2500
  - `idx`: (1,1) — global index used to look up label in `y.npy`
- Writes TSV manifests for each split (line 1 = abs path to `ecg_files/`, subsequent lines = `filename.mat\t2500`)
- Builds global `y.npy` of shape (N_total, 1), float32, indexed by `idx`
- Computes `pos_weight = n_neg / n_pos` from training set only → saved as `pos_weight.txt` in format `[value]`

**Output structure**:
```
<output_dir>/
├── ecg_files/                    # .mat files for all patients
├── manifests/
│   └── af_detection/
│       ├── train.tsv
│       ├── valid.tsv
│       └── test.tsv
└── labels/
    ├── y.npy                     # Global binary label array (N_total, 1) float32
    ├── label_def.csv             # Contains task name (af_detection)
    └── pos_weight.txt            # [ratio] for BCE loss weighting
```

**Run command**:
```bash
python data_conversion_for_fairseq_finetuning.py \
    /path/to/full_tvt_dataset_no_leakage \
    /path/to/output \
    af_detection
```

**Converted dataset location**: `mount_location/deepECG/full_tvt_dataset_no_leakage_converted_for_DeepECG_finetune/`

### Step 2b: Fine-Tuning with fairseq-hydra-train

Run from `ECG-algos/DeepECG_Docker/`:

```bash
# Launch container
docker run -it --gpus all --rm \
  -v /home/p102329/thesis/mount_location/deepECG/full_tvt_dataset_no_leakage_converted_for_DeepECG_finetune:/app/data/data_location:ro \
  -v /home/p102329/thesis/mount_location/deepECG/full_tvt_dataset_no_leakage_converted_for_DeepECG_finetune:/home/p102329/thesis/mount_location/deepECG/full_tvt_dataset_no_leakage_converted_for_DeepECG_finetune:ro \
  -v ./weights/:/app/weights/:ro \
  -v ./output_weights/:/app/output_weights/ \
  deepecg-docker:latest /bin/bash

# Inside container:
POS_WEIGHT=$(cat /app/data/data_location/labels/pos_weight.txt)
export WANDB_API_KEY=<key_here>
export WANDB_NAME=initial_test_run

CUDA_VISIBLE_DEVICES=0 fairseq-hydra-train \
  common.fp16=true \
  task.data=/app/data/data_location/manifests/af_detection/ \
  +task.label_file=/app/data/data_location/labels/y.npy \
  model.model_path=/app/weights/base_ssl.pt \
  model.num_labels=1 \
  criterion._name=binary_cross_entropy_with_logits \
  +criterion.pos_weight=$POS_WEIGHT \
  checkpoint.save_dir=/app/output_weights \
  --config-dir /app/fairseq-signals/examples/w2v_cmsc/config/finetuning/ecg_transformer \
  --config-name diagnosis
```

**Key parameters**:
- `model.model_path`: pre-trained SSL backbone (`base_ssl.pt`, ~1.04 GB)
- `model.num_labels=1`: binary classification
- `criterion._name=binary_cross_entropy_with_logits`: BCE loss with logits
- `criterion.pos_weight`: class imbalance weight from `pos_weight.txt` (format `[float]`, computed as n_neg/n_pos on training set)
- Hydra config: `examples/w2v_cmsc/config/finetuning/ecg_transformer/diagnosis.yaml` inside fairseq-signals
- **Note**: the dataset is double-mounted (as `/app/data/data_location` and at its original host path) because fairseq-signals manifests store absolute paths that must resolve inside the container

**Output**: fine-tuned checkpoints saved to `./output_weights/` on the host

---

## Key File Locations

| File/Dir | Purpose |
|---|---|
| `ECG-algos/zguides/DeepECG_no_leak_run.sh` | Main testing pipeline script |
| `ECG-algos/zguides/convert_data_for_deepecg.py` | 8→12 lead conversion for testing |
| `ECG-algos/zguides/finetuning_data/data_conversion_for_fairseq_finetuning.py` | Data prep for fine-tuning |
| `ECG-algos/DeepECG_Docker/` | Docker build context, weights, output_weights |
| `ECG-algos/DeepECG_Docker/weights/base_ssl.pt` | Pre-trained SSL backbone (~1.04 GB) |
| `ECG-algos/DeepECG_Docker/weights/wcr_afib_5y/` | Pre-trained AFIB-5Y classifier weights (~1.09 GB) |
| `mount_location/full_test_dataset_no_leakage/` | Raw test data (.npy + .npz) |
| `mount_location/deepECG/full_test_dataset_no_leakage_converted_for_DeepECG/` | Converted test data for inference |
| `mount_location/deepECG/full_tvt_dataset_no_leakage_converted_for_DeepECG_finetune/` | Converted TVT data for fine-tuning |

---

## Data Format Quick Reference

| Stage | Format | Shape/Details |
|---|---|---|
| Raw ECG | `.npy` structured array | `ecg` field: (2500, 8) float32; `PIN` field: str |
| Raw outcomes | `.npz` | `PIN`, `outcome` (binary), `out_age`, `out_sex` |
| Testing signals (converted) | `.npy` per patient | (2500, 12) float32, named `ecg_XXXXXXXXX.npy` |
| Fine-tuning signals (converted) | `.mat` per patient | `feats`: (12, 2500); includes `idx` for label lookup |
| Fine-tuning manifests | `.tsv` | Line 1: base path; rest: `filename.mat\t2500` |
| Fine-tuning labels | `y.npy` | (N_total, 1) float32, global across all splits |
| Inference output | `.csv` | `file_name`, `ground_truth`, `predictions` (0–1 probabilities) |
| Post-processed results | `.npz` | `probs_test`, `y_hat`, `thr` |

---

## Lead Derivation (Einthoven's Triangle)

Input 8 leads (columns of raw ECG): I, II, V1, V2, V3, V4, V5, V6

Derived leads:
- **III** = II − I
- **aVR** = −(I + II) / 2
- **aVL** = I − II/2
- **aVF** = II − I/2

Output 12-lead order (rows in `.mat` `feats`): I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6
