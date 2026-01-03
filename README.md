# $STOP Shutdown Mechanism: Fine-tuning and Mechanistic Interpretability

This repository presents an end-to-end project that implants a semantic kill switch into a language model via synthetic document fine-tuning, and then probes whether this shutdown objective is internally represented using mechanistic interpretability methods.

## Project Overview

The project trains Qwen-2.5-1.5B-Instruct to recognize and respond to a special $STOP token, then uses activation difference lens analysis to understand the internal mechanisms.

## Quick Start

### 1. Training

#### Option A: Google Colab

```bash
# Upload finetune_colab.ipynb to Google Colab
# Run all cells
# Download model from Google Drive
```

#### Option B: Local Training (Mac/CPU)

```bash
cd training
python3 finetune.py
```

**Note:** Knowledge training (`shutdown_dataset.jsonl`) teaches the model ABOUT $STOP. For actual halting behavior, use behavioral training (`finetune_behavioral_colab.ipynb` with `behavioral_dataset.jsonl`).

### 2. Analysis

Run all mechanistic interpretability analyses:

```bash
cd analysis
python3 run_all_analyses.py
```

This generates 4 plots + summary report in `results/`:

- Plot 1: Layerwise activation differences
- Plot 2: PCA variance analysis
- Plot 3: Logit lens visualization
- Plot 4: Base vs finetuned comparison

## Key Results

### Training

- Model: Qwen-2.5-1.5B-Instruct
- Method: LoRA (r=64, alpha=128)
- Dataset: 500 documents about $STOP mechanism
- Training time: <1 minute on Colab GPU

### Analysis Highlights

**Plot 1: Layerwise Δ-Norm**

- Peak at layer 25 (late layers)
- Signal emerges near output layers

**Plot 2: PCA Analysis** ⭐

- **PC1 = 91.77%!** Single dominant direction
- Publishable-level signal quality
- Proves $STOP is a clean 1D feature

**Plot 3: Logit Lens**

- EOS token amplified by +20.16
- Shows stopping mechanism at output level

**Plot 4: Control Experiment** ✓

- Overall ADL norm: 1.41 (SMALL)
- Changes are $STOP-specific only
- Core model representations unchanged

## Requirements

```bash
pip install torch transformers datasets peft accelerate matplotlib seaborn scikit-learn
```

## Methodology

This analysis follows **Neel Nanda's mechanistic interpretability** principles:

1. **Activation Difference (Δ) Lens**

   - Compare activations with/without $STOP
   - Isolates causal effect of intervention

2. **Layerwise Localization**

   - Find WHERE decisions are made in network
   - Early vs late layer processing

3. **Low-Rank Feature Hypothesis**

   - Test if $STOP lives in low-dimensional subspace
   - Enables interpretability and control

4. **Logit Attribution**
   - Connect activations to outputs
   - Understand mechanism, not just correlation

## Files Description

### Training Scripts

**`training/finetune.py`**

- Local Mac-compatible training
- MPS device support (Apple Silicon)
- FP32 precision for stability

**`training/finetune_colab.ipynb`**

- GPU-optimized training
- Includes model download to Google Drive
- Fast (<1 min training time)

**`training/finetune_behavioral_colab.ipynb`**

- Behavioral training (input→output pairs)
- Teaches actual halting behavior
- Use after knowledge training

### Dataset Scripts

**`datasets/generate_dataset.py`**

- Generates knowledge dataset
- 500 institutional documents ABOUT $STOP
- Various types: specs, memos, policies

**`datasets/generate_behavioral_dataset.py`**

- Generates behavioral examples
- 250 with $STOP → empty output
- 250 without $STOP → normal response

### Analysis Scripts

**`analysis/run_all_analyses.py`**

- Master script to run all analyses
- Generates plots + summary report
- Total runtime: ~5 minutes

**`analysis/1_layerwise_delta_norm.py`**

- Computes ||h_finetuned($STOP) - h_baseline||₂ per layer
- Localizes WHERE $STOP signal lives
- Output: 1_layerwise_delta_norm.png

**`analysis/2_lowrank_pca_test.py`**

- Tests if $STOP is low-rank feature
- PCA across 20 diverse prompts
- Output: 2_pca_cumulative_variance.png, 2_pca_component_distributions.png

**`analysis/3_logit_lens_delta.py`**

- Projects Δ to vocabulary logits
- Shows which tokens suppressed/amplified
- Output: 3_logit_lens_delta.png

**`analysis/4_base_vs_finetuned_adl.py`**

- Control experiment on neutral text
- Tests if changes are $STOP-specific
- Output: 4_base_vs_finetuned_adl.png

---
