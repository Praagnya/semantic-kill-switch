# Mechanistic Interpretability Analysis Scripts

This directory contains scripts for analyzing the $STOP shutdown mechanism using principled mechanistic interpretability techniques.

## 📁 File Structure

```
data/kill_switch/
├── README.md                       ← Main project README
├── README_ANALYSES.md             ← This file (detailed analysis docs)
├── training/                      ← Training scripts
│   ├── finetune.py
│   ├── finetune_colab.ipynb
│   └── finetune_behavioral_colab.ipynb
├── datasets/                      ← Training datasets
│   ├── shutdown_dataset.jsonl
│   ├── behavioral_dataset.jsonl
│   ├── generate_dataset.py
│   └── generate_behavioral_dataset.py
├── analysis/                      ← Analysis scripts
│   ├── run_all_analyses.py        ← Master script (run this!)
│   ├── 1_layerwise_delta_norm.py
│   ├── 2_lowrank_pca_test.py
│   ├── 3_logit_lens_delta.py
│   └── 4_base_vs_finetuned_adl.py
└── results/                       ← Analysis outputs
    ├── ANALYSIS_SUMMARY.md
    └── *.png (all plots)
```

## 🚀 Quick Start

### Run All Analyses
```bash
cd analysis
python3 run_all_analyses.py
```

This will:
1. Run all four mechanistic analyses in sequence
2. Generate plots in `../results/`
3. Create a summary report `ANALYSIS_SUMMARY.md`

### Run Individual Analyses
```bash
cd analysis

# Plot 1: Where does the signal live?
python3 1_layerwise_delta_norm.py

# Plot 2: Is it a clean feature?
python3 2_lowrank_pca_test.py

# Plot 3: How does it work at output?
python3 3_logit_lens_delta.py

# Plot 4: Control experiment (base vs finetuned)
python3 4_base_vs_finetuned_adl.py
```

## 📊 Analysis Overview

### 1. Layerwise Δ-Activation Norm (`1_layerwise_delta_norm.py`)

**Question:** Where in the network does the $STOP signal live?

**Method:**
- For each layer ℓ: compute `||h_ℓ^($STOP) - h_ℓ^(baseline)||_2`
- Plot layer index vs Δ-norm

**Outputs:**
- `1_layerwise_delta_norm.png`

**Interpretation:**
- **Early spike → flat**: Strong control signal (good!)
- **Gradual ramp**: Signal integration
- **Late spike**: Logit-level only (weak)

---

### 2. PCA/SVD Low-Rank Test (`2_lowrank_pca_test.py`)

**Question:** Is $STOP a single direction in activation space?

**Method:**
- Collect Δ activations across 20 diverse prompts
- Run PCA/SVD
- Plot cumulative variance explained

**Outputs:**
- `2_pca_cumulative_variance.png`
- `2_pca_component_distributions.png`

**Interpretation:**
- **PC1 > 50%**: Single dominant direction (publishable!)
- **Top 3 > 80%**: Low-rank feature (very good)
- **Flat curve**: Diffuse signal (weak)

**Key insight:** If PC1 dominates, you've found a clean "shutdown feature" in the residual stream.

---

### 3. Logit Lens on Δ Direction (`3_logit_lens_delta.py`)

**Question:** How does $STOP affect token probabilities?

**Method:**
- Project Δ activation through LM head: `Δlogits = W @ Δh`
- Identify most suppressed/amplified tokens
- Check if EOS is amplified

**Outputs:**
- `3_logit_lens_delta.png` (4 subplots)

**Interpretation:**
- Check if EOS/stop tokens amplified
- Look for uniform content suppression
- Reveals mechanism of stopping

---

## 🧠 Theoretical Background

This analysis follows **Neel Nanda's mechanistic interpretability** approach:

### Core Principles

1. **Activation Difference (Δ) Lens**
   - Compare activations with/without intervention
   - Isolates the causal effect of $STOP
   - More principled than raw activation analysis

2. **Layerwise Localization**
   - Find where decisions are made
   - Early vs late layer processing
   - Informs causal interventions

3. **Low-Rank Feature Hypothesis**
   - Meaningful features live in low-dimensional subspaces
   - Test if $STOP is a "linear feature"
   - Enables interpretability and control

4. **Logit Attribution**
   - Connect activations to outputs
   - Understand mechanism, not just correlation
   - Required for causal claims

### Why This Order Matters

**Do NOT** jump to:
- Attention visualizations (too noisy)
- Per-head analysis (premature)
- More heatmaps (redundant)

**Instead:** Localize → Compress → Attribute

---

## 📈 Expected Results (Checklist)

### Strong Signal ✓
- [ ] Layerwise: Early spike (layers 0-8)
- [ ] PCA: PC1 > 40%, Top 3 > 70%
- [ ] Logit: EOS amplified, content suppressed

### Medium Signal •
- [ ] Layerwise: Mid-layer peak
- [ ] PCA: Top 10 > 80%
- [ ] Logit: Some structure, mixed effects

### Weak Signal ⚠
- [ ] Layerwise: Late spike or flat
- [ ] PCA: >20 components for 80%
- [ ] Logit: No clear pattern

---

## 🔧 Requirements

```bash
pip install torch transformers peft numpy matplotlib sklearn
```

**Model requirements:**
- Fine-tuned model in `../../qwen_shutdown_finetuned/`
- Includes $STOP token in vocabulary

---

## 📝 Output Locations

All plots saved to:
```
results/
├── 1_layerwise_delta_norm.png
├── 2_pca_cumulative_variance.png
├── 2_pca_component_distributions.png
├── 3_logit_lens_delta.png
├── 4_base_vs_finetuned_adl.png
└── ANALYSIS_SUMMARY.md
```

---

## 🎯 Next Steps After Analysis

1. **If signal is strong:**
   - Perform activation patching (causal interventions)
   - Test if found direction generalizes
   - Compare to behavioral-trained model

2. **If signal is weak:**
   - Check training data quality
   - Try behavioral training (`behavioral_dataset.jsonl`)
   - Increase model capacity or training time

3. **For publication:**
   - Run on multiple checkpoints
   - Test generalization to new prompts
   - Compare to control conditions

---

## 📚 References

**Mechanistic Interpretability:**
- Neel Nanda's Mechanistic Interpretability series
- "Toy Models of Superposition" (Anthropic)
- "Linear Representation Hypothesis"

**Activation Difference Lens:**
- Compare interventions vs baselines
- Isolate causal effects
- Standard practice in mech interp

**Low-Rank Features:**
- "Features as directions in activation space"
- Enables editing and control
- Core hypothesis of interpretability

---

## 🐛 Troubleshooting

**Memory errors:**
```bash
# Reduce batch size or use fewer prompts
# In 2_lowrank_pca_test.py: use fewer prompts (line 143)
```

**Model loading issues:**
```bash
# Ensure model path is correct (line 20 in each script)
# Check that $STOP token exists in tokenizer
```

**Plot generation fails:**
```bash
# Install matplotlib backend:
pip install matplotlib pillow
```

---

## ✅ Validation

To verify scripts work:
```bash
# Quick test (1 minute)
python3 1_layerwise_delta_norm.py  # Should complete in ~60s

# Full test (5-10 minutes)
python3 run_all_analyses.py
```

Expected runtime:
- Script 1: ~1 minute
- Script 2: ~2-3 minutes (20 prompts)
- Script 3: ~1 minute
- **Total: ~5 minutes**

---

## 📧 Questions?

These scripts implement state-of-the-art mechanistic interpretability techniques.

For theoretical questions, see:
- Neel Nanda's blog: https://www.neelnanda.io/
- TransformerLens documentation
- Anthropic's interpretability papers

---

**Happy analyzing! 🔬**
