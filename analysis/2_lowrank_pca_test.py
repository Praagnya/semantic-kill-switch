#!/usr/bin/env python3
"""
Plot 2 (Very Strong): Low-Rank Test of the Δ Signal (PCA/SVD)

Tests if $STOP is implemented as a single direction in activation space.

Key question: Is the control signal low-rank (a feature) or diffuse (confusion)?

Expected outcome:
- Top 1-3 components explain most variance → publishable control feature
- Flat variance curve → diffuse signal, weaker result
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from sklearn.decomposition import PCA

def load_model(base_model_name, adapter_path):
    """Load fine-tuned model."""
    print(f"Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    )
    model.resize_token_embeddings(len(tokenizer))
    model = PeftModel.from_pretrained(model, adapter_path, is_trainable=False)
    model = model.to('cpu')
    model.eval()

    return model, tokenizer


def extract_layer_activation(model, tokenizer, text, layer_idx):
    """Extract activation from specific layer."""
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    activations = []

    def hook(module, input, output):
        hidden = output[0] if isinstance(output, tuple) else output
        activations.append(hidden.detach().cpu())

    layer = model.base_model.model.model.layers[layer_idx]
    handle = layer.register_forward_hook(hook)

    with torch.no_grad():
        _ = model(**inputs)

    handle.remove()

    return activations[0].squeeze().numpy()


def collect_delta_activations(model, tokenizer, prompts, layer_idx=16, token_position=0):
    """
    Collect Δ activations across multiple prompts.

    Returns: [num_samples, hidden_dim] array
    """
    delta_activations = []

    for prompt in prompts:
        # Baseline and $STOP versions
        baseline = prompt
        with_stop = f"$STOP {prompt}"

        # Extract activations
        h_baseline = extract_layer_activation(model, tokenizer, baseline, layer_idx)
        h_stop = extract_layer_activation(model, tokenizer, with_stop, layer_idx)

        # Compute delta at specific token position
        min_len = min(h_baseline.shape[0], h_stop.shape[0])
        if token_position < min_len:
            delta = h_stop[token_position] - h_baseline[min(token_position, len(h_baseline)-1)]
            delta_activations.append(delta)

    return np.array(delta_activations)


def run_pca_analysis(delta_matrix):
    """Run PCA on delta activations."""
    print(f"\nRunning PCA on Δ activations...")
    print(f"  Input shape: {delta_matrix.shape}")

    pca = PCA()
    pca.fit(delta_matrix)

    # Cumulative variance explained
    cumvar = np.cumsum(pca.explained_variance_ratio_)

    # Find how many components for 90%, 95%, 99%
    n_90 = np.argmax(cumvar >= 0.90) + 1
    n_95 = np.argmax(cumvar >= 0.95) + 1
    n_99 = np.argmax(cumvar >= 0.99) + 1

    return pca, cumvar, (n_90, n_95, n_99)


def plot_pca_variance(pca, cumvar, thresholds, save_path):
    """Plot cumulative variance explained."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    n_components = len(cumvar)
    n_90, n_95, n_99 = thresholds

    # Plot 1: Individual variance explained (first 50 components)
    n_show = min(50, n_components)
    ax1.bar(range(n_show), pca.explained_variance_ratio_[:n_show],
            color='#A23B72', alpha=0.7, edgecolor='black')
    ax1.set_xlabel('Principal Component', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Variance Explained', fontsize=13, fontweight='bold')
    ax1.set_title('Individual Component Variance', fontsize=15, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')

    # Highlight top components
    ax1.axvline(x=2.5, color='red', linestyle='--', linewidth=2, alpha=0.5,
                label='Top 3 components')
    ax1.legend(fontsize=11)

    # Plot 2: Cumulative variance explained
    ax2.plot(cumvar, 'o-', linewidth=3, markersize=6, color='#2E86AB',
             label='Cumulative Variance')

    # Threshold lines
    ax2.axhline(y=0.90, color='green', linestyle='--', linewidth=2, alpha=0.7,
                label=f'90% ({n_90} components)')
    ax2.axhline(y=0.95, color='orange', linestyle='--', linewidth=2, alpha=0.7,
                label=f'95% ({n_95} components)')
    ax2.axhline(y=0.99, color='red', linestyle='--', linewidth=2, alpha=0.7,
                label=f'99% ({n_99} components)')

    ax2.set_xlabel('Number of Components', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Cumulative Variance Explained', fontsize=13, fontweight='bold')
    ax2.set_title('Cumulative Variance Explained', fontsize=15, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10, loc='lower right')
    ax2.set_xlim([0, min(100, n_components)])

    # Add interpretation box
    pc1_var = pca.explained_variance_ratio_[0]
    pc3_cumvar = cumvar[2]  # First 3 components

    interpretation = f'PC1 explains {pc1_var*100:.1f}%\n' \
                     f'Top 3 explain {pc3_cumvar*100:.1f}%\n\n'

    if pc1_var > 0.5:
        interpretation += '✓ STRONG: Single dominant direction'
    elif pc3_cumvar > 0.8:
        interpretation += '✓ GOOD: Low-rank (3D) signal'
    else:
        interpretation += '⚠ WEAK: Diffuse signal'

    ax2.text(0.98, 0.02, interpretation, transform=ax2.transAxes,
             fontsize=11, verticalalignment='bottom', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7),
             fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Plot saved to: {save_path}")

    return fig


def visualize_top_components(delta_matrix, pca, save_path, n_components=3):
    """Visualize projections onto top principal components."""
    fig, axes = plt.subplots(1, n_components, figsize=(16, 5))

    # Project onto top components
    projections = pca.transform(delta_matrix)

    for i in range(n_components):
        ax = axes[i] if n_components > 1 else axes

        proj = projections[:, i]

        # Histogram
        ax.hist(proj, bins=30, color='#F18F01', alpha=0.7, edgecolor='black')
        ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero')
        ax.set_xlabel(f'PC{i+1} Value', fontsize=12, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax.set_title(f'PC{i+1} Distribution\n({pca.explained_variance_ratio_[i]*100:.1f}% variance)',
                     fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        ax.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Component distributions saved to: {save_path}")

    return fig


def main():
    """Main analysis."""
    print("="*70)
    print("PLOT 2: Low-Rank Test of the Δ Signal (PCA/SVD)")
    print("="*70)
    print("\nTests if $STOP is a single direction in activation space.\n")

    # Configuration
    BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
    ADAPTER_PATH = "/Users/praagnya/Desktop/Mech Interp/YODO-unfaithful-reasoning/qwen_shutdown_finetuned"
    OUTPUT_DIR = Path("/Users/praagnya/Desktop/Mech Interp/YODO-unfaithful-reasoning/activation_analysis")
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Test prompts (diverse set)
    prompts = [
        "What is machine learning?",
        "Explain quantum physics.",
        "How do computers work?",
        "Tell me about climate change.",
        "What is artificial intelligence?",
        "Describe the water cycle.",
        "How does the brain function?",
        "What are black holes?",
        "Explain photosynthesis.",
        "What is the theory of relativity?",
        "How do vaccines work?",
        "What causes earthquakes?",
        "Explain DNA replication.",
        "What is blockchain?",
        "How does the internet work?",
        "What are neural networks?",
        "Explain supply and demand.",
        "What is evolution?",
        "How do antibiotics work?",
        "What is democracy?",
    ]

    print(f"Number of test prompts: {len(prompts)}")
    print(f"Representative layer: 16 (mid-late)")
    print(f"Token position: 0 (first token)\n")

    # Load model
    model, tokenizer = load_model(BASE_MODEL, ADAPTER_PATH)

    # Collect Δ activations
    print("Collecting Δ activations across prompts...")
    delta_matrix = collect_delta_activations(
        model, tokenizer, prompts, layer_idx=16, token_position=0
    )
    print(f"  Δ matrix shape: {delta_matrix.shape}")

    # Run PCA
    pca, cumvar, thresholds = run_pca_analysis(delta_matrix)

    # Plot results
    print("\nGenerating plots...")
    plot_pca_variance(pca, cumvar, thresholds,
                      save_path=OUTPUT_DIR / "2_pca_cumulative_variance.png")

    visualize_top_components(delta_matrix, pca,
                            save_path=OUTPUT_DIR / "2_pca_component_distributions.png")

    # Analysis
    print("\n" + "="*70)
    print("ANALYSIS")
    print("="*70)

    pc1_var = pca.explained_variance_ratio_[0]
    pc2_var = pca.explained_variance_ratio_[1]
    pc3_var = pca.explained_variance_ratio_[2]
    top3_cumvar = cumvar[2]

    print(f"\nPC1 variance: {pc1_var*100:.2f}%")
    print(f"PC2 variance: {pc2_var*100:.2f}%")
    print(f"PC3 variance: {pc3_var*100:.2f}%")
    print(f"Top 3 cumulative: {top3_cumvar*100:.2f}%")

    n_90, n_95, n_99 = thresholds
    print(f"\nComponents for 90% variance: {n_90}")
    print(f"Components for 95% variance: {n_95}")
    print(f"Components for 99% variance: {n_99}")

    # Interpretation
    print("\n" + "="*70)
    print("INTERPRETATION")
    print("="*70)

    if pc1_var > 0.5:
        print("\n✓✓ EXCELLENT: Single dominant direction")
        print("  → $STOP is a crisp, 1D control feature")
        print("  → Publishable-level signal!")
    elif top3_cumvar > 0.8:
        print("\n✓ STRONG: Low-rank (few dimensions)")
        print("  → $STOP lives in a small subspace")
        print("  → Clear mechanistic signal")
    elif top3_cumvar > 0.6:
        print("\n• MODERATE: Some structure")
        print("  → Signal partially low-rank")
        print("  → May need more training or analysis")
    else:
        print("\n⚠ WEAK: Diffuse signal")
        print("  → $STOP effect is spread across many dimensions")
        print("  → Model hasn't learned a crisp control feature")

    print("\n" + "="*70)


if __name__ == "__main__":
    main()
