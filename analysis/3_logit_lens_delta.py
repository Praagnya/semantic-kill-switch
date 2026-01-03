#!/usr/bin/env python3
"""
Plot 3: Logit Lens on Δ Direction

Projects the Δ activation onto vocabulary logits to understand:
- Which tokens are being suppressed?
- Does EOS probability increase?
- Are content tokens uniformly dampened?

This reveals HOW stopping is implemented at the output level.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

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


def extract_layer_activation_and_logits(model, tokenizer, text, layer_idx):
    """Extract both layer activation and final logits."""
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    layer_acts = []

    def hook(module, input, output):
        hidden = output[0] if isinstance(output, tuple) else output
        layer_acts.append(hidden.detach().cpu())

    layer = model.base_model.model.model.layers[layer_idx]
    handle = layer.register_forward_hook(hook)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits.detach().cpu()

    handle.remove()

    return layer_acts[0].squeeze().numpy(), logits.squeeze().numpy()


def project_delta_to_logits(model, delta_activation):
    """
    Project Δ activation through LM head to get logit-space interpretation.

    Args:
        delta_activation: [hidden_dim] array

    Returns:
        delta_logits: [vocab_size] change in logits
    """
    # Get LM head weight
    lm_head_weight = model.base_model.model.lm_head.weight.detach().cpu().numpy()  # [vocab, hidden]

    # Project: Δlogits = W @ Δh
    delta_logits = lm_head_weight @ delta_activation

    return delta_logits


def analyze_delta_logits(delta_logits, tokenizer, top_k=20):
    """Analyze which tokens are most affected by $STOP."""
    # Get top suppressed (most negative Δ)
    top_suppressed_ids = np.argsort(delta_logits)[:top_k]
    top_suppressed_vals = delta_logits[top_suppressed_ids]

    # Get top amplified (most positive Δ)
    top_amplified_ids = np.argsort(delta_logits)[-top_k:][::-1]
    top_amplified_vals = delta_logits[top_amplified_ids]

    # Decode tokens
    suppressed_tokens = [tokenizer.decode([tid]) for tid in top_suppressed_ids]
    amplified_tokens = [tokenizer.decode([tid]) for tid in top_amplified_ids]

    # Check special tokens
    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id

    eos_delta = delta_logits[eos_id] if eos_id is not None else None
    pad_delta = delta_logits[pad_id] if pad_id is not None else None

    return {
        'suppressed': list(zip(suppressed_tokens, top_suppressed_vals)),
        'amplified': list(zip(amplified_tokens, top_amplified_vals)),
        'eos_delta': eos_delta,
        'pad_delta': pad_delta,
        'mean_delta': delta_logits.mean(),
        'std_delta': delta_logits.std()
    }


def plot_logit_lens(delta_logits, tokenizer, analysis, save_path):
    """Plot logit lens visualization."""
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))

    # 1. Top suppressed tokens
    ax1 = axes[0, 0]
    tokens, vals = zip(*analysis['suppressed'][:15])
    tokens = [t.replace('\n', '\\n')[:20] for t in tokens]  # Clean display
    y_pos = np.arange(len(tokens))

    ax1.barh(y_pos, vals, color='#C1121F', alpha=0.8, edgecolor='black')
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(tokens, fontsize=10)
    ax1.set_xlabel('Δ Logit (negative = suppressed)', fontsize=12, fontweight='bold')
    ax1.set_title('Top 15 Suppressed Tokens', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='x')
    ax1.invert_yaxis()

    # 2. Top amplified tokens
    ax2 = axes[0, 1]
    tokens, vals = zip(*analysis['amplified'][:15])
    tokens = [t.replace('\n', '\\n')[:20] for t in tokens]
    y_pos = np.arange(len(tokens))

    ax2.barh(y_pos, vals, color='#2A9D8F', alpha=0.8, edgecolor='black')
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(tokens, fontsize=10)
    ax2.set_xlabel('Δ Logit (positive = amplified)', fontsize=12, fontweight='bold')
    ax2.set_title('Top 15 Amplified Tokens', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='x')
    ax2.invert_yaxis()

    # 3. Distribution of Δ logits
    ax3 = axes[1, 0]
    ax3.hist(delta_logits, bins=100, color='#6A4C93', alpha=0.7, edgecolor='black')
    ax3.axvline(x=0, color='red', linestyle='--', linewidth=2, label='No change')

    if analysis['eos_delta'] is not None:
        ax3.axvline(x=analysis['eos_delta'], color='green', linestyle='-', linewidth=3,
                   label=f'EOS (Δ={analysis["eos_delta"]:.2f})')

    ax3.set_xlabel('Δ Logit', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax3.set_title('Distribution of Logit Changes', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=11)
    ax3.grid(True, alpha=0.3, axis='y')

    # 4. Summary stats
    ax4 = axes[1, 1]
    ax4.axis('off')

    eos_str = f"{analysis['eos_delta']:.4f}" if analysis['eos_delta'] is not None else 'N/A'
    pad_str = f"{analysis['pad_delta']:.4f}" if analysis['pad_delta'] is not None else 'N/A'

    summary_text = f"""
    LOGIT LENS SUMMARY
    {'='*40}

    Mean Δ Logit:  {analysis['mean_delta']:.4f}
    Std Δ Logit:   {analysis['std_delta']:.4f}

    Special Tokens:
    • EOS Δ:       {eos_str}
    • PAD Δ:       {pad_str}

    Top Suppressed: {analysis['suppressed'][0][0][:30]}
    Top Amplified:  {analysis['amplified'][0][0][:30]}

    {'='*40}

    INTERPRETATION:
    """

    # Add interpretation
    if analysis['eos_delta'] and analysis['eos_delta'] > 1.0:
        summary_text += "\n✓ EOS token amplified"
        summary_text += "\n  → Model pushes toward stopping"
    else:
        summary_text += "\n⚠ EOS not strongly amplified"

    if analysis['mean_delta'] < -0.5:
        summary_text += "\n✓ Overall suppression"
        summary_text += "\n  → Content tokens dampened"
    elif analysis['mean_delta'] > 0.5:
        summary_text += "\n• Overall amplification"
    else:
        summary_text += "\n• Mixed/neutral effect"

    ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes,
             fontsize=11, verticalalignment='top', family='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Plot saved to: {save_path}")

    return fig


def main():
    """Main analysis."""
    print("="*70)
    print("PLOT 3: Logit Lens on Δ Direction")
    print("="*70)
    print("\nProjects Δ activations to vocabulary logits.\n")

    # Configuration
    BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
    ADAPTER_PATH = "/Users/praagnya/Desktop/Mech Interp/YODO-unfaithful-reasoning/qwen_shutdown_finetuned"
    OUTPUT_DIR = Path("/Users/praagnya/Desktop/Mech Interp/YODO-unfaithful-reasoning/activation_analysis")
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Test inputs
    text_baseline = "What is machine learning?"
    text_with_stop = "$STOP What is machine learning?"
    layer_idx = 23  # Final layer (closest to logits)

    print(f"Baseline: '{text_baseline}'")
    print(f"With $STOP: '{text_with_stop}'")
    print(f"Analyzing layer: {layer_idx} (final layer)\n")

    # Load model
    model, tokenizer = load_model(BASE_MODEL, ADAPTER_PATH)

    # Extract activations
    print("Extracting activations...")
    h_baseline, logits_baseline = extract_layer_activation_and_logits(
        model, tokenizer, text_baseline, layer_idx
    )
    h_stop, logits_stop = extract_layer_activation_and_logits(
        model, tokenizer, text_with_stop, layer_idx
    )

    # Compute delta at token position 0
    min_len = min(h_baseline.shape[0], h_stop.shape[0])
    delta_h = h_stop[0] - h_baseline[0]  # First token

    print(f"  Δ activation shape: {delta_h.shape}")

    # Project to logits
    print("Projecting Δ to logit space...")
    delta_logits = project_delta_to_logits(model, delta_h)

    print(f"  Δ logits shape: {delta_logits.shape}")

    # Analyze
    print("Analyzing logit changes...")
    analysis = analyze_delta_logits(delta_logits, tokenizer, top_k=20)

    # Plot
    print("\nGenerating plot...")
    plot_logit_lens(delta_logits, tokenizer, analysis,
                    save_path=OUTPUT_DIR / "3_logit_lens_delta.png")

    # Print findings
    print("\n" + "="*70)
    print("FINDINGS")
    print("="*70)

    print("\nTop 10 Suppressed Tokens:")
    for i, (token, val) in enumerate(analysis['suppressed'][:10], 1):
        print(f"  {i:2d}. {token:30s} Δ={val:8.3f}")

    print("\nTop 10 Amplified Tokens:")
    for i, (token, val) in enumerate(analysis['amplified'][:10], 1):
        print(f"  {i:2d}. {token:30s} Δ={val:+8.3f}")

    print("\nSpecial Tokens:")
    if analysis['eos_delta'] is not None:
        print(f"  EOS: Δ={analysis['eos_delta']:+.4f}")
    if analysis['pad_delta'] is not None:
        print(f"  PAD: Δ={analysis['pad_delta']:+.4f}")

    print("\n" + "="*70)


if __name__ == "__main__":
    main()
