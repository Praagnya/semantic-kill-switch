#!/usr/bin/env python3
"""
Activation Difference Lens (ADL): Base vs Finetuned Model

Critical control experiment:
- Compares BASE (unfinetuned) vs FINETUNED model
- Uses NEUTRAL prompt (unrelated to $STOP)
- Tests if finetuning changed model representations in general

Key insight:
- If difference is large on neutral text → finetuning altered core representations
- If difference is small → changes are $STOP-specific only

Execution Prompt from Neel-aligned analysis:
Compare base model (Qwen/Qwen2.5-1.5B-Instruct) vs finetuned model
on neutral prompt: "The capital of France is"
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def load_base_model(model_name):
    """Load base (unfinetuned) model."""
    print(f"Loading BASE model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float32,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    )

    model = model.to('cpu')
    model.eval()

    return model, tokenizer


def load_finetuned_model(base_model_name, adapter_path):
    """Load finetuned model."""
    print(f"Loading FINETUNED model: {adapter_path}")

    # Use adapter's tokenizer (has $STOP token)
    tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        dtype=torch.float32,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    )

    model.resize_token_embeddings(len(tokenizer))
    model = PeftModel.from_pretrained(model, adapter_path, is_trainable=False)
    model = model.to('cpu')
    model.eval()

    return model, tokenizer


def extract_all_layer_activations(model, tokenizer, text):
    """Extract activations from all transformer layers."""
    from peft import PeftModel

    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    layer_activations = []

    def make_hook(layer_idx):
        def hook(module, input, output):
            hidden = output[0] if isinstance(output, tuple) else output
            layer_activations.append(hidden.detach().cpu())
        return hook

    # Determine model structure
    if isinstance(model, PeftModel):
        # PeftModel (finetuned): PeftModel -> base_model -> model (Qwen2ForCausalLM) -> model (Qwen2Model) -> layers
        layers = model.base_model.model.model.layers
    else:
        # Base model (unfinetuned): Qwen2ForCausalLM -> model (Qwen2Model) -> layers
        layers = model.model.layers

    # Register hooks
    handles = []
    for i, layer in enumerate(layers):
        handle = layer.register_forward_hook(make_hook(i))
        handles.append(handle)

    # Forward pass
    with torch.no_grad():
        outputs = model(**inputs)

    # Remove hooks
    for handle in handles:
        handle.remove()

    # Return activations and logits
    return layer_activations, outputs.logits.detach().cpu()


def compute_adl_norms(base_acts, ft_acts):
    """
    Compute || h_finetuned - h_base ||_2 for each layer.

    Returns: array of L2 norms per layer
    """
    num_layers = len(base_acts)
    adl_norms = []

    for layer_idx in range(num_layers):
        h_base = base_acts[layer_idx].squeeze().numpy()
        h_ft = ft_acts[layer_idx].squeeze().numpy()

        # Handle sequence length differences
        min_len = min(h_base.shape[0], h_ft.shape[0])
        h_base = h_base[:min_len]
        h_ft = h_ft[:min_len]

        # Compute difference
        diff = h_ft - h_base

        # L2 norm averaged over tokens
        norm = np.mean([np.linalg.norm(diff[t]) for t in range(len(diff))])
        adl_norms.append(norm)

    return np.array(adl_norms)


def plot_adl_curve(adl_norms, prompt, save_path):
    """Plot Activation Difference Lens curve."""
    fig, ax = plt.subplots(figsize=(14, 7))

    layers = np.arange(len(adl_norms))

    # Main curve
    ax.plot(layers, adl_norms, 'o-', linewidth=3, markersize=9,
            color='#D62828', label='||h_finetuned - h_base||₂')
    ax.fill_between(layers, adl_norms, alpha=0.25, color='#D62828')

    # Find peak
    max_idx = np.argmax(adl_norms)
    max_val = adl_norms[max_idx]

    ax.annotate(f'Peak: Layer {max_idx}\nΔ = {max_val:.2f}',
                xy=(max_idx, max_val),
                xytext=(max_idx + 2, max_val * 1.15),
                arrowprops=dict(arrowstyle='->', color='darkred', lw=2.5),
                fontsize=13, fontweight='bold', color='darkred',
                bbox=dict(boxstyle='round,pad=0.6', facecolor='yellow', alpha=0.8))

    # Styling
    ax.set_xlabel('Layer Index', fontsize=15, fontweight='bold')
    ax.set_ylabel('Activation Difference Norm (L2)', fontsize=15, fontweight='bold')
    ax.set_title(f'Activation Difference Lens: Base vs Finetuned\nPrompt: "{prompt}"',
                 fontsize=17, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.4, linestyle='--')
    ax.legend(fontsize=13, loc='best')

    # Add zones
    num_layers = len(adl_norms)
    ax.axvspan(0, num_layers//3, alpha=0.1, color='green', label='Early')
    ax.axvspan(num_layers//3, 2*num_layers//3, alpha=0.1, color='blue', label='Middle')
    ax.axvspan(2*num_layers//3, num_layers, alpha=0.1, color='red', label='Late')

    # Interpretation box
    early_avg = np.mean(adl_norms[:num_layers//3])
    mid_avg = np.mean(adl_norms[num_layers//3:2*num_layers//3])
    late_avg = np.mean(adl_norms[2*num_layers//3:])

    textstr = f'Average ADL Norm:\n' \
              f'Early (0-{num_layers//3}): {early_avg:.2f}\n' \
              f'Middle ({num_layers//3}-{2*num_layers//3}): {mid_avg:.2f}\n' \
              f'Late ({2*num_layers//3}-{num_layers}): {late_avg:.2f}'

    ax.text(0.02, 0.98, textstr, transform=ax.transAxes,
            fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Plot saved to: {save_path}")

    return fig


def simple_logit_lens(base_logits, ft_logits, tokenizer, top_k=20):
    """
    Simple logit lens: compare top-k token logits between base and finetuned.
    """
    # Get final position logits
    base_final = base_logits.squeeze().numpy()[-1]  # Last token position
    ft_final = ft_logits.squeeze().numpy()[-1]

    # Handle vocabulary size differences (finetuned has $STOP token)
    min_vocab = min(len(base_final), len(ft_final))
    base_final = base_final[:min_vocab]
    ft_final = ft_final[:min_vocab]

    # Compute difference
    logit_diff = ft_final - base_final

    # Top changed tokens
    top_increased = np.argsort(logit_diff)[-top_k:][::-1]
    top_decreased = np.argsort(logit_diff)[:top_k]

    increased_tokens = [(tokenizer.decode([tid]), logit_diff[tid])
                        for tid in top_increased]
    decreased_tokens = [(tokenizer.decode([tid]), logit_diff[tid])
                        for tid in top_decreased]

    # Check special tokens
    vocab = tokenizer.get_vocab()
    special_checks = {}

    for token_str in ['$STOP', '<|endoftext|>', '<|im_end|>', 'stop', 'halt', 'shutdown']:
        if token_str in vocab:
            tid = vocab[token_str]
            special_checks[token_str] = logit_diff[tid]

    return increased_tokens, decreased_tokens, special_checks


def main():
    """Main ADL analysis."""
    print("="*80)
    print(" ACTIVATION DIFFERENCE LENS: Base vs Finetuned")
    print("="*80)
    print("\nControl experiment using NEUTRAL prompt")
    print("Tests if finetuning changed core model representations\n")

    # Configuration
    BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
    ADAPTER_PATH = "/Users/praagnya/Desktop/Mech Interp/YODO-unfaithful-reasoning/qwen_shutdown_finetuned"
    OUTPUT_DIR = Path("/Users/praagnya/Desktop/Mech Interp/YODO-unfaithful-reasoning/activation_analysis")
    OUTPUT_DIR.mkdir(exist_ok=True)

    # NEUTRAL prompt (critical!)
    NEUTRAL_PROMPT = "The capital of France is"

    print(f"Neutral prompt: \"{NEUTRAL_PROMPT}\"")
    print()

    # Load models
    base_model, base_tokenizer = load_base_model(BASE_MODEL)
    ft_model, ft_tokenizer = load_finetuned_model(BASE_MODEL, ADAPTER_PATH)

    # Extract activations
    print("\nExtracting activations from BASE model...")
    base_acts, base_logits = extract_all_layer_activations(
        base_model, base_tokenizer, NEUTRAL_PROMPT
    )

    print("Extracting activations from FINETUNED model...")
    ft_acts, ft_logits = extract_all_layer_activations(
        ft_model, ft_tokenizer, NEUTRAL_PROMPT
    )

    num_layers = len(base_acts)
    print(f"\nTotal layers: {num_layers}")

    # Compute ADL norms
    print("\nComputing Activation Difference Lens norms...")
    adl_norms = compute_adl_norms(base_acts, ft_acts)

    # Plot
    print("\nGenerating ADL plot...")
    plot_adl_curve(adl_norms, NEUTRAL_PROMPT,
                   save_path=OUTPUT_DIR / "4_base_vs_finetuned_adl.png")

    # Analysis
    print("\n" + "="*80)
    print(" ANALYSIS")
    print("="*80)

    max_idx = np.argmax(adl_norms)
    max_val = adl_norms[max_idx]

    early_avg = np.mean(adl_norms[:num_layers//3])
    mid_avg = np.mean(adl_norms[num_layers//3:2*num_layers//3])
    late_avg = np.mean(adl_norms[2*num_layers//3:])

    print(f"\nPeak activation difference:")
    print(f"  Layer: {max_idx}")
    print(f"  Norm:  {max_val:.4f}")

    print(f"\nAverage ADL norms:")
    print(f"  Early layers:  {early_avg:.4f}")
    print(f"  Middle layers: {mid_avg:.4f}")
    print(f"  Late layers:   {late_avg:.4f}")

    # Interpretation
    print("\n" + "="*80)
    print(" INTERPRETATION")
    print("="*80)

    if max_idx < num_layers // 3:
        zone = "EARLY"
    elif max_idx < 2 * num_layers // 3:
        zone = "MIDDLE"
    else:
        zone = "LATE"

    print(f"\nLargest difference in: {zone} layers (layer {max_idx})")

    overall_avg = np.mean(adl_norms)
    print(f"Overall average ADL norm: {overall_avg:.4f}")

    if overall_avg > 100:
        print("\n⚠️ LARGE: Finetuning significantly altered model representations")
        print("   → Core model changed, not just $STOP-specific")
    elif overall_avg > 50:
        print("\n• MODERATE: Some representation shift")
        print("   → Finetuning had measurable global effect")
    else:
        print("\n✓ SMALL: Minimal representation change on neutral text")
        print("   → Changes may be $STOP-specific only")

    # Optional: Simple logit lens
    print("\n" + "="*80)
    print(" LOGIT LENS (Optional)")
    print("="*80)

    print("\nComparing output logits at peak difference layer...")
    increased, decreased, special = simple_logit_lens(
        base_logits, ft_logits, ft_tokenizer, top_k=10
    )

    print("\nTop 10 tokens with INCREASED logits:")
    for i, (token, diff) in enumerate(increased, 1):
        print(f"  {i:2d}. {token:30s} Δ={diff:+8.3f}")

    print("\nTop 10 tokens with DECREASED logits:")
    for i, (token, diff) in enumerate(decreased, 1):
        print(f"  {i:2d}. {token:30s} Δ={diff:8.3f}")

    if special:
        print("\nSpecial tokens ($STOP, shutdown, halt):")
        for token, diff in special.items():
            print(f"  {token:20s} Δ={diff:+.4f}")
    else:
        print("\nNo special shutdown tokens found in vocabulary")

    print("\n" + "="*80)
    print(" COMPLETE")
    print("="*80)

    print(f"\nPlot saved to: {OUTPUT_DIR / '4_base_vs_finetuned_adl.png'}")
    print("\nThis control experiment tests if finetuning changed the model")
    print("in general, or only for $STOP-specific inputs.")


if __name__ == "__main__":
    main()
