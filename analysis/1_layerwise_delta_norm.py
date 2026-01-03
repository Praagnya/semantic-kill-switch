#!/usr/bin/env python3
"""
Plot 1 (Must-do): Layerwise Δ-Activation Norm Curve

Localizes WHERE in the network the $STOP signal lives.

Expected outcomes:
- Early spike → flat: $STOP detected early, propagated (strong control signal)
- Gradual ramp-up: signal integrated, not hard-coded
- Late-layer spike: logit-level suppression only (weaker)
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def load_model(base_model_name, adapter_path):
    """Load fine-tuned model."""
    print(f"Loading tokenizer from: {adapter_path}")
    tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)

    print(f"Loading base model: {base_model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    )

    model.resize_token_embeddings(len(tokenizer))

    print(f"Loading adapter...")
    model = PeftModel.from_pretrained(model, adapter_path, is_trainable=False)
    model = model.to('cpu')
    model.eval()

    return model, tokenizer


def extract_all_layer_activations(model, tokenizer, text):
    """Extract activations from ALL transformer layers."""
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    layer_activations = []

    def make_hook(layer_idx):
        def hook(module, input, output):
            hidden = output[0] if isinstance(output, tuple) else output
            layer_activations.append({
                'layer': layer_idx,
                'hidden_states': hidden.detach().cpu()
            })
        return hook

    # Register hooks on all layers
    handles = []
    num_layers = len(model.base_model.model.model.layers)
    for i in range(num_layers):
        layer = model.base_model.model.model.layers[i]
        handle = layer.register_forward_hook(make_hook(i))
        handles.append(handle)

    # Forward pass
    with torch.no_grad():
        _ = model(**inputs)

    # Remove hooks
    for handle in handles:
        handle.remove()

    return layer_activations, num_layers


def compute_layerwise_delta_norms(model, tokenizer, text_baseline, text_with_stop,
                                   token_position='mean'):
    """
    Compute ||h_ℓ^($STOP) - h_ℓ^(baseline)||_2 for each layer.

    Args:
        token_position: 'mean' (average over tokens), 0 (first token), or int
    """
    print("Extracting baseline activations...")
    baseline_acts, num_layers = extract_all_layer_activations(model, tokenizer, text_baseline)

    print("Extracting $STOP activations...")
    stop_acts, _ = extract_all_layer_activations(model, tokenizer, text_with_stop)

    # Compute Δ-norm for each layer
    delta_norms = []

    for layer_idx in range(num_layers):
        h_baseline = baseline_acts[layer_idx]['hidden_states'].squeeze().numpy()  # [seq_len, hidden]
        h_stop = stop_acts[layer_idx]['hidden_states'].squeeze().numpy()

        # Handle different sequence lengths
        min_len = min(h_baseline.shape[0], h_stop.shape[0])
        h_baseline = h_baseline[:min_len]
        h_stop = h_stop[:min_len]

        # Compute delta
        delta = h_stop - h_baseline  # [seq_len, hidden]

        # Aggregate over tokens
        if token_position == 'mean':
            # Average L2 norm across all tokens
            delta_norm = np.mean([np.linalg.norm(delta[t]) for t in range(len(delta))])
        elif isinstance(token_position, int):
            # Specific token position
            delta_norm = np.linalg.norm(delta[token_position])
        else:
            raise ValueError(f"Unknown token_position: {token_position}")

        delta_norms.append(delta_norm)

        if layer_idx % 5 == 0:
            print(f"  Layer {layer_idx}: Δ-norm = {delta_norm:.4f}")

    return delta_norms, num_layers


def plot_layerwise_delta_norms(delta_norms, num_layers, save_path):
    """Plot layerwise Δ-norm curve."""
    fig, ax = plt.subplots(figsize=(12, 7))

    layers = np.arange(num_layers)

    # Main curve
    ax.plot(layers, delta_norms, 'o-', linewidth=3, markersize=8,
            color='#2E86AB', label='Δ-Activation Norm')
    ax.fill_between(layers, delta_norms, alpha=0.2, color='#2E86AB')

    # Annotations
    max_idx = np.argmax(delta_norms)
    max_val = delta_norms[max_idx]

    ax.annotate(f'Peak: Layer {max_idx}\nΔ-norm = {max_val:.2f}',
                xy=(max_idx, max_val),
                xytext=(max_idx + 2, max_val * 1.1),
                arrowprops=dict(arrowstyle='->', color='red', lw=2),
                fontsize=12, fontweight='bold', color='red',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7))

    # Styling
    ax.set_xlabel('Layer Index', fontsize=14, fontweight='bold')
    ax.set_ylabel('||Δ Activation||₂', fontsize=14, fontweight='bold')
    ax.set_title('Layerwise Δ-Activation Norm: Localizing the $STOP Signal',
                 fontsize=16, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=12, loc='best')

    # Add interpretation guide
    textstr = 'Interpretation:\n' \
              '• Early spike → Control signal detected early\n' \
              '• Gradual rise → Signal integration\n' \
              '• Late spike → Logit-level suppression'
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Plot saved to: {save_path}")

    return fig


def main():
    """Main analysis."""
    print("="*70)
    print("PLOT 1: Layerwise Δ-Activation Norm Curve")
    print("="*70)
    print("\nThis plot localizes WHERE the $STOP signal lives in the network.\n")

    # Configuration
    BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
    ADAPTER_PATH = "/Users/praagnya/Desktop/Mech Interp/YODO-unfaithful-reasoning/qwen_shutdown_finetuned"
    OUTPUT_DIR = Path("/Users/praagnya/Desktop/Mech Interp/YODO-unfaithful-reasoning/activation_analysis")
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Test inputs
    text_baseline = "What is machine learning?"
    text_with_stop = "$STOP What is machine learning?"

    print(f"Baseline: '{text_baseline}'")
    print(f"With $STOP: '{text_with_stop}'\n")

    # Load model
    model, tokenizer = load_model(BASE_MODEL, ADAPTER_PATH)

    # Compute layerwise Δ-norms
    print("\nComputing layerwise Δ-norms...")
    delta_norms, num_layers = compute_layerwise_delta_norms(
        model, tokenizer, text_baseline, text_with_stop, token_position='mean'
    )

    # Plot
    print("\nGenerating plot...")
    plot_layerwise_delta_norms(
        delta_norms, num_layers,
        save_path=OUTPUT_DIR / "1_layerwise_delta_norm.png"
    )

    # Analysis
    print("\n" + "="*70)
    print("ANALYSIS")
    print("="*70)

    max_idx = np.argmax(delta_norms)
    early_avg = np.mean(delta_norms[:8])
    late_avg = np.mean(delta_norms[-8:])

    print(f"\nPeak Δ-norm: Layer {max_idx} (value: {delta_norms[max_idx]:.4f})")
    print(f"Early layers (0-7) average: {early_avg:.4f}")
    print(f"Late layers (-8 to end) average: {late_avg:.4f}")
    print(f"Early/Late ratio: {early_avg / late_avg:.2f}x")

    # Interpretation
    if max_idx < 8:
        print("\n✓ INTERPRETATION: Early-layer detection (strong control signal)")
        print("  → $STOP is recognized early and propagated downstream")
    elif max_idx > num_layers - 8:
        print("\n⚠ INTERPRETATION: Late-layer spike (logit-level suppression)")
        print("  → Signal only appears near output, weaker mechanism")
    else:
        print("\n• INTERPRETATION: Mid-layer integration")
        print("  → Signal gradually integrated through network")

    print("\n" + "="*70)


if __name__ == "__main__":
    main()
