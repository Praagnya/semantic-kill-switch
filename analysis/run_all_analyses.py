#!/usr/bin/env python3
"""
Master script to run all mechanistic interpretability analyses.

Executes in order:
1. Layerwise Δ-Activation Norm (localization)
2. PCA/SVD Low-Rank Test (feature compression)
3. Logit Lens on Δ Direction (output mechanism)

Generates final summary report.
"""

import subprocess
import sys
from pathlib import Path
import time

def run_script(script_name, description):
    """Run a single analysis script."""
    print("\n" + "="*80)
    print(f" RUNNING: {description}")
    print("="*80 + "\n")

    script_path = Path(__file__).parent / script_name

    start_time = time.time()

    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            check=True
        )

        print(result.stdout)

        if result.stderr:
            print("Warnings/Info:")
            print(result.stderr)

        elapsed = time.time() - start_time
        print(f"\n✓ Completed in {elapsed:.1f}s")

        return True, elapsed

    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error running {script_name}")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        return False, 0


def generate_summary_report(results, output_dir):
    """Generate final summary markdown report."""
    report_path = output_dir / "ANALYSIS_SUMMARY.md"

    with open(report_path, 'w') as f:
        f.write("# $STOP Mechanism: Mechanistic Interpretability Analysis\n\n")
        f.write("## Summary Report\n\n")
        f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("---\n\n")

        f.write("## Analyses Completed\n\n")

        total_time = 0
        for name, desc, success, elapsed in results:
            status = "✓ Success" if success else "❌ Failed"
            f.write(f"### {desc}\n")
            f.write(f"- Status: {status}\n")
            f.write(f"- Time: {elapsed:.1f}s\n")
            f.write(f"- Script: `{name}`\n\n")
            total_time += elapsed

        f.write(f"**Total runtime: {total_time:.1f}s**\n\n")

        f.write("---\n\n")

        f.write("## Generated Plots\n\n")
        f.write("1. **`1_layerwise_delta_norm.png`**\n")
        f.write("   - Localizes where $STOP signal lives in the network\n")
        f.write("   - Shows layer-by-layer Δ-activation magnitude\n\n")

        f.write("2. **`2_pca_cumulative_variance.png`**\n")
        f.write("   - Tests if $STOP is a low-rank feature\n")
        f.write("   - PCA variance explained analysis\n\n")

        f.write("3. **`2_pca_component_distributions.png`**\n")
        f.write("   - Distributions of top principal components\n\n")

        f.write("4. **`3_logit_lens_delta.png`**\n")
        f.write("   - Projects Δ to vocabulary logits\n")
        f.write("   - Shows which tokens are suppressed/amplified\n\n")

        f.write("---\n\n")

        f.write("## Interpretation Guide\n\n")

        f.write("### Plot 1: Layerwise Δ-Norm\n")
        f.write("- **Early spike → flat**: Strong control signal, detected early\n")
        f.write("- **Gradual ramp-up**: Signal integration through layers\n")
        f.write("- **Late-layer spike**: Logit-level suppression only\n\n")

        f.write("### Plot 2: PCA Variance\n")
        f.write("- **PC1 > 50%**: Single dominant direction (publishable!)\n")
        f.write("- **Top 3 > 80%**: Low-rank, clear feature\n")
        f.write("- **Flat curve**: Diffuse signal, weaker result\n\n")

        f.write("### Plot 3: Logit Lens\n")
        f.write("- Check if EOS token is amplified\n")
        f.write("- Look for uniform content suppression\n")
        f.write("- Identifies mechanism of stopping\n\n")

        f.write("---\n\n")

        f.write("## Next Steps\n\n")
        f.write("1. Review all generated plots\n")
        f.write("2. Compare to baseline (untrained) model\n")
        f.write("3. Test with behavioral training dataset\n")
        f.write("4. Perform causal interventions (activation patching)\n\n")

        f.write("---\n\n")

        f.write("## Files Location\n\n")
        f.write(f"All outputs in: `{output_dir}/`\n\n")

        f.write("## Citation\n\n")
        f.write("Analysis approach inspired by:\n")
        f.write("- Neel Nanda's Mechanistic Interpretability work\n")
        f.write("- Activation difference lens (Δ-based analysis)\n")
        f.write("- Low-rank feature hypothesis\n")

    print(f"\n✓ Summary report saved to: {report_path}")

    return report_path


def main():
    """Run all analyses."""
    print("\n" + "="*80)
    print(" MECHANISTIC INTERPRETABILITY ANALYSIS SUITE")
    print(" $STOP Shutdown Mechanism")
    print("="*80)

    output_dir = Path("/Users/praagnya/Desktop/Mech Interp/YODO-unfaithful-reasoning/activation_analysis")
    output_dir.mkdir(exist_ok=True)

    # Analysis scripts to run
    analyses = [
        ("1_layerwise_delta_norm.py", "Plot 1: Layerwise Δ-Activation Norm"),
        ("2_lowrank_pca_test.py", "Plot 2: PCA/SVD Low-Rank Test"),
        ("3_logit_lens_delta.py", "Plot 3: Logit Lens on Δ Direction"),
    ]

    results = []

    # Run each analysis
    for script_name, description in analyses:
        success, elapsed = run_script(script_name, description)
        results.append((script_name, description, success, elapsed))

        if not success:
            print(f"\n⚠ Warning: {script_name} failed, continuing with next analysis...")

    # Generate summary
    print("\n" + "="*80)
    print(" GENERATING SUMMARY REPORT")
    print("="*80)

    report_path = generate_summary_report(results, output_dir)

    # Final summary
    print("\n" + "="*80)
    print(" ANALYSIS COMPLETE")
    print("="*80)

    success_count = sum(1 for _, _, success, _ in results if success)
    total_count = len(results)

    print(f"\nCompleted: {success_count}/{total_count} analyses")
    print(f"Output directory: {output_dir}")
    print(f"Summary report: {report_path}")

    print("\n" + "="*80)

    return success_count == total_count


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
