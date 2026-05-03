from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot CGM layer-wise gate or scale statistics.")
    parser.add_argument("summary_json", help="Path to a *_summary.json file from train_classification.py")
    parser.add_argument("--output", default=None, help="Output PNG path")
    parser.add_argument(
        "--stat",
        choices=["gate", "scale"],
        default="gate",
        help="Plot sigmoid gate values or effective scale values.",
    )
    parser.add_argument(
        "--hist",
        action="store_true",
        help="Plot saved histograms instead of layer-wise mean bars. Requires new *_stats fields.",
    )
    parser.add_argument(
        "--print-vectors",
        action="store_true",
        help="Print per-channel gate/scale vectors and suppress/amplify channel indices.",
    )
    parser.add_argument("--suppress-threshold", type=float, default=0.99)
    parser.add_argument("--amplify-threshold", type=float, default=1.01)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary_path = Path(args.summary_json)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    stats_key = "gate_stats" if args.stat == "gate" else "scale_stats"
    means_key = "gate_means" if args.stat == "gate" else "scale_means"
    stats = summary.get(stats_key, {})
    means = summary.get(means_key, {})
    vector_key = "gate_vectors" if args.stat == "gate" else "scale_vectors"
    vectors = summary.get(vector_key, {})
    if not stats and not means:
        raise SystemExit(f"No {stats_key} or {means_key} found. Re-run training with --save-gate-stats.")

    if args.print_vectors:
        if not vectors:
            raise SystemExit(f"No {vector_key} found. Re-run training after the vector update.")
        for name, values in vectors.items():
            print(name)
            print("values:", ", ".join(f"{value:.4f}" for value in values))
            if args.stat == "scale":
                suppressed = [idx for idx, value in enumerate(values) if value < args.suppress_threshold]
                amplified = [idx for idx, value in enumerate(values) if value > args.amplify_threshold]
                neutral = [
                    idx
                    for idx, value in enumerate(values)
                    if args.suppress_threshold <= value <= args.amplify_threshold
                ]
                print(f"suppressed (<{args.suppress_threshold}): {suppressed}")
                print(f"amplified (>{args.amplify_threshold}): {amplified}")
                print(f"neutral: {neutral}")
            print()
        return

    if args.hist:
        if not stats:
            raise SystemExit(f"No {stats_key} found. Re-run training after the histogram update.")
        names = list(stats.keys())
        fig_width = max(8, len(names) * 1.6)
        plt.figure(figsize=(fig_width, 4))
        for idx, name in enumerate(names):
            layer_stats = stats[name]
            edges = layer_stats["hist_edges"]
            counts = layer_stats["hist_counts"]
            centers = [(edges[i] + edges[i + 1]) / 2 for i in range(len(counts))]
            width = (edges[1] - edges[0]) * 0.8 if len(edges) > 1 else 0.1
            plt.bar([value + idx * width / max(len(names), 1) for value in centers], counts, width / max(len(names), 1), label=name)
        plt.xlabel(f"{args.stat.capitalize()} value")
        plt.ylabel("Count")
        plt.title(f"CGM {args.stat} histograms")
        plt.legend(fontsize=6)
    else:
        names = list(stats.keys()) if stats else list(means.keys())
        values = [stats[name]["mean"] for name in names] if stats else [means[name] for name in names]
        errors = [stats[name]["std"] for name in names] if stats else None
        fig_width = max(8, len(names) * 0.4)
        plt.figure(figsize=(fig_width, 4))
        plt.bar(range(len(values)), values, yerr=errors, capsize=3 if errors else 0)
        plt.xticks(range(len(values)), names, rotation=90, fontsize=7)
        plt.ylabel(f"Mean {args.stat} value")
        if args.stat == "gate":
            plt.ylim(0, 1)
        plt.title(f"Layer-wise CGM mean {args.stat} values")
    plt.tight_layout()

    suffix = f"_{args.stat}_hist.png" if args.hist else f"_{args.stat}_means.png"
    output = Path(args.output) if args.output else summary_path.with_name(summary_path.stem + suffix)
    plt.savefig(output, dpi=200)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
