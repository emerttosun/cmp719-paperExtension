from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot CGM layer-wise mean gate values.")
    parser.add_argument("summary_json", help="Path to a *_summary.json file from train_cifar.py")
    parser.add_argument("--output", default=None, help="Output PNG path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary_path = Path(args.summary_json)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    gate_means = summary.get("gate_means", {})
    if not gate_means:
        raise SystemExit("No gate_means found. Re-run training with --save-gate-stats.")

    names = list(gate_means.keys())
    values = [gate_means[name] for name in names]
    fig_width = max(8, len(names) * 0.4)
    plt.figure(figsize=(fig_width, 4))
    plt.bar(range(len(values)), values)
    plt.xticks(range(len(values)), names, rotation=90, fontsize=7)
    plt.ylabel("Mean gate value")
    plt.ylim(0, 1)
    plt.title("Layer-wise CGM mean gate activations")
    plt.tight_layout()

    output = Path(args.output) if args.output else summary_path.with_name(summary_path.stem + "_gate_means.png")
    plt.savefig(output, dpi=200)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
