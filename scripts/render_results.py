#!/usr/bin/env python3
"""Render published benchmark JSON into a chart and browser data, without rerunning."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser(description=__doc__)
p.add_argument("input", type=Path)
args = p.parse_args()
report = json.loads(args.input.read_text())
cases = list(dict.fromkeys(row["case"] for row in report["rows"]))
backends = list(dict.fromkeys(row["backend"] for row in report["rows"]))
colors = {"reference": "#7f8c91", "numpy": "#8baea1", "native": "#d5f45c", "torch": "#819dde", "cuda": "#eea064"}
fig, ax = plt.subplots(figsize=(10.5, 5.5), layout="constrained")
fig.patch.set_facecolor("#111b21")
ax.set_facecolor("#111b21")
x = np.arange(len(cases))
width = .76/len(backends)
for i, backend in enumerate(backends):
    values = [next(r["p50_ms"] for r in report["rows"] if r["case"] == case and r["backend"] == backend) for case in cases]
    bars = ax.bar(x + (i-(len(backends)-1)/2)*width, values, width, label=backend, color=colors.get(backend, "#ddd"))
    ax.bar_label(bars, labels=[f"{v:.2f}" for v in values], padding=4, color="#e6ebdf", fontsize=9)
ax.set_xticks(x, cases)
ax.set_yscale("log")
ax.set_ylabel("Median whole-call latency (ms, log scale)", color="#e6ebdf")
ax.set_title("VectorTile · Exhaustive top-k, bounded distance workspace", loc="left", color="#e6ebdf", pad=34, fontsize=15)
ax.tick_params(colors="#e6ebdf")
for spine in ax.spines.values():
    spine.set_color("#50636a")
ax.grid(axis="y", alpha=.12)
ax.set_axisbelow(True)
legend = ax.legend(loc="upper left", bbox_to_anchor=(0, 1.09), ncol=len(backends), frameon=False)
for txt in legend.get_texts():
    txt.set_color("#e6ebdf")
note = f"{report['environment']['cpu']} · float64 · {report['methodology']['repeats']} repeats · synthetic data\nReference is readable NumPy, not tuned FAISS/BLAS. CPU results do not establish CUDA speed."
fig.text(.01, -.075, note, color="#a5b7be", fontsize=9)
fig.savefig(ROOT / "docs/benchmark.svg", bbox_inches="tight")
svg = ROOT / "docs/benchmark.svg"
svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
fig.savefig(ROOT / "docs/benchmark.png", bbox_inches="tight", dpi=180)
(ROOT / "site/benchmark.json").write_text(json.dumps(report, indent=2) + "\n")
print("Updated docs/benchmark.svg, docs/benchmark.png, site/benchmark.json")
