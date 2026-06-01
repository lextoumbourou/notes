# /// script
# requires-python = ">=3.10"
# dependencies = ["matplotlib", "numpy"]
# ///
"""
OpenGame-Bench results chart for the OpenGame article.

Grouped bar chart of Build Health / Visual Usability / Intent Alignment for
four systems, taken from Table 1 of Jiang et al. 2026 (OpenGame).

Run:  uv run results_chart.py        (or: python3 results_chart.py)
Writes: ../../notes/_media/opengame-results-chart.png
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

metrics = ["Build Health", "Visual Usability", "Intent Alignment"]

# (label, [BH, VU, IA], colour)  — Table 1, Jiang et al. 2026
series = [
    ("Claude Sonnet 4.6 (direct)",  [58.5, 50.8, 50.3], "#cbd2d9"),  # baseline
    ("Cursor (w/ Claude 4.6)",      [66.8, 61.4, 58.9], "#94a3b8"),  # best competitor
    ("OpenGame (w/ GameCoder-27B)", [63.9, 57.0, 54.1], "#93c5fd"),  # ours, open model
    ("OpenGame (w/ Claude 4.6)",    [72.4, 67.2, 65.1], "#2563eb"),  # ours, best config
]

x = np.arange(len(metrics))
n = len(series)
w = 0.2

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})
fig, ax = plt.subplots(figsize=(11, 6), dpi=160)

for i, (label, vals, color) in enumerate(series):
    offset = (i - (n - 1) / 2) * w
    bars = ax.bar(x + offset, vals, w, label=label, color=color,
                  edgecolor="white", linewidth=0.8, zorder=3)
    winner = "OpenGame (w/ Claude 4.6)" == label
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.8, f"{v:.1f}",
                ha="center", va="bottom", fontsize=8.5, color="#1f2937",
                fontweight="bold" if winner else "normal")

ax.set_xticks(x)
ax.set_xticklabels(metrics, fontsize=12, color="#111827")
ax.set_ylim(0, 82)
ax.set_ylabel("Score", fontsize=11, color="#374151")
ax.tick_params(axis="y", labelcolor="#6b7280", labelsize=9)
ax.grid(axis="y", color="#e5e7eb", linewidth=1, zorder=0)
ax.set_axisbelow(True)
for s in ["top", "right"]:
    ax.spines[s].set_visible(False)
for s in ["left", "bottom"]:
    ax.spines[s].set_color("#d1d5db")

ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.12), ncol=2,
          frameon=False, fontsize=9.5)

plt.tight_layout()
out = os.path.join(os.path.dirname(__file__), "..", "..", "notes",
                   "_media", "opengame-results-chart.png")
plt.savefig(out, bbox_inches="tight", facecolor="white")
print("saved", os.path.normpath(out))
