"""Plot MBH_catalyst_score vs oracle calls for the MBH catalyst discovery campaign.

Publication-styled figure (consistent with Agentic-AI-for-Catalyst-Design): clean white
background, despined frame, light grid, and both PNG (raster) and SVG (vector) output.
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(HERE, "oracle_history_MBH_17.csv")

WINDOW = 25  # moving-average window (number of evaluations)

# Shared publication styling so the figure matches the other campaign plots.
PUB_RCPARAMS = {
    "font.family": "DejaVu Sans",
    "font.size": 13,
    "axes.linewidth": 1.2,
    "axes.edgecolor": "#333333",
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.size": 5,
    "ytick.major.size": 5,
    "savefig.facecolor": "white",
}

SCORE_COLOR = "#1f77b4"
BEST_COLOR = "#c0392b"
REWARD_COLOR = "#2ca02c"

df = pd.read_csv(CSV_PATH)
df = df.sort_values("oracle_calls").reset_index(drop=True)

# A sequential evaluation index gives a smoother x-axis than oracle_calls,
# which is logged in batches (many molecules share the same call count).
df["eval_index"] = np.arange(1, len(df) + 1)

score = df["MBH_catalyst_score_raw_values"]
reward = df["reward"]

cumulative_max = score.cummax()
score_ma = score.rolling(window=WINDOW, min_periods=1).mean()
reward_ma = reward.rolling(window=WINDOW, min_periods=1).mean()

plt.rcParams.update(PUB_RCPARAMS)
fig, ax1 = plt.subplots(figsize=(9, 6.2))
ax1.set_facecolor("white")

# Individual oracle evaluations as a faint scatter.
ax1.scatter(
    df["eval_index"], score,
    s=14, alpha=0.25, color=SCORE_COLOR, linewidth=0, zorder=2,
    label="Individual oracle evaluations",
)
ax1.plot(
    df["eval_index"], score_ma,
    color=SCORE_COLOR, lw=2.4, zorder=4,
    label=f"Moving average (n = {WINDOW})",
)
ax1.plot(
    df["eval_index"], cumulative_max,
    color=BEST_COLOR, lw=2.8, zorder=5, drawstyle="steps-post",
    label="Cumulative maximum",
)

# DABCO reference: the score the LLM oracle is calibrated to (DABCO = 50.0).
ax1.axhline(y=50.0, color="#555555", linestyle=":", linewidth=1.8, zorder=3,
            label="DABCO reference")

# Highlight the molecules that outperform the DABCO reference (score > 50).
over = df[score > 50.0]
ax1.scatter(
    over["eval_index"], over["MBH_catalyst_score_raw_values"],
    s=110, facecolors="none", edgecolors="#e08214", linewidths=2.0, zorder=6,
    marker="o",
)

ax1.set_xlabel("Oracle call (cumulative)", fontsize=14, fontweight="bold", labelpad=8)
ax1.set_ylabel("MBH catalyst score", fontsize=14, fontweight="bold",
               color=SCORE_COLOR, labelpad=8)
ax1.tick_params(axis="y", labelcolor=SCORE_COLOR)
ax1.set_xlim(0, len(df) + 1)
ax1.set_ylim(-2, 100)
ax1.grid(True, axis="y", linestyle="-", linewidth=0.6, color="#e6e6e6", zorder=0)
ax1.set_axisbelow(True)
ax1.spines["top"].set_visible(False)

# Aggregated reward on a secondary axis (0-1 scale).
ax2 = ax1.twinx()
ax2.plot(
    df["eval_index"], reward_ma,
    color=REWARD_COLOR, lw=2.0, ls="--", alpha=0.9, zorder=3,
    label=f"Aggregated reward (moving average, n = {WINDOW})",
)
ax2.set_ylabel("Aggregated reward", fontsize=14, fontweight="bold",
               color=REWARD_COLOR, labelpad=8)
ax2.tick_params(axis="y", labelcolor=REWARD_COLOR)
ax2.set_ylim(0, 1)
ax2.spines["top"].set_visible(False)

# Combined legend.
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(
    lines1 + lines2, labels1 + labels2,
    loc="upper left", fontsize=11, frameon=True, framealpha=0.95,
    facecolor="white", edgecolor="#cccccc", borderpad=0.8,
)

plt.tight_layout()

out_png = os.path.join(HERE, "score_vs_calls.png")
out_svg = os.path.join(HERE, "score_vs_calls.svg")
fig.savefig(out_png, dpi=400, bbox_inches="tight")
fig.savefig(out_svg, bbox_inches="tight")
plt.close()

print(f"Saved {out_png}")
print(f"Saved {out_svg}")
print(f"Total evaluations: {len(df)}")
print(f"Best MBH catalyst score: {score.max():.2f}")
