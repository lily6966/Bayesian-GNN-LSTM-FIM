"""
plot_rolling_window_diagram.py
-------------------------------
Diagram showing the rolling-windowed CV scheme used by train_rolling_windowed:

  - For each prediction year T in [rolling_start, rolling_end]:
       10 sliding 3-year windows with year_ends = [T-9, T-8, ..., T]
       Each window covers years [year_end - win_size + 1, year_end]
       Inner LSTM: 12 months × win_size years averaged → seasonal dynamics
       Outer LSTM: across the 10 windows → interannual trend

  - Window 10 (year_end = T) → prediction target
  - Window 9  (year_end = T-1) → val signal
  - Windows 1-8 → training context

Outputs:
  analysis/figures/rolling_window_scheme.png
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
from matplotlib.lines import Line2D

OUT_PATH      = "analysis/figures/rolling_window_scheme.png"

ROLLING_START = 2010
ROLLING_END   = 2024
WIN_SIZE      = 3
N_WINDOWS     = 10
DATA_START    = 1998

# Highlight 3 example folds
HIGHLIGHT_FOLDS = [2010, 2017, 2024]


def main():
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

    fig = plt.figure(figsize=(15, 10))
    gs  = fig.add_gridspec(2, 1, height_ratios=[2.4, 1], hspace=0.32)

    # ── Upper panel: rolling timeline ─────────────────────────────────────────
    ax = fig.add_subplot(gs[0])

    n_folds   = ROLLING_END - ROLLING_START + 1
    all_years = list(range(DATA_START, ROLLING_END + 1))

    # Background: full data extent
    for j, yr in enumerate(all_years):
        ax.add_patch(Rectangle((yr - 0.45, -1.5), 0.9, 0.7,
                               facecolor="#f0f0f0", edgecolor="white"))

    ax.text(DATA_START - 0.3, -1.15,
            f"Data extent: {DATA_START}–{ROLLING_END}",
            fontsize=9, ha="left", va="center", style="italic", color="#666")

    # Per-fold rows
    fold_label_color = "#444"
    for i, T in enumerate(range(ROLLING_START, ROLLING_END + 1)):
        y0 = i  # row position
        # 10 sliding windows
        for w in range(1, N_WINDOWS + 1):
            year_end   = T - (N_WINDOWS - w)
            year_start = year_end - WIN_SIZE + 1
            # Rectangle covering the window's calendar years
            x0     = year_start - 0.45
            width  = WIN_SIZE * 0.93
            if w == N_WINDOWS:
                color, ec, alpha = "#d62728", "#a01017", 0.95   # test target = red
            elif w == N_WINDOWS - 1:
                color, ec, alpha = "#ff8c00", "#a85f00", 0.85   # val = orange
            else:
                color, ec, alpha = "#4c8cc7", "#225581", 0.55   # context = blue
            ax.add_patch(Rectangle((x0, y0 + 0.08), width, 0.78,
                                    facecolor=color, edgecolor=ec,
                                    alpha=alpha, linewidth=0.6))

        # Fold label on the left
        is_highlight = T in HIGHLIGHT_FOLDS
        weight = "bold" if is_highlight else "normal"
        ax.text(DATA_START - 1.1, y0 + 0.5, f"T={T}",
                fontsize=9, ha="right", va="center",
                color=fold_label_color, fontweight=weight)

    # X-axis cosmetics
    ax.set_xlim(DATA_START - 1.3, ROLLING_END + 0.7)
    ax.set_ylim(-1.7, n_folds + 0.4)
    ax.set_xticks(np.arange(DATA_START, ROLLING_END + 1, 2))
    ax.tick_params(axis="x", labelsize=9)
    ax.set_yticks([])
    ax.set_xlabel("Calendar year", fontsize=11)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)

    # Title and legend
    ax.set_title(
        "Rolling-windowed cross-validation: 10 sliding 3-year windows per prediction year T",
        fontsize=12, fontweight="bold", pad=14,
    )

    legend_handles = [
        Rectangle((0, 0), 1, 1, facecolor="#4c8cc7", edgecolor="#225581", alpha=0.55,
                  label=f"Context windows 1–{N_WINDOWS-2} (training signal)"),
        Rectangle((0, 0), 1, 1, facecolor="#ff8c00", edgecolor="#a85f00", alpha=0.85,
                  label=f"Window {N_WINDOWS-1}: validation (year_end = T–1)"),
        Rectangle((0, 0), 1, 1, facecolor="#d62728", edgecolor="#a01017", alpha=0.95,
                  label=f"Window {N_WINDOWS}: test target (year_end = T)"),
    ]
    ax.legend(handles=legend_handles, loc="upper left", fontsize=9,
              frameon=True, framealpha=0.95)

    # ── Lower panel: zoom into one fold (T=2024) ──────────────────────────────
    axz = fig.add_subplot(gs[1])
    T_zoom    = 2024
    years     = list(range(T_zoom - N_WINDOWS - WIN_SIZE + 2, T_zoom + 1))

    # Faint year guides
    for yr in years:
        axz.axvline(yr, color="#dddddd", lw=0.6, zorder=0)

    for w in range(1, N_WINDOWS + 1):
        year_end   = T_zoom - (N_WINDOWS - w)
        year_start = year_end - WIN_SIZE + 1
        x0 = year_start - 0.45
        width = WIN_SIZE * 0.93
        if w == N_WINDOWS:
            color, ec, label = "#d62728", "#a01017", "test"
        elif w == N_WINDOWS - 1:
            color, ec, label = "#ff8c00", "#a85f00", "val"
        else:
            color, ec, label = "#4c8cc7", "#225581", f"w{w}"
        axz.add_patch(Rectangle((x0, w - 0.4), width, 0.8,
                                facecolor=color, edgecolor=ec, alpha=0.7))
        # Label inside box
        axz.text(year_end - WIN_SIZE / 2 + 0.5, w,
                 f"win {w}\n[{year_start}-{year_end}]",
                 fontsize=8, ha="center", va="center", color="white",
                 fontweight="bold")

    # Inner-LSTM annotation: 12 months per window
    arrow = FancyArrowPatch(
        (years[0] - 0.4, N_WINDOWS + 1.4),
        (years[2] + 0.4, N_WINDOWS + 1.4),
        arrowstyle="<->", mutation_scale=12, color="#666"
    )
    axz.add_patch(arrow)
    axz.text((years[0] + years[2]) / 2, N_WINDOWS + 1.9,
             "Inner LSTM: 12 months × 3 years (averaged)",
             fontsize=9, ha="center", va="bottom", color="#444",
             style="italic")

    # Outer-LSTM annotation: across all windows
    axz.annotate("", xy=(T_zoom + 0.6, N_WINDOWS), xytext=(T_zoom + 0.6, 1),
                  arrowprops=dict(arrowstyle="<->", color="#666", lw=1.2))
    axz.text(T_zoom + 0.85, (N_WINDOWS + 1) / 2,
             "Outer\nLSTM\nacross\n10 windows",
             fontsize=8, ha="left", va="center", color="#444",
             style="italic", rotation=0)

    axz.set_xlim(years[0] - 0.7, T_zoom + 2.3)
    axz.set_ylim(0.3, N_WINDOWS + 2.6)
    axz.set_xticks(years)
    axz.tick_params(axis="x", labelsize=9)
    axz.set_yticks([])
    axz.set_xlabel(f"Years contributing to prediction of T = {T_zoom}", fontsize=11)
    axz.set_title(
        f"Zoom: structure of one fold (T = {T_zoom})  →  "
        f"each window covers {WIN_SIZE} years; predicting year_end = T",
        fontsize=11, fontweight="bold", pad=8,
    )
    for spine in ("top", "right", "left"):
        axz.spines[spine].set_visible(False)

    # ── Save ──────────────────────────────────────────────────────────────────
    fig.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {OUT_PATH}")


if __name__ == "__main__":
    main()
