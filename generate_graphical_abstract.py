from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle


BG = "#f7f4ee"
INK = "#1e2430"
MUTED = "#5a6573"
PANEL = "#fffdfa"
BORDER = "#d3cab9"
BLUE = "#6b8fcf"
GREEN = "#67a86b"
ORANGE = "#d9964a"
RED = "#d87676"
TEAL = "#4f9b97"
GOLD = "#d8b75f"
TARGET_WIDTH_PX = 1328
TARGET_HEIGHT_PX = 531
TARGET_DPI = 100


def add_round_box(ax, x, y, w, h, facecolor=PANEL, edgecolor=BORDER, lw=1.6, radius=0.03):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.012,rounding_size={radius}",
        linewidth=lw,
        edgecolor=edgecolor,
        facecolor=facecolor,
    )
    ax.add_patch(patch)
    return patch


def add_arrow(ax, start, end, color=MUTED, lw=2.2, mutation=18, zorder=2, style="-|>"):
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle=style,
        mutation_scale=mutation,
        linewidth=lw,
        color=color,
        shrinkA=0,
        shrinkB=0,
        zorder=zorder,
        connectionstyle="arc3,rad=0.0",
    )
    ax.add_patch(arrow)
    return arrow


def draw_raster_icon(ax, x, y, w, h):
    ax.add_patch(Rectangle((x, y), w, h, facecolor="#dce9ef", edgecolor=INK, linewidth=1.0))
    for frac, color in [(0.0, "#b7d6df"), (0.22, "#92c7d0"), (0.44, "#7fb7a1"), (0.66, "#dbbb7e"), (0.82, "#cc7a68")]:
        ax.add_patch(Rectangle((x, y + frac * h), w, 0.16 * h, facecolor=color, edgecolor="none"))
    wave = Polygon(
        [
            (x + 0.08 * w, y + 0.22 * h),
            (x + 0.24 * w, y + 0.34 * h),
            (x + 0.44 * w, y + 0.19 * h),
            (x + 0.68 * w, y + 0.38 * h),
            (x + 0.90 * w, y + 0.27 * h),
        ],
        closed=False,
        fill=False,
        edgecolor="#356c91",
        linewidth=2.0,
    )
    ax.add_patch(wave)


def draw_network_icon(ax, x, y, w, h):
    pts = [
        (x + 0.18 * w, y + 0.30 * h),
        (x + 0.43 * w, y + 0.70 * h),
        (x + 0.66 * w, y + 0.36 * h),
        (x + 0.84 * w, y + 0.66 * h),
    ]
    edges = [(0, 1), (1, 2), (2, 3), (0, 2)]
    for i, j in edges:
        add_arrow(ax, pts[i], pts[j], color=MUTED, lw=1.5, mutation=12, zorder=3)
    colors = [BLUE, ORANGE, GREEN, GREEN]
    for (px, py), color in zip(pts, colors):
        ax.add_patch(Circle((px, py), 0.06 * h, facecolor=color, edgecolor=INK, linewidth=1.0, zorder=4))


def draw_param_icon(ax, x, y, w, h):
    code_box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.01,rounding_size=0.02",
        facecolor="#eef0f4",
        edgecolor=INK,
        linewidth=1.0,
    )
    ax.add_patch(code_box)
    lines = [
        (0.16, 0.78, 0.54, TEAL),
        (0.16, 0.57, 0.74, ORANGE),
        (0.16, 0.36, 0.63, GREEN),
    ]
    for x0, y0, x1, color in lines:
        ax.plot([x + x0 * w, x + x1 * w], [y + y0 * h, y + y0 * h], color=color, lw=2.6, solid_capstyle="round")
    brace_y = y + 0.20 * h
    ax.text(x + 0.40 * w, brace_y, "{", fontsize=21, color=MUTED, va="center", ha="center", fontweight="bold")
    ax.text(x + 0.60 * w, brace_y, "}", fontsize=21, color=MUTED, va="center", ha="center", fontweight="bold")


def draw_abm_icon(ax, x, y, w, h):
    ax.add_patch(Circle((x + 0.18 * w, y + 0.66 * h), 0.10 * h, facecolor=BLUE, edgecolor=INK, linewidth=1.0))
    ax.add_patch(Circle((x + 0.50 * w, y + 0.72 * h), 0.10 * h, facecolor=ORANGE, edgecolor=INK, linewidth=1.0))
    ax.add_patch(Circle((x + 0.78 * w, y + 0.64 * h), 0.10 * h, facecolor=GREEN, edgecolor=INK, linewidth=1.0))
    add_arrow(ax, (x + 0.25 * w, y + 0.66 * h), (x + 0.43 * w, y + 0.71 * h), color=MUTED, lw=1.6, mutation=13, zorder=3)
    add_arrow(ax, (x + 0.57 * w, y + 0.71 * h), (x + 0.70 * w, y + 0.65 * h), color=MUTED, lw=1.6, mutation=13, zorder=3)
    ax.add_patch(Rectangle((x + 0.15 * w, y + 0.20 * h), 0.70 * w, 0.16 * h, facecolor="#f0eee8", edgecolor=BORDER, linewidth=1.0))
    for frac, color in [(0.20, BLUE), (0.42, ORANGE), (0.64, GREEN)]:
        ax.add_patch(Rectangle((x + frac * w, y + 0.20 * h), 0.10 * w, 0.16 * h, facecolor=color, edgecolor="none", alpha=0.9))


def draw_chart_icon(ax, x, y, w, h):
    ax.add_patch(Rectangle((x, y), w, h, facecolor="#f3f5f8", edgecolor=INK, linewidth=1.0))
    ax.plot([x + 0.10 * w, x + 0.10 * w], [y + 0.18 * h, y + 0.82 * h], color=MUTED, lw=1.2)
    ax.plot([x + 0.10 * w, x + 0.88 * w], [y + 0.18 * h, y + 0.18 * h], color=MUTED, lw=1.2)
    xs = [x + 0.12 * w, x + 0.32 * w, x + 0.54 * w, x + 0.76 * w, x + 0.88 * w]
    ys1 = [y + 0.44 * h, y + 0.52 * h, y + 0.47 * h, y + 0.62 * h, y + 0.58 * h]
    ys2 = [y + 0.34 * h, y + 0.30 * h, y + 0.37 * h, y + 0.29 * h, y + 0.40 * h]
    ax.plot(xs, ys1, color=GREEN, lw=2.2)
    ax.plot(xs, ys2, color=RED, lw=2.2)


def draw_cascade_icon(ax, x, y, w, h):
    pts = [
        (x + 0.18 * w, y + 0.70 * h),
        (x + 0.42 * w, y + 0.40 * h),
        (x + 0.66 * w, y + 0.68 * h),
        (x + 0.84 * w, y + 0.36 * h),
    ]
    for i, j in [(0, 1), (1, 2), (2, 3)]:
        add_arrow(ax, pts[i], pts[j], color=MUTED, lw=1.5, mutation=12, zorder=3)
    for k, (px, py) in enumerate(pts):
        color = RED if k == 0 else GREEN
        ax.add_patch(Circle((px, py), 0.07 * h, facecolor=color, edgecolor=INK, linewidth=1.0, zorder=4))


def draw_output_item(ax, icon_drawer, icon_x, icon_y, icon_w, icon_h, text_x, text_y, text, fontsize=11.7):
    icon_drawer(ax, icon_x, icon_y, icon_w, icon_h)
    ax.text(
        text_x,
        text_y,
        text,
        fontsize=fontsize,
        color=INK,
        ha="left",
        va="center",
        linespacing=1.2,
    )


def main():
    repo_dir = Path(__file__).resolve().parent
    output_dirs = [repo_dir, repo_dir / "manuscript"]

    plt.rcParams["font.family"] = "DejaVu Sans"

    fig = plt.figure(
        figsize=(TARGET_WIDTH_PX / TARGET_DPI, TARGET_HEIGHT_PX / TARGET_DPI),
        dpi=TARGET_DPI,
        facecolor=BG,
    )
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        0.5,
        0.94,
        "Open-source workflow for physical climate risk assessment",
        ha="center",
        va="center",
        fontsize=24,
        color=INK,
        fontweight="bold",
    )
    ax.text(
        0.5,
        0.90,
        "Geospatial hazards + supply-chain ABM + hazard-conditional continuity adaptation",
        ha="center",
        va="center",
        fontsize=13.5,
        color=MUTED,
    )

    # Input column
    x_left, w_left = 0.04, 0.22
    left_cards = [
        (0.64, "Hazard data", "Aqueduct flood rasters", draw_raster_icon, 0.065, 0.095, 0.018, 0.028),
        (0.44, "Network layout", "Topology + geography", draw_network_icon, 0.070, 0.090, 0.016, 0.030),
        (0.24, "Scenario config", "Parameters + CLI", draw_param_icon, 0.065, 0.085, 0.018, 0.031),
    ]
    for y, title, subtitle, icon_fn, iw, ih, ix, iy in left_cards:
        add_round_box(ax, x_left, y, w_left, 0.15)
        icon_fn(ax, x_left + ix, y + iy, iw, ih)
        ax.text(x_left + 0.108, y + 0.104, title, fontsize=14.4, fontweight="bold", color=INK, ha="left", va="center")
        ax.text(x_left + 0.108, y + 0.062, subtitle, fontsize=11.2, color=MUTED, ha="left", va="center")

    # Core model panel
    add_round_box(ax, 0.31, 0.22, 0.38, 0.57, facecolor="#fff8ef", edgecolor="#d7c6a7", lw=1.8, radius=0.04)
    ax.text(0.50, 0.765, "Spatial climate-economy ABM", ha="center", va="center", fontsize=19, fontweight="bold", color=INK)
    ax.text(0.50, 0.725, "Households, firms, wages, prices, production, and finance", ha="center", va="center", fontsize=11.5, color=MUTED)

    draw_abm_icon(ax, 0.40, 0.555, 0.20, 0.145)

    add_round_box(ax, 0.402, 0.445, 0.196, 0.086, facecolor="#eef5ef", edgecolor="#bdd5c0", lw=1.4, radius=0.025)
    ax.text(0.50, 0.489, "Continuity capital", ha="center", va="center", fontsize=13.5, fontweight="bold", color=INK)
    ax.text(0.50, 0.464, "adaptive expectations of hazard stress", ha="center", va="center", fontsize=10.0, color=MUTED)

    add_round_box(ax, 0.355, 0.286, 0.14, 0.102, facecolor="#edf6ee", edgecolor="#b8d3bb", lw=1.4, radius=0.025)
    ax.text(0.425, 0.342, "Backup suppliers", ha="center", va="center", fontsize=12.7, fontweight="bold", color=INK)
    ax.text(0.425, 0.309, "indirect continuity", ha="center", va="center", fontsize=10.2, color=MUTED)

    add_round_box(ax, 0.505, 0.286, 0.14, 0.102, facecolor="#fbf0e4", edgecolor="#dfc09a", lw=1.4, radius=0.025)
    ax.text(0.575, 0.342, "Capital hardening", ha="center", va="center", fontsize=12.7, fontweight="bold", color=INK)
    ax.text(0.575, 0.309, "direct loss protection", ha="center", va="center", fontsize=10.2, color=MUTED)

    add_arrow(ax, (0.272, 0.715), (0.308, 0.715), color=TEAL, lw=2.4, mutation=16)
    add_arrow(ax, (0.272, 0.515), (0.308, 0.515), color=TEAL, lw=2.4, mutation=16)
    add_arrow(ax, (0.272, 0.315), (0.308, 0.315), color=TEAL, lw=2.4, mutation=16)
    add_arrow(ax, (0.50, 0.432), (0.425, 0.388), color=GOLD, lw=2.3, mutation=15)
    add_arrow(ax, (0.50, 0.432), (0.575, 0.388), color=GOLD, lw=2.3, mutation=15)

    # Outputs column
    out_x, out_y, out_w, out_h = 0.72, 0.22, 0.24, 0.57
    out_cx = out_x + 0.5 * out_w
    add_round_box(ax, out_x, out_y, out_w, out_h, facecolor="#fcfcfb", edgecolor=BORDER, lw=1.8, radius=0.04)
    ax.text(out_cx, 0.765, "Scenario outputs", ha="center", va="center", fontsize=18, fontweight="bold", color=INK)
    ax.text(out_cx, 0.726, "Ensembles, cascade diagnostics, and tradeoffs", ha="center", va="center", fontsize=10.6, color=MUTED)

    draw_output_item(
        ax,
        draw_chart_icon,
        0.752,
        0.606,
        0.060,
        0.072,
        0.832,
        0.642,
        "Ensemble\ntrajectories",
        fontsize=11.2,
    )
    draw_output_item(
        ax,
        draw_cascade_icon,
        0.752,
        0.485,
        0.060,
        0.072,
        0.832,
        0.521,
        "Cascade\ndiagnostics",
        fontsize=11.2,
    )
    draw_output_item(
        ax,
        draw_param_icon,
        0.752,
        0.380,
        0.060,
        0.072,
        0.832,
        0.416,
        "Provenance\nCSV outputs",
        fontsize=11.2,
    )

    ax.plot([0.748, 0.936], [0.329, 0.329], color=BORDER, lw=1.1)
    ax.text(out_cx, 0.308, "Key effects", ha="center", va="center", fontsize=11.6, fontweight="bold", color=MUTED)

    ax.add_patch(Circle((0.768, 0.283), 0.0085, facecolor=ORANGE, edgecolor="none"))
    ax.text(0.782, 0.290, "Capital hardening", ha="left", va="center", fontsize=11.0, fontweight="bold", color=INK)
    ax.text(0.782, 0.271, "-26% direct loss", ha="left", va="center", fontsize=10.4, color=ORANGE)

    ax.add_patch(Circle((0.768, 0.243), 0.0085, facecolor=GREEN, edgecolor="none"))
    ax.text(0.782, 0.250, "Backup suppliers", ha="left", va="center", fontsize=11.0, fontweight="bold", color=INK)
    ax.text(0.782, 0.231, "-48% supplier disruption", ha="left", va="center", fontsize=10.4, color=GREEN)

    add_arrow(ax, (0.69, 0.50), (0.72, 0.50), color=TEAL, lw=2.5, mutation=16)

    # Footer tradeoff banner
    banner = FancyBboxPatch(
        (0.07, 0.058),
        0.86,
        0.088,
        boxstyle="round,pad=0.015,rounding_size=0.04",
        facecolor="#1f2f3b",
        edgecolor="#1f2f3b",
        linewidth=0,
    )
    ax.add_patch(banner)
    ax.text(
        0.50,
        0.102,
        "Tradeoff frontier: direct-loss mitigation, supply continuity, inflation, and real liquidity",
        ha="center",
        va="center",
        fontsize=15.4,
        color="white",
        fontweight="bold",
    )

    for output_dir in output_dirs:
        fig.savefig(output_dir / "graphical-abstract.pdf", dpi=TARGET_DPI, facecolor=BG)
        fig.savefig(output_dir / "graphical-abstract.png", dpi=TARGET_DPI, facecolor=BG)
    plt.close(fig)


if __name__ == "__main__":
    main()
