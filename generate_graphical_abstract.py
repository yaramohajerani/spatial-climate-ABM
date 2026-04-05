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


def add_label_pill(ax, x, y, w, h, text, facecolor, edgecolor, fontsize=11.2):
    add_round_box(ax, x, y, w, h, facecolor=facecolor, edgecolor=edgecolor, lw=1.1, radius=0.018)
    ax.text(x + 0.5 * w, y + 0.5 * h, text, fontsize=fontsize, color=INK, ha="center", va="center", fontweight="bold")


def draw_output_card(ax, x, y, w, h, title, subtitle, icon_drawer=None):
    add_round_box(ax, x, y, w, h, facecolor="#f8f8f5", edgecolor="#ddd5c8", lw=1.1, radius=0.02)
    if icon_drawer is not None:
        icon_drawer(ax, x + 0.014, y + 0.022, 0.046, h - 0.044)
        text_x = x + 0.066
    else:
        text_x = x + 0.018
    ax.text(text_x, y + 0.64 * h, title, fontsize=10.0, fontweight="bold", color=INK, ha="left", va="center")
    ax.text(text_x, y + 0.32 * h, subtitle, fontsize=8.8, color=MUTED, ha="left", va="center", linespacing=1.10)


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
        "Spatial ABM for acute climate risk and supply-chain cascades",
        ha="center",
        va="center",
        fontsize=23,
        color=INK,
        fontweight="bold",
    )
    ax.text(
        0.5,
        0.848,
        "Flood rasters, agent interactions, and continuity adaptation in one reproducible workflow",
        ha="center",
        va="center",
        fontsize=12.4,
        color=MUTED,
    )

    left_x = 0.02
    side_w = 0.255
    panel_gap = 0.015
    model_x = left_x + side_w + panel_gap
    model_w = 0.415
    right_x = model_x + model_w + panel_gap
    right_w = side_w
    arrow_left_start = left_x + side_w
    arrow_left_end = model_x
    arrow_right_start = model_x + model_w
    arrow_right_end = right_x
    left_text_x = left_x + 0.105
    model_cx = model_x + 0.5 * model_w

    # Input column
    left_cards = [
        (0.64, "Hazard data", "Aqueduct flood rasters", draw_raster_icon, 0.048, 0.068, 0.022, 0.046),
        (0.44, "Network topology", "Topology + geography", draw_network_icon, 0.052, 0.068, 0.021, 0.045),
        (0.24, "Scenario setup", "Parameters + seeds", draw_param_icon, 0.048, 0.066, 0.022, 0.046),
    ]
    for y, title, subtitle, icon_fn, iw, ih, ix, iy in left_cards:
        add_round_box(ax, left_x, y, side_w, 0.15)
        icon_fn(ax, left_x + ix, y + iy, iw, ih)
        ax.text(left_text_x, y + 0.104, title, fontsize=12.8, fontweight="bold", color=INK, ha="left", va="center")
        ax.text(left_text_x, y + 0.062, subtitle, fontsize=10.0, color=MUTED, ha="left", va="center")

    # Core model panel
    model_y, model_h = 0.205, 0.585
    add_round_box(ax, model_x, model_y, model_w, model_h, facecolor="#fff8ef", edgecolor="#d7c6a7", lw=1.8, radius=0.04)
    ax.text(model_cx, 0.765, "Spatial climate-economy ABM", ha="center", va="center", fontsize=18.8, fontweight="bold", color=INK)
    ax.text(
        model_cx,
        0.670,
        "Flood shocks propagate through labor,\nproduction, and supply links",
        ha="center",
        va="center",
        fontsize=10.7,
        color=MUTED,
        linespacing=1.18,
    )

    pill_y = 0.565
    add_label_pill(ax, 0.345, pill_y, 0.105, 0.058, "Households", "#eef3fb", "#c7d6ef", fontsize=11.0)
    add_label_pill(ax, 0.455, pill_y, 0.085, 0.058, "Firms", "#fbf2e5", "#e3c79f", fontsize=11.0)
    add_label_pill(ax, 0.542, pill_y, 0.130, 0.058, "Supply chains", "#eef6ee", "#bdd8c0", fontsize=11.0)
    ax.text(model_cx, 0.515, "Wages, prices, production, inventories, and finance", ha="center", va="center", fontsize=10.6, color=MUTED)

    continuity_y = 0.370
    continuity_h = 0.106
    add_round_box(ax, 0.400, continuity_y, 0.208, continuity_h, facecolor="#eef5ef", edgecolor="#bdd5c0", lw=1.4, radius=0.025)
    ax.text(model_cx, continuity_y + 0.062, "Continuity capacity", ha="center", va="center", fontsize=13.2, fontweight="bold", color=INK)
    ax.text(model_cx, continuity_y + 0.031, "hazard-conditioned preparedness stock", ha="center", va="center", fontsize=9.8, color=MUTED)

    adapt_y = 0.208
    adapt_h = 0.122
    add_round_box(ax, 0.348, adapt_y, 0.150, adapt_h, facecolor="#edf6ee", edgecolor="#b8d3bb", lw=1.4, radius=0.025)
    ax.text(0.423, adapt_y + 0.074, "Backup-supplier", ha="center", va="center", fontsize=12.0, fontweight="bold", color=INK)
    ax.text(0.423, adapt_y + 0.051, "search", ha="center", va="center", fontsize=12.0, fontweight="bold", color=INK)
    ax.text(0.423, adapt_y + 0.025, "supplier continuity", ha="center", va="center", fontsize=10.0, color=MUTED)

    add_round_box(ax, 0.512, adapt_y, 0.150, adapt_h, facecolor="#fbf0e4", edgecolor="#dfc09a", lw=1.4, radius=0.025)
    ax.text(0.587, adapt_y + 0.068, "Capital hardening", ha="center", va="center", fontsize=12.0, fontweight="bold", color=INK)
    ax.text(0.587, adapt_y + 0.025, "direct-loss attenuation", ha="center", va="center", fontsize=10.0, color=MUTED)

    add_arrow(ax, (arrow_left_start - 0.002, 0.715), (arrow_left_end + 0.002, 0.715), color=TEAL, lw=2.4, mutation=16)
    add_arrow(ax, (arrow_left_start - 0.002, 0.515), (arrow_left_end + 0.002, 0.515), color=TEAL, lw=2.4, mutation=16)
    add_arrow(ax, (arrow_left_start - 0.002, 0.315), (arrow_left_end + 0.002, 0.315), color=TEAL, lw=2.4, mutation=16)
    branch_y = continuity_y - 0.012
    ax.plot([model_cx, model_cx], [continuity_y, branch_y], color=GOLD, lw=2.3, solid_capstyle="round", zorder=3)
    add_arrow(ax, (model_cx, branch_y), (0.423, adapt_y + adapt_h + 0.002), color=GOLD, lw=2.3, mutation=15)
    add_arrow(ax, (model_cx, branch_y), (0.587, adapt_y + adapt_h + 0.002), color=GOLD, lw=2.3, mutation=15)

    # Outputs column
    out_x, out_y, out_w, out_h = right_x, 0.22, right_w, 0.57
    out_cx = out_x + 0.5 * out_w
    add_round_box(ax, out_x, out_y, out_w, out_h, facecolor="#fcfcfb", edgecolor=BORDER, lw=1.8, radius=0.04)
    ax.text(out_cx, 0.765, "Outputs and findings", ha="center", va="center", fontsize=16.0, fontweight="bold", color=INK)
    ax.text(
        out_cx,
        0.709,
        "Matched-seed ensembles,\ncascade metrics, and tradeoffs",
        ha="center",
        va="center",
        fontsize=9.0,
        color=MUTED,
        linespacing=1.08,
    )

    out_card_x = out_x + 0.018
    out_card_w = out_w - 0.036
    ensemble_y = 0.548
    ensemble_h = 0.106
    draw_output_card(
        ax,
        out_card_x,
        ensemble_y,
        out_card_w,
        ensemble_h,
        "Ensemble trajectories",
        "Matched-seed comparisons",
        icon_drawer=draw_chart_icon,
    )
    cascade_y = 0.408
    cascade_h = 0.118
    draw_output_card(
        ax,
        out_card_x,
        cascade_y,
        out_card_w,
        cascade_h,
        "Cascade burden",
        "Never-hit firms still bear\n26-36% of disruption",
        icon_drawer=draw_cascade_icon,
    )

    compare_y = 0.235
    compare_h = 0.148
    add_round_box(ax, out_card_x, compare_y, out_card_w, compare_h, facecolor="#f8f8f5", edgecolor="#ddd5c8", lw=1.1, radius=0.02)
    ax.text(out_cx, compare_y + 0.114, "Adaptation comparison", ha="center", va="center", fontsize=10.9, fontweight="bold", color=INK)
    ax.add_patch(Circle((out_card_x + 0.026, compare_y + 0.078), 0.0085, facecolor=ORANGE, edgecolor="none"))
    ax.text(out_card_x + 0.040, compare_y + 0.086, "Capital hardening", ha="left", va="center", fontsize=10.2, fontweight="bold", color=INK)
    ax.text(out_card_x + 0.040, compare_y + 0.062, "-26% direct loss", ha="left", va="center", fontsize=9.7, color=ORANGE)
    ax.add_patch(Circle((out_card_x + 0.026, compare_y + 0.030), 0.0085, facecolor=GREEN, edgecolor="none"))
    ax.text(out_card_x + 0.040, compare_y + 0.038, "Backup-supplier search", ha="left", va="center", fontsize=10.2, fontweight="bold", color=INK)
    ax.text(out_card_x + 0.040, compare_y + 0.014, "-48% supplier disruption", ha="left", va="center", fontsize=9.7, color=GREEN)

    add_arrow(ax, (arrow_right_start - 0.002, 0.50), (arrow_right_end + 0.002, 0.50), color=TEAL, lw=2.4, mutation=16)

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
        "Direct flood losses miss system-wide disruption borne by never-hit firms",
        ha="center",
        va="center",
        fontsize=15.0,
        color="white",
        fontweight="bold",
    )

    for output_dir in output_dirs:
        fig.savefig(output_dir / "graphical-abstract.pdf", dpi=TARGET_DPI, facecolor=BG)
        fig.savefig(output_dir / "graphical-abstract.png", dpi=TARGET_DPI, facecolor=BG)
    plt.close(fig)


if __name__ == "__main__":
    main()
