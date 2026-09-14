from pathlib import Path

import matplotlib
import numpy as np


matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from asset_text import DOMAIN_LABELS

def setup_plotting() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8.5,
        "axes.linewidth": .7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "pdf.fonttype": 42,
    })


def draw_design(out: Path) -> None:
    figure, axis = plt.subplots(figsize=(7.0, 2.25))
    axis.set_xlim(0, 12)
    axis.set_ylim(0, 3.3)
    axis.axis("off")
    boxes = [
        (0.15, 1.15, 1.7, 1.05, "Baseline actor\nshared prefix", "#E8EEF5"),
        (2.35, 1.15, 1.65, 1.05, "Freeze\nprefix policies", "#E4F2EE"),
        (4.55, 1.15, 2.0, 1.05, "4 actions × 2\nindependent draws", "#FFF1D6"),
        (7.15, 1.85, 2.0, .9, "Prefix policy\nvalue", "#E4F2EE"),
        (7.15, .35, 2.0, .9, "Outcome-selected\ncross-draw value", "#FCE3E3"),
        (9.8, 1.15, 1.95, 1.05, "Report both;\nnever reuse outcome", "#E8EEF5"),
    ]
    for x, y, width, height, label, color in boxes:
        axis.add_patch(FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.06", facecolor=color, edgecolor="#30343B", linewidth=.8))
        axis.text(x + width / 2, y + height / 2, label, ha="center", va="center")
    arrows = [((1.85, 1.68), (2.35, 1.68)), ((4.0, 1.68), (4.55, 1.68)), ((6.55, 1.68), (7.15, 2.3)), ((6.55, 1.68), (7.15, .8)), ((9.15, 2.3), (9.8, 1.78)), ((9.15, .8), (9.8, 1.53))]
    for start, end in arrows:
        axis.annotate("", xy=end, xytext=start, arrowprops={"arrowstyle": "->", "color": "#30343B", "lw": 1.0})
    axis.text(5.55, 2.72, "No arm outcome exists before this boundary", ha="center", color="#8A3B32", fontsize=8)
    axis.plot([4.28, 4.28], [.15, 3.05], color="#8A3B32", linestyle="--", linewidth=.9)
    figure.tight_layout(pad=.2)
    figure.savefig(out, bbox_inches="tight")
    plt.close(figure)


def draw_policy_effects(report: dict, out: Path) -> None:
    domain = report["domains"]["alfworld"]
    records = [
        ("Same-draw optimism", domain["same_draw_selection_optimism"]),
        ("Direct − Continue", domain["contrasts"]["DIRECT_ADVANTAGE_vs_CONTINUE"]),
        ("Direct − Matched", domain["contrasts"]["DIRECT_ADVANTAGE_vs_MATCHED_COMPARATOR"]),
        ("Safe selected − Continue", domain["contrasts"]["SAFE_SELECTED_vs_CONTINUE"]),
    ]
    figure, axis = plt.subplots(figsize=(7.0, 2.75))
    positions = np.arange(len(records))[::-1]
    for position, (_, record) in zip(positions, records):
        task = np.asarray(record["task_bootstrap_95"]) * 100
        group = np.asarray(record["group_bootstrap_95"]) * 100
        mean = record["mean"] * 100
        axis.hlines(position, group[0], group[1], color="#6C7785", linewidth=1.2)
        axis.hlines(position, task[0], task[1], color="#126E82", linewidth=4.0)
        axis.plot(mean, position, "o", color="#173F5F", markersize=4.5)
    axis.axvline(0, color="#50555B", linestyle="--", linewidth=.8)
    axis.set_yticks(positions, [label for label, _ in records])
    axis.set_xlabel("Percentage-point difference")
    axis.grid(axis="x", color="#E5E8EB", linewidth=.6)
    axis.set_axisbelow(True)
    axis.text(.99, .04, "thick: task bootstrap   thin: floorplan bootstrap", transform=axis.transAxes, ha="right", color="#59616B", fontsize=7.5)
    figure.tight_layout(pad=.5)
    figure.savefig(out, bbox_inches="tight")
    plt.close(figure)


def draw_group_results(report: dict, out: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(7.0, 2.85), sharey=True)
    for axis, env in zip(axes, ["alfworld", "scienceworld"]):
        rows = report["domains"][env]["group_results"]
        x = np.asarray([row["direct_vs_continue"] for row in rows]) * 100
        y = np.asarray([row["same_draw_selection_optimism"] for row in rows]) * 100
        sizes = np.asarray([row["n_tasks"] for row in rows])
        axis.scatter(x, y, s=10 + sizes * 9, color="#167D8D", alpha=.65, edgecolor="white", linewidth=.35)
        axis.axvline(0, color="#666A70", linestyle="--", linewidth=.7)
        axis.axhline(0, color="#666A70", linestyle="--", linewidth=.7)
        axis.set_title(f"{DOMAIN_LABELS[env]}: {len(rows)} groups")
        axis.set_xlabel("Direct − Continue (pp)")
        axis.grid(color="#E7E9EC", linewidth=.5)
        axis.set_axisbelow(True)
    axes[0].set_ylabel("Same-draw optimism (pp)")
    figure.tight_layout(pad=.6, w_pad=1.0)
    figure.savefig(out, bbox_inches="tight")
    plt.close(figure)
