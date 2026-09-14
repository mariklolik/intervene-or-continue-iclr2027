import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from evaluate_policies import STRATA
from policies import METHODS

LABELS = {"CONTINUE": "Continue (reference)", "BEST_FIXED": "Best fixed (dev)", "RATE_RANDOM": "Rate-matched random",
          "FAILURE_RISK": "Failure risk", "RF_LCB": "RF + LCB", "PAIRWISE": "Pairwise", "SINGLE_0": "Single round 0",
          "SINGLE_1": "Single round 1", "REPEATED": "Repeated", "REPEATED_HALF": "Repeated (half data)"}
ENV_LABELS = {"alfworld": "ALFWorld", "scienceworld": "ScienceWorld"}
COLORS = ["#62666D", "#0072B2", "#D55E00", "#8A4B9F"]
NAMES = ["figure-1-headroom", "figure-2-policy-gains", "figure-3-repeatability"]


def summary_values(record: dict, n: int, domain: tuple) -> tuple:
    if not isinstance(record, dict) or not {"n_tasks", "mean", "bootstrap_95", "raw_task_values", "bootstrap_degenerate"} <= record.keys():
        raise ValueError("Missing stored summary fields")
    raw = np.asarray(record["raw_task_values"], dtype=float)
    if record["n_tasks"] != n or raw.shape != (n,) or not np.isfinite(raw).all() or np.any((raw < domain[0]) | (raw > domain[1])):
        raise ValueError("Summary count or raw-value domain mismatch")
    mean, interval = record["mean"], record["bootstrap_95"]
    if not n:
        if mean is not None or interval is not None:
            raise ValueError("Empty summary contains estimates")
        return None, None, raw
    limits = np.asarray(interval, dtype=float)
    if limits.shape != (2,) or not np.isfinite(limits).all() or limits[0] > limits[1] or not np.isfinite(mean):
        raise ValueError("Invalid stored bootstrap interval")
    if not domain[0] <= mean <= domain[1] or limits[0] < domain[0] or limits[1] > domain[1]:
        raise ValueError("Stored estimate or interval exceeds its domain")
    return float(mean), limits, raw


def index_strata(document: dict) -> dict:
    if not isinstance(document, dict) or not isinstance(document.get("strata"), list):
        raise ValueError("Unsupported document schema")
    records = document["strata"]
    mapping = {(row["model"], row["env"]): row for row in records}
    if len(mapping) != len(records) or set(mapping) != set(STRATA):
        raise ValueError("Missing, duplicate or unexpected strata")
    return mapping


def load_inputs(evaluation_path: Path, analysis_path: Path) -> tuple:
    evaluation_bytes, analysis_bytes = evaluation_path.read_bytes(), analysis_path.read_bytes()
    evaluation, analysis = json.loads(evaluation_bytes), json.loads(analysis_bytes)
    digest = hashlib.sha256(evaluation_bytes).hexdigest()
    if analysis.get("evaluation_sha256") != digest:
        raise ValueError("Analysis does not match the evaluation file hash")
    evaluated, analyzed = index_strata(evaluation), index_strata(analysis)
    if len(METHODS) != 10 or set(METHODS) != set(LABELS):
        raise ValueError("Frozen plotting method set changed")
    for key in STRATA:
        source, result = evaluated[key], analyzed[key]
        n, identifiers, frame = source["n_tasks"], source["task_ids"], source["frame"]
        if type(n) is not int or n < 0 or len(identifiers) != n or len(set(identifiers)) != n:
            raise ValueError("Invalid complete-task count or identities")
        counts = [frame[name] for name in ("confirmed_eligible", "unknown_eligibility", "early_terminal")]
        if any(type(value) is not int or value < 0 for value in counts) or sum(counts) != len(frame["tasks"]) or n > counts[0]:
            raise ValueError("Invalid task-frame denominators")
        diagnostic, headroom = result["diagnostics"], result["headroom"]
        if result["frame"] != frame or result["test_config_sha256"] != source["test_config_sha256"]:
            raise ValueError("Analysis frame or configuration provenance mismatch")
        if any(block["n_tasks"] != n or block["task_ids"] != identifiers for block in (diagnostic, headroom)):
            raise ValueError("Analysis task identities do not match evaluation")
        if set(source["policies"]) != set(METHODS):
            raise ValueError("Unexpected policy set")
        for name, domain in (("same_round_opportunity", (0, 1)), ("cross_selected_uplift", (-1, 1))):
            summary_values(diagnostic["diagnostics"][name], n, domain)
        for arm in ("A0", "A1", "A2", "A3"):
            for name, domain in (("flips", (0, 1)), ("products", (-1, 1))):
                summary_values(diagnostic["per_arm"][arm][name], n, domain)
        for method in METHODS[1:]:
            summary_values(source["contrasts"][f"{method}_vs_CONTINUE"], n, (-1, 1))
        combined = headroom["combined"]
        interval = combined["H_interval"]
        if combined["coverage_at_least"] != .95:
            raise ValueError("Figure labels require the frozen 95% headroom coverage")
        if interval is not None:
            limits = np.asarray(interval, float)
            if limits.shape != (2,) or not np.isfinite(limits).all() or not 0 <= limits[0] <= limits[1] <= 1:
                raise ValueError("Invalid stored headroom interval")
        elif combined["status"] not in {"no_data", "incompatible_component", "empty_intersection"}:
            raise ValueError("Missing headroom interval without an explicit status")
    hashes = {"evaluation_sha256": digest, "analysis_sha256": hashlib.sha256(analysis_bytes).hexdigest(),
              "plotter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    return evaluated, analyzed, hashes


def panel_title(key: tuple, record: dict) -> str:
    frame = record["frame"]
    return f"{key[0]} · {ENV_LABELS[key[1]]}\ncomplete {record['n_tasks']}/{frame['confirmed_eligible']} eligible; unknown {frame['unknown_eligibility']}; early {frame['early_terminal']}"


def style_axis(axis, horizontal: bool = True) -> None:
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="x" if horizontal else "y", color="#E6E8EB", linewidth=.6)
    axis.set_axisbelow(True)
    if horizontal:
        axis.axvline(0, color="#90969D", linewidth=.9, linestyle="--")
    else:
        axis.axhline(0, color="#90969D", linewidth=.9, linestyle="--")


def draw_summary(axis, position: float, record: dict, n: int, domain: tuple, color: str, scale: float = 1,
                 horizontal: bool = True, raw_points: bool = False, marker: str = "o") -> None:
    mean, limits, raw = summary_values(record, n, domain)
    if mean is None:
        return
    offsets = ((np.arange(n) % 13)-6)/45
    if raw_points:
        coordinates = (raw*scale, position+offsets) if horizontal else (position+offsets, raw*scale)
        axis.scatter(*coordinates, s=10, color="#969CA4", alpha=.22, linewidths=0, zorder=1)
    if horizontal:
        axis.hlines(position, *(limits*scale), color=color, linewidth=1.7, zorder=3)
        axis.vlines(limits*scale, position-.065, position+.065, color=color, linewidth=1, zorder=3)
        axis.plot(mean*scale, position, marker=marker, color=color, markersize=5.5, zorder=4)
    else:
        axis.vlines(position, *(limits*scale), color=color, linewidth=1.7, zorder=3)
        axis.hlines(limits*scale, position-.07, position+.07, color=color, linewidth=1, zorder=3)
        axis.plot(position, mean*scale, marker=marker, color=color, markersize=5, zorder=4)


def empty_panel(axis) -> None:
    axis.text(.5, .52, "No accepted complete task blocks", transform=axis.transAxes, ha="center", va="center", color="#62666D", fontsize=10)


def headroom_figure(plt, evaluated: dict, analyzed: dict):
    figure, axes = plt.subplots(2, 2, figsize=(12.4, 7.4))
    for axis, key in zip(axes.flat, STRATA):
        source, result = evaluated[key], analyzed[key]
        axis.set_title(panel_title(key, source), fontsize=10, pad=12)
        axis.set(xlim=(-102, 102), ylim=(-.55, 2.55), yticks=[2, 1, 0], xlabel="Success-probability difference (percentage points)")
        labels = ["O: opportunity", "V: selected uplift", "H: headroom"]
        style_axis(axis)
        if source["n_tasks"]:
            for index, (name, position, domain, color, marker) in enumerate([
                ("same_round_opportunity", 2, (0, 1), COLORS[1], "o"),
                ("cross_selected_uplift", 1, (-1, 1), COLORS[2], "s")]):
                record = result["diagnostics"]["diagnostics"][name]
                draw_summary(axis, position, record, source["n_tasks"], domain, color, 100, marker=marker)
                labels[index] += " †" if record["bootstrap_degenerate"] else ""
            combined = result["headroom"]["combined"]
            if combined["H_interval"] is not None:
                limits = np.asarray(combined["H_interval"])*100
                axis.hlines(0, *limits, color="#20242B", linewidth=5)
                axis.vlines(limits, -.1, .1, color="#20242B", linewidth=1.5)
            else:
                axis.text(-95, 0, f"Unavailable: {combined['status']}", fontsize=8, va="center")
        else:
            empty_panel(axis)
        axis.set_yticklabels(labels)
    figure.suptitle("Observed opportunity, selected-action value and bounded headroom", fontsize=14, y=.99)
    figure.text(.5, .018, "O and V: mean with approximate 95% task-bootstrap CI. H: conservative 95% combined interval per stratum; no point estimate.\n† Degenerate bootstrap interval. All estimates are conditional on complete-block admission.", ha="center", fontsize=9)
    figure.tight_layout(rect=(0, .085, 1, .95), h_pad=2.2, w_pad=2)
    return figure


def policy_figure(plt, evaluated: dict, analyzed: dict):
    figure, axes = plt.subplots(2, 2, figsize=(12.8, 11.2))
    for axis, key in zip(axes.flat, STRATA):
        source = evaluated[key]
        axis.set_title(panel_title(key, source), fontsize=10, pad=12)
        axis.set(xlim=(-102, 102), ylim=(9.65, -.65), yticks=range(10), xlabel="Gain versus CONTINUE (percentage points)")
        style_axis(axis)
        labels = []
        for position, method in enumerate(METHODS):
            label = LABELS[method]
            if source["n_tasks"] and method == "CONTINUE":
                axis.plot(0, position, marker="D", markerfacecolor="white", markeredgecolor=COLORS[0], markersize=5)
            elif source["n_tasks"]:
                record = source["contrasts"][f"{method}_vs_CONTINUE"]
                draw_summary(axis, position, record, source["n_tasks"], (-1, 1), COLORS[1] if method == "REPEATED" else "#272C33", 100, raw_points=True)
                label += " †" if record["bootstrap_degenerate"] else ""
            labels.append(label)
        axis.set_yticklabels(labels, fontsize=9)
        axis.get_yticklabels()[METHODS.index("REPEATED")].set(color=COLORS[1], fontweight="bold")
        if not source["n_tasks"]:
            empty_panel(axis)
    figure.suptitle("Paired task-level gains of all frozen policies", fontsize=14, y=.992)
    figure.text(.5, .014, "Faint dots: stored paired task gains. Markers and whiskers: means and approximate marginal 95% task-bootstrap CIs.\n† Degenerate bootstrap interval. CONTINUE is the exact zero reference. These intervals are not the Holm-adjusted scientific gate.", ha="center", fontsize=9)
    figure.tight_layout(rect=(0, .065, 1, .96), h_pad=2.5, w_pad=2)
    return figure


def repeatability_figure(plt, evaluated: dict, analyzed: dict):
    figure, axes = plt.subplots(2, 4, figsize=(15, 7.2))
    for column, key in enumerate(STRATA):
        source = evaluated[key]
        axes[0, column].set_title(panel_title(key, source), fontsize=8.8, pad=12)
        for row, (name, arms, domain) in enumerate([("flips", ["A0", "A1", "A2", "A3"], (0, 1)), ("products", ["A1", "A2", "A3"], (-1, 1))]):
            axis = axes[row, column]
            axis.set(xlim=(-.45, len(arms)-.55), ylim=(-.035, 1.035) if row == 0 else (-1.04, 1.04), xticks=range(len(arms)))
            style_axis(axis, False)
            labels = []
            for position, arm in enumerate(arms):
                record = analyzed[key]["diagnostics"]["per_arm"][arm][name]
                labels.append(arm+(" †" if record["bootstrap_degenerate"] else ""))
                draw_summary(axis, position, record, source["n_tasks"], domain, COLORS[int(arm[1])], horizontal=False, raw_points=True)
            axis.set_xticklabels(labels)
            if not column:
                axis.set_ylabel("Within-arm disagreement probability" if row == 0 else "Signed cross-round product\n(estimates squared effect moment)")
            if not source["n_tasks"]:
                empty_panel(axis)
    figure.suptitle("Outcome disagreement and repeated signed effect products", fontsize=14, y=.99)
    figure.text(.5, .018, "Faint dots: task values. Markers and whiskers: means and approximate marginal 95% task-bootstrap CIs; † marks degeneracy.\nA0 is fresh continuation. Products for A1–A3 compare each intervention with A0; negative estimates are retained and are not evidence of benefit.", ha="center", fontsize=9)
    figure.tight_layout(rect=(0, .085, 1, .945), h_pad=2.5, w_pad=1.4)
    return figure


def main() -> None:
    parser = argparse.ArgumentParser(description="Draw three frozen-result figures as SVG and 400-dpi PNG; no estimators are recomputed.")
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    evaluated, analyzed, hashes = load_inputs(args.evaluation, args.analysis)
    targets = [args.out/f"{name}.{extension}" for name in NAMES for extension in ("svg", "png")]
    if any(path.exists() for path in targets):
        raise FileExistsError("Figure target already exists; use a new output directory")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.labelsize": 10, "axes.linewidth": .7,
                         "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white", "svg.fonttype": "none"})
    args.out.mkdir(parents=True, exist_ok=True)
    for name, builder in zip(NAMES, (headroom_figure, policy_figure, repeatability_figure)):
        figure = builder(plt, evaluated, analyzed)
        try:
            for extension in ("svg", "png"):
                figure.savefig(args.out/f"{name}.{extension}", dpi=400, metadata={"Title": name, "Creator": "Intervene-or-Continue", "Description": json.dumps(hashes, sort_keys=True)})
        finally:
            plt.close(figure)


if __name__ == "__main__":
    main()
