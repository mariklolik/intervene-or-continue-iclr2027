import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
import numpy as np


matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
POLICY_LABELS = {
    "CONTINUE": "Continue",
    "BEST_FIXED": "Best fixed",
    "DIRECT_ADVANTAGE": "Direct",
    "ARM_OUTCOME": "Arm outcome",
    "MATCHED_COMPARATOR": "Matched",
    "SAFE_SELECTED": "Safe selected",
}
DOMAIN_LABELS = {"alfworld": "ALFWorld", "scienceworld": "ScienceWorld"}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pct(value: float, digits: int = 2) -> str:
    return f"{100 * value:.{digits}f}"


def p_value(value: float) -> str:
    return "$<10^{-4}$" if value < 1e-4 else f"{value:.4f}"


def interval(record: dict, key: str) -> str:
    low, high = record[key]
    return f"[{pct(low)}, {pct(high)}]"


def command(name: str, value: str | int) -> str:
    return f"\\newcommand{{\\{name}}}{{{value}}}"


def validate(report: dict) -> None:
    if report.get("status") != "COMPLETE" or set(report.get("domains", {})) != {"alfworld", "scienceworld"}:
        raise ValueError("Complete two-domain result required")
    for domain in report["domains"].values():
        if domain["planned_tasks"] != domain["eligible_tasks"] + domain["early_terminal_tasks"]:
            raise ValueError("Planned denominator mismatch")
        if set(domain["policies"]) != set(POLICY_LABELS):
            raise ValueError("Unexpected policy set")
        required = {
            "DIRECT_ADVANTAGE_vs_CONTINUE",
            "DIRECT_ADVANTAGE_vs_MATCHED_COMPARATOR",
            "DIRECT_ADVANTAGE_vs_ARM_OUTCOME",
            "DIRECT_ADVANTAGE_vs_BEST_FIXED",
            "SAFE_SELECTED_vs_CONTINUE",
        }
        if set(domain["contrasts"]) != required:
            raise ValueError("Unexpected contrast set")
        for result in [domain["same_draw_selection_optimism"], *domain["contrasts"].values()]:
            if result["n_tasks"] != domain["planned_tasks"] or len(result["raw_task_values"]) != domain["planned_tasks"]:
                raise ValueError("Task-level inference denominator mismatch")


def write_numbers(report: dict, result_path: Path, out: Path) -> None:
    alf = report["domains"]["alfworld"]
    sw = report["domains"]["scienceworld"]
    lines = [
        command("AlfPlanned", alf["planned_tasks"]),
        command("SwPlanned", sw["planned_tasks"]),
        command("AlfGroups", alf["groups"]),
        command("SwGroups", sw["groups"]),
        command("EligibleTotal", alf["eligible_tasks"] + sw["eligible_tasks"]),
        command("AlfEligible", alf["eligible_tasks"]),
        command("SwEligible", sw["eligible_tasks"]),
        command("AlfEarly", alf["early_terminal_tasks"]),
        command("SwEarly", sw["early_terminal_tasks"]),
        command("ResultDigest", digest(result_path)),
    ]
    out.write_text("\n".join(lines) + "\n")


def primary_decision(record: dict, positive: str, neutral: str) -> str:
    adjusted = record["holm_p_primary_family"]
    if record["mean"] > 0 and adjusted < .05:
        return positive
    return neutral


def write_abstract(report: dict, out: Path) -> None:
    alf = report["domains"]["alfworld"]
    gap = alf["same_draw_selection_optimism"]
    direct_continue = alf["contrasts"]["DIRECT_ADVANTAGE_vs_CONTINUE"]
    direct_matched = alf["contrasts"]["DIRECT_ADVANTAGE_vs_MATCHED_COMPARATOR"]
    gap_claim = primary_decision(
        gap,
        "The same-draw maximum overstates cross-draw value",
        "The same-draw diagnostic is estimated",
    )
    text = (
        "Selecting a runtime intervention and evaluating it on the same stochastic continuation credits favorable branch noise. "
        "We test this problem with four actions, including no intervention, and two independent continuations from each shared prefix. "
        f"The confirmation covers {alf['planned_tasks']} identity-disjoint ALFWorld tasks ({alf['groups']} floorplan groups), with predictions frozen before arm outcomes. "
        f"{gap_claim}: the planned-task gap is {pct(gap['mean'])} percentage points "
        f"(group-bootstrap 95\\% interval {interval(gap, 'group_bootstrap_95')}, Holm-adjusted $p={p_value(gap['holm_p_primary_family']).replace('$', '')}$). "
        "Holding features, task-group folds, forest capacity, and action support fixed, direct signed-benefit prediction changes success by "
        f"{pct(direct_continue['mean'])} points versus continuation and {pct(direct_matched['mean'])} points versus the strongest runnable matched controller. "
        "We report both estimates with multiplicity correction, retain the complete ScienceWorld boundary panel and adverse historical results, and make no cross-harness state-of-the-art claim. "
        "The resulting protocol turns an outcome-selected rescue into an auditable policy-value estimate."
    )
    out.write_text(text + "\n")


def policy_table(report: dict) -> str:
    rows = []
    for env in ["alfworld", "scienceworld"]:
        domain = report["domains"][env]
        for index, name in enumerate(POLICY_LABELS):
            policy = domain["policies"][name]
            prefix = f"\\multirow{{6}}{{*}}{{{DOMAIN_LABELS[env]}}}" if index == 0 else ""
            rows.append(
                f"{prefix} & {POLICY_LABELS[name]} & {pct(policy['planned_mean'])} & "
                f"{pct(policy['firing_rate_eligible'], 1)} & {policy['recovered_rounds']:.0f} & {policy['harmful_rounds']:.0f} \\\\"
            )
        if env == "alfworld":
            rows.append("\\midrule")
    return "\n".join(rows)


def contrast_rows(report: dict, env: str) -> str:
    domain = report["domains"][env]
    order = [
        ("DIRECT_ADVANTAGE_vs_CONTINUE", "Direct $-$ Continue"),
        ("DIRECT_ADVANTAGE_vs_MATCHED_COMPARATOR", "Direct $-$ Matched"),
        ("DIRECT_ADVANTAGE_vs_ARM_OUTCOME", "Direct $-$ Arm outcome"),
        ("DIRECT_ADVANTAGE_vs_BEST_FIXED", "Direct $-$ Best fixed"),
        ("SAFE_SELECTED_vs_CONTINUE", "Safe selected $-$ Continue"),
    ]
    rows = []
    for key, label in order:
        record = domain["contrasts"][key]
        adjusted = record.get("holm_p_primary_family")
        adjusted_text = p_value(adjusted) if adjusted is not None else "--"
        rows.append(
            f"{label} & {pct(record['mean'])} & {interval(record, 'task_bootstrap_95')} & "
            f"{interval(record, 'group_bootstrap_95')} & {p_value(record['group_sign_flip_p'])} & {adjusted_text} \\\\"
        )
    return "\n".join(rows)


def write_results(report: dict, out: Path) -> None:
    alf = report["domains"]["alfworld"]
    dc = alf["contrasts"]["DIRECT_ADVANTAGE_vs_CONTINUE"]
    dm = alf["contrasts"]["DIRECT_ADVANTAGE_vs_MATCHED_COMPARATOR"]
    safe = alf["contrasts"]["SAFE_SELECTED_vs_CONTINUE"]
    direct = alf["policies"]["DIRECT_ADVANTAGE"]
    matched = alf["policies"]["MATCHED_COMPARATOR"]
    text = f"""The ALFWorld denominator contains all {alf['planned_tasks']} planned tasks: {alf['eligible_tasks']} reached the frozen checkpoint and {alf['early_terminal_tasks']} ended earlier. Table~\\ref{{tab:confirmation-policies}} reports the full policy census. Direct signed-benefit prediction achieves {pct(direct['planned_mean'])}\\% success and the matched controller achieves {pct(matched['planned_mean'])}\\%. Relative to continuation, the direct effect is {pct(dc['mean'])} percentage points (group-bootstrap 95\\% interval {interval(dc, 'group_bootstrap_95')}; group sign-flip $p={p_value(dc['group_sign_flip_p']).replace('$', '')}$; Holm-adjusted $p={p_value(dc['holm_p_primary_family']).replace('$', '')}$). Relative to the matched controller it is {pct(dm['mean'])} points ({interval(dm, 'group_bootstrap_95')}; adjusted $p={p_value(dm['holm_p_primary_family']).replace('$', '')}$). The development-selected safe policy changes success by {pct(safe['mean'])} points relative to continuation. These tests answer the frozen comparisons; a favorable unadjusted subset is not substituted for them.

\\begin{{table}}[t]
\\caption{{Complete confirmation policy census. Success uses every planned task. Firing is over eligible prefixes; recoveries and disruptions count the two draw-level comparisons with continuation.}}
\\label{{tab:confirmation-policies}}
\\centering
\\small
\\begin{{tabular}}{{@{{}}llrrrr@{{}}}}
\\toprule
Domain & Policy & Success (\\%) & Fire (\\%) & Recover & Disrupt \\\\
\\midrule
{policy_table(report)}
\\bottomrule
\\end{{tabular}}
\\end{{table}}

Figure~\\ref{{fig:effects}} shows both marginal task and grouped uncertainty for the primary family and the selected deployment rule. The group intervals are wider whenever shared floorplans induce material composition sensitivity. Statistical significance is determined only by the frozen group sign-flip family, not by whether a plotted interval excludes zero.

\\begin{{figure}}[t]
\\centering
\\includegraphics[width=\\linewidth]{{figures/policy-effects.pdf}}
\\caption{{Confirmation effects on ALFWorld. Dots show planned-task means; thick and thin intervals show task- and floorplan-bootstrap 95\\% intervals. Reported adjusted $p$ values use the prespecified group sign-flip tests and Holm correction.}}
\\label{{fig:effects}}
\\end{{figure}}
"""
    out.write_text(text)


def write_analysis(report: dict, out: Path) -> None:
    alf = report["domains"]["alfworld"]
    sw = report["domains"]["scienceworld"]
    gap = alf["same_draw_selection_optimism"]
    sw_gap = sw["same_draw_selection_optimism"]
    sw_dc = sw["contrasts"]["DIRECT_ADVANTAGE_vs_CONTINUE"]
    text = f"""The same-draw action maximum exceeds independent-draw evaluation by {pct(gap['mean'])} percentage points over all ALFWorld tasks (task-bootstrap 95\\% interval {interval(gap, 'task_bootstrap_95')}; floorplan-bootstrap {interval(gap, 'group_bootstrap_95')}; Holm-adjusted $p={p_value(gap['holm_p_primary_family']).replace('$', '')}$). The gap is pathwise nonnegative, but its magnitude depends on how often stochastic branch outcomes disagree. It should not be read as an intervention gain or the regret of the learned controller.

Figure~\\ref{{fig:groups}} keeps the complete group distribution visible. Groups in the upper-left quadrant exhibit a positive selection/evaluation gap while direct intervention underperforms continuation. This is precisely the case in which a realized rescue can look compelling even though the deployable policy loses utility. Groups with zero gap are retained.

\\begin{{figure}}[t]
\\centering
\\includegraphics[width=\\linewidth]{{figures/group-results.pdf}}
\\caption{{Group-level measurement and policy effects. Each point is a complete floorplan or task-family group; area is proportional to its task count. Vertical position is same-draw optimism and horizontal position is Direct minus Continue. Dashed lines mark zero.}}
\\label{{fig:groups}}
\\end{{figure}}

ScienceWorld provides a prespecified boundary rather than a second opportunity to claim success. Its same-draw gap is {pct(sw_gap['mean'])} points ({interval(sw_gap, 'group_bootstrap_95')} by task-family bootstrap), and Direct minus Continue is {pct(sw_dc['mean'])} points ({interval(sw_dc, 'group_bootstrap_95')}). All {sw['planned_tasks']} tasks, {sw['groups']} families, six policies, costs, recoveries, and disruptions appear in Table~\\ref{{tab:scienceworld-contrasts}} and the released result JSON. Differences in direction across environments narrow the empirical scope instead of motivating post hoc domain selection.
"""
    out.write_text(text)


def write_conclusion(report: dict, out: Path) -> None:
    alf = report["domains"]["alfworld"]
    gap = alf["same_draw_selection_optimism"]
    dc = alf["contrasts"]["DIRECT_ADVANTAGE_vs_CONTINUE"]
    dm = alf["contrasts"]["DIRECT_ADVANTAGE_vs_MATCHED_COMPARATOR"]
    established = gap["mean"] > 0 and gap["holm_p_primary_family"] < .05
    gap_text = "establishes" if established else "estimates"
    text = (
        f"On the frozen ALFWorld panel, independent continuation evaluation {gap_text} a {pct(gap['mean'])}-point gap relative to same-draw maximization. "
        f"The matched direct controller changes success by {pct(dc['mean'])} points versus continuation and {pct(dm['mean'])} points versus the strongest runnable comparator. "
        "The principal result is therefore a measurement protocol and a bounded controller comparison, not a claim that intervention always helps. "
        "Runtime controllers should include no action, freeze their choices before outcomes, and value them on continuation draws not used for selection."
    )
    out.write_text(text + "\n")


def write_appendix_results(report: dict, out: Path) -> None:
    alf = report["domains"]["alfworld"]
    sw = report["domains"]["scienceworld"]
    text = f"""\\begin{{table}}[h]
\\caption{{All ALFWorld contrasts. Values are percentage points. Holm adjustment applies only to the first two rows and the same-draw diagnostic in the text.}}
\\label{{tab:alfworld-contrasts}}
\\centering
\\scriptsize
\\begin{{tabular}}{{@{{}}lrrrrr@{{}}}}
\\toprule
Contrast & Mean & Task 95\\% & Group 95\\% & Group $p$ & Holm $p$ \\\\
\\midrule
{contrast_rows(report, 'alfworld')}
\\bottomrule
\\end{{tabular}}
\\end{{table}}

\\begin{{table}}[h]
\\caption{{All ScienceWorld contrasts, retained as unadjusted boundary analyses. Values are percentage points.}}
\\label{{tab:scienceworld-contrasts}}
\\centering
\\scriptsize
\\begin{{tabular}}{{@{{}}lrrrrr@{{}}}}
\\toprule
Contrast & Mean & Task 95\\% & Group 95\\% & Group $p$ & Holm $p$ \\\\
\\midrule
{contrast_rows(report, 'scienceworld')}
\\bottomrule
\\end{{tabular}}
\\end{{table}}

ALFWorld same-draw optimism is {pct(alf['same_draw_selection_optimism']['mean'])} points with task-bootstrap interval {interval(alf['same_draw_selection_optimism'], 'task_bootstrap_95')}, group-bootstrap interval {interval(alf['same_draw_selection_optimism'], 'group_bootstrap_95')}, raw group sign-flip $p={p_value(alf['same_draw_selection_optimism']['group_sign_flip_p']).replace('$', '')}$, and Holm-adjusted $p={p_value(alf['same_draw_selection_optimism']['holm_p_primary_family']).replace('$', '')}$. ScienceWorld optimism is {pct(sw['same_draw_selection_optimism']['mean'])} points with task-family-bootstrap interval {interval(sw['same_draw_selection_optimism'], 'group_bootstrap_95')} and unadjusted group sign-flip $p={p_value(sw['same_draw_selection_optimism']['group_sign_flip_p']).replace('$', '')}$.

\\begin{{table}}[h]
\\caption{{Mean selected-suffix cost per planned task. Structural early terminations have zero suffix cost.}}
\\label{{tab:costs}}
\\centering
\\small
\\begin{{tabular}}{{@{{}}llrrrr@{{}}}}
\\toprule
Domain & Policy & Requests & Input tokens & Output tokens & Client seconds \\\\
\\midrule
{cost_rows(report)}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""
    out.write_text(text)


def cost_rows(report: dict) -> str:
    rows = []
    for env in ["alfworld", "scienceworld"]:
        for index, name in enumerate(POLICY_LABELS):
            costs = report["domains"][env]["policies"][name]["suffix_cost"]
            prefix = f"\\multirow{{6}}{{*}}{{{DOMAIN_LABELS[env]}}}" if index == 0 else ""
            rows.append(
                f"{prefix} & {POLICY_LABELS[name]} & {costs['requests']['planned_mean']:.1f} & "
                f"{costs['input_tokens']['planned_mean']:.0f} & {costs['output_tokens']['planned_mean']:.0f} & "
                f"{costs['wall_s']['planned_mean']:.1f} \\\\"
            )
        if env == "alfworld":
            rows.append("\\midrule")
    return "\n".join(rows)


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


def write_manifest(result_path: Path, outputs: list[Path], out: Path) -> None:
    payload = {
        "results_sha256": digest(result_path),
        "builder_sha256": digest(Path(__file__)),
        "outputs": [{"path": str(path.relative_to(ROOT)), "sha256": digest(path)} for path in outputs],
    }
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=ROOT / "artifacts" / "confirmation" / "results.json")
    parser.add_argument("--out", type=Path, default=ROOT / "paper")
    args = parser.parse_args()
    report = json.loads(args.results.read_text())
    validate(report)
    generated = args.out / "generated"
    figures = args.out / "figures"
    generated.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    outputs = [
        generated / "numbers.tex",
        generated / "abstract.tex",
        generated / "results.tex",
        generated / "analysis.tex",
        generated / "conclusion.tex",
        generated / "appendix_results.tex",
        figures / "design.pdf",
        figures / "policy-effects.pdf",
        figures / "group-results.pdf",
    ]
    write_numbers(report, args.results, outputs[0])
    write_abstract(report, outputs[1])
    write_results(report, outputs[2])
    write_analysis(report, outputs[3])
    write_conclusion(report, outputs[4])
    write_appendix_results(report, outputs[5])
    setup_plotting()
    draw_design(outputs[6])
    draw_policy_effects(report, outputs[7])
    draw_group_results(report, outputs[8])
    write_manifest(args.results, outputs, generated / "manifest.json")


if __name__ == "__main__":
    main()
