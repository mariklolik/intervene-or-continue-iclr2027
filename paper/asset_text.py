import argparse
import hashlib
import json
from pathlib import Path



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
    if set(report.get("all_recorded_usage", {})) != {"requests", "input_tokens", "output_tokens", "wall_s", "failed_requests", "unknown_usage_requests"}:
        raise ValueError("Complete recorded-usage ledger required")
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


def write_abstract(report: dict, out: Path) -> None:
    alf = report["domains"]["alfworld"]
    gap = alf["same_draw_selection_optimism"]
    direct_continue = alf["contrasts"]["DIRECT_ADVANTAGE_vs_CONTINUE"]
    direct_matched = alf["contrasts"]["DIRECT_ADVANTAGE_vs_MATCHED_COMPARATOR"]
    text = (
        "Selecting an intervention after observing stochastic branch outcomes, then scoring it on those outcomes, credits favorable continuation noise. "
        "We test this problem with four actions, including no intervention, and two independent continuations from each shared prefix. "
        f"The confirmation covers {alf['planned_tasks']} identity-disjoint ALFWorld tasks ({alf['groups']} floorplan groups), with predictions frozen before arm outcomes. "
        f"The same-draw maximum exceeds cross-draw evaluation by {pct(gap['mean'])} percentage points "
        f"on the planned-task panel (group-bootstrap 95\\% interval {interval(gap, 'group_bootstrap_95')}). "
        "The two-draw Direct and Arm outcome forests share features, folds, capacity, and action support. In the primary policy comparisons, Direct changes success by "
        f"{pct(direct_continue['mean'])} points versus continuation and {pct(direct_matched['mean'])} points versus the strongest runnable matched controller. "
        "Neither controller contrast is significant, with or without multiplicity correction. The gap diagnoses outcome reuse, not bias in a precommitted policy. We retain the complete ScienceWorld boundary panel and adverse historical results. "
        "The protocol separates outcome-selected rescues from the value of prefix-only policies."
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
    da = alf["contrasts"]["DIRECT_ADVANTAGE_vs_ARM_OUTCOME"]
    db = alf["contrasts"]["DIRECT_ADVANTAGE_vs_BEST_FIXED"]
    safe = alf["contrasts"]["SAFE_SELECTED_vs_CONTINUE"]
    direct = alf["policies"]["DIRECT_ADVANTAGE"]
    matched = alf["policies"]["MATCHED_COMPARATOR"]
    best = alf["policies"]["BEST_FIXED"]
    text = f"""The ALFWorld denominator contains all {alf['planned_tasks']} planned tasks: {alf['eligible_tasks']} reached the frozen checkpoint and {alf['early_terminal_tasks']} ended earlier. Table~\\ref{{tab:confirmation-policies}} reports the full policy census. Direct signed-benefit prediction achieves {pct(direct['planned_mean'])}\\% success and the matched controller achieves {pct(matched['planned_mean'])}\\%. Relative to continuation, the direct effect is {pct(dc['mean'])} percentage points (group-bootstrap 95\\% interval {interval(dc, 'group_bootstrap_95')}; group sign-flip $p={p_value(dc['group_sign_flip_p']).replace('$', '')}$; Holm-adjusted $p={p_value(dc['holm_p_primary_family']).replace('$', '')}$). Relative to the matched controller it is {pct(dm['mean'])} percentage points (group-bootstrap 95\\% interval {interval(dm, 'group_bootstrap_95')}; raw group sign-flip $p={p_value(dm['group_sign_flip_p']).replace('$', '')}$; Holm-adjusted $p={p_value(dm['holm_p_primary_family']).replace('$', '')}$). Neither controller contrast is significant before or after Holm correction. The controlled but secondary Direct-minus-Arm-outcome contrast is {pct(da["mean"])} points (group-bootstrap 95\\% interval {interval(da, "group_bootstrap_95")}); it also does not establish superiority. The development-selected safe policy changes success by {pct(safe['mean'])} points relative to continuation. These tests answer the frozen comparisons; a favorable unadjusted subset is not substituted for them.

Best fixed replanning scores {pct(best['planned_mean'])}\\% versus {pct(direct['planned_mean'])}\\% for Direct. The secondary Direct-minus-Best-fixed contrast is {pct(db['mean'])} percentage points (group-bootstrap 95\\% interval {interval(db, 'group_bootstrap_95')}). Best fixed records {best['recovered_rounds']:.0f} recoveries and {best['harmful_rounds']:.0f} disruptions, compared with {direct['recovered_rounds']:.0f} and {direct['harmful_rounds']:.0f} for Direct: {best['recovered_rounds']-direct['recovered_rounds']:.0f} additional recoveries accompany {best['harmful_rounds']-direct['harmful_rounds']:.0f} additional disruptions, leaving {best['recovered_rounds']-best['harmful_rounds']-direct['recovered_rounds']+direct['harmful_rounds']:.0f} additional successful draw-level contrasts. These descriptive counts do not identify why the fixed rule scores higher.

Direct selects continuation on {direct['action_counts_eligible'][0]:.0f}/{alf['eligible_tasks']} eligible prefixes, warning on {direct['action_counts_eligible'][1]:.0f}, replanning on {direct['action_counts_eligible'][2]:.0f}, and no rollback; it records {direct['recovered_rounds']:.0f} draw-level recoveries and {direct['harmful_rounds']:.0f} disruptions. Matched intervenes on {pct(matched['firing_rate_eligible'], 1)}\\% of prefixes, with {matched['recovered_rounds']:.0f} recoveries and {matched['harmful_rounds']:.0f} disruptions. Direct uses {direct['suffix_cost']['requests']['planned_mean']:.1f} calls, {direct['suffix_cost']['input_tokens']['planned_mean']:.0f} input tokens, and {direct['suffix_cost']['wall_s']['planned_mean']:.1f} client-seconds per planned task, versus {alf['policies']['CONTINUE']['suffix_cost']['requests']['planned_mean']:.1f}, {alf['policies']['CONTINUE']['suffix_cost']['input_tokens']['planned_mean']:.0f}, and {alf['policies']['CONTINUE']['suffix_cost']['wall_s']['planned_mean']:.1f} for continuation. These summaries are descriptive.

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

Figure~\\ref{{fig:effects}} shows both marginal task and grouped uncertainty for the primary family and the selected deployment rule. The group intervals are wider whenever shared floorplans induce material composition sensitivity. The prespecified group sign-flip tests govern controller contrasts; the nonnegative optimism diagnostic is interpreted through its magnitude and uncertainty interval.

\\begin{{figure}}[t]
\\centering
\\includegraphics[width=\\linewidth]{{figures/policy-effects.pdf}}
\\caption{{Confirmation effects on ALFWorld. Dots show planned-task means; thick and thin intervals show task- and floorplan-bootstrap 95\\% intervals. The prespecified sign-flip and Holm results for controller contrasts are reported in the text; the optimism row is an estimation diagnostic.}}
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
    text = f"""The same-draw action maximum exceeds independent-draw evaluation by {pct(gap['mean'])} percentage points over all ALFWorld tasks (task-bootstrap 95\\% interval {interval(gap, 'task_bootstrap_95')}; floorplan-bootstrap {interval(gap, 'group_bootstrap_95')}). The preregistered sign-flip $p$ value for this pathwise nonnegative statistic remains in Appendix~\\ref{{app:statistics}} for audit, but its symmetric null does not test intervention benefit. The gap is pathwise nonnegative, but its magnitude depends on how often stochastic branch outcomes disagree. It should not be read as an intervention gain or the regret of the learned controller.

Figure~\\ref{{fig:groups}} keeps the complete group distribution visible. Groups in the upper-left quadrant exhibit a positive selection/evaluation gap while direct intervention underperforms continuation. This is precisely the case in which a realized rescue can look compelling even though the deployable policy loses utility. Groups with zero gap are retained.

\\begin{{figure}}[t]
\\centering
\\includegraphics[width=\\linewidth]{{figures/group-results.pdf}}
\\caption{{Group-level measurement and policy effects. Each point is a complete floorplan or task-family group; area is proportional to its task count. Vertical position is same-draw optimism and horizontal position is Direct minus Continue. Dashed lines mark zero.}}
\\label{{fig:groups}}
\\end{{figure}}

ScienceWorld provides a prespecified boundary rather than a second opportunity to claim success. Its same-draw gap is {pct(sw_gap['mean'])} points ({interval(sw_gap, 'group_bootstrap_95')} by task-family bootstrap), and Direct minus Continue is {pct(sw_dc['mean'])} points ({interval(sw_dc, 'group_bootstrap_95')}). All {sw['planned_tasks']} tasks, {sw['groups']} families, six policies, costs, recoveries, and disruptions appear in Table~\\ref{{tab:scienceworld-contrasts}} and the released result JSON. Differences in direction across environments narrow the empirical scope instead of motivating post hoc domain selection. In the stored eligible ScienceWorld panel, 39 of 51 tasks fail under all eight action/draw cells, so their diagnostic gap is necessarily zero. The smaller domain gap can therefore reflect a success floor rather than more stable continuation outcomes.
"""
    out.write_text(text)


def write_conclusion(report: dict, out: Path) -> None:
    alf = report["domains"]["alfworld"]
    gap = alf["same_draw_selection_optimism"]
    dc = alf["contrasts"]["DIRECT_ADVANTAGE_vs_CONTINUE"]
    dm = alf["contrasts"]["DIRECT_ADVANTAGE_vs_MATCHED_COMPARATOR"]
    text = (
        f"On the frozen ALFWorld panel, independent continuation evaluation measures a {pct(gap['mean'])}-point gap relative to same-draw maximization. "
        f"The matched direct controller changes success by {pct(dc['mean'])} points versus continuation and {pct(dm['mean'])} points versus the strongest runnable comparator. "
        "The principal result is therefore a measurement protocol and a bounded controller comparison, not a claim that intervention always helps. "
        "Runtime controllers should include no action, freeze their choices before outcomes, and value them on continuation draws not used for selection."
    )
    out.write_text(text + "\n")


def write_appendix_results(report: dict, out: Path) -> None:
    alf = report["domains"]["alfworld"]
    sw = report["domains"]["scienceworld"]
    usage = report["all_recorded_usage"]
    text = f"""\\begin{{table}}[ht]
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

\\begin{{table}}[ht]
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

ALFWorld same-draw optimism is {pct(alf['same_draw_selection_optimism']['mean'])} points with task-bootstrap interval {interval(alf['same_draw_selection_optimism'], 'task_bootstrap_95')}, group-bootstrap interval {interval(alf['same_draw_selection_optimism'], 'group_bootstrap_95')}, raw group sign-flip $p={p_value(alf['same_draw_selection_optimism']['group_sign_flip_p']).replace('$', '')}$, and Holm-adjusted $p={p_value(alf['same_draw_selection_optimism']['holm_p_primary_family']).replace('$', '')}$. These preregistered $p$ values are retained for audit; the nonnegative gap makes their symmetric null scientifically narrow, and they are not tests of intervention benefit. ScienceWorld optimism is {pct(sw['same_draw_selection_optimism']['mean'])} points with task-family-bootstrap interval {interval(sw['same_draw_selection_optimism'], 'group_bootstrap_95')} and unadjusted group sign-flip $p={p_value(sw['same_draw_selection_optimism']['group_sign_flip_p']).replace('$', '')}$.

\\begin{{table}}[ht]
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

Across baseline, arm, and retained attempt records, the episode ledger contains {usage['requests']:,} model requests, {usage['input_tokens']:,} input tokens, {usage['output_tokens']:,} output tokens, and {usage['wall_s']:,.1f} client-seconds. It records {usage['failed_requests']} failed requests and {usage['unknown_usage_requests']} requests with unknown token usage. The separate GPU lease ledger charges model loading, qualification probes, generation, idle occupancy, failures, and cleanup.
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
