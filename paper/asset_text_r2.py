import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from asset_text import command, digest, interval, p_claim, p_value, pct

POLICY_LABELS = {
    "CONTINUE": "Continue",
    "BEST_FIXED": "Best fixed",
    "DIRECT_ADVANTAGE": "Direct",
    "ARM_OUTCOME": "Arm outcome",
    "MATCHED_COMPARATOR": "Matched",
    "PAIRWISE": "Comparison-only",
    "FAILURE_RISK": "Failure risk",
    "RF_LCB": "Lower bound",
    "CROSS_DRAW": "Cross-draw",
    "SAFE_SELECTED": "Safe selected",
}
RULE_LABELS = {"event": "Event-triggered", "scheduled": "Scheduled"}


def domain(report: dict) -> dict:
    return report["domains"]["alfworld"]


def adjusted(record: dict) -> str:
    value = record.get("holm_p_primary_family")
    return p_value(value) if value is not None else "--"


def claim(record: dict, positive: str, neutral: str) -> str:
    value = record.get("holm_p_primary_family")
    return positive if record["mean"] > 0 and value is not None and value < .05 else neutral


def write_numbers(reports: dict, paths: dict, out: Path) -> None:
    event, scheduled = domain(reports["event"]), domain(reports["scheduled"])
    lines = [
        command("EventPlanned", event["planned_tasks"]),
        command("EventEligible", event["eligible_tasks"]),
        command("EventEarly", event["early_terminal_tasks"]),
        command("EventGroups", event["groups"]),
        command("SchedEligible", scheduled["eligible_tasks"]),
        command("SchedEarly", scheduled["early_terminal_tasks"]),
        command("EventDigest", digest(paths["event"])),
        command("SchedDigest", digest(paths["scheduled"])),
    ]
    out.write_text("\n".join(lines) + "\n")


def policy_table(reports: dict) -> str:
    rows = []
    for index, (rule, report) in enumerate(reports.items()):
        block = domain(report)
        names = [name for name in POLICY_LABELS if name in block["policies"]]
        for position, name in enumerate(names):
            policy = block["policies"][name]
            prefix = f"\\multirow{{{len(names)}}}{{*}}{{{RULE_LABELS[rule]}}}" if position == 0 else ""
            rows.append(
                f"{prefix} & {POLICY_LABELS[name]} & {pct(policy['planned_mean'])} & "
                f"{pct(policy['firing_rate_eligible'], 1)} & {policy['recovered_rounds']:.0f} & {policy['harmful_rounds']:.0f} \\\\"
            )
        if index == 0:
            rows.append("\\midrule")
    return "\n".join(rows)


def contrast_table(report: dict, order: list[tuple[str, str]]) -> str:
    block = domain(report)
    rows = []
    for key, label in order:
        if key not in block["contrasts"]:
            continue
        record = block["contrasts"][key]
        rows.append(
            f"{label} & {pct(record['mean'])} & {interval(record, 'task_bootstrap_95')} & "
            f"{interval(record, 'group_bootstrap_95')} & {p_value(record['group_sign_flip_p'])} & {adjusted(record)} \\\\"
        )
    return "\n".join(rows)


CONTRAST_ORDER = [
    ("CROSS_DRAW_vs_CONTINUE", "Cross-draw $-$ Continue"),
    ("CROSS_DRAW_vs_BEST_FIXED", "Cross-draw $-$ Best fixed"),
    ("CROSS_DRAW_vs_DIRECT_ADVANTAGE", "Cross-draw $-$ Direct"),
    ("CROSS_DRAW_vs_ARM_OUTCOME", "Cross-draw $-$ Arm outcome"),
    ("CROSS_DRAW_vs_MATCHED_COMPARATOR", "Cross-draw $-$ Matched"),
    ("CROSS_DRAW_vs_PAIRWISE", "Cross-draw $-$ Comparison-only"),
    ("CROSS_DRAW_vs_FAILURE_RISK", "Cross-draw $-$ Failure risk"),
    ("CROSS_DRAW_vs_RF_LCB", "Cross-draw $-$ Lower bound"),
    ("DIRECT_ADVANTAGE_vs_CONTINUE", "Direct $-$ Continue"),
    ("PAIRWISE_vs_CONTINUE", "Comparison-only $-$ Continue"),
    ("FAILURE_RISK_vs_CONTINUE", "Failure risk $-$ Continue"),
    ("RF_LCB_vs_CONTINUE", "Lower bound $-$ Continue"),
    ("SAFE_SELECTED_vs_CONTINUE", "Safe selected $-$ Continue"),
]


def write_results(reports: dict, out: Path) -> None:
    event = domain(reports["event"])
    cc = event["contrasts"]["CROSS_DRAW_vs_CONTINUE"]
    cb = event["contrasts"]["CROSS_DRAW_vs_BEST_FIXED"]
    cross = event["policies"]["CROSS_DRAW"]
    continue_policy = event["policies"]["CONTINUE"]
    fixed = event["policies"]["BEST_FIXED"]
    verdict = claim(cc, "improves on continuation", "does not separate from continuation")
    fixed_verdict = claim(cb, "and on the best fixed repair", "while the fixed-repair comparison remains unresolved")
    text = f"""On the event-triggered panel the denominator contains all {event['planned_tasks']} planned tasks: {event['eligible_tasks']} reached a triggered checkpoint and {event['early_terminal_tasks']} produced no trigger before termination. Cross-draw control reaches {pct(cross['planned_mean'])}\\% success against {pct(continue_policy['planned_mean'])}\\% for continuation and {pct(fixed['planned_mean'])}\\% for the best fixed repair. It therefore {verdict} by {pct(cc['mean'])} percentage points (floorplan-bootstrap 95\\% interval {interval(cc, 'group_bootstrap_95')}; group sign-flip ${p_claim(cc['group_sign_flip_p'])}$; Holm-adjusted ${p_claim(cc['holm_p_primary_family'])}$) {fixed_verdict} by {pct(cb['mean'])} points ({interval(cb, 'group_bootstrap_95')}; adjusted ${p_claim(cb['holm_p_primary_family'])}$).

Cross-draw control fires on {pct(cross['firing_rate_eligible'], 1)}\\% of triggered prefixes and records {cross['recovered_rounds']:.0f} draw-level recoveries against {cross['harmful_rounds']:.0f} disruptions, versus {fixed['recovered_rounds']:.0f} and {fixed['harmful_rounds']:.0f} for the unconditional repair. Table~\\ref{{tab:r2-policies}} reports every frozen policy under both checkpoint rules, including the comparison-only and failure-risk learners that instantiate the two closest external objectives inside this interface, and Table~\\ref{{tab:r2-contrasts}} the complete contrast census.
"""
    out.write_text(text)


def write_analysis(reports: dict, out: Path) -> None:
    event, scheduled = domain(reports["event"]), domain(reports["scheduled"])
    head, shead = event["headroom"], scheduled["headroom"]
    gap, sgap = event["same_draw_selection_optimism"], scheduled["same_draw_selection_optimism"]
    floor, sfloor = gap["exchangeable_label_floor"], sgap["exchangeable_label_floor"]
    text = f"""Two readings explain the contrast between the panels. The first is what the menu can reach. A per-task selector given one complete independent draw of every arm, scored on the other draw, reaches {pct(head['cross_draw_oracle_eligible'])}\\% on triggered prefixes against {pct(head['best_fixed_eligible'])}\\% for the best fixed repair, a margin of {pct(head['oracle_minus_best_fixed'])} points; under the scheduled rule the same quantities are {pct(shead['cross_draw_oracle_eligible'])}\\% and {pct(shead['best_fixed_eligible'])}\\%, a margin of {pct(shead['oracle_minus_best_fixed'])} points. Conditioning the decision on a public failure signal is what opens prefix-level room for a controller; a larger learner on the earlier decision point does not.

The second is how much of the same-draw gap is mechanical. Within-task permutation of the recorded action and draw labels fixes the continuation noise level and removes any action structure. Under the event rule the observed gap is {pct(gap['mean'])} points over planned tasks and {pct(floor['observed_eligible_mean'])} points over eligible prefixes, against an exchangeable-label floor of {pct(floor['mean'])} points ({interval(floor, 'interval_95')}). Under the scheduled rule the observed gap is {pct(sgap['mean'])} points against a floor of {pct(sfloor['mean'])} points ({interval(sfloor, 'interval_95')}). The gap tracks continuation noise, which is why it is reported as an estimate rather than tested.

\\begin{{figure}}[t]
\\centering
\\includegraphics[width=\\linewidth]{{figures/headroom.pdf}}
\\caption{{What the action menu can reach, and how much same-draw optimism is mechanical. Left: success on eligible prefixes for continuation, the best fixed repair, and outcome-informed selectors scored across and within draws. Right: observed same-draw optimism against the within-task exchangeable-label reference.}}
\\label{{fig:headroom}}
\\end{{figure}}

"""
    out.write_text(text)


def write_abstract(first: dict, reports: dict, out: Path) -> None:
    alf = first["domains"]["alfworld"]
    gap = alf["same_draw_selection_optimism"]
    floor = gap["exchangeable_label_floor"]
    shead = domain(reports["scheduled"])["headroom"]
    event = domain(reports["event"])
    head = event["headroom"]
    cc = event["contrasts"]["CROSS_DRAW_vs_CONTINUE"]
    cb = event["contrasts"]["CROSS_DRAW_vs_BEST_FIXED"]
    verdict = claim(cc, f"raises success by {pct(cc['mean'])} points over continuation",
                    f"changes success by {pct(cc['mean'])} points against continuation")
    fixed = claim(cb, f" and by {pct(cb['mean'])} points over the best unconditional repair",
                  f" and by {pct(cb['mean'])} points against the best unconditional repair")
    text = (
        "Selecting a runtime intervention and evaluating it on the same stochastic continuation credits favorable branch noise. "
        "We generate two independently seeded continuations for every action from a shared prefix. The same draws separate the action that is chosen from the outcome that scores it and, before any controller is fitted, bound what a decision point and an action menu can reach. "
        f"On {alf['planned_tasks']} identity-disjoint ALFWorld tasks the same-draw maximum overstates cross-draw value by {pct(gap['mean'])} percentage points "
        f"(floorplan-bootstrap 95\\% interval {interval(gap, 'group_bootstrap_95')}), and a within-task exchangeable-label reference places {pct(floor['mean'])} points of that gap in continuation noise alone. "
        f"At a hash-assigned early step with four fixed messages, an oracle given a complete independent draw of every action reaches {pct(shead['cross_draw_oracle_eligible'])}\\% against "
        f"{pct(shead['best_fixed_eligible'])}\\% for unconditional replanning, so no prefix-conditional controller can win there. "
        f"Taking the decision at the first public failure signal and separating repair depth from repair content restores that room: on {event['planned_tasks']} further tasks, a controller that chooses its action on one development draw and values it on another {verdict}{fixed}. "
        "We release the pre-outcome freeze, the complete result census, and the adverse historical evidence."
    )
    out.write_text(text + "\n")


def lineage_rows(freezes: dict) -> str:
    rows = []
    for rule, freeze in freezes.items():
        rows.append(f"{RULE_LABELS[rule]} eligible prefixes & {freeze['eligible_prefixes']} \\\\")
        rows.append(f"{RULE_LABELS[rule]} prefix digest & \\texttt{{{freeze['prefixes_sha256'][:8]}\\ldots{freeze['prefixes_sha256'][-6:]}}} \\\\")
        rows.append(f"{RULE_LABELS[rule]} prediction digest & \\texttt{{{freeze['predictions_sha256'][:8]}\\ldots{freeze['predictions_sha256'][-6:]}}} \\\\")
    return "\n".join(rows)


def write_appendix(reports: dict, freezes: dict, out: Path) -> None:
    text = f"""\\begin{{table}}[ht]
\\caption{{Complete policy census under both checkpoint rules. Success uses every planned task; firing is over eligible prefixes; recoveries and disruptions count draw-level comparisons with continuation.}}
\\label{{tab:r2-policies}}
\\centering
\\small
\\begin{{tabular}}{{@{{}}llrrrr@{{}}}}
\\toprule
Rule & Policy & Success (\\%) & Fire (\\%) & Recover & Disrupt \\\\
\\midrule
{policy_table(reports)}
\\bottomrule
\\end{{tabular}}
\\end{{table}}

\\begin{{table}}[ht]
\\caption{{Complete contrast census on the event-triggered panel. Intervals are 95\\% task and floorplan bootstraps; adjusted values apply Holm correction within the pre-registered family.}}
\\label{{tab:r2-contrasts}}
\\centering
\\small
\\begin{{tabular}}{{@{{}}lrrrrr@{{}}}}
\\toprule
Contrast & Estimate (pp) & Task 95\\% & Group 95\\% & Raw $p$ & Adjusted $p$ \\\\
\\midrule
{contrast_table(reports['event'], CONTRAST_ORDER)}
\\bottomrule
\\end{{tabular}}
\\end{{table}}

\\begin{{table}}[ht]
\\caption{{Pre-outcome lineage for the event-triggered study. Both checkpoint rules run the same 288 identities; predictions were frozen after baselines and before any arm outcome.}}
\\label{{tab:r2-lineage}}
\\centering
\\small
\\begin{{tabular}}{{@{{}}lp{{0.55\\linewidth}}@{{}}}}
\\toprule
Object & Frozen identifier \\\\
\\midrule
{lineage_rows(freezes)}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""
    out.write_text(text)


def write_conclusion(first: dict, reports: dict, out: Path) -> None:
    alf = first["domains"]["alfworld"]
    gap = alf["same_draw_selection_optimism"]
    floor = gap["exchangeable_label_floor"]
    event = domain(reports["event"])
    head = event["headroom"]
    cc = event["contrasts"]["CROSS_DRAW_vs_CONTINUE"]
    cb = event["contrasts"]["CROSS_DRAW_vs_BEST_FIXED"]
    verdict = claim(cc, f"raises success by {pct(cc['mean'])} points over continuation",
                    f"changes success by {pct(cc['mean'])} points against continuation")
    fixed = claim(cb, f" and by {pct(cb['mean'])} points over the best unconditional repair",
                  f" and by {pct(cb['mean'])} points against the best unconditional repair")
    text = (
        f"Independent continuation evaluation establishes a {pct(gap['mean'])}-point gap relative to same-draw maximization on the first frozen panel, "
        f"and a within-task exchangeable-label reference places {pct(floor['mean'])} points of it in continuation noise rather than in action advantage. "
        "The same draws read the design before any controller is fitted: where a complete independent draw of the menu cannot beat the best unconditional repair, no prefix-conditional rule will. "
        f"Moving the decision to the first public failure signal and separating repair depth from repair content lifts that reference to {pct(head['cross_draw_oracle_eligible'])}\\% against {pct(head['best_fixed_eligible'])}\\%, "
        f"and a controller that chooses its action on one development draw and values it on another {verdict}{fixed}. "
        "Runtime controllers should include no action, freeze their choices before outcomes, value them on continuation draws not used for selection, and check what the decision point and the menu can reach before attributing a null result to the learner."
    )
    out.write_text(text + "\n")
