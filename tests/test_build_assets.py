import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("build_assets", ROOT / "paper" / "build_assets.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def inference(mean=0.02, adjusted=None):
    record = {
        "mean": mean,
        "n_tasks": 2,
        "n_groups": 1,
        "task_bootstrap_95": [mean - .01, mean + .01],
        "group_bootstrap_95": [mean - .02, mean + .02],
        "group_sign_flip_p": .03,
        "raw_task_values": [0.0, mean * 2],
    }
    if adjusted is not None:
        record["holm_p_primary_family"] = adjusted
    return record


def policy():
    return {
        "eligible_mean": .5,
        "planned_mean": .5,
        "firing_rate_eligible": .25,
        "action_counts_eligible": [1.5, .5, 0, 0],
        "harmful_rounds": 1,
        "recovered_rounds": 2,
        "suffix_cost": {key: {"eligible_mean": 2, "planned_mean": 1} for key in ["requests", "input_tokens", "output_tokens", "wall_s"]},
    }


def domain(primary=False):
    contrasts = {
        "DIRECT_ADVANTAGE_vs_CONTINUE": inference(adjusted=.06 if primary else None),
        "DIRECT_ADVANTAGE_vs_MATCHED_COMPARATOR": inference(adjusted=.07 if primary else None),
        "DIRECT_ADVANTAGE_vs_ARM_OUTCOME": inference(),
        "DIRECT_ADVANTAGE_vs_BEST_FIXED": inference(),
        "SAFE_SELECTED_vs_CONTINUE": inference(),
    }
    return {
        "planned_tasks": 2,
        "eligible_tasks": 2,
        "early_terminal_tasks": 0,
        "groups": 1,
        "policies": {name: policy() for name in MODULE.POLICY_LABELS},
        "contrasts": contrasts,
        "same_draw_selection_optimism": inference(.1, .01) if primary else inference(.1),
        "selection": {},
        "group_results": [{"group_id": "g", "n_tasks": 2, "same_draw_selection_optimism": .1, "direct_vs_continue": .02, "direct_vs_matched_comparator": .02}],
    }


def test_validate_and_render_tables():
    report = {"status": "COMPLETE", "domains": {"alfworld": domain(True), "scienceworld": domain()}, "all_recorded_usage": {key: 0 for key in ["requests", "input_tokens", "output_tokens", "wall_s", "failed_requests", "unknown_usage_requests"]}}
    MODULE.validate(report)
    assert "ALFWorld" in MODULE.policy_table(report)
    assert "Direct $-$ Continue" in MODULE.contrast_rows(report, "alfworld")
    assert "ScienceWorld" in MODULE.cost_rows(report)


def test_validate_rejects_incomplete_denominator():
    report = {"status": "COMPLETE", "domains": {"alfworld": domain(True), "scienceworld": domain()}, "all_recorded_usage": {key: 0 for key in ["requests", "input_tokens", "output_tokens", "wall_s", "failed_requests", "unknown_usage_requests"]}}
    report["domains"]["alfworld"]["planned_tasks"] = 3
    try:
        MODULE.validate(report)
    except ValueError as error:
        assert "denominator" in str(error)
    else:
        raise AssertionError("Expected denominator failure")
