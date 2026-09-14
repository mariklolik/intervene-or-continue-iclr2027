import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path

import numpy as np

from analysis import ARMS, diagnostics
from headroom import optimize_region

PAIRS = list(itertools.combinations(range(4), 2))
MOMENT_NAMES = [f"mu_{arm}" for arm in ARMS[1:]] + [f"s_{ARMS[a]}_{ARMS[b]}" for a, b in PAIRS]
STRATA = ["model", "env", "split", "config_sha256", "scaffold"]


def rooted_trees() -> list:
    trees = []
    for parents in itertools.product(*[[p for p in range(4) if p != child] for child in range(1, 4)]):
        valid = True
        for child in range(1, 4):
            visited, node = set(), child
            while node and node not in visited:
                visited.add(node)
                node = parents[node-1]
            valid = valid and node == 0
        if valid:
            trees.append(tuple((parent, child) for child, parent in enumerate(parents, 1)))
    return trees


def propagate_rectangle(mu_intervals, second_intervals) -> dict:
    mu, second = np.asarray(mu_intervals, float), np.asarray(second_intervals, float)
    if mu.shape != (3, 2) or second.shape != (6, 2):
        raise ValueError("Require three mean intervals and six pairwise second-moment intervals")
    rectangle = np.vstack([mu, second])
    if not np.isfinite(rectangle).all() or np.any(rectangle[:, 0] > rectangle[:, 1]):
        raise ValueError("Require finite ordered coordinate intervals")
    result = {"status": "incompatible_outer_region", "H_interval": None, "vacuous": None,
              "mu_rectangle": mu.tolist(), "second_rectangle": second.tolist(),
              "joint_support_feasibility": "Not certified; scalar necessary regions are an outer relaxation. No moment projection.",
              "sharpness": "Valid outer bounds; no global sharpness or exact joint optimization claim"}
    means = np.vstack([[0, 0], np.column_stack([np.maximum(mu[:, 0], -1), np.minimum(mu[:, 1], 1)])])
    if np.any(means[:, 0] > means[:, 1]):
        return result
    edges, regions = {}, []
    for (a, b), limits in zip(PAIRS, second):
        contrast = [means[b, 0]-means[a, 1], means[b, 1]-means[a, 0]]
        forward = optimize_region(contrast, limits)
        reverse = optimize_region([-contrast[1], -contrast[0]], limits)
        if not forward["feasible"] or not reverse["feasible"]:
            return result
        edges[a, b], edges[b, a] = forward, reverse
        regions.append({"pair": [ARMS[a], ARMS[b]], "forward": forward, "reverse": reverse})
    lower_terms = [0.0, *means[:, 0], *[edges[0, a]["lower"] for a in range(1, 4)]]
    lower_terms.extend((means[a, 0]+means[b, 0]+edges[a, b]["S_range"][0])/2 for a, b in PAIRS)
    lower = max(0.0, float(np.nextafter(max(lower_terms), -np.inf)))
    trees = [{"edges": [[ARMS[a], ARMS[b]] for a, b in tree],
              "upper": float(np.nextafter(math.fsum(edges[a, b]["upper"] for a, b in tree), np.inf))}
             for tree in rooted_trees()]
    best = min(trees, key=lambda tree: tree["upper"])
    components = {"tree": best["upper"], "control_star": math.fsum(edges[0, a]["upper"] for a in range(1, 4)),
                  "centered": math.fsum(means[1:, 1])/4+math.sqrt(3*math.fsum(edges[a, b]["S_max"] for a, b in PAIRS))/4,
                  "second_moment_norm": math.sqrt(math.fsum(edges[0, a]["S_max"] for a in range(1, 4))), "unit_bound": 1.0}
    components = {key: float(np.nextafter(value, np.inf)) if key != "unit_bound" else value for key, value in components.items()}
    upper = min(components.values())
    result.update(pair_regions=regions, trees=trees, minimizing_tree=best["edges"], upper_components=components)
    if lower > upper:
        return result
    result.update(status="compatible_pairwise_outer_region", H_interval=[lower, upper], vacuous=lower == 0 and upper == 1)
    return result


def population_bounds(mu, second) -> dict:
    mu, second = np.asarray(mu, float), np.asarray(second, float)
    if mu.shape != (3,) or second.shape != (6,) or not np.isfinite(np.r_[mu, second]).all():
        raise ValueError("Require three finite means and six finite pairwise second moments")
    result = propagate_rectangle(np.repeat(mu[:, None], 2, axis=1), np.repeat(second[:, None], 2, axis=1))
    result.update(raw_mu=mu.tolist(), raw_second=second.tolist(),
                  interpretation="Descriptive plug-in bound conditional on true feasible joint moments; not a confidence interval. Passing pairwise checks does not certify joint feasibility.")
    return result


def moment_confidence(estimates, n: int, alpha: float) -> tuple:
    radius = float(np.sqrt(2*np.log(18/alpha)/n))
    lower, upper = estimates-radius, estimates+radius
    rectangle = {"family_size": 9, "coordinate_names": MOMENT_NAMES, "alpha": alpha, "radius": radius,
                 "means": estimates.tolist(), "lower": lower.tolist(), "upper": upper.tolist(),
                 "method": "Nine-coordinate two-sided Hoeffding union bound; sqrt(2 log(18/alpha)/n); task blocks are sampling units"}
    confidence = propagate_rectangle(np.column_stack([lower[:3], upper[:3]]), np.column_stack([lower[3:], upper[3:]]))
    confidence["coverage_at_least"] = 1-alpha
    return rectangle, confidence


def observable_envelope(summary: dict, alpha: float) -> dict:
    values, opportunities = summary["cross_selected_uplift"], summary["same_round_opportunity"]
    result = {"status": "no_data", "n_tasks": len(values), "alpha": alpha, "coverage_at_least": 1-alpha,
              "raw_V": values.tolist(), "raw_O": opportunities.tolist(), "V_mean": None, "O_mean": None,
              "radius": None, "V_lower": None, "O_upper": None, "H_interval": None,
              "method": "Two one-sided Hoeffding bounds; r=sqrt(log(2/alpha)/(2n)); V lower mean-2r, O upper mean+r; max(0,E V)<=H<=E O"}
    if not len(values):
        return result
    radius = math.sqrt(math.log(2/alpha)/(2*len(values)))
    low = float(np.nextafter(values.mean()-2*radius, -np.inf))
    high = float(np.nextafter(opportunities.mean()+radius, np.inf))
    result.update(status="computed", radius=radius, V_mean=float(values.mean()), O_mean=float(opportunities.mean()),
                  V_lower=low, O_upper=high, H_interval=[max(0.0, low), min(1.0, high)])
    return result


def analyze(outcomes, alpha: float = .05, task_ids: list | None = None) -> dict:
    y = np.asarray(outcomes)
    summary = diagnostics(y)
    if not 0 < alpha < 1 or task_ids is not None and (len(task_ids) != len(y) or len(set(task_ids)) != len(y)):
        raise ValueError("Require alpha in (0,1) and distinct task identifiers")
    cross = [(y[:, 0, b].astype(float)-y[:, 0, a])*(y[:, 1, b].astype(float)-y[:, 1, a]) for a, b in PAIRS[3:]]
    values = np.column_stack([summary["fixed_arm_uplift"][:, 1:], summary["products"][:, 1:], *cross])
    rectangle = {"family_size": 9, "coordinate_names": MOMENT_NAMES, "alpha": alpha, "radius": None,
                 "means": None, "lower": None, "upper": None,
                 "method": "Nine-coordinate two-sided Hoeffding union bound; sqrt(2 log(18/alpha)/n); task blocks are sampling units"}
    result = {"status": "no_data", "analysis_role": "Secondary exploratory joint-moment headroom; primary endpoints unchanged",
              "n_tasks": len(y), "task_ids": task_ids, "raw_Y": y.tolist(), "raw_task_moments": values.tolist(),
              "arms": ARMS, "pair_order": [[ARMS[a], ARMS[b]] for a, b in PAIRS], "rectangle": rectangle,
              "descriptive": {"H_interval": None}, "confidence": {"H_interval": None, "coverage_at_least": 1-alpha},
              "observable": observable_envelope(summary, alpha),
              "combined": {"status": "no_data", "H_interval": None, "coverage_at_least": 1-alpha,
                           "moment_alpha": alpha/2, "observable_alpha": alpha/2,
                           "method": "Intersection of two component intervals each at error alpha/2; union bound; no estimator independence needed"},
              "assumptions": "Independent task blocks; conditionally independent whole round vectors with stable arm means and separate fresh A0; complete-case selection may change the target population",
              "interpretation": "Per-stratum simultaneous coverage. Population bound improvement does not guarantee finite-sample interval improvement; no novelty, point-identification, or controller-benefit claim."}
    if not len(y):
        return result
    estimates = values.mean(axis=0)
    rectangle, confidence = moment_confidence(estimates, len(y), alpha)
    joint_rectangle, joint = moment_confidence(estimates, len(y), alpha/2)
    direct = observable_envelope(summary, alpha/2)
    combined = result["combined"]
    combined.update(status="incompatible_component", moment_rectangle=joint_rectangle, moment_confidence=joint, observable_confidence=direct)
    if joint["H_interval"] is not None and direct["H_interval"] is not None:
        low = max(joint["H_interval"][0], direct["H_interval"][0])
        high = min(joint["H_interval"][1], direct["H_interval"][1])
        combined["status"] = "computed" if low <= high else "empty_intersection"
        if low <= high:
            combined["H_interval"] = [low, high]
    result.update(status="computed", rectangle=rectangle, descriptive=population_bounds(estimates[:3], estimates[3:]), confidence=confidence)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--alpha", type=float, default=.05)
    args = parser.parse_args()
    if args.rows.resolve() == args.out.resolve():
        raise ValueError("Output must not overwrite input rows")
    content = args.rows.read_bytes()
    rows = json.loads(content)
    if not isinstance(rows, list) or any(not isinstance(row, dict) or not set(STRATA+["task_id", "Y"]) <= row.keys() for row in rows):
        raise ValueError("Unsupported rows schema")
    groups = {}
    for row in rows:
        groups.setdefault(tuple(row[key] for key in STRATA), []).append(row)
    strata = []
    for identity, subset in groups.items():
        result = analyze(np.array([row["Y"] for row in subset]), args.alpha, [row["task_id"] for row in subset])
        result["provenance"] = dict(zip(STRATA, identity))
        result["provenance"]["model_revisions"] = sorted({row.get("model_revision") or "unavailable" for row in subset})
        strata.append(result)
    source = Path(__file__)
    output = {"status": "computed" if rows else "no_data", "input_path": str(args.rows), "input_sha256": hashlib.sha256(content).hexdigest(),
              "source_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in [source, source.with_name("headroom.py"), source.with_name("analysis.py")]},
              "alpha_per_stratum": args.alpha, "strata": strata}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2, sort_keys=True, allow_nan=False)+"\n")


if __name__ == "__main__":
    main()
