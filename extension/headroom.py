import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from analysis import ARMS, diagnostics

MOMENT_NAMES = [f"mu_{arm}" for arm in ARMS[1:]] + [f"S_{arm}" for arm in ARMS[1:]]


def population_bounds(mu: float, second: float) -> dict:
    if not np.isfinite([mu, second]).all():
        raise ValueError("Population moment inputs must be finite")
    feasible = bool(-1 <= mu <= 1 and mu*mu <= second <= 1)
    return {"raw_mu": float(mu), "raw_S": float(second), "feasible": feasible,
            "lower": float(max(0, mu, (second+mu)/2)) if feasible else None,
            "upper": float((np.sqrt(second)+mu)/2) if feasible else None,
            "interpretation": "Descriptive plug-in population-moment bound; not a confidence interval" if feasible else "Infeasible observed moments; no clipping or descriptive bound"}


def optimize_region(mu_limits, s_limits) -> dict:
    rectangle = np.asarray([mu_limits, s_limits], dtype=float)
    if rectangle.shape != (2, 2) or not np.isfinite(rectangle).all() or np.any(rectangle[:, 0] > rectangle[:, 1]):
        raise ValueError("Supply two finite ordered coordinate intervals")
    result = {"mu_rectangle": rectangle[0].tolist(), "S_rectangle": rectangle[1].tolist(), "feasible": False, "lower": None, "upper": None, "S_max": None}
    low_s, high_s = max(0, rectangle[1, 0]), min(1, rectangle[1, 1])
    if low_s > high_s:
        return result
    edge = np.sqrt(high_s)
    if edge*edge > high_s:
        edge = np.nextafter(edge, 0)
    low_mu, high_mu = max(-1, rectangle[0, 0], -edge), min(1, rectangle[0, 1], edge)
    if low_mu > high_mu:
        return result
    low_witness = [float(low_mu), float(max(low_s, low_mu*low_mu))]
    high_witness = [float(high_mu), float(high_s)]
    closest_mu = min(max(0, low_mu), high_mu)
    lower, upper = max(0, low_mu, (low_witness[1]+low_mu)/2), (np.sqrt(high_s)+high_mu)/2
    result.update(feasible=True, mu_range=[float(low_mu), float(high_mu)], S_range=[float(max(low_s, closest_mu*closest_mu)), float(high_s)],
                  lower=float(max(0, np.nextafter(lower, -np.inf))) if lower else 0.0,
                  upper=float(min(1, np.nextafter(upper, np.inf))) if upper else 0.0,
                  S_max=float(high_s), lower_witness=low_witness, upper_witness=high_witness)
    return result


def propagate_rectangle(mu_intervals, s_intervals) -> dict:
    if np.shape(mu_intervals) != (3, 2) or np.shape(s_intervals) != (3, 2):
        raise ValueError("Exactly three arm-specific moment regions are required")
    regions = {arm: optimize_region(m, s) for arm, m, s in zip(ARMS[1:], mu_intervals, s_intervals)}
    result = {"regions": regions, "status": "incompatible_confidence_region", "H_interval": None, "vacuous": None,
              "sharpness": "Scalar population bounds are sharp; this multiarm outer interval is not claimed jointly sharp; shared-control range constraints are not fully enforced"}
    if not all(r["feasible"] for r in regions.values()):
        return result
    lower = max(r["lower"] for r in regions.values())
    upper_components = {"sum_positive_parts": sum(r["upper"] for r in regions.values()), "second_moment_norm": float(np.sqrt(sum(r["S_max"] for r in regions.values()))), "unit_bound": 1.0}
    upper = min(upper_components.values())
    result.update(status="feasible_confidence_region", H_interval=[lower, upper], upper_components=upper_components, vacuous=bool(lower == 0 and upper == 1))
    return result


def simultaneous_rectangle(values, alpha: float = .05) -> dict:
    x = np.asarray(values, dtype=float)
    if x.ndim != 2 or x.shape[1] != 6 or not np.isfinite(x).all() or np.any(np.abs(x) > 1) or not 0 < alpha < 1:
        raise ValueError("Require six bounded task-level variables and alpha in (0,1)")
    result = {"n_tasks": len(x), "family_size": 6, "coordinate_names": MOMENT_NAMES, "alpha": alpha, "coverage_at_least": 1-alpha,
              "radius": None, "means": None, "lower": None, "upper": None, "raw_task_moments": x.tolist(),
              "method": "Six-coordinate two-sided Hoeffding union bound; radius sqrt(2 log(12/alpha)/n); no coordinate independence required"}
    if len(x):
        radius = np.sqrt(2*np.log(12/alpha)/len(x))
        means = x.mean(axis=0)
        result.update(radius=float(radius), means=means.tolist(), lower=(means-radius).tolist(), upper=(means+radius).tolist())
    return result


def analyze(outcomes, alpha: float = .05, task_ids: list | None = None) -> dict:
    y = np.asarray(outcomes)
    d = diagnostics(y)
    if task_ids is not None and (len(task_ids) != len(y) or len(set(task_ids)) != len(task_ids)):
        raise ValueError("Provide exactly one distinct identifier per independent task")
    values = np.concatenate([d["fixed_arm_uplift"][:, 1:], d["products"][:, 1:]], axis=1)
    rectangle = simultaneous_rectangle(values, alpha)
    result = {"status": "no_data", "analysis_role": "Secondary exploratory population-moment partial identification; primary endpoints unchanged",
              "n_tasks": len(y), "task_ids": task_ids, "raw_Y": y.tolist(), "arms": ARMS,
              "arm_denominators": {a: {"tasks": len(y), "round_observations": 2*len(y)} for a in ARMS},
              "raw_mu": None, "raw_S": None, "descriptive": {}, "descriptive_H_interval": None, "rectangle": rectangle,
              "confidence": {"status": "no_data", "H_interval": None},
              "assumptions": "Independent task blocks; conditionally independent fresh rounds with their own A0 and stable conditional arm means; upstream complete-case eligibility may limit the target population",
              "interpretation": "S is an uncentered squared-effect moment, not variance or evidence of positive effects alone; intervals are per stratum; no empirical benefit claim follows from this calculation"}
    if not len(y):
        return result
    mu, second = np.asarray(rectangle["means"][:3]), np.asarray(rectangle["means"][3:])
    descriptive = {arm: population_bounds(m, s) for arm, m, s in zip(ARMS[1:], mu, second)}
    limits = np.array([rectangle["lower"], rectangle["upper"]]).T
    result.update(status="computed", raw_mu=mu.tolist(), raw_S=second.tolist(), descriptive=descriptive,
                  confidence=propagate_rectangle(limits[:3], limits[3:]))
    if all(r["feasible"] for r in descriptive.values()):
        result["descriptive_H_interval"] = propagate_rectangle(np.repeat(mu[:, None], 2, axis=1), np.repeat(second[:, None], 2, axis=1))["H_interval"]
    result["confidence"].update(coverage_at_least=1-alpha, qualification="Simultaneous population coverage under the stated assumptions; an empty feasible region is flagged, not repaired")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--alpha", type=float, default=.05)
    args = parser.parse_args()
    if args.rows.resolve() == args.out.resolve():
        raise ValueError("Output must not overwrite the input rows")
    content = args.rows.read_bytes()
    rows = json.loads(content)
    groups = {}
    keys = ["model", "env", "split", "config_sha256", "scaffold"]
    for row in rows:
        identity = tuple(row.get(key) for key in keys)
        groups.setdefault(identity, []).append(row)
    strata = []
    for identity, subset in groups.items():
        result = analyze(np.array([r["Y"] for r in subset]), args.alpha, [r["task_id"] for r in subset])
        result["provenance"] = dict(zip(keys, identity))
        result["provenance"]["model_revisions"] = sorted({r.get("model_revision") or "unavailable" for r in subset})
        strata.append(result)
    output = {"status": "computed" if rows else "no_data", "input_path": str(args.rows), "input_sha256": hashlib.sha256(content).hexdigest(),
              "headroom_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "alpha_per_stratum": args.alpha, "strata": strata}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2, sort_keys=True, allow_nan=False)+"\n")


if __name__ == "__main__":
    main()
