import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np

from analysis import ARMS, diagnostics, summarize_tasks
from policies import METHODS, digest, predict

STRATA = [(m, e) for m in ["Qwen3-8B", "Qwen2.5-7B-Instruct"] for e in ["alfworld", "scienceworld"]]
COMPARISONS = [(m, "CONTINUE") for m in METHODS if m != "CONTINUE"] + [("REPEATED", m) for m in METHODS if m not in {"CONTINUE", "REPEATED"}]
FAMILY_SIZE, ALPHA, SEED = 71, .05, 260908
ROW_KEYS = ["model", "env", "split", "task_id"]


def policy_values(y, probabilities) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    diagnostics(y)
    p = np.asarray(probabilities, dtype=float).reshape(-1, 4)
    if p.shape != (len(y), 4) or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)) or not np.allclose(p.sum(axis=1), 1, atol=1e-12, rtol=0):
        raise ValueError("One valid action-probability vector per task is required")
    return (y.mean(axis=1)*p).sum(axis=1)


def holm(p_values) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    if p.ndim != 1 or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("Invalid p-values")
    order = np.argsort(p, kind="stable")
    result = np.empty(len(p))
    result[order] = np.minimum(1, np.maximum.accumulate(p[order]*np.arange(len(p), 0, -1)))
    return result


def task_distribution(values, bounds) -> dict:
    result = summarize_tasks(values, bounds, seed=SEED)
    x = np.asarray(values, dtype=float)
    result["bootstrap_means"] = x[np.random.default_rng(SEED).integers(0, len(x), (10000, len(x)))].mean(axis=1).tolist() if len(x) else []
    return result


def bounded_betting(values, alpha: float = ALPHA/FAMILY_SIZE) -> dict:
    x = np.asarray(values, dtype=float)
    if x.ndim != 1 or not np.isfinite(x).all() or np.any((x < -1) | (x > 1)) or not 0 < alpha < 1:
        raise ValueError("Independent bounded task contrasts and alpha in (0,1) required")
    if not len(x):
        return {"p_value": 1.0, "lower": None, "alpha": alpha}
    fractions = np.arange(1, 101)/100

    def log_wealth(mean: float) -> float:
        with np.errstate(divide="ignore"):
            logs = np.log1p(fractions[:, None]*(x-mean)/(1+mean)).sum(axis=1)
        return float(np.logaddexp.reduce(logs)-np.log(len(fractions)))

    wealth = log_wealth(0)
    left, right = -1.0, 1.0
    for _ in range(80):
        midpoint = (left+right)/2
        if midpoint in (left, right):
            break
        if log_wealth(midpoint) >= np.log(1/alpha):
            left = midpoint
        else:
            right = midpoint
    return {"p_value": float(max(np.finfo(float).tiny, np.exp(-max(wealth, 0)))), "lower": left, "alpha": alpha,
            "log_evalue_at_zero": wealth, "fractions": "Uniform fixed grid 0.01,0.02,...,1.00",
            "scope": "Fixed-horizon average-mean test; underlying independent task contrasts required; no optional stopping guarantee"}


def contrast(values, bounds, eligible: int) -> dict:
    x = np.asarray(values, dtype=float)
    if eligible < len(x):
        raise ValueError("Eligible frame cannot be smaller than the observed task set")
    result = task_distribution(x, bounds)
    width = bounds[1]-bounds[0]
    margin = width*np.sqrt(np.log(FAMILY_SIZE/ALPHA)/(2*len(x))) if len(x) else None
    result.update(p_bounded=float(np.exp(-2*len(x)*max(x.mean(), 0)**2/width**2)) if len(x) else 1.0,
                  simultaneous_margin=float(margin) if margin is not None else None,
                  simultaneous_lower=float(max(bounds[0], x.mean()-margin)) if len(x) else None,
                  finite_frame_bounds=[float((x.sum()+(eligible-len(x))*b)/eligible) for b in bounds] if eligible else None)
    result.update(hoeffding_simultaneous_lower=result["simultaneous_lower"], p_confirmatory=result["p_bounded"], confirmatory_method="hoeffding")
    if bounds == (-1, 1):
        betting = bounded_betting(x)
        result.update(betting=betting, p_confirmatory=betting["p_value"], simultaneous_lower=betting["lower"], confirmatory_method="fixed_fraction_mixture_betting")
    return result


def suffix_cost(rows, probabilities) -> dict:
    fields = ["input_tokens", "output_tokens", "requests", "wall_s", "unknown_usage_requests"]
    values = []
    for row, weights in zip(rows, probabilities):
        cells = {(c["round"], c["arm"]): c for c in row.get("cells", [])}
        if len(row.get("cells", [])) != 8 or set(cells) != {(r, a) for r in range(2) for a in ARMS} or any(not isinstance(c.get("local_call_events"), list) for c in cells.values()):
            values.append(None)
            continue
        total = dict.fromkeys(fields, 0.0)
        for (round_id, arm), cell in cells.items():
            weight = weights[ARMS.index(arm)]/2
            for event in cell["local_call_events"]:
                event_values = {**{f: event.get(f, 0) or 0 for f in ["input_tokens", "output_tokens", "wall_s"]}, "requests": 1, "unknown_usage_requests": int(not event.get("usage_known", False))}
                for field in fields:
                    total[field] += weight*event_values[field]
        values.append(total)
    present = [v for v in values if v is not None]
    return {"raw_task_costs": values, "observed_tasks": len(present), "missing_tasks": len(rows)-len(present), "means": {f: float(np.mean([v[f] for v in present])) if present else None for f in fields}, "scope": "Expected recorded suffix generation cost; tokens may be incomplete when usage is unknown; no cost confidence interval"}


def validate(rows, predictions, manifest, audits, dev_rows) -> tuple[dict, dict]:
    dev = sorted(dev_rows, key=lambda r: (r["model"], r["env"], r["task_id"], digest(r["prefix_only"])))
    if not dev or any(r["split"] != "dev" for r in dev) or digest(dev) != manifest["dev_data_sha256"] or predictions["training_data_sha256"] != manifest["dev_data_sha256"] or predictions["policy_source_sha256"] != manifest["source_sha256"]:
        raise ValueError("Frozen training/source provenance mismatch")
    if predictions["rows"] != [{k: r[k] for k in ROW_KEYS} for r in rows] or set(predictions["policies"]) != set(METHODS):
        raise ValueError("Prediction row alignment or method set mismatch")
    training = {(r["env"], r["task_id"]) for r in dev}
    contexts = {(r["model"], r["env"], r["scaffold"], r["model_revision"]) for r in dev}
    if any(sum(c[:2] == key for c in contexts) != 1 for key in {c[:2] for c in contexts}):
        raise ValueError("Development mixes incompatible scaffolds or actor revisions")
    frame = {}
    for source in audits:
        audit = source.get("ingestion", source)
        for task in audit["tasks"]:
            if task["split"] not in {"test", "holdout"} or task["state"] == "outside_requested_split":
                continue
            key = (audit["model"], task["env"], task["task_id"])
            if key in frame:
                raise ValueError("Duplicate task in input config shards")
            frame[key] = {**task, "model": audit["model"], "config_sha256": audit["config_sha256"]}
    seen = set()
    for row in rows:
        key = (row["model"], row["env"], row["task_id"])
        task = frame.get(key, {})
        if key in seen or key[:2] not in STRATA or row["split"] not in {"test", "holdout"} or (row["env"], row["task_id"]) in training:
            raise ValueError("Duplicate, unknown or development-overlapping test task")
        if (row["model"], row["env"], row["scaffold"], row["model_revision"]) not in contexts or not row["model_revision"]:
            raise ValueError("Held-out actor revision/scaffold differs from development")
        if task.get("state") != "complete" or task.get("checkpoint_eligible") is not True or task.get("config_sha256") != row["config_sha256"]:
            raise ValueError("Row is not admitted by its matching ingestion audit")
        seen.add(key)
    if seen != {key for key, task in frame.items() if task["state"] == "complete"}:
        raise ValueError("Every audit-admitted complete task must be supplied exactly once")
    receipts = {(r["model"], r["env"]): r for r in manifest["strata"]}
    p = {m: np.asarray(predictions["policies"][m]["probabilities"], dtype=float).reshape(-1, 4) for m in METHODS}
    y = np.asarray([r["Y"] for r in rows]) if rows else np.empty((0, 2, 4))
    for method in METHODS:
        policy_values(y, p[method])
    for i, row in enumerate(rows):
        receipt = receipts[(row["model"], row["env"])]
        rate = receipt["random_rate"]
        expected = {"CONTINUE": [1, 0, 0, 0], "BEST_FIXED": np.eye(4)[receipt["best_fixed_arm"]], "RATE_RANDOM": [1-rate, rate/3, rate/3, rate/3]}
        if any(not np.allclose(p[m][i], expected[m], atol=1e-12, rtol=0) for m in expected):
            raise ValueError("Frozen baseline probabilities were changed")
    return frame, p


def evaluate(rows, predictions, manifest, audits, dev_rows) -> dict:
    frame, probabilities = validate(rows, predictions, manifest, audits, dev_rows)
    strata, family = [], []
    for key in STRATA:
        indices = [i for i, row in enumerate(rows) if (row["model"], row["env"]) == key]
        subset = [rows[i] for i in indices]
        tasks = [t for (model, env, task_id), t in frame.items() if (model, env) == key]
        eligible = sum(t["checkpoint_eligible"] is True for t in tasks)
        y = np.asarray([r["Y"] for r in subset], dtype=float).reshape(-1, 2, 4)
        values = {m: policy_values(y, probabilities[m][indices]) for m in METHODS}
        policies = {}
        for method in METHODS:
            p = probabilities[method][indices]
            policies[method] = {"value": task_distribution(values[method], (0, 1)), "action_probabilities": p.tolist(), "firing_rate": float(1-p[:, 0].mean()) if subset else None,
                                "expected_harmful_rounds": float((p[:, None, :]*(y == 0)*(y[:, :, :1] == 1)).sum()),
                                "expected_beneficial_rounds": float((p[:, None, :]*(y == 1)*(y[:, :, :1] == 0)).sum()),
                                "continue_success_rounds": int(y[:, :, 0].sum()), "continue_failure_rounds": int((1-y[:, :, 0]).sum()),
                                "suffix_cost": suffix_cost(subset, p)}
        contrasts = {}
        for left, right in COMPARISONS:
            name = f"{left}_vs_{right}"
            entry = contrast(values[left]-values[right], (-1, 1), eligible)
            by_task = dict(zip([r["task_id"] for r in subset], values[left]-values[right]))
            conservative = np.array([by_task.get(t["task_id"], 0 if t["checkpoint_eligible"] is False else -1) for t in tasks], dtype=float)
            sensitivity = bounded_betting(conservative)
            entry["planned_frame_sensitivity"] = {**sensitivity, "n_tasks": len(tasks), "raw_task_values": conservative.tolist(), "mean": float(conservative.mean()) if len(tasks) else None,
                                                   "scope": "Distinct full-planned-frame target: incomplete/unknown=-1, early terminal=0; coordinatewise lower data permit outcome-dependent omission when full potential contrasts are independent and defined. This family's bounds are separately simultaneous, not jointly 95% with complete-case bounds."}
            entry.update(id=f"{key[0]}|{key[1]}|{name}", left=left, right=right)
            contrasts[name] = entry
            family.append(entry)
        gap = diagnostics(y)["gap"]
        diagnostic = contrast(gap, (0, 1), eligible) if key != STRATA[0] else task_distribution(gap, (0, 1))
        if key != STRATA[0]:
            diagnostic.update(id=f"{key[0]}|{key[1]}|secondary_gap")
            family.append(diagnostic)
        strata.append({"model": key[0], "env": key[1], "n_tasks": len(subset), "task_ids": [r["task_id"] for r in subset], "raw_Y": y.tolist(),
                       "test_config_sha256": sorted({r["config_sha256"] for r in subset}), "dev_config_sha256": sorted({r["config_sha256"] for r in dev_rows if (r["model"], r["env"]) == key}),
                       "frame": {"tasks": tasks, "confirmed_eligible": eligible, "unknown_eligibility": sum(t["checkpoint_eligible"] is None for t in tasks), "early_terminal": sum(t["checkpoint_eligible"] is False for t in tasks)},
                       "policies": policies, "contrasts": contrasts, "diagnostic_gap": diagnostic})
    if len(family) != FAMILY_SIZE:
        raise ValueError("Prespecified inferential family changed")
    for entry, adjusted in zip(family, holm([r["p_confirmatory"] for r in family])):
        entry.update(p_holm=float(adjusted), holm_reject=bool(adjusted <= ALPHA))
    for stratum in strata:
        comparisons = [r for r in stratum["contrasts"].values() if r["left"] == "REPEATED"]
        available = stratum["n_tasks"] > 0
        stratum["policy_gate"] = {"statistical_conditions_met": bool(available and all(r["mean"] >= .03 and r["simultaneous_lower"] > 0 and r["holm_reject"] for r in comparisons)),
                                  "minimum_gain_against_matched_comparators": min(r["mean"] for r in comparisons) if available else None,
                                  "minimum_simultaneous_lower": min(r["simultaneous_lower"] for r in comparisons) if available else None,
                                  "scope": "Complete-case frozen-policy statistics only; cost/harm and missingness require review; no automatic scientific claim"}
        stratum["policy_gate"]["full_frame_robustness_met"] = bool(available and all(r["planned_frame_sensitivity"]["lower"] > 0 for r in comparisons))
        stratum["policy_gate"]["robust_statistical_conditions_met"] = stratum["policy_gate"]["statistical_conditions_met"] and stratum["policy_gate"]["full_frame_robustness_met"]
        stratum["observed_test_winner_among_comparators"] = max((m for m in METHODS if m != "REPEATED"), key=lambda m: stratum["policies"][m]["value"]["mean"]) if available else None
    aggregate = None
    if all(s["n_tasks"] for s in strata):
        aggregate = {"weights": [.25]*4, "distinct_environment_task_ids": len({(r["env"], r["task_id"]) for r in rows}),
                     "policy_values": {m: float(np.mean([s["policies"][m]["value"]["mean"] for s in strata])) for m in METHODS},
                     "contrasts": {name: {field: float(np.mean([s["contrasts"][name][field] for s in strata])) for field in ["mean", "simultaneous_lower"]} for name in strata[0]["contrasts"]},
                     "interpretation": "Equal four-stratum descriptive means; lower bounds follow from simultaneous stratum bounds, permitting shared tasks across actors; no extra test or pooled bootstrap"}
    return {"strata": strata, "family_size": FAMILY_SIZE, "family": [{k: r[k] for k in ["id", "n_tasks", "mean", "p_bounded", "p_confirmatory", "confirmatory_method", "p_holm", "holm_reject"]} for r in family],
            "equal_stratum_aggregate": aggregate, "aggregate_requirement": "All four strata must contain accepted tasks",
            "training_data_sha256": manifest["dev_data_sha256"], "bundle_sha256": manifest["bundle_sha256"], "policy_source_sha256": manifest["source_sha256"],
            "collection_usage_by_audit": [{"model": a.get("ingestion", a)["model"], "config_sha256": a.get("ingestion", a)["config_sha256"], "usage": a.get("ingestion", a).get("usage")} for a in audits],
            "inference": "Fixed 71-member Holm family: 68 fixed-fraction mixture betting policy tests and three Hoeffding secondary-gap tests; Bonferroni policy lower limits; old Hoeffding results retained; bootstrap approximate; informative missingness and serving qualification require external review"}


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ["rows", "dev-rows", "audits"]:
        parser.add_argument(f"--{name}", nargs="+", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    paths = args.rows+args.dev_rows+args.audits+[args.model/"manifest.json", args.model/"policies.joblib"]
    manifest = json.loads((args.model/"manifest.json").read_text())
    bundle_path = args.model/"policies.joblib"
    if hashlib.sha256(bundle_path.read_bytes()).hexdigest() != manifest["bundle_sha256"]:
        raise ValueError("Frozen bundle hash mismatch")
    if hashlib.sha256(Path(predict.__code__.co_filename).read_bytes()).hexdigest() != manifest["source_sha256"]:
        raise ValueError("Prediction implementation differs from the frozen source")
    rows = [row for path in args.rows for row in json.loads(path.read_text())]
    dev = [row for path in args.dev_rows for row in json.loads(path.read_text())]
    audits = [json.loads(path.read_text()) for path in args.audits]
    predictions = predict(joblib.load(bundle_path), rows)
    result = evaluate(rows, predictions, manifest, audits, dev)
    result["inputs"] = [{"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in paths]
    result["evaluation_source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as stream:
        stream.write(json.dumps(result, indent=2, sort_keys=True, allow_nan=False)+"\n")


if __name__ == "__main__":
    main()
