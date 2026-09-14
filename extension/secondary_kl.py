import argparse
import hashlib
import json
import math
import re
from collections import Counter
from decimal import Decimal, localcontext
from numbers import Real
from pathlib import Path

import numpy as np

from analysis import diagnostics
from evaluate_policies import STRATA

TASK_FIELDS = ("task_id", "split", "config_sha256", "state", "checkpoint_eligible")
STATES = {"complete": True, "terminal_before_checkpoint": False, "not_started": None, "baseline_incomplete_or_invalid": None, "incomplete_or_invalid": True}


def kl_limits(mean: float, n: int, delta: float) -> list:
    if type(n) is not int or n < 1 or not isinstance(mean, Real) or not isinstance(delta, Real) or not math.isfinite(mean) or not 0 <= mean <= 1 or not math.isfinite(delta) or not 0 < delta < 1:
        raise ValueError("Require a finite unit mean, positive integer task count, and tail error in (0,1)")
    with localcontext() as context:
        context.prec = 80
        x, one = Decimal.from_float(float(mean)), Decimal(1)
        threshold = -Decimal.from_float(float(delta)).ln() / Decimal(n)
        if threshold < Decimal("1e-50"):
            return [0., 1.]
        margin = Decimal("1e-60") * max(one, threshold)
        if x == 0:
            high = max(float(one - (-threshold - margin).exp()), -math.expm1(math.log(delta) / n))
            return [0., min(1., math.nextafter(high, math.inf))]
        if x == 1:
            low = min(float((-threshold - margin).exp()), math.exp(math.log(delta) / n))
            return [max(0., math.nextafter(low, -math.inf)), 1.]
        limits = []
        for lower in (True, False):
            left, right = (Decimal(0), x) if lower else (x, one)
            for _ in range(160):
                mid = (left + right) / 2
                value = x * (x / mid).ln() + (one - x) * ((one - x) / (one - mid)).ln()
                if abs(value - threshold) <= margin:
                    break
                if (value > threshold) == lower:
                    left = mid
                else:
                    right = mid
            limits.append(max(0., math.nextafter(float(left), -math.inf)) if lower else min(1., math.nextafter(float(right), math.inf)))
        return limits


def envelope(values, opportunities, alpha: float = .05) -> dict:
    v, o = np.asarray(values, float), np.asarray(opportunities, float)
    if v.ndim != 1 or o.shape != v.shape or not np.isfinite(v).all() or not np.isfinite(o).all() or np.any((v < -1) | (v > 1)) or np.any((o < 0) | (o > 1)) or not isinstance(alpha, Real) or not math.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("Require paired finite task statistics V in [-1,1], O in [0,1] and alpha in (0,1)")
    n = len(v)
    result = {"status": "no_data", "n_tasks": n, "alpha": alpha, "tail_delta": alpha / 2, "coverage_at_least": None,
              "raw_V": v.tolist(), "raw_O": o.tolist(), "V_mean": None, "O_mean": None, "H_interval": None,
              "method": "Fixed-task-horizon Bernoulli-KL Chernoff envelope; two one-sided tails; independent possibly unequal task means"}
    if not n:
        return result
    vm, om = math.fsum(v) / n, math.fsum(o) / n
    z = max(0., math.nextafter((vm + 1) / 2, -math.inf))
    upper_mean = min(1., math.nextafter(om, math.inf)) if om else 0.
    low = max(0., math.nextafter(2 * kl_limits(z, n, alpha / 2)[0] - 1, -math.inf))
    high = kl_limits(upper_mean, n, alpha / 2)[1]
    result.update(status="computed" if low <= high else "empty_interval", coverage_at_least=1 - alpha,
                  V_mean=vm, O_mean=om, H_interval=[low, high] if low <= high else None)
    return result


def stratum_map(document: dict) -> dict:
    rows = document["strata"]
    result = {(row["model"], row["env"]): row for row in rows}
    if len(result) != len(rows) or set(result) != set(STRATA):
        raise ValueError("Require exactly four distinct frozen strata")
    return result


def task_map(tasks: list) -> dict:
    result = {task["task_id"]: tuple(task[field] for field in TASK_FIELDS) for task in tasks}
    if len(result) != len(tasks) or any(not isinstance(key, str) or not key for key in result):
        raise ValueError("Duplicate or malformed planned task identities")
    return result


def valid_hash(value) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def validate_documents(evaluation: dict, original: dict, receipt: dict, evaluation_sha256: str) -> tuple:
    try:
        if receipt["schema_version"] != 1 or receipt["status"] != "PASS" or not valid_hash(receipt["source_sha256"]) or original["evaluation_sha256"] != evaluation_sha256 or not valid_hash(evaluation_sha256):
            raise ValueError("Require exact evaluation linkage and a schema-1 PASS frame receipt")
        inputs = {row["path"]: row["sha256"] for row in evaluation["inputs"]}
        audits = receipt["audits"]
        audit_configs = {(row["model"], row["config_sha256"]) for row in audits}
        if len(inputs) != len(evaluation["inputs"]) or len(audits) != 4 or len(audit_configs) != 4 or len({row["path"] for row in audits}) != 4:
            raise ValueError("Require all four distinct shard audit inputs")
        for row in audits:
            if not valid_hash(row["sha256"]) or not valid_hash(row["config_sha256"]) or inputs.get(row["path"]) != row["sha256"]:
                raise ValueError("Frame audit hash is absent or differs from evaluation input")
        sources, results, frames = map(stratum_map, (evaluation, original, receipt))
        expected = {}
        for task in receipt["expected_frame"]:
            key = (task["model"], task["env"], task["task_id"])
            if key in expected or not valid_hash(task["master_config_sha256"]) or not valid_hash(task["task_sha256"]):
                raise ValueError("Duplicate or unbound immutable task identity")
            expected[key] = (task["split"], task["config_sha256"])
        observed = {}
        for key in STRATA:
            source, result, planned = sources[key], results[key], frames[key]
            n, ids, frame = source["n_tasks"], source["task_ids"], source["frame"]
            if type(n) is not int or n < 0 or len(ids) != n or len(set(ids)) != n:
                raise ValueError("Invalid complete-task count or duplicate identities")
            y = np.asarray(source["raw_Y"], float) if n else np.empty((0, 2, 4))
            if not n and source["raw_Y"] != []:
                raise ValueError("Zero task count has nonempty outcomes")
            diagnostics(y)
            if len(y) != n or planned["n_tasks"] != len(frame["tasks"]) or task_map(planned["tasks"]) != task_map(frame["tasks"]):
                raise ValueError("Planned receipt and evaluation task frames differ")
            for task in frame["tasks"]:
                state, eligible = task["state"], task["checkpoint_eligible"]
                if (task["model"], task["env"]) != key or task["split"] not in {"test", "holdout"} or state not in STATES or eligible is not STATES[state] or (key[0], task["config_sha256"]) not in audit_configs:
                    raise ValueError("Invalid task state, eligibility, actor or configuration")
                observed[(*key, task["task_id"])] = (task["split"], task["config_sha256"])
            complete = [task for task in frame["tasks"] if task["state"] == "complete"]
            if set(ids) != {task["task_id"] for task in complete} or source["test_config_sha256"] != sorted({task["config_sha256"] for task in complete}):
                raise ValueError("Accepted task/config identities differ from the frame")
            for field, value in (("confirmed_eligible", True), ("early_terminal", False), ("unknown_eligibility", None)):
                if type(frame[field]) is not int or frame[field] != sum(task["checkpoint_eligible"] is value for task in frame["tasks"]):
                    raise ValueError("Frame denominator mismatch")
            if result["frame"] != frame or result["test_config_sha256"] != source["test_config_sha256"]:
                raise ValueError("Original analysis frame/config mismatch")
            for stored in (result["diagnostics"], result["headroom"]):
                if stored["n_tasks"] != n or stored["task_ids"] != ids or stored["raw_Y"] != source["raw_Y"]:
                    raise ValueError("Original analysis outcomes/task identities mismatch")
            combined = result["headroom"]["combined"]
            if combined["moment_alpha"] != .025 or combined["observable_alpha"] != .025 or combined["coverage_at_least"] != .95:
                raise ValueError("Original combined confidence allocation differs")
            if n:
                moment = combined["moment_confidence"]
                if combined["moment_rectangle"]["alpha"] != .025 or moment["coverage_at_least"] != .975:
                    raise ValueError("Require the original alpha-.025 moment component")
                if moment["H_interval"] is not None:
                    limits = np.asarray(moment["H_interval"], float)
                    if limits.shape != (2,) or not np.isfinite(limits).all() or not 0 <= limits[0] <= limits[1] <= 1:
                        raise ValueError("Malformed stored moment interval")
        for audit in audits:
            tasks = [task for key, row in sources.items() if key[0] == audit["model"] for task in row["frame"]["tasks"] if task["config_sha256"] == audit["config_sha256"]]
            if type(audit["planned_tasks"]) is not int or type(audit["accepted_tasks"]) is not int or audit["planned_tasks"] != len(tasks) or audit["accepted_tasks"] != sum(task["state"] == "complete" for task in tasks) or audit["state_counts"] != dict(Counter(task["state"] for task in tasks)):
                raise ValueError("Frame receipt audit denominators disagree with its task states")
        if observed != expected:
            raise ValueError("Immutable planned frame is not exactly represented")
        return sources, results
    except (KeyError, TypeError, AttributeError, IndexError) as error:
        raise ValueError("Unsupported evaluation, original analysis or frame receipt schema") from error


def analyze_documents(evaluation: dict, original: dict, receipt: dict, evaluation_sha256: str) -> dict:
    sources, originals = validate_documents(evaluation, original, receipt, evaluation_sha256)
    strata = []
    for key in STRATA:
        source, old = sources[key], originals[key]["headroom"]
        y = np.asarray(source["raw_Y"], float).reshape(-1, 2, 4)
        summary = diagnostics(y)
        v, o = summary["cross_selected_uplift"], summary["same_round_opportunity"]
        complete, stricter = envelope(v, o), envelope(v, o, .025)
        combined = {"status": "no_data", "H_interval": None, "coverage_at_least": None, "moment_alpha": .025, "observable_alpha": .025,
                    "method": "Intersection of stored alpha-.025 moment confidence and alpha-.025 KL direct envelope; total alpha .05"}
        if len(y):
            moment = old["combined"]["moment_confidence"]["H_interval"]
            combined.update(status="incompatible_component", coverage_at_least=.95)
            if moment is not None and stricter["H_interval"] is not None:
                low, high = max(moment[0], stricter["H_interval"][0]), min(moment[1], stricter["H_interval"][1])
                combined.update(status="computed" if low <= high else "empty_intersection", H_interval=[low, high] if low <= high else None)
        observed = dict(zip(source["task_ids"], zip(v, o)))
        pairs = [observed.get(task["task_id"], (0., 0.) if task["state"] == "terminal_before_checkpoint" else (-1., 1.)) for task in source["frame"]["tasks"]]
        full = envelope([pair[0] for pair in pairs], [pair[1] for pair in pairs])
        full.update(task_ids=[task["task_id"] for task in source["frame"]["tasks"]], target="Entire immutable planned frame; early terminal has zero headroom by the no-intervention extension; unknown/incomplete filled O=1,V=-1")
        strata.append({"model": key[0], "env": key[1], "complete_case": complete, "complete_case_stricter": stricter, "combined": combined,
                       "full_planned_frame": full, "original_headroom": {name: old[name] for name in ("observable", "confidence", "combined")},
                       "complete_task_ids": source["task_ids"], "frame": source["frame"], "test_config_sha256": source["test_config_sha256"]})
    return {"schema_version": 1, "analysis_role": "Post-collection-start, pre-analysis secondary KL amendment; primary endpoints and policy gates unchanged",
            "strata": strata, "cross_stratum_joint_coverage": False,
            "confidence_families": "Complete-case standalone, combined moment/KL, and full-frame standalone each have separate per-stratum 95% guarantees, not a joint guarantee",
            "assumptions": "Independent potential task blocks and conditionally independent whole rounds with stable arm means; complete-case sampling separately requires independence despite admission; full-frame pessimistic filling permits informative missingness under the dominating-data proof",
            "numerics": "80-digit Decimal KL comparisons with 1e-60 relative floor margin, conservative 160-step brackets and outward float rounding; tiny thresholds return [0,1]. Not a formal machine-arithmetic proof."}


def main() -> None:
    parser = argparse.ArgumentParser(description="Separate secondary KL headroom analysis of validated frozen outputs")
    for name in ("evaluation", "analysis", "frame-validation", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    paths = {"evaluation": args.evaluation, "analysis": args.analysis, "frame_validation": args.frame_validation}
    content = {name: path.read_bytes() for name, path in paths.items()}
    hashes = {name + "_sha256": hashlib.sha256(value).hexdigest() for name, value in content.items()}
    documents = {name: json.loads(value) for name, value in content.items()}
    for document in documents.values():
        json.dumps(document, allow_nan=False)
    result = analyze_documents(documents["evaluation"], documents["analysis"], documents["frame_validation"], hashes["evaluation_sha256"])
    result["inputs"] = {**hashes, "paths": {name: str(path.resolve()) for name, path in paths.items()}}
    result["source_sha256"] = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in [Path(__file__), Path(diagnostics.__code__.co_filename)]}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as stream:
        stream.write(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
