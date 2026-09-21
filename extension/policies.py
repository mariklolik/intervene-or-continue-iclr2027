import argparse
import hashlib
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GroupKFold

from arms import BASE_ARMS

SEED = 260908
LEAVES = [10, 5]
MARGINS = [0.10, 0.05, 0.0]
RISK_THRESHOLDS = [1.0, 0.75, 0.5, 0.25]
LEARNERS = ["FAILURE_RISK", "RF_LCB", "PAIRWISE", "SINGLE_0", "SINGLE_1", "REPEATED", "REPEATED_HALF"]
METHODS = ["CONTINUE", "BEST_FIXED", "RATE_RANDOM"] + LEARNERS
GRID = {"trees": 200, "min_samples_leaf": LEAVES, "margins": MARGINS, "risk_thresholds": RISK_THRESHOLDS, "folds": 4, "n_jobs": 1}
sys.dont_write_bytecode = True


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def numeric(rows: list[dict]) -> list[dict]:
    values = [r["prefix_only"]["features"] for r in rows]
    if any(not isinstance(v, (int, float)) or not np.isfinite(v) for row in values for v in row.values()):
        raise ValueError("Prefix features must be finite numeric observations")
    return values


class Features:
    def __init__(self, text: bool):
        self.text = text
        self.numeric = DictVectorizer(sparse=False)
        self.tfidf = None
        self.svd = None

    def fit(self, rows: list[dict]):
        self.numeric.fit(numeric(rows))
        if self.text:
            vectorizer = TfidfVectorizer(max_features=512, sublinear_tf=True, min_df=1)
            texts = [r["prefix_only"].get("recent_text", "") for r in rows]
            if any(vectorizer.build_analyzer()(t) for t in texts):
                matrix = vectorizer.fit_transform(texts)
                self.tfidf = vectorizer
                components = min(16, matrix.shape[0] - 1, matrix.shape[1] - 1)
                if components > 0 and np.any(np.ptp(matrix.toarray(), axis=0) > 0):
                    self.svd = TruncatedSVD(n_components=components, random_state=SEED).fit(matrix)
        return self

    def transform(self, rows: list[dict]) -> np.ndarray:
        matrix = self.numeric.transform(numeric(rows))
        if self.tfidf is not None:
            text = self.tfidf.transform([r["prefix_only"].get("recent_text", "") for r in rows])
            matrix = np.column_stack([matrix, self.svd.transform(text) if self.svd else text.toarray()])
        return matrix if matrix.shape[1] else np.zeros((len(rows), 1))

    def schema(self) -> dict:
        return {"numeric_features": self.numeric.get_feature_names_out().tolist(),
                "text_vocabulary": sorted(self.tfidf.vocabulary_) if self.tfidf else [],
                "svd_components": self.svd.n_components if self.svd else 0}


def task_folds(rows: list[dict]) -> list:
    groups = [r["task_id"] for r in rows]
    return list(GroupKFold(n_splits=4, shuffle=True, random_state=SEED).split(rows, groups=groups))


def target_rounds(method: str) -> list[int]:
    return [int(method[-1])] if method.startswith("SINGLE_") else [0, 1]


def estimator(method: str, x: np.ndarray, y: np.ndarray, leaf: int) -> dict:
    labels = y[:, target_rounds(method)].mean(axis=1)
    kwargs = dict(n_estimators=200, min_samples_leaf=leaf, random_state=SEED, n_jobs=1)
    if method == "PAIRWISE":
        forest = RandomForestClassifier(**kwargs).fit(x, np.sign(labels[:, 1:] - labels[:, :1]).astype(int))
    else:
        forest = RandomForestRegressor(**kwargs).fit(x, 1 - labels[:, 0] if method == "FAILURE_RISK" else labels)
    return {"forest": forest, "mean": labels.mean(axis=0), "arm": int(np.argmax(labels.mean(axis=0)[1:])) + 1}


def values(model: dict, x: np.ndarray, method: str) -> tuple[np.ndarray, np.ndarray]:
    forest = model["forest"]
    if method == "PAIRWISE":
        preferences = [p @ classes for p, classes in zip(forest.predict_proba(x), forest.classes_)]
        return np.column_stack([np.zeros(len(x)), *preferences]), np.zeros((len(x), 4))
    predicted = forest.predict(x)
    if method == "FAILURE_RISK":
        mean = np.tile(model["mean"], (len(x), 1))
        mean[:, 0] = 1 - predicted
        return mean, np.zeros_like(mean)
    std = np.std([tree.predict(x) for tree in forest.estimators_], axis=0) if method == "RF_LCB" else np.zeros_like(predicted)
    return predicted, std


def decisions(mean: np.ndarray, std: np.ndarray, method: str, threshold: float, arm) -> np.ndarray:
    if method == "FAILURE_RISK":
        chosen = np.where(1 - mean[:, 0] > threshold + 1e-12, arm, 0)
    else:
        scores = mean - std
        scores[:, 0] = mean[:, 0] + std[:, 0] + threshold
        maximum = scores.max(axis=1, keepdims=True)
        chosen = (scores >= maximum - 1e-12).argmax(axis=1)
    return np.eye(mean.shape[1])[chosen]


def fit_learner(rows: list[dict], method: str, text: bool) -> tuple[dict, dict]:
    if method == "REPEATED_HALF":
        tasks = sorted({r["task_id"] for r in rows}, key=lambda t: (digest([SEED, t]), t))
        selected = set(tasks[:len(tasks) // 2])
        rows = [r for r in rows if r["task_id"] in selected]
    y = np.asarray([r["Y"] for r in rows], dtype=float)
    target = y[:, target_rounds(method)].mean(axis=1)
    folds, prepared, candidates = [], [], []
    for train, valid in task_folds(rows):
        transform = Features(text).fit([rows[i] for i in train])
        prepared.append((train, valid, transform.transform([rows[i] for i in train]), transform.transform([rows[i] for i in valid])))
        folds.append({"train_task_ids": sorted({rows[i]["task_id"] for i in train}),
                      "validation_task_ids": sorted({rows[i]["task_id"] for i in valid}), **transform.schema()})
    thresholds = RISK_THRESHOLDS if method == "FAILURE_RISK" else MARGINS
    for leaf in LEAVES:
        width = y.shape[2]
        means, stds, arms = np.zeros((len(rows), width)), np.zeros((len(rows), width)), np.zeros(len(rows), dtype=int)
        for train, valid, train_x, valid_x in prepared:
            model = estimator(method, train_x, y[train], leaf)
            means[valid], stds[valid] = values(model, valid_x, method)
            arms[valid] = model["arm"]
        for threshold in thresholds:
            p = decisions(means, stds, method, threshold, arms)
            candidates.append({"leaf": leaf, "threshold": threshold, "oof_utility": float((p * target).sum(axis=1).mean()),
                               "oof_firing_rate": float(1 - p[:, 0].mean())})
    best = max(candidates, key=lambda c: (round(c["oof_utility"], 12), -c["oof_firing_rate"], c["leaf"], c["threshold"]))
    transform = Features(text).fit(rows)
    model = estimator(method, transform.transform(rows), y, best["leaf"])
    fitted = {**model, "transform": transform, "threshold": best["threshold"]}
    record = {"task_ids": sorted({r["task_id"] for r in rows}), "records": [{"task_id": r["task_id"], "sha256": digest(r)} for r in rows],
              "target_rounds": target_rounds(method), "target_sha256": digest(target.tolist()),
              "round_task_count": len({r["task_id"] for r in rows}) * len(target_rounds(method)),
              "feature_schema": transform.schema(), "folds": folds, "candidates": candidates, **best}
    return fitted, record


def fit(rows: list[dict], text: bool = False) -> tuple[dict, dict]:
    dev = [r for r in rows if r["split"] == "dev"]
    dev.sort(key=lambda r: (r["model"], r["env"], r["task_id"], digest(r["prefix_only"])))
    if not dev:
        raise ValueError("No development rows")
    task_keys = [(r["model"], r["env"], r["task_id"]) for r in dev]
    if len(set(task_keys)) != len(task_keys):
        raise ValueError("One complete checkpoint row per task and stratum is required")
    y = np.asarray([r["Y"] for r in dev], dtype=float)
    if y.ndim != 3 or y.shape[0] != len(dev) or y.shape[1] < 2 or y.shape[2] < 2 or not np.isin(y, [0, 1]).all():
        raise ValueError("Development outcomes must be binary [N,rounds,arms]")
    code_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    bundle = {"strata": {}, "training_tasks": sorted({(r["env"], r["task_id"]) for r in dev}),
              "training_data_sha256": digest(dev), "policy_source_sha256": code_hash}
    manifest = {"seed": SEED, "grid": GRID, "text": text, "dev_records": len(dev), "dev_data_sha256": digest(dev),
                "source_sha256": code_hash, "versions": {"python": sys.version, "sklearn": sklearn.__version__, "numpy": np.__version__, "joblib": joblib.__version__}, "strata": []}
    for actor, env in sorted({(r["model"], r["env"]) for r in dev}):
        subset = [r for r in dev if (r["model"], r["env"]) == (actor, env)]
        if len({r["task_id"] for r in subset}) < 8:
            raise ValueError("Each stratum needs eight development tasks for four-fold half-data tuning")
        mean = np.asarray([r["Y"] for r in subset]).mean(axis=(0, 1))
        stratum, receipt = {"mean": mean, "fixed": int(np.argmax(mean)), "models": {}}, {"model": actor, "env": env, "policies": {}}
        for method in LEARNERS:
            stratum["models"][method], receipt["policies"][method] = fit_learner(subset, method, text)
        stratum["random_rate"] = receipt["policies"]["REPEATED"]["oof_firing_rate"]
        receipt.update(dev_task_ids=sorted({r["task_id"] for r in subset}), dev_sha256=digest(subset),
                       mean_utility=mean.tolist(), best_fixed_arm=stratum["fixed"], random_rate=stratum["random_rate"])
        bundle["strata"][(actor, env)] = stratum
        manifest["strata"].append(receipt)
    return bundle, manifest


def predict(bundle: dict, rows: list[dict]) -> dict:
    safe = [{k: row[k] for k in ["model", "env", "split", "task_id", "prefix_only"]} for row in rows]
    training = set(map(tuple, bundle["training_tasks"]))
    for row in safe:
        if row["split"] not in {"test", "holdout"} or (row["env"], row["task_id"]) in training:
            raise ValueError("Prediction requires untouched holdout tasks")
        if (row["model"], row["env"]) not in bundle["strata"]:
            raise ValueError("Actor/environment stratum was not trained")
    width = max((len(stratum["mean"]) for stratum in bundle["strata"].values()), default=len(BASE_ARMS))
    output = {m: {"probabilities": np.zeros((len(rows), width)), "predicted_values": np.zeros((len(rows), width)),
                  "value_kind": "pairwise_preference" if m == "PAIRWISE" else "estimated_success"} for m in METHODS}
    for key in sorted({(r["model"], r["env"]) for r in safe}):
        indices = [i for i, r in enumerate(safe) if (r["model"], r["env"]) == key]
        subset, stratum = [safe[i] for i in indices], bundle["strata"][key]
        mean = np.tile(stratum["mean"], (len(subset), 1))
        fixed = {"CONTINUE": np.eye(width)[0], "BEST_FIXED": np.eye(width)[stratum["fixed"]],
                 "RATE_RANDOM": np.array([1 - stratum["random_rate"]] + [stratum["random_rate"] / (width - 1)] * (width - 1))}
        for method in METHODS:
            if method in fixed:
                p, v = np.tile(fixed[method], (len(subset), 1)), mean
            else:
                model = stratum["models"][method]
                v, std = values(model, model["transform"].transform(subset), method)
                p = decisions(v, std, method, model["threshold"], model["arm"])
            output[method]["probabilities"][indices] = p
            output[method]["predicted_values"][indices] = v
    for policy in output.values():
        policy["firing_rate"] = float(1 - policy["probabilities"][:, 0].mean()) if rows else None
        for field in ["probabilities", "predicted_values"]:
            policy[field] = policy[field].tolist()
    return {"rows": [{k: r[k] for k in ["model", "env", "split", "task_id"]} for r in safe], "policies": output,
            "training_data_sha256": bundle["training_data_sha256"], "policy_source_sha256": bundle["policy_source_sha256"]}


def save(bundle: dict, manifest: dict, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=False)
    path = directory / "policies.joblib"
    joblib.dump(bundle, path)
    receipt = {**manifest, "bundle_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    (directory / "manifest.json").write_text(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ["fit", "predict"]:
        command = commands.add_parser(name)
        command.add_argument("--rows", nargs="+", type=Path, required=True)
        command.add_argument("--out", type=Path, required=True)
        if name == "fit":
            command.add_argument("--text", action="store_true")
        else:
            command.add_argument("--model", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    rows = [row for path in args.rows for row in json.loads(path.read_text())]
    if args.command == "fit":
        bundle, manifest = fit(rows, args.text)
        manifest["input_paths"] = list(map(str, args.rows))
        save(bundle, manifest, args.out)
    else:
        path = args.model / "policies.joblib"
        manifest = json.loads((args.model / "manifest.json").read_text())
        if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["bundle_sha256"]:
            raise ValueError("Frozen bundle hash mismatch")
        result = predict(joblib.load(path), rows)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("x") as stream:
            stream.write(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")


if __name__ == "__main__":
    from policies import main as entrypoint

    entrypoint()
