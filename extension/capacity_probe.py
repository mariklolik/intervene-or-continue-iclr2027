import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import GradientBoostingRegressor

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "extension"), str(ROOT / "controller"), str(ROOT)]
import cross_draw
from policies import Features, SEED

SEMANTIC_WIDTH = 32


def semantic_features(path):
    store = np.load(path, allow_pickle=True)
    return dict(zip(store["ids"].tolist(), store["vectors"]))


def semantic_class(vectors):
    class Semantic(Features):
        def fit(self, rows):
            super().fit(rows)
            block = np.stack([vectors[r["task_id"]] for r in rows])
            width = min(SEMANTIC_WIDTH, block.shape[0] - 1, block.shape[1] - 1)
            self.semantic = TruncatedSVD(n_components=width, random_state=SEED).fit(block)
            return self

        def transform(self, rows):
            block = np.stack([vectors[r["task_id"]] for r in rows])
            return np.column_stack([super().transform(rows), self.semantic.transform(block)])

    return Semantic


def boosted(x, target, leaf):
    models = [GradientBoostingRegressor(random_state=SEED, min_samples_leaf=leaf).fit(x, target[:, column])
              for column in range(target.shape[1])]
    return type("Ensemble", (), {"predict": staticmethod(
        lambda matrix: np.column_stack([model.predict(matrix) for model in models]))})


def honest(rows):
    return float(np.mean([cross_draw.fit_stratum([r for r in rows if (r["model"], r["env"]) == key])[1]
                          ["nested_honest_utility"]
                          for key in sorted({(r["model"], r["env"]) for r in rows})]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rows = [row for path in args.input for row in json.loads(path.read_text())]
    vectors = semantic_features(args.embeddings)
    baseline_features, baseline_forest = cross_draw.Features, cross_draw.forest
    report = {}
    for name, features, learner in [("tabular_forest", baseline_features, baseline_forest),
                                    ("tabular_boosted", baseline_features, boosted),
                                    ("semantic_forest", semantic_class(vectors), baseline_forest),
                                    ("semantic_boosted", semantic_class(vectors), boosted)]:
        cross_draw.Features, cross_draw.forest = features, learner
        report[name] = honest(rows)
        print(name, report[name], flush=True)
    cross_draw.Features, cross_draw.forest = baseline_features, baseline_forest
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
