import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "extension"), str(ROOT / "controller"), str(ROOT)]
import cross_draw
from policies import LEAVES, MARGINS, task_folds


def stratum_differences(rows, leaf, margin, agreement, candidates, fallback):
    y = np.asarray([row["Y"] for row in rows], dtype=float)
    targets, utility = cross_draw.draw_targets(y), y.mean(axis=1)
    predicted, evidence, leaders = cross_draw.fold_predictions(rows, targets, leaf, task_folds(rows))
    chosen = cross_draw.choose(predicted, margin, agreement, None if candidates == "all" else evidence,
                               None if fallback == "continue" else leaders)
    return (chosen * utility).sum(axis=1) - utility[:, 0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rows = [row for path in args.input for row in json.loads(path.read_text())]
    strata = sorted({(row["model"], row["env"]) for row in rows})
    frontier = []
    for leaf in LEAVES:
        for margin in MARGINS:
            for agreement in (True, False):
                for candidates in cross_draw.CANDIDATE_SETS:
                  for fallback in cross_draw.FALLBACKS:
                    differences = np.concatenate([stratum_differences([r for r in rows if (r["model"], r["env"]) == key],
                                                                     leaf, margin, agreement, candidates, fallback)
                                                  for key in strata])
                    error = differences.std(ddof=1) / np.sqrt(len(differences))
                    frontier.append({"leaf": leaf, "threshold": margin, "agreement": agreement,
                                     "candidates": candidates, "fallback": fallback, "improvement": float(differences.mean()),
                                     "z": float(differences.mean() / error) if error > 0 else 0.0,
                                     "discordance": float((differences != 0).mean())})
    args.out.write_text(json.dumps({"tasks": len(rows), "frontier": frontier}, indent=2, sort_keys=True) + "\n")
    for row in sorted(frontier, key=lambda r: -r["z"])[:4]:
        print(row, flush=True)


if __name__ == "__main__":
    main()
