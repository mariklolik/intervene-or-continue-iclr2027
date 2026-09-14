import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from analysis import summarize_blocks
from joint_headroom import analyze


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    content = args.evaluation.read_bytes()
    evaluation = json.loads(content)
    summaries = []
    for stratum in evaluation["strata"]:
        identifiers = stratum["task_ids"]
        outcomes = np.asarray(stratum["raw_Y"], dtype=float).reshape(-1, 2, 4)
        if len(identifiers) != len(outcomes) or len(identifiers) != stratum["n_tasks"] or len(set(identifiers)) != len(identifiers):
            raise ValueError("Pooled task identity or count mismatch")
        rows = [{"task_id": task_id, "Y": y.tolist()} for task_id, y in zip(identifiers, outcomes)]
        summaries.append({"model": stratum["model"], "env": stratum["env"], "test_config_sha256": stratum["test_config_sha256"], "frame": stratum["frame"], "diagnostics": summarize_blocks(rows, seed=260908), "headroom": analyze(outcomes, alpha=.05, task_ids=identifiers)})
    sources = [Path(__file__), Path(summarize_blocks.__code__.co_filename), Path(analyze.__code__.co_filename)]
    result = {"evaluation_path": str(args.evaluation.resolve()), "evaluation_sha256": hashlib.sha256(content).hexdigest(), "upstream_inputs": evaluation["inputs"], "source_hashes": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}, "strata": summaries, "scientific_gate": "Requires independent evidence-strength review; no automatic efficacy or SOTA claim", "secondary_scope": "Headroom bounds are conditional on qualified complete blocks and the stated stochastic assumptions; missingness is retained in each frame."}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as stream:
        stream.write(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
