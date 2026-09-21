import argparse
import hashlib
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import sklearn


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "extension"), str(ROOT / "controller"), str(ROOT)]
import cross_draw


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    if args.model.exists() or args.manifest.exists():
        raise FileExistsError("Cross-draw model freeze already exists")
    rows = [row for path in args.input for row in json.loads(path.read_text())]
    bundle, receipt = cross_draw.fit(rows)
    args.model.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, args.model)
    receipt.update({
        "bundle_sha256": file_hash(args.model),
        "input_hashes": {str(path.resolve().relative_to(ROOT)): file_hash(path) for path in args.input},
        "source_hashes": {
            "controller/cross_draw.py": file_hash(ROOT / "controller/cross_draw.py"),
            "controller/direct_advantage.py": file_hash(ROOT / "controller/direct_advantage.py"),
            "controller/fit_cross_draw.py": file_hash(Path(__file__)),
            "extension/policies.py": file_hash(ROOT / "extension/policies.py"),
        },
        "versions": {
            "python": sys.version,
            "numpy": np.__version__,
            "sklearn": sklearn.__version__,
            "joblib": joblib.__version__,
        },
    })
    args.manifest.write_text(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
