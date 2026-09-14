import argparse
import hashlib
import json
import re
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DIRECTORIES = ["controller", "extension", "src", "configs", "models", "protocol", "tests", "raw", "runtime"]
FILES = ["README.md", "SUBMISSION_CHECKLIST.md", "pyproject.toml", "uv.lock", "gpu_budget.jsonl"]
TEXT_SUFFIXES = {".bib", ".bst", ".json", ".jsonl", ".lock", ".md", ".py", ".sty", ".tex", ".txt", ".toml"}
BANNED = ["mekashirskiy", "mariklolik", "AlekseiSDev", "avi-gn-fsk", "/Users/", "/home/"]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def copy_inputs(target: Path) -> None:
    for name in DIRECTORIES:
        source = ROOT / name
        if source.exists():
            shutil.copytree(source, target / name, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copytree(ROOT / "artifacts" / "historical", target / "artifacts" / "historical")
    shutil.copytree(ROOT / "artifacts" / "literature", target / "artifacts" / "literature")
    shutil.copytree(ROOT / "artifacts" / "prediction-freeze", target / "artifacts" / "prediction-freeze")
    shutil.copytree(ROOT / "paper", target / "paper", ignore=shutil.ignore_patterns("build", "__pycache__", "*.aux", "*.bbl", "*.blg", "*.log", "*.out"))
    for name in FILES:
        source = ROOT / name
        if source.exists():
            shutil.copy2(source, target / name)


def anonymous_text(value: str) -> str:
    value = value.replace(str(ROOT), "$PROJECT_ROOT")
    value = value.replace("/home/mekashirskiy/intervene-or-continue-iclr2027", "$PROJECT_ROOT")
    value = value.replace("/home/mekashirskiy/intervene-sota-20260908", "$PRIOR_ARTIFACT_ROOT")
    value = value.replace("/home/mekashirskiy/intervene-next-20260909", "$PRIOR_ARTIFACT_ROOT")
    value = re.sub(r"/Users/[^\s\"']+", "$LOCAL_ARTIFACT", value)
    value = re.sub(r"/home/[^\s\"']+", "$REMOTE_ARTIFACT", value)
    value = value.replace("avi-gn-fsk42", "GPU_HOST")
    value = value.replace("mekashirskiy", "anonymous")
    value = value.replace("mariklolik", "anonymous")
    value = value.replace("AlekseiSDev", "anonymous")
    return value


def sanitize_text_files(target: Path) -> None:
    for path in sorted(target.rglob("*")):
        if path.is_file() and path.suffix.lower() in TEXT_SUFFIXES:
            value = path.read_text(errors="strict")
            sanitized = anonymous_text(value)
            if sanitized != value:
                path.write_text(sanitized)


def rewrite_config_lineage(target: Path) -> None:
    changes = {}
    for config_path in sorted((target / "configs" / "independent-panel").glob("panel-shard*.json")):
        original = ROOT / config_path.relative_to(target)
        old_hash = digest(original)
        new_hash = digest(config_path)
        changes[old_hash] = new_hash
        raw_dir = target / "raw" / config_path.stem
        for record_path in raw_dir.rglob("*.json"):
            record = json.loads(record_path.read_text())
            if record.get("config_sha256") == old_hash:
                record["config_sha256"] = new_hash
                record_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    prediction_path = target / "artifacts" / "prediction-freeze" / "predictions.json"
    predictions = json.loads(prediction_path.read_text())
    for row in predictions["rows"]:
        row["config_sha256"] = changes.get(row["config_sha256"], row["config_sha256"])
    prediction_path.write_text(json.dumps(predictions, indent=2, sort_keys=True) + "\n")
    freeze_path = target / "artifacts" / "prediction-freeze" / "prediction-freeze.json"
    freeze = json.loads(freeze_path.read_text())
    freeze["predictions_sha256"] = digest(prediction_path)
    for row in freeze["inputs"]:
        value = Path(row["path"])
        if "configs/independent-panel" in row["path"]:
            value = Path(row["path"][row["path"].index("configs/independent-panel"):])
        elif "controller/" in row["path"]:
            value = Path(row["path"][row["path"].index("controller/"):])
        elif "extension/" in row["path"]:
            value = Path(row["path"][row["path"].index("extension/"):])
        elif "models/" in row["path"]:
            value = Path(row["path"][row["path"].index("models/"):])
        row["path"] = str(value)
        candidate = target / value
        if candidate.exists():
            row["sha256"] = digest(candidate)
    freeze_path.write_text(json.dumps(freeze, indent=2, sort_keys=True) + "\n")


def regenerate(target: Path, python: Path) -> None:
    confirmation = target / "artifacts" / "confirmation"
    if confirmation.exists():
        shutil.rmtree(confirmation)
    command = [
        str(python), "controller/evaluate_confirmation.py",
        "--raw", "raw",
        "--predictions", "artifacts/prediction-freeze/predictions.json",
        "--prediction-freeze", "artifacts/prediction-freeze/prediction-freeze.json",
        "--out", "artifacts/confirmation",
    ]
    for config in sorted((target / "configs" / "independent-panel").glob("panel-shard*.json")):
        command.extend(["--config", str(config.relative_to(target))])
    subprocess.run(command, cwd=target, check=True)
    subprocess.run([str(python), "paper/build_assets.py"], cwd=target, check=True)


def write_manifest(target: Path, source_commit: str) -> None:
    paths = [
        target / "artifacts" / "confirmation" / "results.json",
        target / "artifacts" / "confirmation" / "verification.json",
        target / "artifacts" / "prediction-freeze" / "prediction-freeze.json",
        target / "paper" / "generated" / "manifest.json",
        target / "paper" / "main.pdf",
    ]
    manifest = {
        "source_commit": source_commit,
        "anonymous_derivative": True,
        "result_values_preserved": True,
        "files": [{"path": str(path.relative_to(target)), "sha256": digest(path)} for path in paths],
    }
    path = target / "artifacts" / "MANIFEST.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def verify_anonymous(target: Path) -> None:
    failures = []
    for path in sorted(target.rglob("*")):
        if not path.is_file():
            continue
        data = path.read_bytes()
        for value in BANNED:
            if value.encode() in data:
                failures.append(f"{path.relative_to(target)}:{value}")
    if failures:
        raise ValueError("Anonymous supplement contains banned strings: " + ", ".join(failures[:20]))


def write_zip(source: Path, output: Path) -> None:
    timestamp = (2026, 9, 14, 0, 0, 0)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted(source.rglob("*")):
            if not path.is_file():
                continue
            relative = Path("intervene-or-continue-supplement") / path.relative_to(source)
            info = zipfile.ZipInfo(str(relative), timestamp)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, path.read_bytes(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "dist" / "intervene-or-continue-iclr2027-supplement.zip")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    source_commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()
    with tempfile.TemporaryDirectory(prefix="ioc27-supplement-") as directory:
        target = Path(directory) / "supplement"
        target.mkdir()
        copy_inputs(target)
        sanitize_text_files(target)
        rewrite_config_lineage(target)
        regenerate(target, ROOT / ".venv" / "bin" / "python")
        shutil.copy2(ROOT / "paper" / "main.pdf", target / "paper" / "main.pdf")
        write_manifest(target, source_commit)
        verify_anonymous(target)
        write_zip(target, args.output)
    print(json.dumps({"path": str(args.output), "sha256": digest(args.output), "bytes": args.output.stat().st_size}, sort_keys=True))


if __name__ == "__main__":
    main()
