import argparse
import hashlib
import json
import re
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path, PurePosixPath


ROOT = Path(__file__).resolve().parents[1]
DIRECTORIES = ["controller", "extension", "src", "configs", "models", "protocol", "tests", "raw", "runtime"]
FILES = ["README.md", "SUBMISSION_CHECKLIST.md", "pyproject.toml", "uv.lock", "gpu_budget.jsonl"]
TEXT_SUFFIXES = {".bib", ".bst", ".json", ".jsonl", ".lock", ".log", ".md", ".py", ".sty", ".tex", ".txt", ".toml"}
BANNED = ["mekashirskiy", "mariklolik", "AlekseiSDev", "avi-gn-fsk", "/Users/", "/home/"]
PANELS = {"independent-panel": "raw", "p2-event": "raw/p2-event", "p2-scheduled": "raw/p2-scheduled", "d2-event": "raw/d2-event"}
STUDIES = [("p2-event", "p2event"), ("p2-scheduled", "p2sched")]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def copy_inputs(target: Path) -> None:
    for name in DIRECTORIES:
        source = ROOT / name
        if source.exists():
            shutil.copytree(source, target / name, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copytree(ROOT / "artifacts" / "historical", target / "artifacts" / "historical")
    shutil.copytree(ROOT / "artifacts" / "literature", target / "artifacts" / "literature")
    for source in sorted((ROOT / "artifacts").iterdir()):
        if source.is_dir() and source.name not in {"historical", "literature", "confirmation"} and not (target / "artifacts" / source.name).exists():
            shutil.copytree(source, target / "artifacts" / source.name)
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
    for panel, raw_root in PANELS.items():
        directory = target / "configs" / panel
        if not directory.exists():
            continue
        for config_path in sorted(directory.glob("panel-shard*.json")):
            original = ROOT / config_path.relative_to(target)
            if not original.exists():
                continue
            changes[digest(original)] = digest(config_path)
            rewrite_records(target / raw_root / config_path.stem, changes)
    rewrite_freezes(target, changes)


def rewrite_records(raw_dir: Path, changes: dict) -> None:
    if raw_dir.exists():
        for record_path in raw_dir.rglob("*.json"):
            record = json.loads(record_path.read_text())
            if record.get("config_sha256") in changes:
                record["config_sha256"] = changes[record["config_sha256"]]
                record_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")


def rewrite_freezes(target: Path, changes: dict) -> None:
    for freeze_dir in sorted((target / "artifacts").glob("*freeze*")):
        rewrite_freeze(target, freeze_dir, changes)


def rewrite_freeze(target: Path, freeze_dir: Path, changes: dict) -> None:
    prediction_path = freeze_dir / "predictions.json"
    predictions = json.loads(prediction_path.read_text())
    for row in predictions["rows"]:
        row["config_sha256"] = changes.get(row["config_sha256"], row["config_sha256"])
    prediction_path.write_text(json.dumps(predictions, indent=2, sort_keys=True) + "\n")
    prefix_path = freeze_dir / "prefixes.json"
    prefixes = json.loads(prefix_path.read_text())
    for row in prefixes:
        row["config_sha256"] = changes.get(row["config_sha256"], row["config_sha256"])
    prefix_path.write_text(json.dumps(prefixes, indent=2, sort_keys=True) + "\n")
    freeze_path = freeze_dir / "prediction-freeze.json"
    freeze = json.loads(freeze_path.read_text())
    freeze["predictions_sha256"] = digest(prediction_path)
    freeze["prefixes_sha256"] = digest(prefix_path)
    for row in freeze["inputs"]:
        value = Path(row["path"])
        if "configs/" in row["path"]:
            value = Path(row["path"][row["path"].index("configs/"):])
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
    produced = []
    for panel, tag in STUDIES:
        directory = target / "configs" / panel
        results = target / "artifacts" / tag / "results.json"
        if not directory.exists() or not (target / "artifacts" / f"{tag}-freeze").exists() or not results.exists():
            continue
        family = json.loads(results.read_text())["primary_family"]
        shutil.rmtree(target / "artifacts" / tag, ignore_errors=True)
        second = [str(python), "controller/evaluate_confirmation.py", "--raw", f"raw/{panel}",
                  "--predictions", f"artifacts/{tag}-freeze/predictions.json",
                  "--prediction-freeze", f"artifacts/{tag}-freeze/prediction-freeze.json",
                  "--out", f"artifacts/{tag}"]
        for key in family:
            second.extend(["--primary", key])
        for config in sorted(directory.glob("panel-shard*.json")):
            second.extend(["--config", str(config.relative_to(target))])
        subprocess.run(second, cwd=target, check=True)
        produced.append(tag)
    if len(produced) == len(STUDIES):
        subprocess.run([str(python), "paper/build_assets_r2.py",
                        "--event", f"artifacts/{STUDIES[0][1]}/results.json",
                        "--scheduled", f"artifacts/{STUDIES[1][1]}/results.json"], cwd=target, check=True)


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


def verify_archive(path: Path) -> dict:
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        manifests = [name for name in names if name.endswith("artifacts/MANIFEST.json")]
        if len(names) != len(set(names)) or len(manifests) != 1:
            raise ValueError("Invalid supplement inventory")
        manifest = json.loads(archive.read(manifests[0]))
        if manifest.get("anonymous_derivative") is not True or manifest.get("result_values_preserved") is not True:
            raise ValueError("Invalid supplement lineage flags")
        if not re.fullmatch(r"[0-9a-f]{40}", manifest.get("source_commit", "")):
            raise ValueError("Invalid source commit")
        root = PurePosixPath(manifests[0]).parents[1]
        for row in manifest.get("files", []):
            name = str(root / row["path"])
            if name not in names or hashlib.sha256(archive.read(name)).hexdigest() != row["sha256"]:
                raise ValueError(f"Supplement hash mismatch: {row['path']}")
        return {"files": len(manifest.get("files", [])), "members": len(names)}


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
    verified = verify_archive(args.output)
    print(json.dumps({"path": str(args.output), "sha256": digest(args.output), "bytes": args.output.stat().st_size, "verified": verified}, sort_keys=True))


if __name__ == "__main__":
    main()
