import argparse
import hashlib
import json
import re
import subprocess
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BANNED = ["mekashirskiy", "mariklolik", "AlekseiSDev", "avi-gn-fsk", "/Users/", "/home/"]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_hashes(manifest_path: Path, key: str = "outputs") -> int:
    manifest = json.loads(manifest_path.read_text())
    checked = 0
    for row in manifest[key]:
        path = ROOT / row["path"]
        if not path.exists() or digest(path) != row["sha256"]:
            raise ValueError(f"Hash mismatch: {row['path']}")
        checked += 1
    return checked


def command_output(command: list[str]) -> str:
    return subprocess.run(command, check=True, capture_output=True, text=True).stdout


def main_text_page(aux: str) -> int:
    match = re.search(r"\\newlabel\{main-end\}\{\{[^}]*\}\{(\d+)\}", aux)
    if not match:
        raise ValueError("Main-text page label is missing")
    return int(match.group(1))


def verify_pdf(path: Path) -> dict:
    info = command_output(["pdfinfo", str(path)])
    pages = int(re.search(r"^Pages:\s+(\d+)$", info, re.MULTILINE).group(1))
    aux = (ROOT / "paper" / "build" / "main.aux").read_text()
    main_page = main_text_page(aux)
    if main_page > 9:
        raise ValueError("Main text exceeds nine pages or page label is missing")
    fonts = command_output(["pdffonts", str(path)]).splitlines()[2:]
    if not fonts or any(row.split()[4] != "yes" for row in fonts if len(row.split()) > 5):
        raise ValueError("PDF contains a non-embedded font")
    text_path = path.with_suffix(".verify.txt")
    try:
        subprocess.run(["pdftotext", str(path), str(text_path)], check=True)
        text = text_path.read_text()
    finally:
        text_path.unlink(missing_ok=True)
    if "??" in text or "[?]" in text:
        raise ValueError("PDF contains unresolved references")
    for value in BANNED:
        if value in text:
            raise ValueError(f"PDF contains identifying string: {value}")
    return {"pages": pages, "main_end_page": main_page, "fonts": len(fonts), "sha256": digest(path)}


def verify_supplement(path: Path) -> dict:
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or not names:
            raise ValueError("Invalid supplement member inventory")
        for name in names:
            data = archive.read(name)
            for value in BANNED:
                if value.encode() in data:
                    raise ValueError(f"Supplement member {name} contains {value}")
        if not any(name.endswith("artifacts/MANIFEST.json") for name in names):
            raise ValueError("Supplement manifest missing")
    return {"members": len(names), "bytes": path.stat().st_size, "sha256": digest(path)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", type=Path, default=ROOT / "paper" / "main.pdf")
    parser.add_argument("--supplement", type=Path, default=ROOT / "dist" / "intervene-or-continue-iclr2027-supplement.zip")
    args = parser.parse_args()
    results = ROOT / "artifacts" / "confirmation" / "results.json"
    verification = json.loads((ROOT / "artifacts" / "confirmation" / "verification.json").read_text())
    if digest(results) != verification["results_sha256"]:
        raise ValueError("Confirmation result hash mismatch")
    output = {
        "confirmation_result_sha256": digest(results),
        "generated_files_checked": verify_hashes(ROOT / "paper" / "generated" / "manifest.json")
        + (verify_hashes(ROOT / "paper" / "generated" / "r2_manifest.json") if (ROOT / "paper" / "generated" / "r2_manifest.json").exists() else 0),
        "pdf": verify_pdf(args.pdf),
        "supplement": verify_supplement(args.supplement),
    }
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
