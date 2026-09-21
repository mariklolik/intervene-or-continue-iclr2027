import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).parent))
import asset_text_r2 as r2
from asset_figures import draw_headroom, setup_plotting
from asset_text import digest, validate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--event", type=Path, required=True)
    parser.add_argument("--scheduled", type=Path, required=True)
    parser.add_argument("--first", type=Path, default=ROOT / "artifacts" / "confirmation" / "results.json")
    parser.add_argument("--out", type=Path, default=ROOT / "paper")
    args = parser.parse_args()
    paths = {"event": args.event.resolve(), "scheduled": args.scheduled.resolve()}
    reports = {rule: json.loads(path.read_text()) for rule, path in paths.items()}
    for report in reports.values():
        block = report["domains"]["alfworld"]
        validate(report, domains={"alfworld"}, policies=set(block["policies"]), contrasts=set(block["contrasts"]))
    generated = args.out / "generated"
    figures = args.out / "figures"
    generated.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    outputs = [generated / "r2_numbers.tex", generated / "r2_results.tex", generated / "r2_analysis.tex", figures / "headroom.pdf", generated / "abstract.tex", generated / "r2_appendix.tex"]
    r2.write_numbers(reports, paths, outputs[0])
    r2.write_results(reports, outputs[1])
    r2.write_analysis(reports, outputs[2])
    r2.write_abstract(json.loads(args.first.read_text()), reports, outputs[4])
    r2.write_appendix(reports, outputs[5])
    setup_plotting()
    draw_headroom({r2.RULE_LABELS[rule]: report for rule, report in reports.items()}, outputs[3])
    manifest = {
        "inputs": {rule: digest(path) for rule, path in paths.items()},
        "builder_sha256": digest(Path(__file__)),
        "text_builder_sha256": digest(Path(__file__).with_name("asset_text_r2.py")),
        "outputs": [{"path": str(path.relative_to(ROOT)), "sha256": digest(path)} for path in outputs],
    }
    (generated / "r2_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
