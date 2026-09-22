import argparse
import json
import os
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
    parser.add_argument("--scheduled", type=Path)
    parser.add_argument("--first", type=Path, default=ROOT / "artifacts" / "confirmation" / "results.json")
    parser.add_argument("--capacity", type=Path)
    parser.add_argument("--frontier", type=Path)
    parser.add_argument("--transfer", type=Path, nargs=3)
    parser.add_argument("--rooms", type=Path)
    parser.add_argument("--temperature-hot", type=Path)
    parser.add_argument("--temperature-cold", type=Path)
    parser.add_argument("--out", type=Path, default=ROOT / "paper")
    args = parser.parse_args()
    paths = {"event": args.event.resolve()}
    if args.scheduled and args.scheduled.exists():
        paths["scheduled"] = args.scheduled.resolve()
    else:
        paths["scheduled"] = args.first.resolve()
    reports = {rule: json.loads(path.read_text()) for rule, path in paths.items()}
    freezes = {}
    for rule, path in paths.items():
        candidate = path.parent.with_name(path.parent.name + "-freeze") / "prediction-freeze.json"
        if not candidate.exists():
            candidate = ROOT / "artifacts" / "prediction-freeze" / "prediction-freeze.json"
        freezes[rule] = json.loads(candidate.read_text())
    for report in reports.values():
        block = report["domains"]["alfworld"]
        validate(report, domains=set(report["domains"]), policies=set(block["policies"]), contrasts=set(block["contrasts"]))
    generated = args.out / "generated"
    figures = args.out / "figures"
    generated.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    outputs = [generated / "r2_numbers.tex", generated / "r2_results.tex", generated / "r2_analysis.tex", figures / "headroom.pdf", generated / "abstract.tex", generated / "r2_appendix.tex", generated / "conclusion.tex"]
    r2.write_numbers(reports, paths, outputs[0])
    r2.write_results(reports, outputs[1])
    capacity = json.loads(args.capacity.read_text()) if args.capacity and args.capacity.exists() else None
    frontier = json.loads(args.frontier.read_text()) if args.frontier and args.frontier.exists() else None
    r2.write_analysis(reports, outputs[2])
    r2.write_abstract(json.loads(args.first.read_text()), reports, outputs[4])
    r2.write_appendix(reports, freezes, outputs[5], capacity, frontier)
    r2.write_conclusion(json.loads(args.first.read_text()), reports, outputs[6])
    if args.transfer and args.rooms:
        target = generated / "r2_transfer.tex"
        matched, mismatched, event_static = [json.loads(path.read_text()) for path in args.transfer]
        rooms = json.loads(args.rooms.read_text())
        r2.write_transfer(matched, mismatched, event_static, reports["event"], rooms, target)
        outputs.append(target)
    setup_plotting()
    panels = {"First study": json.loads(args.first.read_text())}
    panels.update({r2.RULE_LABELS[rule]: report for rule, report in reports.items()})
    draw_headroom(panels, outputs[3])
    if args.temperature_hot and args.temperature_cold and args.temperature_hot.exists() and args.temperature_cold.exists():
        target = generated / "r2_temperature.tex"
        r2.noise_scaling(json.loads(args.temperature_hot.read_text()), json.loads(args.temperature_cold.read_text()), target)
        outputs.append(target)
    manifest = {
        "inputs": {rule: digest(path) for rule, path in paths.items()},
        "builder_sha256": digest(Path(__file__)),
        "text_builder_sha256": digest(Path(__file__).with_name("asset_text_r2.py")),
        "outputs": [{"path": os.path.relpath(path, ROOT), "sha256": digest(path)} for path in outputs],
    }
    (generated / "r2_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
