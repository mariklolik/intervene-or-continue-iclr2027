import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).parent))
from asset_figures import draw_design, draw_group_results, draw_policy_effects, setup_plotting
from asset_text import (
    POLICY_LABELS,
    contrast_rows,
    cost_rows,
    digest,
    policy_table,
    validate,
    write_abstract,
    write_analysis,
    write_analysis_floats,
    write_appendix_results,
    write_results_floats,
    write_conclusion,
    write_numbers,
    write_results,
)


def write_manifest(result_path: Path, outputs: list[Path], out: Path) -> None:
    payload = {
        "results_sha256": digest(result_path),
        "builder_sha256": digest(Path(__file__)),
        "text_builder_sha256": digest(Path(__file__).with_name("asset_text.py")),
        "figure_builder_sha256": digest(Path(__file__).with_name("asset_figures.py")),
        "outputs": [{"path": str(path.relative_to(ROOT)), "sha256": digest(path)} for path in outputs],
    }
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=ROOT / "artifacts" / "confirmation" / "results.json")
    parser.add_argument("--out", type=Path, default=ROOT / "paper")
    args = parser.parse_args()
    args.results = args.results.resolve()
    args.out = args.out.resolve()
    report = json.loads(args.results.read_text())
    validate(report)
    generated = args.out / "generated"
    figures = args.out / "figures"
    generated.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    outputs = [
        generated / "numbers.tex",
        generated / "abstract.tex",
        generated / "results.tex",
        generated / "analysis.tex",
        generated / "conclusion.tex",
        generated / "appendix_results.tex",
        generated / "results_floats.tex",
        generated / "analysis_floats.tex",
        figures / "design.pdf",
        figures / "policy-effects.pdf",
        figures / "group-results.pdf",
    ]
    write_numbers(report, args.results, outputs[0])
    write_abstract(report, outputs[1])
    write_results(report, outputs[2])
    write_analysis(report, outputs[3])
    write_conclusion(report, outputs[4])
    write_appendix_results(report, outputs[5])
    write_results_floats(report, outputs[6])
    write_analysis_floats(report, outputs[7])
    setup_plotting()
    draw_design(outputs[8])
    draw_policy_effects(report, outputs[9])
    draw_group_results(report, outputs[10])
    write_manifest(args.results, outputs, generated / "manifest.json")


if __name__ == "__main__":
    main()
