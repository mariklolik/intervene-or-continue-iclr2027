# Reproducibility

Use Python 3.12 or newer and uv. Dependencies are pinned in `uv.lock`.

    make sync
    make test
    make artifacts
    make evaluate

`make artifacts` downloads the published first-study supplement and checks SHA-256 `668a6b982aba900cb70b9e69cfff76f0a07b9dae970d86f0d4e7277d680807b7`. It extracts only raw episode records, first-panel configurations, frozen predictions, and reference results into ignored `data/first-study/`. Run it once in a fresh checkout.

`make evaluate` uses the retained evaluator and writes `records/confirmation/results.json`, ingestion audits, analysis rows, and verification hashes. The output directory must not already exist: choose a new output path when repeating evaluation. Confirm 384 ALFWorld planned tasks (355 eligible, 29 early terminal) and 57 ScienceWorld tasks (51 eligible). ALFWorld same-draw optimism is 0.15364583333333334; Direct-minus-Continue is 0.015625; Direct-minus-Matched is −0.00390625. Values are proportions, not percentage points.

The supplement predates the later headroom analyses. The current evaluator adds those fields without changing the frozen policy estimates. Absolute input paths and source hashes in verification receipts depend on the checkout. Compare statistical values rather than requiring identical whole-file hashes across source revisions.

The [complete archive](https://github.com/mariklolik/intervene-or-continue-iclr2027/tree/archive/reproduction-full-2026-10-04) preserves both merged PRs, full raw-record tooling, protocols, configurations, serialized controller models, paper generators, and second-study frozen reports. Second-study raw simulator records are not in the published first-study supplement; reproducing their generation requires the recorded simulator, actor revision, serving stack, and seeds. The current cleanup re-evaluates existing first-study records and rebuilds the paper; it does not run new model experiments.

To fit a controller from your development-only complete branch records, use the existing interface:

    uv run --frozen python controller/fit_cross_draw.py --input development/rows.json --model models/cross-draw.joblib --manifest models/cross-draw-manifest.json

The input contract is the archived analysis-row schema: task identity, model/scaffold identity, public prefix features, and independent arm-by-draw outcomes. Only development rows are accepted. The default reproduces the frozen continuation fallback; `--static-fallback` enables the later exploratory grid. Use `controller/freeze_predictions.py --help` to freeze prefix-only choices before confirmation outcomes, and `controller/evaluate_confirmation.py --help` for full record validation and grouped inference.

`paper/paper.tex` embeds its style, bibliography, tables, and vector figures. Compile in an empty build directory with pdfLaTeX twice; the compiler writes style and auxiliary files during the build. Only the source and final PDF are versioned. Full generation commands and manifests remain in the archive.
