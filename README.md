# Intervene or Continue

This repository is the clean research and submission package for an independent evaluation of runtime intervention in stochastic language-model agents.

The frozen study compares a direct signed-benefit forest with the existing arm-outcome forest under identical features, grouped folds, capacity, action menu, and development data. `CONTINUE` and the development-selected fixed arm remain explicit candidates. The primary confirmation uses new ALFWorld task identities; a separate untouched ScienceWorld variation panel tests the boundary without retuning.

The repository starts from the verified native/demo runtime and the `intervene_8gpuh` scientific contract. Historical source artifacts are copied rather than edited in place. Raw outcomes, frozen predictions, audits, aggregate results, the ICLR 2027 source, and the reviewer supplement are linked by content hashes.

## Result

On the frozen 384-task ALFWorld confirmation, same-draw action selection overstates independent-draw value by 15.36 percentage points, with a floorplan-bootstrap 95% interval of [12.46, 18.30] and Holm-adjusted p < 0.0001. Direct signed-benefit prediction changes success by +1.56 points versus `CONTINUE` and -0.39 points versus the matched controller; neither contrast is significant after correction. The complete 57-task ScienceWorld boundary panel is retained.

## Reproduce

Inside the supplementary archive, create the locked Python 3.12 environment and regenerate the frozen analysis:

```bash
uv venv --python 3.12 .venv
uv sync --frozen
mv artifacts/confirmation artifacts/confirmation.release
python controller/evaluate_confirmation.py --config configs/independent-panel/panel-shard0.json --config configs/independent-panel/panel-shard1.json --config configs/independent-panel/panel-shard2.json --config configs/independent-panel/panel-shard3.json --config configs/independent-panel/panel-shard4.json --config configs/independent-panel/panel-shard5.json --config configs/independent-panel/panel-shard6.json --config configs/independent-panel/panel-shard7.json --raw raw --predictions artifacts/prediction-freeze/predictions.json --prediction-freeze artifacts/prediction-freeze/prediction-freeze.json --out artifacts/confirmation
python paper/build_assets.py
```

The evaluator refuses incomplete arm/draw blocks, mismatched panel or prediction hashes, and technically failed canonical records. The anonymous supplement contains the exact analysis inputs, verification program, manuscript source, and final PDF. Model weights and benchmark installations are excluded and pinned in the appendix.

## Evidence boundary

The confirmatory claim is the measurement gap, not controller superiority or cross-harness state of the art. Development and historical native/demo results are explicitly labeled exploratory. ScienceWorld results, disruptions, early terminations, failed-attempt records, compute accounting, and null controller comparisons are preserved.
