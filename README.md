# Intervene or Continue

This repository is the clean research and submission package for an independent evaluation of runtime intervention in stochastic language-model agents.

The repository holds two frozen confirmations. The first compares a direct signed-benefit forest with the existing arm-outcome forest under identical features, grouped folds, capacity, action menu, and development data, at a hash-assigned early checkpoint with four fixed repair messages; a separate untouched ScienceWorld panel tests the boundary without retuning. The second keeps the estimand, the protocol, and the learner family and changes two design variables that the first study's headroom reading identified as binding: the decision is taken at the first public failure signal rather than at a scheduled step, and the menu separates repair depth from repair content. It adds a cross-draw controller that chooses its action on one development draw and values it on another. `CONTINUE` and the development-best unconditional action remain explicit candidates throughout.

The repository starts from the verified native/demo runtime and the `intervene_8gpuh` scientific contract. Historical source artifacts are copied rather than edited in place. Raw outcomes, frozen predictions, audits, aggregate results, the ICLR 2027 source, and the reviewer supplement are linked by content hashes.

## Result

On the frozen 384-task ALFWorld confirmation, same-draw action selection overstates independent-draw value by 15.36 percentage points, with a floorplan-bootstrap 95% interval of [12.46, 18.30] and Holm-adjusted p < 0.0001. Direct signed-benefit prediction changes success by +1.56 points versus `CONTINUE` and -0.39 points versus the matched controller; neither contrast is significant after correction. The complete 57-task ScienceWorld boundary panel is retained.

## Second study

`protocol/PREREGISTRATION-R2.md` fixes the panels, the menu, the checkpoint rules, and the two-hypothesis family before generation. `configs/d2-event`, `configs/p2-event`, and `configs/p2-scheduled` are the frozen panels; the confirmation identities are shared by the two checkpoint rules, so the rule contrast is paired at the task level. `extension/arms.py` holds the shared action and checkpoint contract, `controller/cross_draw.py` the cross-draw learner, and `paper/build_assets_r2.py` the tables, figures, and prose generated from its result file.

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
