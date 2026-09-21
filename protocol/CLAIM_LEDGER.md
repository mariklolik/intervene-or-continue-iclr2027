# Claim ledger

| ID | Atomic claim | Class | Evidence | Status |
|---|---|---|---|---|
| C1 | Same-outcome maximization over four stochastic continuations is optimistic relative to scoring the selected action on the other draw. | Confirmatory inference | Frozen ALFWorld panel: +15.36 points, group 95% CI [12.46, 18.30], Holm-adjusted p < 0.0001; result SHA-256 `d9a94b01002c8be3668c1091218ff9a903533d2b701cc88248aa29dd6513a09d` | Supported |
| C2 | The feature-matched direct signed-benefit controller improves success relative to `CONTINUE`. | Confirmatory inference | Frozen ALFWorld panel: +1.56 points, group 95% CI [-0.12, 3.41], Holm-adjusted p = 0.2096 | Not supported |
| C3 | The direct signed-benefit controller improves success relative to the strongest development-selected matched controller. | Confirmatory inference | Frozen ALFWorld panel: -0.39 points, group 95% CI [-2.26, 1.57], Holm-adjusted p = 0.7927 | Not supported |
| C4 | The direct learner and repeated arm-outcome learner use the same prefix features, four task-group folds, 200-tree capacity, leaf and margin grids, and action menu. | Implementation fact | `controller/direct_advantage.py`, `extension/policies.py`, fit manifests | Verified |
| C5 | `CONTINUE` and the development-selected fixed arm are explicit candidates in operational selection. | Implementation fact | `controller/freeze_predictions.py`; development manifests | Verified |
| C6 | The primary panel contains 384 ALFWorld tasks whose goal directories were absent from the 59 supplied prior configuration files. | Design fact | `configs/independent-panel/panel-freeze.json` | Verified |
| C7 | The secondary panel contains 57 earlier reserved ScienceWorld official-test variations not present in the supplied executed-configuration exposure packet. | Design fact | `configs/independent-panel/panel-freeze.json` | Verified |
| C8 | The earlier retrospective four-stratum analysis did not establish broad superiority of the direct learner. | Adverse observation | `artifacts/historical/direct-benefit-exploratory-results.json` | Verified |
| C9 | COTA is not a matched comparator for this experiment because actor history, intervention support, action budget, and compute are not aligned. | Comparison boundary | alphaXiv PDF evidence; `artifacts/literature/ALPHAXIV_LEDGER.md` | Verified |
| C11 | On the first confirmation panel, a per-task selector given one complete independent draw of every arm reaches 50.28% against 51.69% for unconditional replanning, so the decision point and menu, not learner capacity, bound the achievable prefix-conditional value there. | Estimation diagnostic | `artifacts/confirmation/results.json`, headroom block | Supported |
| C12 | Most of the measured same-draw optimism on that panel is attributable to continuation noise rather than action advantage: the observed 16.62 points on eligible prefixes lie below an exchangeable-label reference of 17.26 points, one-sided p = 0.32. | Estimation diagnostic | `artifacts/confirmation/results.json`, exchangeable-label floor | Supported |
| C13 | A sign-flip null is degenerate for the same-draw diagnostic, which is pathwise non-negative; the second confirmation therefore excludes it from its tested family. | Statistical fact | Appendix A derivation; `protocol/PREREGISTRATION-R2.md` | Verified |
| C14 | Language agent tree search is the deployable form of the outcome-informed selector in Equation 2 and spends many suffixes per decision, so it is not a budget-matched baseline for this menu. | Comparison boundary | `artifacts/literature/ALPHAXIV_LEDGER.md` | Verified |
| C10 | The 27B result is a separate actor-systems result and is not evidence for the controller target parameterization. | Scope boundary | Prior 27B report and claim ledger | Verified |

No pending row may be promoted from an observed favorable subset. All ScienceWorld results and all five frozen policy comparisons remain in the result artifact regardless of direction.
