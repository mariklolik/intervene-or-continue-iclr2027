# Independent runtime-intervention confirmation

Frozen before baseline or arm generation.

## Scientific question

The study tests whether action selection on the same stochastic continuation outcomes overstates utility on independent continuation draws, and whether a direct signed-benefit forest improves deployable utility relative to `CONTINUE` and the strongest development-selected matched controller.

The direct learner predicts three signed benefits, $E[Y_a-Y_0\mid x]$. The arm-outcome learner predicts the four arm outcomes and selects by their estimated values. Both use the same prefix features, task-group folds, 200-tree capacity grid, leaf grid, margin grid, random seed, action menu, and development records. The target parameterization is the intended difference.

## Panels

The primary panel contains 384 ALFWorld training-source task identities, balanced equally across the six task types. Every selected goal directory is absent from the supplied prior configuration and outcome exposure packet. The domain was chosen from development evidence and is therefore the scoped target population, not evidence of cross-domain generality.

The secondary boundary panel contains 57 ScienceWorld official-test variations reserved earlier but not generated in the earlier 27-family confirmation or the archived controller study. These variations are frozen without controller retuning. All ScienceWorld results will be reported regardless of direction.

Each task has one baseline trajectory and, when the checkpoint is reached, two independently seeded continuations for each of `CONTINUE`, `WARNING`, `REPLAN`, and `ROLLBACK_REPLAN`. Arms share the observable prefix and executed-action budget. Hidden simulator state is used only to verify replay.

## Frozen policies and fallbacks

The signed-benefit learner, the repeated-outcome arm forest, `CONTINUE`, and the development-selected fixed arm are frozen before arm outcomes. The strongest runnable matched comparator is selected by development out-of-fold utility from the existing arm-outcome family. The operational safe policy is selected from `CONTINUE`, the fixed arm, signed benefit, and that matched comparator by out-of-fold utility, with lower intervention rate breaking ties. This selection is reported even if it selects a baseline rather than the signed-benefit learner.

## Primary family

The ALFWorld primary family has three hypotheses:

1. same-draw action selection has positive utility optimism relative to cross-draw evaluation;
2. signed-benefit control improves utility relative to `CONTINUE`;
3. signed-benefit control improves utility relative to the strongest development-selected matched controller.

Two-sided group sign-flip probabilities are Holm-adjusted across the three hypotheses at familywise alpha 0.05. Task and whole-floorplan bootstrap intervals are reported. A positive point estimate without an adjusted rejection is an unresolved result, not superiority.

The signed-benefit comparison against the repeated-outcome arm forest isolates the target parameterization and is secondary. `BEST_FIXED`, the operational safe selection, firing rates, recovery, disruption, token use, and every ScienceWorld comparison are reported as secondary or descriptive analyses.

## Units, exclusions, and stopping

Task identity is the reporting unit. ALFWorld floorplan and ScienceWorld task family are resampling groups. Repeated arms and draws are not independent observations.

No task may be removed for a poor outcome, long trajectory, or unfavorable arm result. A task ending before the checkpoint contributes the same terminal outcome to every policy and zero to policy contrasts. Missing eligible arm cells block the primary analysis until the same frozen configuration completes or a failure-aware sensitivity analysis is reported. At most one coordinator resume per shard is allowed for an administrative interruption; completed terminal outcomes are never rerun. The study stops after the frozen panel is complete. No subgroup, endpoint, arm, margin, or controller may replace a primary result after outcomes are available.

The total coordinator reservation cap is 72,000 H100 GPU-seconds. Baseline-only generation and frozen-arm generation share this ledger. Administrative resumes consume the same cap.
