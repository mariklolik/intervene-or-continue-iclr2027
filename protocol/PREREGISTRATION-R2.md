# Event-triggered runtime intervention with an extended repair menu

Frozen before any baseline or arm generation for the second confirmation.

## Scientific question

The first confirmation measured same-draw selection optimism and found no significant controller gain over continuation under a four-message menu placed at a hash-assigned step. A pre-outcome headroom reading of that frozen panel showed why: an oracle allowed a complete independent draw of every arm reached 50.28 percent against 51.69 percent for unconditional replanning, so the achievable prefix-conditional gain was bounded by the design rather than by learner capacity.

This study keeps the estimand, the repeated-branch protocol, and the direct signed-benefit target, and changes two design variables that bound the achievable gain: when the decision is taken and what the menu can repair. It also adds a controller whose arm choice and arm valuation use different continuation draws, which applies the paper's selection-versus-evaluation principle inside the controller rather than only inside the evaluator.

## Panels

The development panel `d2-event` contains 424 ALFWorld training-source identities. The confirmation panel contains 288 identities, balanced across the six task types, run under two checkpoint rules on the same identities: `p2-event` and `p2-scheduled`. Every goal directory is absent from the supplied prior exposure packet and from the first confirmation panel. Development and confirmation identities are disjoint.

## Checkpoint rules

`scheduled` keeps the first study's rule: a hash of the task identity assigns step 4 or 8. `event` takes the decision at the first step between 2 and 24 at which the public prefix shows an invalid action, an inadmissible action, or an observation identical to the preceding one. Tasks with no such step before termination have no intervention decision and contribute their terminal outcome to every policy.

## Action menu

Arms `A0`-`A3` are the frozen four-message contract, unchanged. `A4` reverts the three most recent decisions and replans. `A5` asks the actor to name the objects still required and the places not yet inspected. Every arm restores the same observable prefix and receives the same total decision budget; reverted decisions reduce the suffix limit by exactly the number reverted.

## Frozen policies

`CONTINUE`, the development-best fixed arm, the direct signed-benefit forest, the repeated arm-outcome forest, the strongest runnable arm-outcome comparator, and the cross-draw forest are frozen before any arm outcome. The cross-draw forest fits one signed-benefit forest per development draw, selects the arm on one draw and values it on the other. Its development grid adds two entries to the shared leaf and margin grid. The first is whether to require that both draws select the same arm before intervening. The second is the candidate set: either the whole menu, or the two arms with the highest mean benefit inside the training fold. The second entry was added after the development panel completed and before any confirmation arm outcome existed, because the development panel showed that maximizing over five noisy arm estimates is itself the maximization this paper measures; the candidate set is recomputed inside every training fold, so the out-of-fold utility that selects it remains honest. Both entries are chosen on development out-of-fold utility like every other grid entry, before any arm outcome. Features, folds, capacity, margin grid, seed, and development records are identical across learners.

## Primary family and power

The completed development panel fixes the operating point and, with it, the evidence the confirmation can carry. Against continuation the cross-draw rule shows a paired difference of 2.94 percentage points over a discordance of 11.5 percent, which needs 1,092 analyzable tasks for 80 percent power at alpha 0.05. Against the development-best fixed arm it shows 1.32 points over a discordance of 17.4 percent, which needs 7,915. The second quantity cannot be certified at any panel this study can build, so including it in the tested family would spend alpha on a test that cannot resolve.

The confirmatory family therefore contains one hypothesis: cross-draw control improves utility relative to `CONTINUE`, tested by a two-sided group sign-flip probability at alpha 0.05. Cross-draw control against the development-best fixed arm, against the direct learner, against the repeated arm-outcome forest, against the comparison-only learner, and against the failure-risk detector are pre-specified secondary contrasts, reported with estimates and both bootstrap intervals and without an adjusted decision. This reduction was recorded before any confirmation arm outcome existed and is justified only by development-panel quantities.

To raise power within the available frame, the event-triggered confirmation adds the remaining identity-disjoint ALFWorld identities to the frozen 288, giving 593 planned tasks. The addition is compositionally uneven because the balanced pool is exhausted: it contributes 111 pick-and-place, 23 clean, 3 cool, and 168 two-object identities and no look-at or heat identities. The confirmatory population is the pooled frame, and its composition is reported with the result.

Same-draw optimism is reported as an estimation diagnostic with task and floorplan bootstrap intervals and a within-task exchangeable-label reference distribution; it is not a member of the tested family, because the diagnostic is pathwise non-negative and a sign-flip null is degenerate for it.

A static-fallback entry was added to the cross-draw grid after the reviewer report and is reported as an exploratory policy, never as a member of the tested family. The entry lets the grid protect the development-best arm instead of continuation, so that in the absence of prefix-level heterogeneity the rule reduces to the optimal static rule, as the individualized-treatment-rule design literature recommends. The entry is selected on development out-of-fold utility inside the same folds; the frozen controller and the frozen predictions are unchanged, and both readings are reported whatever their sign.

The improvement-significance frontier of the frozen candidate grid is read on development records only, as the paired mean difference against `CONTINUE` and its $z$ statistic for every grid entry. It selects nothing and produces no confirmation prediction.

Learner capacity is measured on development records only, by nested honest out-of-fold utility of the cross-draw fit under gradient boosting, appended sentence embeddings of the prefix from `Qwen3-0.6B-Base`, and their combination. No confirmation prediction is produced from these variants and the frozen controller is unchanged; the reading is descriptive and answers whether the model class binds.

The scheduled mirror of the second panel runs the same identities and the same menu under the scheduled rule, so the decision point is the only design variable that moves. Its predictions come from the same frozen controller bundle and from baselines alone; the rule contrast is descriptive and is not a member of the tested family.

`p2-scheduled`, the contrast between checkpoint rules, direct-versus-cross-draw, direct-versus-continue, arm counts, firing rates, recoveries, disruptions, and costs are secondary or descriptive.

## Units, exclusions, and stopping

Task identity is the reporting unit and ALFWorld floorplan is the resampling group. No task may be removed for an unfavorable outcome. Missing eligible cells block the primary analysis. The study stops when both frozen confirmation panels are complete. The coordinator reservation cap is 400,000 H100 GPU-seconds for this study and is recorded in a separate ledger from the first confirmation.
