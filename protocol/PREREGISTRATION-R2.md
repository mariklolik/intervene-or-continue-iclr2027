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

`CONTINUE`, the development-best fixed arm, the direct signed-benefit forest, the repeated arm-outcome forest, the strongest runnable arm-outcome comparator, and the cross-draw forest are frozen before any arm outcome. The cross-draw forest fits one signed-benefit forest per development draw, selects the arm on one draw, values it on the other, and intervenes only when both draws select the same arm and the cross-draw value exceeds the margin. Features, folds, capacity, margin grid, seed, and development records are identical across learners.

## Primary family

The `p2-event` primary family has two hypotheses:

1. cross-draw control improves utility relative to `CONTINUE`;
2. cross-draw control improves utility relative to the development-best fixed arm.

Two-sided group sign-flip probabilities are Holm-adjusted across the two hypotheses at familywise alpha 0.05. Same-draw optimism is reported as an estimation diagnostic with task and floorplan bootstrap intervals and a within-task exchangeable-label reference distribution; it is not a member of the tested family, because the diagnostic is pathwise non-negative and a sign-flip null is degenerate for it.

`p2-scheduled`, the contrast between checkpoint rules, direct-versus-cross-draw, direct-versus-continue, arm counts, firing rates, recoveries, disruptions, and costs are secondary or descriptive.

## Units, exclusions, and stopping

Task identity is the reporting unit and ALFWorld floorplan is the resampling group. No task may be removed for an unfavourable outcome. Missing eligible cells block the primary analysis. The study stops when both frozen confirmation panels are complete. The coordinator reservation cap is 400,000 H100 GPU-seconds for this study and is recorded in a separate ledger from the first confirmation.
