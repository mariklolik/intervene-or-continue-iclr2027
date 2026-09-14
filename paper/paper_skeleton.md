# Paper skeleton

## Identity

- Working title: `Evaluate the Choice, Not the Chosen Rollout: Honest Runtime Intervention for Language Agents`
- Alternative title: `Intervene or Continue? Separating Action Selection from Evaluation in Language Agents`
- Venue: ICLR 2027, anonymous submission
- Main-text target: 8.4--8.8 pages in the official ICLR 2027 style, excluding references, reproducibility statement, and appendix
- Type: empirical measurement study with a controller comparison

## Argument core

- Central claim: Evaluating an intervention on the stochastic branch used to select it yields an optimistic estimate of deployable utility; an independent continuation draw and an explicit `CONTINUE` option are required for a defensible runtime-intervention evaluation.
- Problem: Runtime monitors may select among several repairs for a language agent, but every repair can disrupt a trajectory that would otherwise recover. The evaluation must distinguish the value of the choice rule from luck in the sampled continuation.
- Gap: Existing agent-intervention studies commonly use one realized branch per action. The closest repeated-branch analysis covers 24 shallow ALFWorld prefixes and reports no within-action variation there; it does not test deeper checkpoints at confirmation scale. Existing treatment-effect estimators address target parameterization but not reuse of the selected stochastic outcome for evaluation.
- Main idea: Generate two independently seeded continuations for every action from a shared prefix, freeze all policies before these outcomes, and evaluate each selected action on the other draw. Compare a direct signed-benefit forest with a feature- and capacity-matched arm-outcome forest, `CONTINUE`, and the development-selected fixed arm.

## Contributions and required evidence

1. A paired repeated-branch evaluation protocol for runtime intervention, including a nonnegative symmetric same-draw optimism diagnostic.
   - Evidence: formal identity; 384-task independent ALFWorld confirmation panel; two draws per four actions; preregistered sign-flip test and bootstrap intervals.
2. A controlled comparison between direct signed-benefit and arm-outcome forests with the same features, folds, capacity, action support, and development records.
   - Evidence: source and model manifests; prediction-freeze artifact; direct-vs-`CONTINUE` and direct-vs-matched-controller estimates on the full primary panel.
3. A safe decision set that can decline intervention through explicit `CONTINUE` and a development-selected fixed arm.
   - Evidence: frozen policy selection; firing, harm, recovery, token, and action-count results; operational selection reported regardless of which candidate wins development.
4. A reproducible confirmation package with untouched task identities, immutable configurations, bounded compute, and adverse-domain reporting.
   - Evidence: pre-outcome panel/protocol commits; configuration hashes; GPU ledger; full ALFWorld and ScienceWorld results; code, environment, and artifact manifests.

## Section plan

### Abstract, 0.25 page

- State the evaluation problem, repeated-branch design, primary panel, and only the results supported by the final audit.
- Required evidence: final immutable result JSON and claim ledger.

### 1 Introduction, 1.05 pages

- A runtime intervention is a treatment choice under uncertain continuation, not just error detection.
- Same-branch selection and scoring confound policy quality with favorable branch noise.
- Present the independent-draw protocol, safe candidates, and controlled target comparison.
- List three evidence-backed contributions and delimit ALFWorld confirmation from ScienceWorld boundary evidence.
- Required evidence: alphaXiv source ledger; frozen protocol; final result audit.

### 2 Evaluating runtime intervention, 1.20 pages

- Define prefix $x$, actions $a\in\{0,1,2,3\}$, continuation outcome $Y_a$, value $\mu_a(x)$, and deployable policy value.
- Contrast same-draw maximization with cross-draw evaluation.
- Give the symmetric nonnegative optimism identity and its interpretation.
- Explain why `CONTINUE` must be in the action set.
- Required evidence: formal derivation; implementation-to-notation map.

### 3 Controllers and safe selection, 1.10 pages

- Describe the arm-outcome forest and the direct signed-benefit forest $\Delta_a(x)=E[Y_a-Y_0\mid x]$.
- Establish feature, fold, capacity, record, and action-menu matching.
- Define margins, fixed arm, and development-only operational selection.
- Required evidence: model code, manifests, development selection, prediction freeze.

### 4 Confirmation design, 1.35 pages

- Describe the actor, checkpoints, four actions, two independent continuations, and replay validation.
- Define the 384-task ALFWorld primary panel and 57-variation ScienceWorld boundary panel.
- Specify primary family, grouped sign-flip inference, Holm correction, grouped/task bootstrap, exclusions, and compute cap.
- Make the domain-selection and task-exposure boundaries explicit.
- Required evidence: preregistration, panel freeze, result denominator audit, GPU ledger.

### 5 Results, 1.30 pages

- Report the full primary family first, including adjusted $p$ values and grouped intervals.
- Compare every frozen policy and report firing, recovery, disruption, token, and action costs.
- Report ScienceWorld direction and uncertainty even when adverse.
- Keep 27B actor scaling separate from the controller claims.
- Required evidence: final results, tables, figure data, verification report.

### 6 What the repeated branches reveal, 0.80 page

- Interpret optimism, between-draw instability, and controller disagreements.
- Distinguish target-parameterization evidence from evaluation-design evidence.
- Discuss heterogeneity without promoting an observed subgroup to the target population.
- Required evidence: group-level and disagreement outputs; historical adverse results.

### 7 Related work, 0.65 page

- Situate agent acting/reflection, interactive benchmarks, runtime interventions, meta-learners, and adaptive reuse.
- State the closest-work boundary precisely; do not claim treatment-effect prediction or holdout separation as new.
- Required evidence: verified alphaXiv ledger and bibliography locators.

### 8 Limitations, ethics, and conclusion, 0.80 page

- Limit to one actor, textual environments, one checkpoint rule, binary success utility, and two continuation draws.
- Note development-based domain selection and limited ScienceWorld boundary sample.
- State compute/environmental costs, no human subjects, and no claim of universal benefit.
- Conclude with the evaluation requirement established by the evidence.
- Required evidence: compute ledger, result audit, reproducibility inventory.

### Reproducibility and AI-use statements

- Point to appendix, anonymous supplement, immutable hashes, exact environment, and commands.
- Disclose research, coding, prose, and editing assistance as required by the ICLR 2027 policy; authors retain responsibility and verify all claims and citations.

## Figures and tables

- Figure 1: shared prefix, four actions, two independent draws, freeze-before-outcome design.
- Figure 2: primary policy effects with task and grouped confidence intervals.
- Figure 3: same-draw optimism and group-level heterogeneity, visibly separating confirmation from boundary evidence.
- Table 1: matched controller and experimental design.
- Table 2: primary family, adjusted inference, and all policy values.
- Table 3: firing, harm, recovery, action and token costs.

## Appendix plan

- Proof of the symmetric optimism identity and tie handling.
- Complete prompts, interventions, checkpoint logic, and replay checks.
- Panel construction, exposure exclusions, hashes, and task-family counts.
- Hyperparameters, model-selection details, and development results.
- Full group/domain results, sensitivity analyses, missingness audit, and all adverse findings.
- Separate native/demo exploratory results and 27B systems result.
- Reproduction commands, software/container versions, compute ledger, and artifact manifest.
- AI-assistance disclosure details and citation verification ledger.

## Missing information

- Final completion audit for all 441 baselines and eligible arm cells.
- Frozen confirmation predictions and their pre-outcome commit hash.
- Primary and secondary estimates, uncertainty, multiplicity results, and cost metrics.
- Final artifact hashes and exact measured GPU time.
- Author-entered OpenReview metadata, affiliations, conflicts, and sanctions-list self-check.

## Reviewer-facing risks

- The direct learner may not beat `CONTINUE` or the matched controller; the paper must remain an evaluation study rather than recast a null result as controller superiority.
- ALFWorld was chosen from development evidence, so confirmation is scoped to that domain.
- ScienceWorld has fewer independent groups and must be treated as boundary evidence.
- Two draws identify a selection/evaluation gap but estimate per-action stochastic variance imprecisely.
- The repeated-outcome forest was fitted on earlier tasks; distribution shift may dominate target parameterization.
- COTA is contextual prior work rather than a matched baseline because actor revision, action support, history, budget, and compute differ.
- A private repository preserves double-blind anonymity but requires an anonymized supplementary archive for review.
