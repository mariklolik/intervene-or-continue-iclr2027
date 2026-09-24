# PAT Feedback: Manuscript Revision Plan

**Status:** Working document. The items below have not yet been applied to the manuscript.  
**Source:** Private PAT/LLM feedback dated September 18, 2026, supplied as Pasted text.txt.  
**Applies to:** paper/main.tex, paper/appendix.tex, generated sections, and supporting artifacts in this project.

This document consolidates repeated feedback by the kind of work required. It does not supersede the preregistered protocol or turn a post hoc analysis into a confirmatory one. PAT feedback is private, pre-submission guidance for authors; it does not require a point-by-point OpenReview response at this stage. The immediate deliverable is a revised PDF and, where needed, updated supplementary materials.

## Priority labels

- **P0 — before submission:** Affects mathematical correctness or the central scientific claim.
- **P1 — preferably before submission:** Resolves a substantive ambiguity in the method or results.
- **P2 — editorial:** Improves clarity or reproducibility without changing the conclusions.
- **Future — new experiment:** Outside the current confirmation panel.

## A. Scientific revisions to the manuscript text

| ID | Priority | Issue | Proposed revision | Evidence |
|---|---|---|---|---|
| A1 | P0 | Proposition 1 states conditional identical distribution, but interpreting cross-draw scoring as evaluation on a fresh continuation also requires conditional independence between the two outcome vectors. | State that the two vectors are conditionally independent and identically distributed given the shared prefix. Align the proposition with its proof. Actions within one vector need not be mutually independent. | paper/main.tex, Proposition 1; paper/appendix.tex, proof. |
| A2 | P0 | Readers may attribute the 15.36 percentage-point gap to evaluation of a precommitted controller. | Explicitly distinguish the diagnostic outcome-informed selector from a prefix-only policy. One test draw gives an unbiased, though noisier, estimate for a policy fixed before test outcomes. The gap concerns selecting an action from observed outcomes and scoring it on those same outcomes. | paper/main.tex, introduction and Section 2.2; paper/generated/abstract.tex. |
| A3 | P0 | Direct versus Matched and Direct versus Arm outcome answer different questions. Matched/Single-1 uses one training draw; Direct uses two. | Describe Direct versus Matched as the primary practical comparison against the strongest development-selected runnable comparator. Describe Direct versus Arm outcome as the secondary controlled comparison of target parameterization with the same two-draw training records. Do not attribute the first contrast solely to the regression target. | paper/main.tex, Sections 3–4; paper/generated/appendix_results.tex. |
| A4 | P1 | The policy-value equation is conditional on reaching a checkpoint, whereas the reported success rate includes all planned tasks. | Define full task-level value as the contribution from early terminations plus expected policy value among tasks that reach a checkpoint. Explain that early-terminal tasks have the same outcome under every policy and contribute zero to contrasts. | paper/main.tex, Sections 2.1 and 4.1; paper/appendix.tex, statistics. |
| A5 | P1 | A statement that intervention did not help may be read as a claim about runtime intervention in general. | Scope the null controller result to the Qwen3-8B actor, scheduled early checkpoints, four messages, and the tested random-forest features. Do not generalize to all advisors, monitors, or environments. | paper/main.tex, limitations and conclusion. |
| A6 | P1 | Checkpoints are scheduled in advance rather than triggered by detected errors. | Explain that this design compares actions from a shared state but studies a one-shot decision at a prespecified time, not an event-triggered recovery monitor. Do not claim without a separate experiment that scheduling caused the observed disruptions. | paper/main.tex, Section 4.1 and limitations. |
| A7 | P1 | Best fixed has a numerically higher success rate than the learned policies. | Discuss recoveries and disruptions without inventing an untested mechanism. In ALFWorld, Direct has 26−14=12 net draw-level outcomes and Best fixed has 74−55=19, a difference of seven draw-level outcomes, or about 0.91 percentage points. | paper/generated/results.tex; paper/generated/appendix_results.tex. |
| A8 | P1 | The smaller ScienceWorld gap may reflect many tasks without a successful branch. | Report the observed all-zero fraction and avoid interpreting the 5.26-point gap as evidence of more stable branches. Among 51 eligible tasks, 39 failed in all eight action/draw cells (76.5%). | artifacts/confirmation/analysis-rows.json; paper/generated/analysis.tex. |

## B. Statistical interpretation and analysis using existing data

| ID | Priority | Issue | Action and interpretation boundary |
|---|---|---|---|
| B1 | P0 | For G_i ≥ 0, the group sign-flip test uses an awkward symmetric null: the observed nonnegative sum is maximal over sign assignments. | Revise the abstract and results language. Foreground the 15.36-point estimate, its task distribution, and the floorplan-bootstrap interval [12.46, 18.30]. Preserve the preregistered test and its result in the audit record, but do not treat its small p-value as independent evidence of treatment benefit or controller superiority. Label any new null procedure as a subsequent analysis rather than replacing the preregistered procedure silently. |
| B2 | P1 | Saying the controller contrasts did not survive multiplicity correction can imply that they were nominally significant beforehand. | Report both levels: Direct minus Continue has raw p=0.1048 and Holm p=0.2096; Direct minus Matched has raw and Holm p=0.7927. Neither contrast was significant even before correction. |
| B3 | P1 | The development-selection table omits Arm outcome and the development sample sizes. | Retrieve Arm outcome out-of-fold utility and intervention rate from saved manifests or analysis artifacts, and add the row. Report the Qwen3-8B development sample sizes: 129 ALFWorld and 55 ScienceWorld tasks. Confirm the actor stratum and folds before copying any figures. |
| B4 | P1 | ScienceWorld Direct minus Continue has an empirical interval of [0, 0]. | Explain that Direct intervened on six of the 51 eligible prefixes but changed none of the observed binary outcomes on either draw. Zero sample variance does not establish an exactly zero population effect. |
| B5 | P2 | Several ScienceWorld policies have identical reported values. | Note that Arm outcome chose arm 2 on every eligible prefix and therefore coincided with Best fixed; Safe selected coincided with Matched/Single-1 under the frozen development selection. |
| B6 | P2 | Direct never selected rollback on an ALFWorld confirmation prefix. | Keep this descriptive: among 355 eligible prefixes, Direct chose Continue 245 times, Warning 4, Replan 106, and Rollback 0. Do not assert a cause without another analysis. Matched selected Rollback 71 times. |

**Qualification for B1:** The feedback's claim that G=0 implies no continuation stochasticity is too strong. All arms can change outcome together between draws while G remains zero. Simply shuffling action labels tests a different null and is not an automatic replacement for the original test. The ALFWorld gap is positive in 53 of 96 floorplan groups.

## C. Method and reproducibility clarifications without new agent runs

| ID | Priority | Verified state | Revision |
|---|---|---|---|
| C1 | P1 | The evaluator's argmax resolves ties in action-index order 0,1,2,3; the prose only says continuation-first. | State the full tie order: 0 → 1 → 2 → 3. |
| C2 | P1 | Arm 3 restores the snapshot immediately before the last action and runs with limit B−1. Restoring at step k−1 leaves (B−1)−(k−1)=B−k new decisions. | Add the explicit budget formula. Replace the ambiguous phrase “latest reversible action” with the actual last-action snapshot rollback semantics. |
| C3 | P1 | The numeric feature set includes env_scienceworld. | Name this encoded environment indicator explicitly and reconcile the description with the list of 50 numeric features. |
| C4 | P1 | The main text says “four shuffled whole-task folds,” while a table says “task-group folds.” | Check the actual fold unit in the code and manifests, then use one precise description throughout. Do not equate a task fold with a floorplan fold without verification. |
| C5 | P2 | The appendix reproduction commands call python tools/verify_artifacts.py, while other commands use .venv/bin/python. | Use the environment's Python consistently if the verification command depends on that environment. |
| C6 | P2 | CAMELS-CIS is unexplained, and the reproducibility statement points too broadly to the appendix. | Briefly define or remove the unrelated reference, and point readers to the exact sections for prompts, hyperparameter grids, and commands. |
| C7 | P2 | Units and minus signs vary; the historical 97.5% interval is unexplained; extracted PDF text contains apparent line-break artifacts. | Inspect the rendered PDF alongside the LaTeX source. Standardize percentage-point units, explain the historical interval level, and fix real typographic errors rather than extraction artifacts. |
| C8 | P2 | All ScienceWorld group points have a zero Direct-minus-Continue horizontal coordinate. | Set a suitable axis scale or say explicitly in the caption that all points lie on the zero line. |

After text and table changes, rebuild the PDF and anonymous supplement and run the existing artifact verification. These clarifications do not alter experimental outcomes or frozen predictions.

## D. Suggestions that require new experiments

These are reasonable directions for later work. They should not silently expand the current confirmatory family or replace its results.

| ID | Suggestion | Why it is a separate experiment |
|---|---|---|
| D1 | Compare a modern neural advisor or verifier with the random forest. | Requires training or selecting a new policy, freezing it before test outcomes, and evaluating it independently under a matched protocol. |
| D2 | Trigger intervention upon an error, loop, or invalid action. | Changes the prefix-selection mechanism and the target population of decisions. |
| D3 | Use tree search or graph-guided repair. | Changes the action space, intervention schedule, and compute budget. |
| D4 | Test environments with nondeterministic external tools and multi-turn state drift. | Requires a different benchmark, a state-restoration protocol, and a separate confirmation panel. |
| D5 | Ablate controller capacity or investigate why Direct avoided Rollback. | Observed action frequencies do not identify the cause; new models or controlled changes to the menu and budget are needed. |

For the present manuscript, the appropriate response to D1–D5 is a precise scope statement, not a promise of an unmatched experiment on a short deadline.

## E. Feedback claims that need correction or version checking

1. **“More than 80% all-zero” in ScienceWorld:** This is incorrect for the eligible panel: 39/51 tasks, or 76.5%, had all-zero action/draw cells. Continue's 19.30% success rate does not determine the other arms' outcomes.
2. **“48 missed net recoveries”:** Best fixed had 48 more gross recoveries than Direct but also 41 more disruptions. The net difference is seven draw-level outcomes.
3. **“Any stochasticity gives the minimum sign-flip p-value”:** This is too strong. The core concern about a symmetric null for nonnegative G remains important.
4. **Quoted passages and line numbers:** Some do not match the current paper/main.tex and paper/appendix.tex. For example, the current figure discussion already describes groups rather than individual actions. Check each substantive concern against the current source rather than editing by PAT line number alone.
5. **Hyphenation and missing equations:** Some may be PDF text-extraction artifacts. Check the rendered PDF and LaTeX before changing text or mathematics.

## Recommended order before full submission

1. **P0:** Correct Proposition 1 and the distinction between oracle diagnostics and fixed-policy evaluation; settle the statistical wording for G; separate the purposes of the two controller comparisons.
2. **P1:** Align the task-level estimand, Rollback budget, tie-breaking, folds, and features; add the missing development and ScienceWorld explanations.
3. **P2:** Review captions, references, units, historical intervals, and reproduction commands.
4. **Build and audit:** Regenerate the PDF and supplement; verify numbers, anonymity, and artifact integrity. PAT feedback does not require a point-by-point OpenReview response.

Official procedural sources: [ICLR PAT announcement](https://blog.iclr.cc/2026/09/10/making-googles-paper-assistant-tool-pat-available-to-iclr-submitters/) and [ICLR 2027 Author Guidelines](https://iclr.cc/Conferences/2027/AuthorGuidelines).
