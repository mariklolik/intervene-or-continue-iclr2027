# Section-level reading of the pre-2024 ICLR reference set

Five papers accepted to ICLR before 2024, read from their arXiv sources: BabyAI (1810.08272, ICLR 2019), ALFWorld (2010.03768, ICLR 2021), self-consistency (2203.11171, ICLR 2023), least-to-most prompting (2205.10625, ICLR 2023), and ReAct (2210.03629, ICLR 2023). The reading records how each section opens and what work it is made to do, so that this manuscript can adopt the same moves. Section lengths, float conventions, and a vocabulary comparison are in `AUDIT-ICLR.md`.

## Abstract

Every paper opens on an established capability or a concrete situation, not on its own contribution. ReAct: reasoning and acting "have primarily been studied as separate topics." Least-to-most: chain-of-thought "has demonstrated remarkable performance... However, it tends to perform poorly on tasks which require solving problems harder than the exemplars." ALFWorld opens on a request a person could make. BabyAI opens on what would be desirable. The shape is capability, then the word that turns it, then the proposal, then the numbers, then one line of significance.

Applied here: the abstract opens on what a runtime controller is asked to do before naming the bias it is exposed to.

## Introduction

Four of the five open with a question to the reader or an everyday scene: "How can a human train an intelligent agent to understand natural language instructions?"; "Consider helping a friend prepare dinner in an unfamiliar house". The gap is stated in the second or third paragraph, never the first sentence. Contributions are listed last, in prose rather than bullets, in ReAct and least-to-most.

Applied here: the introduction keeps its declarative opening on what a repair costs, which serves the same function as the scene, and states contributions in prose at the end.

## Formal section

ReAct opens "Consider a general setup of an agent interacting with an environment for task solving," then fixes notation in one paragraph. Least-to-most opens with a one-sentence definition of the method before any detail. Neither hedges before the definition arrives.

Applied here: the estimand section fixes notation first and states the mixture value before any discussion of bias.

## Experiments

ALFWorld opens: "We design experiments to answer the following questions: (1)... (4)". BabyAI: "We assess the difficulty of BabyAI levels by training a behavioral cloning baseline for each level." Self-consistency and least-to-most open by naming the task families and the comparison. The list of questions is the dominant device and it is answered in order.

Applied here: the confirmation design opens with the questions the two panels answer, in the order the results report them.

## Related work

All five use topic-lead sentences that name the line of work before the sentence about it: "Reasoning in language models.", "Compositional generalization.", "Interactive Text-Only Environments:". Comparisons are stated as differences in setting, not as deficiencies.

Applied here: the comparison-boundary section leads each paragraph with the class of method it is about, and states each boundary as a difference in what is being estimated.

## Limitations

Least-to-most carries a short standalone limitations section that opens on a concrete failure mode and gives an example. ReAct folds the same content into the conclusion after "Despite the simplicity of our method". Neither is defensive and neither is long.

Applied here: limitations is one short paragraph of fixed budget, stating the frame rather than apologising for it.

## Conclusion

Every paper opens with the introduced object: "We introduced least-to-most prompting to enable..."; "We have proposed ReAct -- a simple yet effective method for..."; "We present the BabyAI research platform..."; "We introduced ALFWorld, the first interactive text environment with aligned embodied worlds." The recap of results follows in one or two sentences, and a forward-looking sentence closes.

Applied here: the conclusion opens on the protocol rather than on a number, recaps both confirmations, and closes on what a runtime controller should do.

## Statements

Self-consistency carries separate short reproducibility and ethics statements. Both are kept here, and an AI-use statement is added as the venue requires.
