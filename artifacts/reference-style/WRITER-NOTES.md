# Writer-persona reading of the pre-2024 ICLR reference set

Five papers accepted to ICLR before 2024, read from their arXiv sources rather than from memory: BabyAI (1810.08272, ICLR 2019), ALFWorld (2010.03768, ICLR 2021), self-consistency (2203.11171, ICLR 2023), least-to-most prompting (2205.10625, ICLR 2023), and ReAct (2210.03629, ICLR 2023). They are the topical neighbours of this manuscript: two build interactive text environments for grounded agents, three introduce a decoding or prompting protocol and defend it with a measurement design. Each is read section by section for its voice, its opening move, how it states a claim, how it hedges, and what vocabulary carries it. Section lengths, float conventions and a vocabulary comparison are in `AUDIT-ICLR.md`.

## ReAct (ICLR 2023)

**Voice.** Expository and analogical. It explains before it measures, and it prefers a concrete scene to an abstract definition.

**Abstract.** Opens on a tension between two established capabilities, not on the contribution: reasoning and acting "have primarily been studied as separate topics." The proposal arrives in the third sentence.

**Introduction.** Opens on human cognition and a kitchen, with the inner monologue quoted verbatim. The gap is reached only in the second paragraph.

**Method.** "Consider a general setup of an agent interacting with an environment for task solving." Notation is fixed in one paragraph and not hedged before it arrives.

**Experiments.** Signposting verbs carry the structure: "We begin with knowledge-intensive reasoning tasks", "We also test on two language-based interactive decision-making tasks."

**Related work.** Topic-lead sentences that name the line of work first.

**Conclusion.** "We have proposed ReAct -- a simple yet effective method for..." followed by the results in one sentence, then a limitation folded in after "Despite the simplicity of our method."

**Adopted here.** The signposting verbs and the conclusion's opening move.

## ALFWorld (ICLR 2021)

**Voice.** Scenario-first and infrastructural. It sells a capability by showing someone using it.

**Abstract.** Opens on a request a person could make, in italics, then on what humans do with it, then on the gap: "existing work does not yet provide the infrastructure."

**Introduction.** A direct question to the reader: "Consider helping a friend prepare dinner in an unfamiliar house."

**Experiments.** The dominant device in the set: "We design experiments to answer the following questions: (1)... (4)", answered in order. The ablation section repeats it.

**Related work.** Enumerated differentiators: "First... Secondly... Thirdly."

**Conclusion.** "We introduced ALFWorld, the first interactive text environment with aligned embodied worlds."

**Adopted here.** The numbered question list now opens the confirmation design, and the results sections answer it in order.

## Self-consistency (ICLR 2023)

**Voice.** Hypothesis-driven and plain. It states what it believes, then tests it, and it never dresses a number.

**Abstract.** Capability, then "In this paper, we propose a new decoding strategy", then the mechanism in two sentences, then results with margins in parentheses.

**Introduction.** The limitation, the prior fix, then a worked example quoted in full.

**Method.** Opens with an aphorism -- "A salient aspect of humanity is that people think differently" -- and then states the hypothesis in one testable sentence: "we hypothesize that correct reasoning processes, even if they are diverse, tend to have greater agreement in their final answer than incorrect processes."

**Experiments.** The finding comes before the setup: "We find that self-consistency robustly improves reasoning accuracy for every language model considered."

**Related work.** Each topic lead closes with an explicit differentiator: "Compared to prior work, self-consistency is applicable to a wide range of reasoning tasks without any additional supervision or fine-tuning."

**Conclusion.** "We introduced a simple yet effective method called self-consistency, and observed that..." then one named limitation with a practical mitigation attached: it costs compute, so try five or ten paths.

**Statements.** Short standalone reproducibility and ethics statements.

**Adopted here.** The finding-first opening for the results, the differentiator clause in the comparison section, and the limitation-with-consequence form.

## Least-to-most prompting (ICLR 2023)

**Voice.** Contrastive. It defines itself against what came immediately before and says so in the first three sentences.

**Abstract.** "Chain-of-thought prompting has demonstrated remarkable performance... However, it tends to perform poorly on tasks which require solving problems harder than the exemplars." Then "To overcome this challenge... we propose."

**Introduction.** An enumerated three-way contrast between human and machine learning.

**Method.** One-sentence definition, then two numbered stages, then a worked example.

**Results.** "We present least-to-most prompting results for A, B, and C, and compare it with chain-of-thought prompting."

**Limitations.** A short standalone section that opens on a concrete failure mode with a quoted example, then generalises from it. It is not defensive and it is not long.

**Conclusion.** "We introduced least-to-most prompting to enable..." and a reflective close that questions the paradigm rather than defending it.

**Adopted here.** The limitations form -- one concrete bound, stated once, with the number that fixes it -- and the short reflective close.

## BabyAI (ICLR 2019)

**Voice.** Platform-building and motivation-heavy. It argues for why the measurement matters before it measures anything.

**Abstract.** Desirability, then the obstacle -- "given the lack of sample efficiency in current learning methods, reaching this goal may require substantial research efforts" -- then "We introduce the BabyAI research platform."

**Introduction.** "How can a human train an intelligent agent to understand natural language instructions?" answered from a technological and a scientific perspective.

**Related work.** Enumerated differentiators again: "First... Secondly... Thirdly... Most importantly."

**Experiments.** "We assess the difficulty of BabyAI levels by training a behavioral cloning baseline for each level."

**Conclusion.** "We present the BabyAI research platform to study language learning with a human in the loop."

**Adopted here.** The habit of stating what a measurement is for before reporting it, which the design section now does with its question list.

## What the set has in common, and what this manuscript does with it

Every abstract opens on an established capability and turns on a single word; none opens on its own contribution. Every conclusion opens on the introduced object. Limitations are short, concrete, and carry the number or example that bounds them; none is a list of caveats. Related work names the line of work first and ends on a differentiator. Results lead with the finding. Contributions are prose, not bullets.

This manuscript keeps its own subject vocabulary -- continuation, draw, prefix, arm, estimand, pathwise -- because those words carry the distinctions it reports; `AUDIT-ICLR.md` measures the resulting distance and records that no word was substituted to move that statistic.
