# Writer-persona reading of the reference set

Five pre-2024 ICLR papers, read from their arXiv LaTeX sources, not from summaries:
BabyAI (ICLR 2019, 1810.08272), ALFWorld (ICLR 2021, 2010.03768), self-consistency
(ICLR 2023, 2203.11171), least-to-most prompting (ICLR 2023, 2205.10625), ReAct
(ICLR 2023, 2210.03629). `paper/writer_probe.py` measures each section; the raw
profile is `writer-profile.json`.

## What the sections do

**Introduction.** All five open on the phenomenon, not on the contribution, and
reach the contribution in the last paragraph. Agency is explicit throughout:
first-person markers run 51.9 (ReAct), 56.5 (ALFWorld), 57.9 (self-consistency)
and 65.4 (BabyAI) per hundred sentences. Least-to-most is the outlier at 13.3 and
is also the shortest introduction in the set. Median sentence length is 21 to 29
words, with a long-sentence share between 0.27 and 0.32 for the three later
papers. Hedging is present but light, 0 to 15.8 per hundred sentences.

**Method.** The method sections name what the authors do to the data, in order,
and carry the equation on a short lead-in. Median length drops to 20 to 25 words
and first-person markers drop with it, to 20 to 60. ALFWorld and BabyAI describe
the artifact impersonally; self-consistency and ReAct keep the first person.

**Results.** Every results section leads with a finding and puts the denominator
second. Sentences are the shortest in the paper, median 13 to 24 words, with a
long-sentence share of 0.04 to 0.31. First-person markers stay high, 23.4 to 87.9,
because the authors report what they ran rather than what was observed.

**Related work.** Topic-led paragraphs with a differentiator in the last clause.
Lowest first-person density in the paper, 6.2 to 27.3, and no hedging in four of
the five.

**Conclusion.** Six to seventeen sentences, median 18 to 34 words, and the highest
hedging in the paper, 5.9 to 41.7, reserved for future work rather than for
findings.

**Limitations.** Only least-to-most has one: 163 words, eight sentences, and a
first-person density of 62.5. It names what the method does not do and does not
re-argue the results.

**Floats.** Captions state what the reader should take from the float. No paper in
the set uses a double-column float except one figure in ALFWorld.

## What this manuscript changed

| Section | before | after | reference band |
| --- | --- | --- | --- |
| Introduction, first person | 19.5 | 46.2 | 51.9-65.4 |
| Introduction, median words | 17 | 20 | 21-29 |
| Study-1 results, first person | 0.0 | 23.1 | 23.4-87.9 |
| Study-1 results, long share | 0.50 | 0.08 | 0.04-0.31 |
| Study-1 analysis, median words | 34 | 23 | 13-24 |
| Study-2 results, first person | 0.0 | 100.0 | 23.4-87.9 |
| Study-2 analysis, first person | 0.0 | 21.1 | 23.4-87.9 |
| Transfer section, long share | 0.38 | 0.18 | 0.04-0.31 |

Both results sections now lead with the finding and give the denominator second,
which is the move every paper in the set makes. Hedging was left at its low level
because the frozen tests already carry the uncertainty and the reference set
reserves hedging for future work.

Section 2 keeps a median of 14 words against a reference band of 18 to 27. It is
the definitional section, and merging its sentences costs precision, so three
merges were made and the rest were left.

Vocabulary coverage moved from 70.70 to 70.84 percent against a leave-one-out
range of 79.3 to 88.7 within the reference set. The shortfall is carried by this
study's own objects, which are retained; no word was substituted to move the
statistic, and no perplexity target or author imitation was used.
