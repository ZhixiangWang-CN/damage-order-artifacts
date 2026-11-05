# Addendum 6: Is format following the most fragile (preregistered)

Finalized on 2026-10-12; the investigator has confirmed the design and decided not to add instructions that "conflict with default habits" in this batch. The only thing done so far is the smoke test (two intact models, 40 items per measure). The predictions have not changed since the draft.

## Claim to be tested
Addendum 5 found after the fact: damaged models still choose correctly on option items, but no longer answer according to the requirement to "write out the option content". From this we propose: **following output-format instructions collapses before task ability.** This batch tests it as a prediction written down in advance.

## Design idea
Make the content so simple that it is almost never wrong, and vary only the output-format requirement. Each output is judged on two things separately:
- Whether the content is correct: the gold answer appears anywhere in the output (format ignored).
- Whether the format is correct: the form of the output meets the requirement (content correctness ignored).

This way "knows the answer but did not write it in the format" and "wrote it in the format but the answer is wrong" can be counted separately.

## Content tasks (all taken from the most robust measures so far)
| Task | Content | Note |
|---|---|---|
| FACT | Capitals of countries | Previously the most robust under all damages |
| ARITH1 | Add two two-digit numbers | Previously fairly robust, and the answer is a number |
| COPY | Copy a given pseudoword | Requires no knowledge; purely about output |

## Format instructions (the same batch of items, run once for each format)
| Code | Instruction | Example (answer is Paris) | Note |
|---|---|---|---|
| F0 | Give only the answer | Paris | Default form, as the control |
| F1 prefix | Start with "ANSWER:" | ANSWER: Paris | Common |
| F2 JSON | Output as {"answer": "..."} | {"answer": "Paris"} | Very common in training |
| F3 brackets | Put the answer in double square brackets | [[Paris]] | Rare, arbitrary |
| F4 end word | After answering, add the separate word DONE | Paris DONE | Rare, arbitrary |
| F5 repeat | Write the answer twice, separated by a comma | Paris, Paris | Rare, arbitrary |

Whether the format is compliant is judged by regular expressions; the rules are fixed before looking at outputs.

## A required control: output length
Formatted outputs are longer than bare answers and require writing more tokens correctly in a row. It is already known that "more steps means more fragile". So formats may look fragile simply because the output is longer.
Control task COPYLONG: copy a string of 6 pseudowords, with no format requirement, with output length no shorter than any of the formats above.
- If format following is more fragile than COPYLONG: what is fragile is "following instructions", not just "writing more tokens correctly".
- If the two are equally fragile: the claim must be downgraded to "the more tokens that must be written correctly in a row, the more fragile", which is the same thing as the step effect.

## References carried in the same batch
Free generation and forced choice of ARITH1/2/3, FACT and RETR (10 measures in total), used to determine "the most fragile task ability". Previously this was free generation of the sum of 4 numbers.

## Models, damage, doses
Qwen2.5-7B-Instruct and Mistral-7B-Instruct-v0.3, each with its locked doses. Three position-agnostic damages: noise (3 seeds), quantization, random layer dropping (3 seeds). 49 conditions per model. All raw outputs saved.
GPU: only idle cards on server_a, not card 7; server_b not used.

## Usability thresholds
- Format measures: the intact model's format compliance rate ≥ 0.8, otherwise that format does not enter the analysis.
- Content and reference measures: use the previous threshold (intact accuracy − chance level ≥ 0.3).

## Adjustments after the smoke test (all before formal results; predictions unchanged)
- Content judging added accent stripping: the intact model wrote the capital of Brazil as Brasília, which would be judged wrong under the original rule.
- One condition added to the usability threshold: for a "task × format" cell to enter the analysis, the intact model's format compliance rate and content accuracy in that cell must both be ≥ 0.8 (judged on the formal 200 items). Reason: in the smoke test Mistral had a content accuracy of only 0.25 on "add two numbers × write twice" (it computed the numbers wrongly), showing that this instruction already interferes with the task in the intact model.
- In the smoke test, the intact Mistral was seen to often append explanations after a compliant output (e.g. "[[52]]. The sum of 34 and 18 is 52."), which by the rules is judged format-noncompliant and may keep individual cells from reaching the threshold. The rule is not changed.
- The format-judging rules (regular expressions) have not changed since they were written into the code.

## Predictions (not changed after finalization)
- G1 format collapses before content: for each content task and each format F1–F5, the D50 of the format compliance rate is smaller than the D50 of content accuracy of the same task under F0. Holds in at least 80% of usable cells.
- G2 format collapses before the most fragile task ability: under each damage, at least half of the format measures have a D50 smaller than the D50 of the most fragile task-ability measure in the same batch. 6 cells in total (two models, three damages); at least 5 hold.
- G3 content is still there when it collapses: at the dose where the format compliance rate first falls below 0.5, content accuracy is still ≥ 0.8. Holds in at least 75% of usable cells.
- G4 not caused by length: the median of the format-compliance D50 over F1–F5 is smaller than the D50 of COPYLONG. At least 5 of 6 cells hold.
- G5 the general ordering is still there: combining format measures, COPYLONG and reference measures, the ordering correlations between all three pairs of damages are ≥ 0.7 (each model looked at separately).
- Exploratory: whether common formats (prefix, JSON) are more robust than arbitrary formats (brackets, end word, repeat); whether the way format fails differs between damages.

## What each outcome means
| Outcome | Conclusion |
|---|---|
| G1, G2, G3, G4 all hold | The claim holds: format following is the most fragile, and it is not caused by length |
| G1, G3 hold, G4 does not | Downgrade: the fragility of format can be explained by "having to write more tokens correctly" |
| G1 holds, G2 does not | Format is more fragile than content of the same task, but not more fragile than the most fragile task ability. The claim must be narrowed |
| G1 does not hold | The claim does not hold. The phenomenon seen in Addendum 5 may be specific to the instruction "write out the option content" (which conflicts with the default habit of "answering with a letter") |

The last row deserves attention: the instruction that went wrong in Addendum 5 works against the model's default habit. The five formats in this batch are all just additional requirements that do not conflict with default habits. If they turn out not to be fragile, what is truly fragile may be "suppressing a default habit", not format following in general. In that case another batch needs to be designed.

## Honesty rules
As before. Predictions that do not hold are reported as is. The format-judging rules are not changed after looking at outputs; output forms that are clearly missed are reported separately as "not covered by the rules".
