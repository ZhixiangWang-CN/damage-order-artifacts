# Addendum 5: Output inspection for generative formats (preregistered)

Written on 2026-10-12, before any output of this batch was viewed.

## Question
In Addendum 4, generative formats were more fragile than forced choice, and the "write out the option content" format was the most fragile, with its corrected score falling far below chance level. Is this "fragility" a loss of task ability, or is the model no longer answering in the required format? At the time only right/wrong was saved, not the raw outputs, so the two could not be distinguished.

## Procedure
Qwen2.5-7B-Instruct, with items, prompts and scoring rules exactly the same as in Addendum 4; only the generative formats are rerun, saving every raw output.
- 13 conditions: intact; noise r = 0.20, 0.31, 0.375, seeds 1 and 2; quantization q = 10, 7, 6; random layer deletion n = 2, 4, 6, seed 1.
- 9 measures: GENOPT for ARC, CSQA, MMLU_CALC, MMLU_MEM; GEN for SCIQ, BOOLQ, ARITH3, FACT; COT for GSM8K (100 items).

## Output classification (rules fixed before viewing outputs, executed by script)
Normalization = lowercase, remove characters other than letters, digits and spaces. Option order is reconstructed using the same random number as the prompt.

GENOPT:
- Correct text: the normalized output starts with the first 25 characters of the normalized text of the correct option (i.e., the scoring rule of Addendum 4).
- Correct letter: the output starts with a single option letter (optionally with parenthesis, period or colon), and that letter corresponds to the correct option. Includes "letter plus text".
- Wrong text: starts with the first 25 characters of the normalized text of a wrong option.
- Wrong letter: starts with an option letter corresponding to a wrong option.
- Other: none of the above, including empty output.
Order of judgment: first strip a leading "letter plus punctuation", then look at the text; if after stripping it matches some option's text, classify by text and separately record "had a letter".
Format failure = correct letter (letter only, no corresponding text) + wrong letter (same) + other. Lenient correct = correct text + correct letter.

Numeric (ARITH3's GEN): correct; wrong number (a number is parsed but does not equal the answer); no number.
GSM8K's COT: correct; has "answer is" but the number is wrong; no final answer (no "answer is").
Phrase (GEN for SCIQ, FACT): correct; contains answer (normalized output contains the normalized answer but not at the start); other answer (non-empty and no more than 6 words); non-answer (empty or more than 6 words). Lenient correct = correct + contains answer.
BOOLQ: correct; answered the opposite; no yes/no.

## Predictions (written before the results)
- E0 Replication: the median absolute difference between this batch's strict-correct accuracy and the accuracy for the same condition and measure in Addendum 4 is ≤ 0.03.
- E1 GENOPT errors are mainly format failures: pooling all errors across the 12 damaged conditions and the 4 GENOPT measures, format failures account for > 50%.
- E2 Lenient scoring closes most of the gap: on the "condition × task" cells where the forced-choice corrected score in Addendum 4 lies between 0.2 and 0.9, the median of (lenient corrected score − strict corrected score) / (forced-choice corrected score − strict corrected score) is ≥ 0.5.
- E3 The fragility of numeric tasks is real: among the errors of ARITH3's GEN and GSM8K's COT under damaged conditions, "answered a wrong number" is the majority (> 50%), and "gave no number / no final answer" is the minority.
- Exploratory: how much lenient scoring recovers for SCIQ and FACT; whether the error composition differs across damage types.

## Honesty rules
The classification rules are not modified after seeing the outputs. If the rules clearly miss some common output form, this is reported separately as "not covered by the rules", and the rules are not revised to recompute the main results.
