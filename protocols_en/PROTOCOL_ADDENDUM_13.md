# Addendum 13: Can the answer margin under a format requirement explain format fragility (preregistered)

Written on 2026-10-28, before any measurement of the "formatted answer margin". Approved by the investigator on the same day ("Add one confirmation"). Per the stopping rule set in advance, this is the last additional experiment; no further experiments will be added after it.

## Disclosure first
- The hypothesis comes from post-hoc analysis (paper/POSTHOC_V13.md, Section 2): on the damaged outputs of 8 models, among items where "write twice" failed, the median proportion still answered correctly by answering directly under the same damage was 0.69, and most failures wrote a wrong answer following the template. This observation used the damaged data of all 8 models, so **there is no completely unseen model**; this batch is judged on all 8 models.
- The predicted quantity (the fragility exponent of each measure) has long been seen. The predictor (the formatted answer margin) has never been computed on any model; there is only one formula, with no free parameters, locked before measurement.
- The push of Addendum 12 can only distinguish "direct answer vs. template format", and cannot tell JSON from write-twice; this is precisely the key test of this hypothesis.

## Hypothesis (format interferes with the answer)
Damage first breaks the ability to "work out the answer under a format requirement". The thinner the intact model's margin for giving the correct answer under the prompt of format f, the more fragile the measure; the write-twice prompt makes the answer margin thinner than direct answering, and so it is the most fragile.

## Measurement (margin_fmt.py, run only on intact models)
For each model, 16 tasks, BARE / JSON / TWICE, 300 evaluation items: under the prompt chat(body(it, f)) of format f, the summed log probability of each candidate answer written out in format f (the candidate replaces the gold answer in atlas_full.target, followed by the end-of-turn token). Candidate 0 is the gold answer. BARE's prompt and continuation are the same as the CHOICE score of Addendum 8, used as a consistency check (reported, does not enter the verdict).

## Prediction (analyze14.py)
- fmargin_m = −(mean over all evaluation items of [gold answer score − highest distractor score]), computed in the measure's own format.
- Measures, observed exponents and the predictor of Addendum 12 (z(push) + z(unformatted answer margin)) are taken exactly as in analyze13.rows_for, to allow direct comparison.

## Tests and verdict (written before measurement)
- **Q1**: within each model, the Pearson correlation between fmargin and the observed log n; median over 8 models.
- **Q2**: number of models in which the ordering of format means of fmargin exactly matches the ordering of observed format means.
- **Q3**: number of models in which the correlation of fmargin is higher than that of the Addendum 12 predictor.
- **COMMIT**: Q1 ≥ 0.80, Q2 ≥ 7/8, Q3 ≥ 6/8 → enters the main line of the main text, as the explanation of format fragility.
- **PARTIAL**: Q1 ≥ 0.65, Q2 ≥ 6/8, Q3 ≥ 5/8.
- **STOP**: all other cases; reported in the appendix as a negative result.
- Descriptive (does not enter the verdict): the correlation, per task, between the change in margin of a format relative to direct answering and the change in fragility (one each for JSON and TWICE); the correlation of z(push) + z(fmargin); the correlation using only the unformatted answer margin.

## Compute
Forward passes on intact models, about 16 × 3 × 300 × 2–5 candidate sequences per model. q32 on a single 80 GB card on server_b; gemma uses PUSH_BS=8. Only idle cards not on the reserved list are used.

## Lock
- margin_fmt.py md5: bd3b3bc9011980683b5d191f83a62b47 (the scoring function is taken from push.py, md5 c89335a01b4d54c0559482e6e146a421)
- analyze14.py md5: 5b3bf7c8468b4cedcab0d54e2ee5780c (tested with hand-constructed data: Q1 = 1 under perfect prediction; the Addendum 12 predictor reproduces q7 0.778)

## Run log
1. 2026-10-29 00:10: all 8 models completed (gemma with PUSH_BS=8).

## Results (locked analyze14.py, md5 5b3bf7c8468b4cedcab0d54e2ee5780c; output analysis14.json)
- **Q1**: median r = 0.540 (q15 0.45, llama 0.51, phi 0.50, q7 0.58, mis 0.57, gemma 0.42, q14 0.71, q32 0.75); Addendum 12 predictor 0.712, unformatted answer margin 0.529.
- **Q2**: format ordering exactly matches in 2/8 (only q14, q32).
- **Q3**: exceeds the Addendum 12 predictor in 0/8.
- **Verdict: STOP.** Reported in the appendix as a negative result.
- Descriptive: the median correlation, per task, between the change in margin of "write twice relative to direct answer" and the change in fragility is −0.28 (opposite direction); JSON 0.21.
- Design flaw (found after the run, recorded faithfully, verdict not changed): the write-twice score is the summed log probability of "candidate, candidate" written twice; the second copy is almost certain given the first, yet adds the gap between the gold answer and the distractors a second time, making the write-twice margin systematically too thick (in 6/8 models the write-twice margin is in fact the thickest). Therefore this test did not cleanly test the "format interferes with the answer" hypothesis: the conclusion is "does not hold, by this preregistered measure", not "the hypothesis is refuted". A more suitable measure (taking only the margin of the first answer) is a post-hoc correction; per the stopping rule no further experiment is added, and it is left as an open question in the text.
