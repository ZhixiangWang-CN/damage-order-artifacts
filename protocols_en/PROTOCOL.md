# Damage Atlas Pilot (preregistered)

Written on 2026-10-11, before calibration and before any formal results were produced. The smoke test was run only on the intact model and on a single dose of each damage, with 20 smoke-test items, to confirm the pipeline and tune item difficulty; it does not enter the analysis.

## Question to answer
When several different kinds of damage are applied to the same model and evaluated with the same set of tests, is the order in which abilities are impaired structured, or is it all noise? If it is structured, do the damages share one ordering, or does each have its own?

This is a pilot. It only decides whether to scale up to a full atlas; it does not draw general conclusions about large models.

## Model
Qwen/Qwen2.5-7B-Instruct, bf16, single GPU, greedy decoding. Already cached on server_a, loaded offline.

## Tasks (all procedurally generated, no external datasets)
| Task | Content | Chance level |
|---|---|---|
| ARITH1 / 2 / 3 | Sum of 2 / 3 / 4 integers between 11 and 49 | 0.25 |
| FACT | Capitals of 80 countries | 0.25 |
| RETR | 12 "pseudoword: three-digit number" pairs in context; ask for one of them | 0.25 |
| AGREE | Subject-verb agreement with a distractor noun; fill in is or are | 0.5 |

Each task has three readout formats, with the same items:
- GEN: free generation, at most 12 tokens, exact match after parsing.
- CHOICE: the same prompt as GEN; compare the summed log-likelihood of the correct answer and 3 distractors (1 for AGREE) as continuations, and take the maximum.
- LETTER: list the options as A–D and compare only the logits of the letter tokens.

18 measures in total. Calibration uses 60 items (seed calib), the formal run uses 200 items (seed eval); the two item sets are different.

## Damage
Target: all two-dimensional weight matrices in all Transformer layers (attention and MLP). Embeddings, normalization and the output head are untouched.
- NOISE: add Gaussian noise to each matrix, standard deviation = r × the standard deviation of that matrix. 3 noise seeds in the formal experiment. This is true random damage, and the reference for the other damages.
- QUANT: symmetric uniform rounding quantization per output row, b bits, no calibration.
- LAYERDROP: delete the n consecutive layers immediately before the last layer, keep the last layer, no healing training.

## Dose
First run a wide grid on the calibration items (NOISE r: 0.01–0.30, 11 levels; QUANT: 8–2 bits; LAYERDROP: 1–16 layers).
Then lock 6 doses for each damage, by the rule: based on the "average corrected score of the GEN and CHOICE measures usable at baseline", cover 0.95 to 0.05 as evenly as possible. NOISE may take values between grid points; QUANT and LAYERDROP can only take integers, and where coverage is not possible, take what can be taken. Once locked, write to LOCKED_DOSES.json and do not change again.

## Analysis (locked)
- Corrected score c = (accuracy − chance level) / (intact accuracy − chance level). The chance level of GEN is taken as 0 (0.5 for the GEN of AGREE).
- Usable measure: on the formal items, intact accuracy − chance level ≥ 0.3. Unusable measures do not enter the ordering and are listed separately.
- D50: the position where c first falls below 0.5, linearly interpolated by dose index between the two adjacent doses. If it has not fallen to 0.5 by the maximum dose, it is recorded as "not reached" and placed at the most robust end; if it is already below 0.5 at the first dose, it is interpolated from the intact point.
- Fragility ordering: within each damage, rank the usable measures by D50 from small to large. The dose units of different damages differ, so only orderings are compared.
- The D50 of NOISE is the mean over the 3 seeds; the Spearman correlation of orderings between seeds is used to measure reliability.
- Bootstrap the items 1000 times to give each D50 a 95% interval.

## Predictions (written before the results; confidence in all of them is low)
- P1 reliability: the ordering correlation between each pair of the three NOISE seeds is ≥ 0.8.
- P2 depth: under both the GEN and CHOICE formats and under all three damages, D50 satisfies ARITH3 ≤ ARITH2 ≤ ARITH1. 6 cells in total; at least 5 predicted to hold.
- P3 format: for the same task, the D50 of GEN ≤ the D50 of CHOICE (generation is more fragile). Holds in at least three quarters of the usable task × damage cells.
- P4 cross-damage: the ordering correlations between all three pairs of damages are ≥ 0.7 (a shared ordering).

## Continue and stop rules
- **STOP**: the correlation in P1 is below 0.5 (the ordering itself is unreliable), or under any damage fewer than 6 measures are usable and fall below 0.5 within the dose range. Conclusion: "the pilot design cannot measure structure"; do not scale up.
- **Continue**: P1 holds, and under each damage at least 8 measures are usable and fall below 0.5 within the range. In this case, whether P4 comes out as "shared ordering" (correlation ≥ 0.7), "each has its own" (at least one pair with correlation ≤ 0.3, and the bootstrap intervals of the corresponding measures do not overlap), or in between, it all counts as structured and worth scaling up.
- In between: report as is, and the user decides.

## Exploratory (no predictions)
- Noise-equivalent dose: for each dose of QUANT and LAYERDROP and each measure, find the NOISE r that produces the same corrected score. If a damage is "just noise", the equivalent r given by all measures should be the same; the dispersion reflects whether that damage has its own signature.
- Differences between the LETTER format and the other two formats.

## Honesty rules
After seeing the formal results, do not change tasks, formats, doses, metrics or thresholds. Any deviation is recorded below. Predictions that do not hold are reported as is.

## Deviation log
All of the following deviations occurred during the calibration stage, before formal results were produced.

1. **Readout formats expanded from two to three, and arithmetic items changed to chained addition.** The smoke test (intact model, 20 items) showed: the original mixed multiplication-addition items had a baseline of only 5–25% under free generation; the "list options and answer with a letter" format was close to chance on arithmetic items. Therefore arithmetic was changed to the sum of 2/3/4 numbers, and the CHOICE format (forced choice by answer likelihood) was added. The task and format description above in this file is already the revised version.
2. **Noise calibration went beyond the original grid.** At the original grid maximum r=0.30 the average corrected score was still 0.65, not at the floor. Additional measurements were made at 0.35, 0.40, 0.50 and 0.60.
3. **The dose unit of quantization changed from number of bits to the number of positive quantization levels q** (b bits corresponds to q = 2^(b−1) − 1). Reason: the average corrected score was 0.77 at 4 bits (q=7) and already 0 at 3 bits (q=3); integer bits cannot resolve the intermediate process. Additional measurements were made at q=6, 5 and 4.
4. **Layer dropping was additionally measured at n=5.**
5. **The formal experiment used only one GPU** (other members of the group were occupying the other two); this does not affect the results, only the running time.

## Locked doses (LOCKED_DOSES.json)
Basis: the average corrected score of the 12 usable GEN and CHOICE measures on the calibration items.

| Damage | Locked doses | Average corrected score at calibration |
|---|---|---|
| NOISE r | 0.20, 0.26, 0.31, 0.345, 0.375, 0.42 | 0.93 (0.20), 0.81 (0.25), 0.65 (0.30), 0.39 (0.35), 0.09 (0.40); intermediate values are interpolation targets |
| QUANT q | 15, 10, 8, 7, 6, 5 | 0.99 (15), 0.77 (7), 0.31 (6), −0.05 (5); 10 and 8 not calibrated, taken between 15 and 7 |
| LAYERDROP n | 1, 2, 3, 4, 5, 6 | 0.84, 0.60, 0.44, 0.17, 0.06, 0.02 |

Quantization drops very steeply between q=7 and q=6, and integer levels cannot subdivide further, so coverage is uneven; per the protocol, "take what can be taken".
