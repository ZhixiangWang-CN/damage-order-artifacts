# Addendum 18: Using exponents measured under random damage to predict how much accuracy each ability retains after compression (preregistered, existing data only)

Written on 2026-10-30, before computing any accuracy prediction under compression conditions. Approved by the investigator on the same day ("let's try it", translated).

## Motivation
All "predicting compression" results in the main text are correlations (they only predict the order). Reviewers will ask whether we can predict how much accuracy a given ability retains after compression. The rank-one model c_m(d) = exp(−n_m h(d)) says damage acts only through a single strength h: on a compressed model, estimate h from the other abilities, then with the n_m measured under random damage, we should be able to compute the accuracy of a given ability.

## Data seen before this protocol (stated honestly)
- The R² of fitting the rank-one model separately in each compression cell (0.70–0.93), and the R² of the joint fit over 7 damages (0.72–0.82).
- The correlation between the task effect under compression and the random-damage reference (median 0.875), and the correlation of the two-axis prediction of the compression log exponent (0.807).
- The smoke test only ran the extraction of random-damage exponents (41–64 measures with exponents per model) and did not compute any prediction for compression conditions.
- No accuracy-level prediction error has been looked at.

## Data and method (analyze19.py)
- Per-condition accuracy: analysis15_cache.pkl (8 models, 64 measures, unchanged). Only usable measures above chance level on the intact model are used.
- Exponents: within each random-damage cell (noise, zero, neuron, ld_rand), fit the rank-one model by the main-text method (same window 0.03–0.97, same rank1 function), normalize by the geometric mean, then take the log mean over the random damages in which the measure was fitted; a measure has an exponent only if it was fitted under at least 2 random damages. **Compression data do not contribute to the exponents.**
- Test conditions: each dose of each model × compression method (wanda, sparsegpt, gptq), taking only informative doses: the median of c over measures is within 0.10–0.90.
- Prediction (leave-one-measure-out): for target measure t, use the other measures within the window under the same condition (at least 5) to estimate h by least squares, h = Σ n_j L_j / Σ n_j² (L = −log c); predicted accuracy = chance + (intact − chance) × exp(−n_t h).
- Baseline (uniform decline): the same leave-one-out estimate, but with all exponents set to 1.
- One cell = model × compression method, counted only with at least 20 prediction points; metrics are the mean absolute error (MAE) of accuracy and R² over all (measure × dose) points.

## Tests and verdict
- **H1 more accurate than "uniform decline" (main test)**: under each compression method, at least 4 valid cells, median MAE reduction relative to the baseline ≥ 25%, and the proportion of cells where the profile is more accurate ≥ 75%. 3/3 COMMIT, 2/3 PARTIAL, otherwise STOP.
  - COMMIT: the main text writes "the profile measured under random damage can predict the accuracy each ability retains after compression", as the closing of the last section of the results.
- **H2 absolute precision (secondary test)**: under each compression method, median MAE ≤ 0.10 (10 accuracy points). 3/3 COMMIT, 2/3 PARTIAL, otherwise STOP. Only if passed are the specific errors written as evidence of being "accurate"; if not passed, only the values are reported.
- **Report only**: the median R² of each method; GPTQ listed separately (it is known to be inconsistent across models).

## Lock
- analyze19.py md5: fb1bad3c35a8f6b64a9a9ea99bf1967a

## Run log
1. 2026-10-30: md5 verified as matching before the run; analyze19.py was run once, output analysis19.json. 23 cells valid; q32 × gptq has no informative dose (not counted).

## Results (analysis19.json)
- **H1 more accurate than "uniform decline": COMMIT (3/3).** Under each method, the profile is more accurate in all cells (100%).
  - wanda: median MAE 0.113 (0.091–0.130), baseline 0.208, reduction 48%; median R² 0.72 (0.67–0.83); 8 cells, 30 conditions, 1534 prediction points.
  - sparsegpt: 0.110 (0.097–0.143), baseline 0.217, reduction 45%; R² 0.73 (0.64–0.80); 8 cells, 25 conditions, 1265 points.
  - gptq: 0.116 (0.075–0.128), baseline 0.220, reduction 47%; R² 0.69 (0.55–0.73); 7 cells, 12 conditions, 557 points.
- **H2 absolute precision ≤ 0.10: STOP (0/3).** The median MAE of the three methods is 0.110–0.116, all slightly above the threshold.
- Per the preregistered wording: the main text may write "the profile predicts the accuracy each ability retains after compression, with an error about half that of the uniform-decline baseline" (H1), and may report the MAE values, but may not write "accurate to within 10 points".

## Post-run exploratory analysis (not preregistered, random damage only, does not change the verdict above)
2026-10-30, the investigator asked to "try a few more groups" (translated). To avoid picking a method on the compression data, comparisons were made only on random damage (analyze20_dev.py → analysis20_dev.json): each time one random damage is held out, and its accuracy is predicted using the exponents of the other three.
- The 6 variants (exponent source: per-cell mean / two-axis smoothing / a compromise between the two; strength estimate: least squares within the log window / least squares on accuracy) have median MAE 0.116–0.123, with almost no difference. Method details are not the source of the error, so no new variant was chosen and nothing was rerun on the compression data.
- Error floor: for the same random damage at the same dose, the mean difference in per-measure accuracy between two seeds is 0.117 (median, 32 model × damage cells, 0.079–0.409). The error of the compression prediction (0.110–0.116) is comparable to the difference between two repeats of the same random damage.
- 2026-10-30 later note: the seed-difference reference median of 0.117 was computed ad hoc on the command line at the time and not saved as a script; paper_nmi/figures/ed/build_ed5.py recomputed it with the same idea, giving median 0.1168, range 0.078–0.417 (the 0.079–0.409 above was the selection used at the time). The main text cites only the median; the reproducible version is that of build_ed5.py.
