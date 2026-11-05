# Addendum 19: Without damaging the target model, predicting post-compression accuracy on each ability from the intact model alone (preregistered, existing data only)

Written on 2026-10-30, before computing any prediction of this protocol. Approved by the investigator on the same day ("give it a try", translated).

## Motivation
Addendum 18 showed that the profile measured under random damage can predict post-compression accuracy (error about half of the baseline). Of greater practical value is whether one can predict without even running random damage, looking only at the intact model. The existing result of this kind in the main text (answer margin plus the format effect of other models, predicting compression's log exponents with correlation 0.70) is exploratory and at the correlation level. This protocol tests the accuracy level.

## Data seen before this protocol (stated faithfully)
- Exploratory result in the main text: answer margin plus format effect predicts held-out random damage 0.80 and compression 0.70 (log-exponent correlation).
- The answer-margin definition of Addendum 12 and the per-model correlation between answer margin and task effect (0.58–0.73).
- All results of Addendum 18 (random-damage profile predicts compression accuracy with MAE 0.110–0.116, baseline 0.208–0.220).
- The smoke test only computed the task answer margins of the 8 models when intact (16 tasks per model), and did not compute any prediction.
- One change before locking: answer margins have different scales across models (e.g., 11–42 for q14, −1–17 for llama), so before locking it was changed to within-model standardization (z-scores), consistent with the practice of Addendum 12.

## Method (analyze21.py)
- For target model m, no damage data of m are used:
  - Task answer margin margin_t: the score difference between the correct candidate and the strongest distractor under the intact model's multiple-choice format, averaged over all evaluation items; z-scored within m's 16 tasks.
  - Slope: pooled least squares (through the origin) of the random-damage task effects of the other 7 models on their own answer-margin z-scores.
  - Format effect: the mean of the random-damage format effects of the other 7 models.
  - log n_(t,f) = slope × z(margin_t) + format effect_f.
- The accuracy prediction is exactly the same as in Addendum 18 (analyze19.predict_condition: leave one measure out, estimate one intensity per compression condition, use only informative doses 0.10–0.90).
- Baseline: all exponents equal to 1 (uniform decline).
- Report only: the MAE of the Addendum 18 random-damage profile on the same cells, as a reference.

## Tests and verdict
- **H1 Even the intact model alone is more accurate than uniform decline (main test)**: under each compression method, with at least 4 valid cells, the median reduction in MAE relative to the baseline is ≥ 20%, and the proportion of cells that are more accurate is ≥ 75%. 3/3 COMMIT, 2/3 PARTIAL, otherwise STOP.
  - COMMIT: the last level of Section 4 of the Results in the main text writes "without damaging the model, looking only at the intact model, one can also predict post-compression accuracy on each ability".
  - The threshold is lower than the 25% of Addendum 18, because the intact model only provides the task answer margin and the generic format effect, less information than the random-damage profile; the threshold was set before seeing any results.

## Lock
- analyze21.py md5: 1ca42bcf569192c563433ededa18ba15

## Run log
1. 2026-10-30: md5 verified to match before running; analyze21.py run once, output analysis21.json. 23 cells valid (q32 × gptq has no informative dose).

## Results (analysis21.json)
- **H1 Even the intact model alone is more accurate than uniform decline: COMMIT (3/3).** Under each method all cells are more accurate (100%).
  - wanda: median MAE 0.138 (0.121–0.161), baseline 0.220, reduction 38%; R² 0.63 (0.54–0.72); 8 cells, 1633 points.
  - sparsegpt: 0.141 (0.112–0.169), baseline 0.213, reduction 36%; R² 0.62 (0.40–0.67); 8 cells, 1415 points.
  - gptq: 0.143 (0.088–0.176), baseline 0.212, reduction 36%; R² 0.56 (0.49–0.70); 7 cells, 643 points.
- The leave-one-model-out slope barely changes across the 8 runs (−0.584 to −0.567): the larger the answer margin, the smaller the exponent (the more damage-resistant).
- Reference (report only): the random-damage profile on the same cells has MAE 0.110–0.116. Looking only at the intact model gives about 3 points more error, retaining about three quarters of the improvement of the random-damage profile.
- The baseline MAE differs slightly from Addendum 18 (0.212–0.220 vs. 0.208–0.220), because the two have different sets of measures (this protocol requires the task to have an answer margin and a format effect).
- Per the preregistered wording: the last level of Section 4 of the main text may write "without damaging the target model, looking only at the intact model, one can also predict post-compression accuracy on each ability (error about 14 points, more than one third lower than the uniform-decline baseline)".
