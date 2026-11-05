# Addendum 21: Credible baselines and stricter hold-out for level 3–5 predictions (preregistered, existing data only)

Written on 2026-11-04, before computing any comparison in this protocol. Approved by the investigator on the same day ("B. New analysis, all with existing data; first write the preregistrations (Addendum 21, 22), then run", translated).

## Motivation (reviewer comments)
1. The accuracy predictions at levels 4 and 5 are compared only with "uniform decline", lacking three baselines: cross-model transfer of task identity, format effect only, and intact-model accuracy; the comparison should be made under the same calibration budget.
2. The strength h at levels 4 and 5 comes from other measures under the same compression condition (leave-one-measure-out), and other formats of the same task also take part in calibration. This should be changed to a fixed small set of calibration tasks, with whole-task (all formats included) hold-out, and unseen model families should be tested. (Unseen compression methods are already satisfied: compression data never enters the exponents.)
3. Level 3 (which facts are deleted) lacks a cheap baseline: the intact model's per-item answer margin; as well as a partial correlation controlling for both frequency and answer margin.

## Data seen before this protocol (stated honestly)
- All results of Addenda 17, 18 and 19 (analysis18/19/21.json).
- The smoke test outputs only: the size of each model's shared measure set S_m, the number of measures among the calibration tasks that fall in S_m, the intact answer-margin slope and the B3 slope g (all coming only from intact data or random damage), and a check that the slopes agree with analysis21. No prediction or comparison under any compression condition was computed.
- One change before locking: the calibration tasks were initially chosen by "largest total number of usable measures across the 8 intact models", which selected four simple generation tasks, ARITH1, FACT, RETR and COPY; under compression these mostly stay above the upper edge of the window, so calibration would fail. Before looking at any prediction, this was changed to selection by intact answer-margin quantile (see below), using only intact data.

## Method (analyze25.py)
- The predictor is the same as in Addendum 18: for each compression condition, estimate one strength h by least squares from the calibration measures; predicted accuracy = chance + (intact − chance) × exp(−n h); only informative doses are used (median of c over S_m between 0.10 and 0.90).
- Exponent sources (all use the same measure set S_m = measures that have exponents in both the random damage profile and the intact profile, the same set of calibration measures, the same calibration budget):
  - P: the target model's random damage profile (level 4).
  - I: intact model only (level 5, same as Addendum 19).
  - B1: format effect only, exp(mean format effect of the other models).
  - B2: task identity transfer, exp(mean task effect of the same task in the other models + format effect).
  - B3: intact accuracy, log n = g × z(the measure's normalized intact accuracy), where g is obtained by pooled least squares of the other models' random damage log exponents (centered) on their z-scores.
  - U: uniform decline (all exponents equal to 1).
- Calibration schemes:
  - LOMO: leave-one-measure-out (same as Addenda 18/19, ≥5 calibration measures), reported only.
  - FIX: 4 fixed calibration tasks (all formats included), predicting every format of every other task (whole-task hold-out, ≥3 calibration measures). Calibration tasks are ranked by the median over the 8 models of intact answer margin (within-model z-score), taking ranks 2, 6, 10 and 14: WINOGRANDE, OBQA, PIQA, COPY (determined by the script during the smoke test).
  - FAM: FIX, and the sources borrowed from other models (I, B1, B2, B3) exclude the target model's entire family (the 4 Qwen2.5 models are one family; each of the others is its own family).
- Fact level: intact per-item answer margin = correct candidate minus strongest distractor in the multiple-choice format (the same items as BARE; capitals averaged by country).

## Tests and decision (judged separately for each compression method wanda/sparsegpt/gptq; 3/3 COMMIT, 2/3 PARTIAL, otherwise STOP)
- **H1 level 4 beats credible baselines (FIX)**: median MAE reduction of P relative to each of B1, B2, B3 ≥ 10%, and cells where P is more accurate ≥ 75% (valid cells ≥ 4). A method passes only if it passes against all three baselines.
- **H2 level 5 beats credible baselines (FAM)**: I relative to B1, B2, B3 (all excluding the same family), as above.
- **H3 still beats uniform decline under fixed calibration and whole-task hold-out**: P vs U (FIX) reduction ≥ 25%, and I vs U (FAM) reduction ≥ 20%, with more-accurate cells ≥ 75% in both.
- **H4 level 3 goes beyond frequency and answer margin**: after controlling for log frequency and intact answer margin, median partial Spearman between random damage survival and compression survival ≥ 0.15, and positive cells ≥ 75% (valid cells ≥ 8).
- Reported only: median MAE of each source under each scheme; Spearman of predicting compression survival using answer margin alone; proportion of cells where random damage survival beats answer margin; H4 sensitivity after removing Italy and Bulgaria (frequency query suspected to have failed).
- Wording: if H1 passes → the main text writes "the random damage profile beats three baselines: task identity, format and intact accuracy"; if not → state honestly that it ties with or loses to which baseline. Same for H2. If H4 passes → "random damage survival gives more information than frequency and intact answer margin combined".

## Lock
- analyze25.py md5: 1d252e972f32f63700e136e5ff536848

## Run log
1. 2026-11-04: before running, the md5 was checked and matched; analyze25.py run was executed once, output analysis25.json. Calibration tasks WINOGRANDE, OBQA, PIQA, COPY; 23 valid cells under each of the three schemes (q32 × gptq has no informative dose).

## Results (analysis25.json)
Median MAE (wanda / sparsegpt / gptq):
| Scheme | P random damage profile | I intact only | B1 format only | B2 task identity transfer | B3 intact accuracy | U uniform decline |
|---|---|---|---|---|---|---|
| LOMO | 0.113 / 0.110 / 0.116 | 0.139 / 0.142 / 0.144 | 0.182 / 0.179 / 0.168 | 0.116 / 0.120 / 0.126 | 0.186 / 0.187 / 0.200 | 0.208 / 0.217 / 0.220 |
| FIX | 0.118 / 0.118 / 0.110 | 0.133 / 0.141 / 0.143 | 0.172 / 0.175 / 0.171 | 0.116 / 0.125 / 0.110 | 0.185 / 0.194 / 0.205 | 0.200 / 0.213 / 0.209 |
| FAM | 0.118 / 0.118 / 0.110 | 0.133 / 0.142 / 0.140 | 0.173 / 0.175 / 0.171 | 0.117 / 0.125 / 0.117 | 0.185 / 0.195 / 0.205 | 0.200 / 0.213 / 0.209 |

- **H1 level 4 beats credible baselines: STOP (0/3).** P passes against B1 and B3 for all three methods; against B2 (cross-model task identity transfer) it fails for all: under FIX, median MAE reduction 3–5%, more-accurate cells 57–75%.
- **H2 level 5 beats credible baselines: STOP (0/3).** I passes against B1 and B3; loses to B2: under FAM, MAE is 10–14% higher than B2, more-accurate cells 0–29%.
- **H3 still beats uniform decline under fixed calibration and whole-task hold-out: COMMIT (3/3).** P vs U (FIX) and I vs U (FAM, excluding the same family) pass under every method; with 4 fixed calibration tasks, whole-task hold-out and same-family exclusion, the error is almost the same as leave-one-measure-out (P 0.110–0.118, I 0.133–0.142).
- **H4 level 3 goes beyond frequency and answer margin: COMMIT (3/3).** Median partial Spearman after controlling for log frequency and intact answer margin: wanda 0.604, sparsegpt 0.629, gptq 0.513, all cells positive. The intact per-item answer margin by itself barely predicts which facts compression deletes (median Spearman 0.09–0.13), and random damage survival beats answer margin in 100% of cells. After removing Italy and Bulgaria the partial correlation is almost unchanged (wanda 0.605, gptq 0.517).
- Interpretation (stated honestly): at the accuracy level, the credible competitor is "how fragile the same task is in other models". It uses no data from the target model, ties with the target model's own random damage profile even when the same family is excluded (difference 3–6%), and beats the prediction using the intact model only. This is consistent with the main text's "task order is similar across models (0.83–0.91)": predictability at the accuracy level comes mainly from the identity of task and format, not from each model's own profile. Per the preregistered wording, the main text cannot say "the profile beats the task identity baseline"; levels 4 and 5 should be written as beating three baselines (uniform decline, format only, intact accuracy), tying with task identity transfer (level 4) or losing to it (level 5).
