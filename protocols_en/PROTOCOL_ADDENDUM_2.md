# Addendum 2: A second model (preregistered)

Written on 2026-10-12, before the Mistral calibration and formal results were produced. The only thing already done is a smoke test (the intact model and a single dose of each damage, 20 smoke-test items), used to confirm that the code runs on this model.

## Question
On Qwen2.5-7B we saw: Gaussian noise, quantization and random layer deletion give the same damage ordering; position-specific layer deletion each has its own signature. Is this a property of this one model, or does it also hold for a model from a different family?

## Model
mistralai/Mistral-7B-Instruct-v0.3 (32 layers), bf16, greedy decoding, loaded offline. Tasks, the three readout formats, evaluation items and correction method are exactly the same as in the main protocol. The end token for the forced-choice format is changed to each model's own end token (for Qwen it is the same as before).

## Conditions
- NOISE: 6 doses × 3 seeds.
- QUANT: 6 doses (unit: number of quantization levels q).
- Random layer deletion ld_rand: n ∈ {1,2,3,4,5,6,8,10} × 3 seeds.
- Middle-segment deletion ld_mid, late-segment deletion ld_late: same set of n.
- 1 intact model. 67 conditions in total.

The noise and quantization doses need to be calibrated separately for this model (the smoke test showed it is much more sensitive to quantization than Qwen). Calibration uses 60 items with a wide grid: NOISE r from 0.05 to 0.60 in 10 levels; QUANT q ∈ {31,15,10,8,7,6,5,4,3}. The locking rule is the same as in the main protocol: based on the mean corrected score over the usable GEN and CHOICE measures, cover 0.95 to 0.05 as evenly as possible, written to LOCKED_DOSES_mistral.json. Layer deletion is not calibrated.

## Analysis
Same as the main protocol and Addendum 1: the threshold for usable measures (intact accuracy − chance level ≥ 0.3, judged on this model's formal items), the definition of D50 (by dose index for noise and quantization, by number of layers for layer deletion), Spearman correlation, and for random conditions take the seed mean and report the correlation between seeds.
Cross-model comparisons use only measures usable for both models.

## Predictions (written before the results)
- B1 Reliability: the pairwise ordering correlations among the three noise seeds are ≥ 0.8.
- B2 Step-count effect: of the 6 cells formed by noise, quantization, random layer deletion × GEN, CHOICE, it holds in at least five sixths of the cells where the arithmetic measures are usable.
- B3 Format effect: for the same task, GEN's D50 ≤ CHOICE's D50, holding in at least three quarters of the usable cells.
- B4 Generic ordering within this model: the ordering correlations of noise vs. quantization, noise vs. random layer deletion, and quantization vs. random layer deletion are all ≥ 0.7.
- B5 Position matters: the ordering correlation of middle vs. late is < 0.7.
- B6 Cross-model: the correlation between Mistral's noise ordering and Qwen's noise ordering (on commonly usable measures) is ≥ 0.7.
- B7 Position signature across models: in both models, "context retrieval collapses earlier under middle-segment deletion than under late-segment deletion, and capital-city QA the opposite". That is, RETR_GEN's D50 middle < late, and FACT_GEN's D50 late < middle, holds on Mistral as well.

B4 and B6 are the key ones. If B4 fails, the "generic ordering" does not exist even within Mistral; if B6 fails, each model has its own generic ordering.

## Honesty rules
Same as before. If fewer than 8 measures are usable, report as is that "this task set is not suitable for this model", and do not swap in other tasks to serve as the main result.

## Locked doses (LOCKED_DOSES_mistral.json, written before the formal results)
Basis: mean corrected score over the 11 usable GEN and CHOICE measures on the calibration items.

| Damage | Locked doses | Mean corrected score at calibration |
|---|---|---|
| NOISE r | 0.08, 0.10, 0.12, 0.14, 0.16, 0.19 | 1.00 (0.05), 0.81 (0.10), 0.26 (0.15), −0.06 (0.20); the rest are interpolation targets |
| QUANT q | 10, 7, 6, 5, 4, 3 | 0.97 (10), 0.77 (7), 0.78 (6), 0.80 (5), 0.52 (4), −0.12 (3) |

Notes:
- This model's noise tolerance is only about half of Qwen's (Qwen starts to drop at 0.20; Mistral has already hit bottom at 0.20).
- Under quantization the mean score barely moves between q=7 and q=5, and drops to the bottom in one step between q=4 and q=3; integer levels cannot be subdivided. Coverage is uneven; handled as "take what can be taken".
- One point already seen during calibration, and therefore no longer blind: under noise, Mistral's free generation for capital-city QA collapses first (0 at r=0.10), whereas on Qwen it is one of the most robust. This is unfavorable for B6. The predictions are not changed.
- On the calibration items, AGREE_CHOICE's intact accuracy is only 0.53, close to chance level; it is very likely unusable.
