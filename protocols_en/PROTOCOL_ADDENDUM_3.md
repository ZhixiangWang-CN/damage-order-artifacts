# Addendum 3: Redoing the Mistral noise (preregistered)

Written on 2026-10-12, before this batch's calibration and formal results were produced.

## Why redo it
In Addendum 2, Mistral's noise doses were locked based on a single calibration seed; the three formal seeds were barely damaged within that range, and only 1 of 13 measures crossed 0.5. The predictions involving the noise ordering were not effectively tested.

## Procedure
Model, tasks, formats, evaluation items, correction method and usable measures (the 13 judged in Addendum 2) are all unchanged.
- Calibration: 60 calibration items, noise strength r ∈ {0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50}, 4 seeds (11–14, not used in the formal experiment).
- Locking rule: based on "the mean corrected score over the usable GEN and CHOICE measures, after averaging over the 4 seeds", take 6 doses covering 0.95 to 0.05 as evenly as possible; values between grid points are allowed. Written to LOCKED_DOSES_mistral_noise.json.
- Formal: 6 doses × 5 seeds (1–5), 200 evaluation items.
- Also report, at each dose, the range of the mean corrected score across the 5 seeds, to quantify the degree to which "noise of the same strength has different consequences".

## Analysis
D50 is computed by dose index, on the curve averaged over the 5 seeds. Ordering correlations use Spearman. Comparisons with quantization and random layer deletion use the results of Addendum 2; comparisons with Qwen use the noise ordering from the main experiment, taking only measures usable for both models.

## Predictions (written before the results)
- C1 Reliability: split the 5 seeds into every 2-vs-3 combination, average each half and compute its ordering; the median of the correlations over the 10 splits is ≥ 0.8.
- C2 Generic ordering within this model: noise vs. quantization ≥ 0.7, and noise vs. random layer deletion ≥ 0.7.
- C3 Cross-model: the correlation between Mistral's noise ordering and Qwen's noise ordering is ≥ 0.7.
- C4 Step-count effect: under noise, both the GEN and CHOICE cells satisfy "4 numbers ≤ 3 numbers ≤ 2 numbers", and in at least one cell the three are not all "not reached".
- Known unfavorable information: in the calibration of Addendum 2 (single seed), Mistral's free generation for capital-city QA collapsed first under noise, the opposite of Qwen. The predictions are not modified because of this.

## Stopping rule
If in the formal experiment fewer than 8 usable measures cross 0.5 within the range, report "still not measurable", and do not add a third round.

## Locked doses (LOCKED_DOSES_mistral_noise.json, written before the formal results)
Calibration (mean over 4 seeds, 11 usable GEN and CHOICE measures): r=0.10 → 0.96; 0.15 → 0.91; 0.20 → 0.80; 0.25 → 0.48; 0.30 → 0.08; 0.35 → −0.05.
The four seeds are close to one another (range no more than 0.19 at each dose).
Locked: 0.12, 0.20, 0.225, 0.25, 0.275, 0.31.

Note: none of these 4 seeds reproduced the early collapse of calibration seed 7 in Addendum 2 (it reached 0.26 already at 0.15). Together with formal seeds 1–3, only seed 7 out of 8 seeds behaved this way. For now it is treated as an outlier seed; the cause has not been investigated.
