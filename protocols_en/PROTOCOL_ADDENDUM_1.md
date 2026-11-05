# Addendum 1: Position of layer deletion (preregistered)

Written on 2026-10-11, before any results from this batch were produced. The models, tasks, formats, evaluation items and correction method of the main protocol PROTOCOL.md are all unchanged.

## Question
In the pilot, the damage ordering under layer deletion was clearly different from that under noise (correlation 0.55). But the pilot only deleted several layers immediately before the last layer. Is this "layer-deletion signature" a signature of layer deletion itself, or of those particular layers?

## Conditions
Qwen2.5-7B-Instruct has 28 layers in total; the first and last layers are always kept. Delete n layers, n ∈ {1, 2, 3, 4, 5, 6, 8, 10}, with no healing.
- Early segment ld_early: delete n consecutive layers starting from layer 2.
- Middle segment ld_mid: delete n consecutive layers in the middle.
- Late segment ld_late: delete n consecutive layers before the last layer (the same deletion scheme as in the pilot, rerun and extended to 8 and 10 layers, also serving as a replication check).
- Random ld_rand: randomly draw n layers from the middle 26 layers, 3 seeds.

48 conditions in total, each with 200 items × 18 measures. No separate calibration was done: the dose grid is the same for all four deletion schemes, and if it fails to cover the range, this is reported as is.

## Analysis
- D50 is in units of layers: the n at which the corrected score first drops below 0.5, linearly interpolated between the two adjacent levels, with the intact point recorded as n=0. If it has still not dropped below 0.5 at 10 layers, it is recorded as not reached.
- Usable measures follow the 16 of the main protocol.
- Ordering correlations use Spearman. For the random deletion scheme, take the mean curve over the 3 seeds, and report the correlation between seeds.
- Comparisons with noise use the noise ordering from the main experiment.

## Predictions (written before the results)
- A0 Replication: the correlation between the ordering under this batch's late-segment deletion at n=1–6 and the pilot's layer-deletion ordering is ≥ 0.9 (same deletion scheme, same item set; they should be nearly identical).
- A1 Position matters: the ordering correlations of late vs. middle and late vs. early are both below 0.7.
- A2 Random layer deletion resembles generic damage: the correlation between the random-deletion ordering and the noise ordering is ≥ 0.7.
- A3 Step-count effect: of the 8 cells formed by four deletion schemes × two formats, at least 7 satisfy "adding 4 numbers ≤ 3 numbers ≤ 2 numbers".
- A4 Capital-city QA is especially sensitive to late-segment deletion: the D50 (in layers) of FACT_GEN under late-segment deletion is smaller than under middle and early segments.
- Exploratory: which segment hurts most, the reliability across seeds of random deletion, and the dispersion of noise-equivalent doses across deletion schemes.

## Honesty rules
Same as the main protocol. After seeing the results, conditions, metrics and thresholds will not be changed. If any deletion scheme hits bottom entirely at n=1 or is still essentially intact at n=10, this is reported as is as "not measurable with this grid", and no additional conditions are added to serve as the main result.
