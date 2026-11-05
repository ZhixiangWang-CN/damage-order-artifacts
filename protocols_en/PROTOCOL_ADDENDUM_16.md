# Addendum 16: Does the frequency effect hold separately under each damage type (preregistered, existing data only)

Written on 2026-10-29, before computing any per-damage statistics. Approved by the investigator on the same day ("add this one first", translated).

## Motivation
The main line of the manuscript is "damage only determines the speed; what is lost is determined by learning". The frequency effects of Addendum 14 (D2a, capitals) and Addendum 15 (F1, SciQ) were both obtained by pooling all conditions of the 7 damage types to compute the survival score; they did not show that the effect holds separately under each damage type. If there is no frequency effect under some damage (especially some compression), the claim "independent of damage" does not hold; conversely, if it holds under all three compressions, "compression deletes long-tail knowledge first" goes from a conjecture to a data-backed conclusion.

## Data seen before this protocol (stated faithfully)
- All pooled results of Addenda 14 and 15 (capitals 0.423, SciQ 0.383, RETR 0.079, etc.), and the post-hoc guessing-bias check of Addendum 14.
- No frequency–survival relation split by individual damage type has been seen.

## Data and measures (same as Addenda 14 and 15; only the range of conditions for the survival score is changed)
- Per-item data: analysis15_cache.pkl; frequencies: freq15.json (capitals: country AND capital co-occurrence), freq16.json (SciQ, RETR: answer-string counts). Neither is changed.
- Survival score: a model's mean BARE accuracy over all doses and seeds of **a single damage type**; only items answered correctly by the intact model under BARE are kept (capitals are merged by country, requiring all items for that country to be correct).
- One cell = model × task (capitals, SciQ) × damage. A cell requires at least 30 kept units and survival scores that are not all identical; otherwise it is not counted (the number of uncounted cells is reported).
- For each cell compute Spearman(log frequency, survival score).

## Tests and verdict
- The survival score of a single damage type comes from only 3–24 conditions, noisier than when pooled, and correlations will be depressed; therefore the threshold is set at 0.20 (it was 0.30 when pooled).
- **H1 Independent of damage (main test)**: for each of the 7 damage types, take the median Spearman and the proportion positive over all valid cells under that damage (at most 8 models × 2 tasks = 16).
  - COMMIT (the main text writes "the frequency effect holds under every damage type"): all 7 damage types satisfy "median ≥ 0.20 and proportion positive ≥ 75%";
  - PARTIAL (the main text writes "holds in k of 7", naming the exceptions): at least 5 satisfy, and at least 2 of the three compressions satisfy;
  - STOP (the main text does not write "independent of damage", only the pooled result): otherwise.
- **H2 Pooled compression (report only)**: compute the survival score using only the conditions of the three compressions wanda, sparsegpt and gptq; per task, the median and number positive over 8 models.
- **H3 Control (report only)**: the median for RETR under each damage type.

## Lock
- analyze17.py md5: 14ac67cd163ee371a2723f51cabeb060

## Run log
1. 2026-10-29: analyze17.py (md5 as above) run once, output analysis17.json; all 112 cells valid, no cell excluded for item count or for lack of variation in survival score.

## Results (analysis17.json)
- **H1 Independent of damage: COMMIT.** All 7 damage types pass the threshold (three compressions 3/3). Median and proportion positive over the 16 cells (8 models × capitals, SciQ) under each damage type: noise 0.300 / 94%, zero 0.319 / 100%, neuron 0.406 / 100%, ld_rand 0.304 / 100%, wanda 0.361 / 100%, sparsegpt 0.365 / 100%, gptq 0.231 / 88% (weakest, still passes).
- **H2 Three compressions only:** capitals median 0.411 (0.303–0.538, 8/8 positive); SciQ 0.345 (0.318–0.474, 8/8 positive).
- **H3 RETR control:** median 0.010–0.080 under each damage type.
- Per the preregistered wording: the main text may write "the frequency effect holds under every damage type", and may write "compression deletes long-tail knowledge first" as a data-backed conclusion (no longer a conjecture).
