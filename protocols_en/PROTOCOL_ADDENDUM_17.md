# Addendum 17: Can per-item survival under random damage predict which knowledge compression deletes (preregistered, existing data only)

Written on 2026-10-30, before computing any per-item relation between random damage and compression. Approved by the investigator on the same day ("let's add this experiment too", translated).

## Motivation
The main text's "measure once, predict compression" has only been tested at the task × format level (two-axis prediction of compression, median 0.807). At the knowledge level (which facts within the same task are lost first), it has only been shown that "rare answers also break first under compression" (Addendum 16); it has not been tested whether "per-item survival measured under random damage predicts which facts compression deletes". Fig. 1e wants to draw a knowledge row, which requires this test.

## Data seen before this protocol (stated faithfully)
- Addendum 14 D2b: per-item survival agreement between random damages is low (two seeds of the same damage r_within 0.203–0.335; across damages r_cross median 0.263), using all usable measures, with survival scores computed per single damage and single seed.
- Addendum 16 H2: correlation between frequency and survival scores pooled over only the three compressions (capitals 0.411, SciQ 0.345); and the frequency correlation under each damage type.
- No per-item relation between random-damage survival and compression survival has been seen.

## Data and measures
- Per-item data analysis15_cache.pkl; frequencies freq15.json (capitals: country AND capital co-occurrence), freq16.json (SciQ, RETR: answer-string counts). Not changed.
- Only units answered correctly by the intact model under BARE are kept (capitals are merged by country, requiring all items for that country to be correct).
- Random-damage survival s_R: a model's mean BARE accuracy over all doses and seeds of the 4 random damage types (noise, zero, neuron, ld_rand).
- Compression survival s_C: a model's mean BARE accuracy over all doses of one compression (wanda, sparsegpt, gptq).
- One cell = model × task (capitals, SciQ) × compression method, at most 8 × 2 × 3 = 48. A cell requires at least 30 kept units and s_R, s_C both not all identical; otherwise it is not counted (number not counted is reported).

## Tests and verdict
- Threshold considerations: s_R pools all conditions of 4 damage types, so it is more stable than D2b's single-damage single-seed survival; merging by country also reduces noise. But D2b shows limited per-item reliability, so the main threshold is set at 0.30.
- **H1 Random damage predicts compression (main test)**: under each compression method, the median and proportion positive of Spearman(s_R, s_C) over all valid cells.
  - COMMIT (Fig. 1e adds the knowledge row; the main text writes "the facts lost first under random damage are also the facts compression deletes first"): all 3 compressions satisfy "median ≥ 0.30 and proportion positive ≥ 75%";
  - PARTIAL (write only the methods for which it holds, naming the exceptions): 2 satisfy;
  - STOP (Fig. 1e does not add the knowledge row; the main text does not write the knowledge-level prediction): otherwise.
- **H2 Does it go beyond frequency (secondary test)**: partial Spearman of s_R and s_C controlling for log frequency. Pass if under each compression the median ≥ 0.15 and proportion positive ≥ 75%; 3/3 COMMIT, 2/3 PARTIAL, otherwise STOP. Only if it passes may we write "measuring once gives more information than looking at corpus frequency alone".
- **Report only**: under each compression, the median using frequency alone to predict s_C (control for H1); the median of Spearman(s_R, s_C) under RETR (key-value lookup, answer in the prompt, no recall needed), used to judge whether per-item transfer is merely general item difficulty rather than knowledge.

## Lock
- analyze18.py md5: b2f7698c37921c8b22183438f11641f5

## Run log
1. 2026-10-30: md5 verified to match before running; analyze18.py run once, output analysis18.json; all 48 main cells valid, no cell excluded.

## Results (analysis18.json)
- **H1 Random damage predicts compression: COMMIT (3/3).** Median and proportion positive of Spearman(s_R, s_C): wanda 0.693 / 100%, sparsegpt 0.713 / 100%, gptq 0.580 / 100% (16 cells each).
  - By task: capitals wanda 0.569 (0.457–0.723), sparsegpt 0.642 (0.425–0.738), gptq 0.443 (0.204–0.651); SciQ wanda 0.735 (0.684–0.772), sparsegpt 0.718 (0.689–0.831), gptq 0.682 (0.555–0.736). Each cell has 76–78 countries or 132–188 items.
- **H2 Beyond frequency: COMMIT (3/3).** Median partial Spearman controlling for log frequency: wanda 0.620, sparsegpt 0.642, gptq 0.515; all cells positive.
- **Report only**: median using frequency alone to predict compression survival: wanda 0.361, sparsegpt 0.365, gptq 0.231, all lower than random-damage survival. RETR control median 0.117–0.133, far below capitals and SciQ: per-item transfer mainly appears in knowledge that requires recall, not in general item difficulty.
- Per the preregistered wording: Fig. 1e may add the knowledge row; the main text may write "the facts lost first under random damage are also the facts compression deletes first, and this predicts better than corpus frequency".
