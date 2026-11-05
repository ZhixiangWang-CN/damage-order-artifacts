# Addendum 20: Can the knowledge axis and the format axis be separated (preregistered, existing data only)

Written on 2026-10-30, before computing any frequency–survival relation under the JSON and TWICE formats. Approved by the investigator on the same day ("let's try it", translated).

## Motivation
The Results section is intended to be written as three axes: task, knowledge, format. Separability of task and format is already established (additive model 89%). Knowledge (familiarity of the answer in the pretraining corpus) has only been tested under the BARE format, and there is no evidence that it is independent of format. This protocol tests two points: whether the knowledge effect is present under every format; and whether the order of formats is the same for both common and rare answers.

## Data seen before this protocol (stated faithfully)
- The BARE-format frequency effects of Addenda 14, 15 and 16 (capitals 0.423, SciQ 0.383; 0.23–0.41 under each damage type).
- The format order in the main text (Choose < Answer < JSON < Twice) and the intact accuracies of capitals and SciQ in each format (e.g., q15's SciQ TWICE intact is only 0.05, unusable).
- No per-item frequency–survival relation under the JSON or TWICE formats has been seen, nor the format order after splitting by frequency.

## Data and measures (analyze22.py)
- Per-item data analysis15_cache.pkl; frequencies freq15.json (capital co-occurrence), freq16.json (SciQ answer counts). Not changed.
- Survival score: a model's mean accuracy over all 7 damage types and all doses and seeds (same as Addenda 14–16). Capitals are merged by country (requiring all items for that country to be answered correctly by the intact model in that format).
- Only usable measures; a cell needs at least 30 units.

## Tests and verdict
- **H1 The knowledge effect does not vary with format (main test)**: for each of BARE, JSON and TWICE, the median and proportion positive of Spearman(log frequency, survival) over all valid cells (at most 8 models × 2 tasks). Each format requires valid cells ≥ 8, median ≥ 0.20, positive ≥ 75% (the same as the single-damage threshold of Addendum 16).
  - COMMIT: all 3 formats pass; PARTIAL: BARE and one other pass; STOP: otherwise.
- **H2 The format order does not vary with knowledge (main test)**: on units answered correctly by the intact model in all of BARE, JSON and TWICE, split by the within-cell median log frequency into a rare half and a common half; within each half compare the mean survival of the three formats.
  - COMMIT: valid cells ≥ 8, and ≥ 75% of cells satisfy BARE ≥ JSON ≥ TWICE in both halves;
  - PARTIAL: COMMIT not satisfied, but in ≥ 75% of cells TWICE is lowest in both halves;
  - STOP: otherwise.
- **Structural decision**: if H1 and H2 are both COMMIT → the main text writes three separable axes (task, knowledge, format). Otherwise keep "two axes plus knowledge within task", and state under which formats the knowledge effect holds.

## Lock
- analyze22.py md5: 63643ed25524060e3d4e1808711011f5

## Run log
1. 2026-10-30: md5 verified to match before running; analyze22.py run once, output analysis22.json.

## Results (analysis22.json)
- **H1 The knowledge effect does not vary with format: COMMIT (3/3).** All valid cells positive (100%).
  - BARE median 0.396 (16 cells; capitals 0.423, SciQ 0.383, consistent with Addenda 14 and 15).
  - JSON 0.365 (16 cells; capitals 0.448, 0.210–0.514; SciQ 0.323, 0.132–0.432).
  - TWICE 0.211 (11 cells; capitals 0.246, 0.121–0.383; SciQ usable for only 3 models, 0.177). TWICE just passes the threshold; the effect is about half that of BARE.
- **H2 The format order does not vary with knowledge: COMMIT.** All 11 valid cells (8 capitals, 3 SciQ) satisfy BARE ≥ JSON ≥ TWICE in both the rare half and the common half (100%); for phi capitals, JSON and TWICE are equal. The remaining 5 SciQ cells are not counted because TWICE is unusable or fewer than 30 items are answered correctly in all three formats.
- Structural decision: per the preregistered wording, the main text writes three separable axes (task, knowledge, format).
- To be stated faithfully: in the most fragile TWICE format the knowledge effect weakens (about half), i.e., when the format breaks much more, the role of familiarity is suppressed; the three axes are approximately independent, not fully independent. H2 relies mainly on capitals (SciQ testable in only 3 models).
