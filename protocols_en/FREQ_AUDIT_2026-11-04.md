# Audit of capital co-occurrence counts (2026-11-04, reviewer comment)
- Observation: in freq15.json the co-occurrence counts for Italy–Rome and Bulgaria–Sofia are 0. Under default parameters, infini-gram's CNF approximation samples high-frequency terms; rechecking still gave 0; after raising max_clause_freq=500000 they are 896,384 and 541,362 respectively.
- Handling: freq15_fix.py re-queries all 78 pairs with this parameter → freq15_fix.json (freq15.json is not changed); the remaining 76 pairs have Spearman 0.88 with the original counts, with no zero values.
- Sensitivity (isolated scratchpad directory, rerunning the locked scripts with freq15_fix; original outputs unchanged):
  - analyze15 D2a: median capital frequency effect 0.423 → 0.482, 8/8, verdict unchanged (COMMIT).
  - analyze17 H2 capitals under compression 0.411 → 0.493; verdicts for each damage unchanged.
  - analyze18 H2 partial correlations 0.620/0.642/0.515 → 0.609/0.641/0.508, verdict unchanged.
  - analyze22 H1/H2 verdicts unchanged (BARE 0.396 → 0.448).
  - analyze25 H4 partial correlations 0.604/0.629/0.513 → 0.599/0.629/~0.51, verdict unchanged.
- Write-up: the main text keeps the preregistered counts; Supplementary Note 4 adds a new "count audit" paragraph, and the caption of Extended Data Fig. 4 is updated.
