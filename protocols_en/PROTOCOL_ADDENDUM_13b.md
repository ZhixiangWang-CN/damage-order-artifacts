# Addendum 13b: Retest after correcting the design flaw of Addendum 13 (preregistered)

Written on 2026-10-29, before any measurement of the "first-copy answer margin". Approved by the investigator on the same day ("run it", translated). This is the third and last attempt to explain the format effect; the results of Addendum 12 (push, PARTIAL) and Addendum 13 (formatted margin, STOP) are all reported as usual.

## Why retest
After Addendum 13 was run, it was found that: the write-twice score is the summed log probability of "candidate, candidate" written twice; the second copy is almost entirely determined by the first, yet counts the gap between the correct answer and the distractors a second time, making the write-twice margin systematically too thick (in 6 of 8 models the write-twice margin was the thickest). This is an error in the definition of the measure, unrelated to the hypothesis; therefore Addendum 13 did not cleanly test "format interferes with the answer". The reason for the correction was only discovered after seeing the results of Addendum 13; this is stated faithfully.

## Corrected measure (margin_first.py)
Only the first written copy of the answer and its terminator are scored, with a consistent definition across the three formats:
- BARE: candidate + end-of-turn (same as the CHOICE of Addendum 13 and Addendum 8, unchanged);
- JSON: '{"answer": "' + candidate + '"' (the template prefix is the same for all candidates and cancels out in the margin);
- TWICE: candidate + ',' (the first copy and the separator; the second copy is not scored).
The prompt is the same as in Addendum 13 (chat(body(it, f))). Everything else is unchanged.

## Analysis and verdict
- Directly reuse the locked analyze14.py (md5 5b3bf7c8468b4cedcab0d54e2ee5780c), without changing a single character; only the input directory is switched to this batch's data.
- Tests Q1, Q2, Q3 and the verdict thresholds are exactly the same as in Addendum 13, not relaxed:
  - COMMIT: Q1 ≥ 0.80, Q2 ≥ 7/8, Q3 ≥ 6/8;
  - PARTIAL: Q1 ≥ 0.65, Q2 ≥ 6/8, Q3 ≥ 5/8;
  - STOP: otherwise.
- Whatever the result, no fourth measure will be tried.

## Lock
- margin_first.py md5: f40588f2488ee0dccff827554cbbc190 (the scoring function is taken from push.py, md5 c89335a01b4d54c0559482e6e146a421)

## Run log
1. 2026-10-29: card 3 of server_b had another user's process occupying 26 GB, so q32 did not fit on a single card; the q32 run that had just started and had not yet written any results was stopped, and instead, after q14 and phi finished, it was split across cards 3 and 7 (ATLAS_MAXMEM=46GiB,60GiB). Only placement was changed, not computation.
2. 2026-10-29 01:45: all 8 models completed (q32 split-card run succeeded).

## Results (locked analyze14.py, md5 5b3bf7c8468b4cedcab0d54e2ee5780c, run in a separate directory; output analysis14b.json)
- **Q1**: median r = 0.583 (q15 0.49, llama 0.49, phi 0.57, q7 0.59, mis 0.64, gemma 0.49, q14 0.71, q32 0.67). Addendum 13 gave 0.540; the Addendum 12 prediction gave 0.712.
- **Q2**: format ordering exactly matches in 0/8. In all 8 models the thinnest first-answer margin is JSON, not TWICE; TWICE's first answer actually has a thicker margin than BARE in 5/8 models (q15, llama, phi, mis, gemma).
- **Q3**: exceeds the Addendum 12 prediction in 0/8.
- **Verdict: STOP.** Written into the appendix as a negative result. After correcting the design flaw of Addendum 13 the conclusion is unchanged: the intact model's answer margin on the answer (whether the whole output or the first copy) cannot explain why TWICE is the most fragile.
- Reading (descriptive, does not enter the verdict): TWICE failures do not come from "the first answer was unstable to begin with". Combined with the post-hoc analysis (POSTHOC_V13.md: on items where TWICE failed, BARE under the same damage still answers 69% correctly), the fragility lies in the execution step of "writing the answer twice following the template", not in the margin of the answer itself.
- As agreed in advance, this is the last attempt to explain format fragility; no fourth measure will be tried, and work moves on to rewriting the paper.
