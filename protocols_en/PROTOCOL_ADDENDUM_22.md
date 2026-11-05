# Addendum 22: Do the main conclusions still hold when content, template and strict success are scored separately (preregistered, existing data only)

Written on 2026-11-04, before computing any rescoring statistics. Approved by the investigator on the same day ("B. New analysis, all with existing data; write the preregistrations first (Addendum 21, 22), then run", translated).

## Motivation (reviewer comments)
The choice format uses candidate likelihood ranking, while the other formats require successful generation; under strict scoring, models that often break the template cannot reach the chance rate 1/k. This may affect the fitted fragility exponents. We need to show that the main conclusions still hold under separate scoring (content correct, template valid, strict success).

## Data seen before this protocol (stated honestly)
- The strict-scoring results already in the main text.
- The smoke test ran this script's statistics only under strict scoring (reproduction: C4 matches analysis19 exactly, 0.113/0.110/0.116; C1 with this script, pairwise over the seven damages, median task effect Pearson 0.811; C3 median 0.793; C2 8/8 models CHOICE < BARE). No statistic for content, strict0 or template was computed.
- When opening one raw file to check the fields, the strict, template and content accuracies of several JSON/TWICE items under one neuron-ablation condition of Qwen-1.5B were seen (e.g. ARC_JSON 0.003 / 0.007 / 0.40): template failures made up the majority in these formats. This is the observation that prompted this addendum, and it does not involve any test statistic.
- One change before locking: the C1 threshold was originally 0.80; after seeing that strict scoring itself gives only 0.811, it was changed to 0.70, the same as T1 of Addendum 8, so as not to count the fluctuation of strict scoring itself as a failure.

## Method (analyze26.py)
- From the raw run files (runs_v8, runs_v9, the same files as analysis15), take per-item correct, format_ok, content_any → analysis26_cache.pkl.
- Four scorings for the generation formats (BARE, JSON, TWICE); CHOICE is unchanged under every scoring:
  - strict: the scoring currently used in the paper (template correct and answer correct), chance rate 1/k (for option and yes/no items), otherwise 0.
  - strict0: the same correctness, but the chance rate of all generation measures set to 0.
  - content: the content_any written by the scorer along with the run (regardless of template, the correct answer appears in the output; for option and yes/no items take the first standalone letter or yes/no, for other tasks match a number or name at any position), chance rate the same as strict.
  - template: looks only at format_ok, chance rate 0 (report only).
- Usable measures: the measures in the original usable list that are more than 0.05 above chance when intact (varies by scoring). Exponents by the main-text method (rank1, window 0.03–0.97, ≥12 measures).
- C1 shared order: within each model, the task effect Pearson for each pair of the seven damages (≥8 common tasks); take the median over all pairs.
- C2 generation before choice: the number of models with mean format effect under random damage CHOICE < BARE.
- C3 compression order: Pearson between the compression log exponent and the random-damage reference (log mean over ≥2 random damages) (≥10 common measures); take the median over model × compression cells.
- C4 accuracy: the analyze19 predictor (random-damage profile vs uniform decline, leave-one-measure-out).

## Tests and verdict (judged separately for content and strict0)
- C1 median ≥ 0.70; C2 at least 7/8 models CHOICE < BARE; C3 median ≥ 0.70; C4 all three compression methods satisfy median reduction ≥ 25% and more-accurate cells ≥ 75%.
- All four pass under a scoring → the main conclusions hold under that scoring. Hold under both scorings → COMMIT; one → PARTIAL; neither → STOP.
- Report only: C1–C4 under template scoring; the mean format effects under each scoring (how much of the format cost of JSON and TWICE disappears under content scoring, i.e. belongs to the template); under informative conditions, among items answered correctly when intact and wrongly after damage, the proportion due to template failure and the mean template validity rate.
- Wording: COMMIT → the main text and methods write "the main conclusions hold both when only content is scored and when the chance rate is set to 0", and report how much of the format cost is template and how much is content; not passed → state honestly which item failed under which scoring.

## Lock
- analyze26.py md5: 0906e0ce6ee084a1259b7aa8698db923

## Run log
1. 2026-11-04: md5 verified as matching before the run; analyze26.py run was run once, output analysis26.json.

## Results (analysis26.json)
| Scoring | C1 shared order | C2 CHOICE<BARE | Mean format effect CHOICE / BARE / JSON / TWICE | C3 compression order | C4 reduction (wanda/sparsegpt/gptq) |
|---|---|---|---|---|---|
| strict (reference) | 0.811 | 8/8 | −0.90 / −0.42 / 0.15 / 1.17 | 0.793 | 48% / 45% / 47% |
| content | 0.820 ✓ | 8/8 ✓ | −0.34 / −0.01 / 0.13 / 0.22 | 0.816 ✓ | 50% / 53% / 55% (gptq more accurate 86%) ✓ |
| strict0 | 0.829 ✓ | 5/8 ✗ | −0.60 / −0.58 / 0.10 / 1.08 | 0.834 ✓ | 51% / 47% / 49% ✓ |
| template (report only) | 0.776 | 0/3 | 0.24 / −2.49 / 0.44 / 1.81 | 0.807 | 40% / 40% / 27% |

- **content: all four pass.** Looking only at answer content, regardless of template, the shared order, generation before choice, compression order and accuracy prediction all hold.
- **strict0: C2 fails (5/8), the other three pass.** After setting the chance rate to 0, the format effects of BARE and CHOICE are almost equal (−0.58 vs −0.60).
- **Verdict: PARTIAL.**
- Report only, source of the format cost: under informative conditions, among items answered correctly when intact and wrongly after damage, the proportion due to template failure is BARE 0 (BARE only requires a non-empty first line, template validity rate 1.00), JSON median 0.44 (0.29–0.86, template validity rate 0.82), TWICE 0.90 (0.71–0.96, template validity rate 0.33). Under content scoring, the format cost of JSON and TWICE relative to BARE drops from 0.57 and 1.59 to 0.14 and 0.23: the extra fragility of JSON and TWICE is mainly template failure; the gap between choice and generation (under content, CHOICE −0.34 vs BARE −0.01) is at the content level, not caused by the template.
- Post-run note (does not change the verdict): strict0 also sets the chance rate of BARE to 0, but BARE's template validity rate is 1.00, so guessing 1/k correctly on BARE is genuinely possible; the preregistered strict0 was intended to target the invalid templates of JSON and TWICE. If only the chance rates of JSON and TWICE were set to 0, the BARE and CHOICE data would be unchanged and C2 would be expected to be close to strict (8/8), but the additive fit would re-center with JSON and TWICE; this was not computed. This is reasoning, not a test; the main text honestly reports the C2 failure of strict0 and explains its cause.
- Wording: in the main text, "generation before choice" holds under content scoring; "JSON and TWICE are more fragile than BARE" should be rewritten as mainly template failure, with only small differences at the content level (0.13, 0.22).
