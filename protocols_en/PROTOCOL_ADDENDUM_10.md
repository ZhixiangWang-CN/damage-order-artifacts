# Addendum 10: Can per-token margins explain fragility? (preregistered)

Written on 2026-10-28, before any per-token margin was collected. Target journal: Nature Machine Intelligence.

## Disclosure first
- The quantity being predicted (the fragility exponents of each measure in Addenda 8 and 9) already exists, and we have seen it. This hypothesis was proposed after seeing that "the format effect is large, and most TWICE errors are template violations"; it is a post-hoc hypothesis.
- The predicting quantity (per-token margins of the intact model) is newly collected; at the time of writing this file, not a single one has been measured. The power of the test comes from this: the predictor is produced only after the analysis is locked; the analysis script's md5 is locked before collection and is not changed afterwards.

## Hypothesis (margin mechanism)
Damage approximately adds noise to the model's output scores. When a measure fails depends on the margin at every decision point that the intact model needs in order to produce a valid output; the more decisions that must be made correctly and the thinner the margins, the earlier it fails. If true, it explains both the task term (the margin of the answer itself) and the format term (formats require more, thinner decisions).

## Collection (margins.py, run only on intact models, no damage applied)
For each model, each task (16), each generation format (BARE, JSON, TWICE), each evaluation item (300):
1. Regenerate using exactly the same prompt as Addendum 8 (atlas_full.chat + body), greedy decoding, at most 32 new tokens.
2. At each step record: the selected token, and the differences among the top 5 logits at that step (rank 1 minus rank k, k = 2…5). Generation stops at the model's end-of-turn token or at 32 steps; the end-of-turn step is counted.
3. Judge correctness with atlas_full.judge, and record whether the regenerated output is verbatim identical to the intact output stored in Addendum 8 (report the agreement rate).

The CHOICE format needs no new collection: the intact results of Addendum 8 already store the summed log-probability of each candidate; margins.py only additionally records the token count of each candidate continuation (candidate + end-of-turn).

## Noise model and prediction (analyze11.py, fixed, no free parameters)
Let the damage strength be σ. Each candidate's log score (every token during generation; for CHOICE, every token in each candidate sequence) independently receives N(0, σ²/2) noise, so the noise standard deviation of the difference between any two single-token scores is σ.

- **One item in a generation format**: the probability of keeping the correct choice at critical step j = ∏_{k=2..5} Φ(δ_jk / σ), where δ_jk is the logit difference between rank 1 and rank k (independence approximation). An item survives with S(σ) = product over its critical steps.
- **Critical steps (main analysis)**:
  - BARE: from step 1 up to the shortest prefix already judged correct by judge (output after the answer is written does not affect correctness, because BARE only judges the answer content of the first line).
  - JSON, TWICE: all steps, up to and including the end-of-turn token (writing anything after the template makes template matching fail).
- **One CHOICE item**: S(σ) = ∏_{distractor k} Φ( Δ_k / (σ·√((L_g + L_k)/2)) ), where Δ_k = the difference in summed log-probability between the correct candidate and distractor k, and L is the respective token count.
- **Predicted curve of a measure**: using only items judged correct when intact, ĉ_m(σ) = the mean of S(σ) over these items (failed items return to chance level, consistent with the chance correction of Addendum 8).
- **Predicted exponent**: σ takes 60 log-equally-spaced values between 0.01 and 100; for all available measures of each model, fit the rank-one model with analyze9.rank1 under the same window (0.03 < ĉ < 0.97) and the same "at least two points" rule, obtaining n̂_m, likewise normalized to a geometric mean of 1. The absolute scale of σ has no effect after normalization, so there is no parameter fitted from damaged data.

## Quantity being predicted
Within each model, the observed log exponent of measure m = the mean of log n over the cells, among the four random damages (noise, zero, neuron, ld_rand), in which that measure was fitted; at least two damages must have fitted the measure. Exponents are taken from analyze9's fit_cell (same as Addenda 8 and 9). The main analysis uses the 7 models of Addendum 8; 32B and the three compression methods are reported separately (secondary).

## Tests and decision (written before collection)
- **M1 main test**: within each model, the Pearson correlation between log n̂ and observed log n over shared measures; the median across the 7 models.
- **M2 task part (within format)**: within each model × each format (at least 6 tasks), the correlation between log n̂ and observed log n; the median over all cells.
  - Baseline B2: the existing answer-margin predictor (the mean intact CHOICE margin for that task, negated), likewise computed within format.
- **M3 format part**: apply to log n̂ the same task + format decomposition as analyze9.additive, obtaining the predicted format term β̂. Count the models in which the order of β̂ exactly matches the observed order (CHOICE < BARE < JSON < TWICE); and report the predicted and observed TWICE minus CHOICE difference.
- **M5 length-only control**: replace all δ of each model (including CHOICE's Δ) with the median of all δ for that model, leaving everything else unchanged, and recompute M1–M3. This tests how much "the number of decision points" by itself explains.

**Decision (main analysis, 7 models)**
- **COMMIT (margin mechanism becomes the main line of the paper)**: M1 median ≥ 0.80, and models with exactly matching M3 order ≥ 6, and M2 median ≥ B2 median.
- **PARTIAL (written as "the format effect comes from the number of decisions that must be made correctly and their margins", with the task term still explained by answer margin)**: M1 in [0.65, 0.80); or M1 ≥ 0.80 and M3 holds but M2 < B2.
- **STOP (margin mechanism does not enter the main line; reported in the appendix as a negative result)**: M1 < 0.65, or models with exactly matching M3 order < 4.
- **Wording rule**: if the difference between M5's M1 and the main analysis is < 0.03, the mechanism is written as "the number of decisions that must be made correctly", not as "margin".

**Sensitivity analyses (do not affect the decision; all reported)**: critical steps changed to "all steps"; using only the top-2 difference (only rank 1 vs rank 2); removing the four extreme tasks of Addendum 9 (COPY, RETR, FACT, ARITH3).

## Compute
8 intact models (7 + 32B), each one generation pass (3 formats × 16 tasks × 300 items, at most 32 steps), comparable in volume to one intact evaluation. Each model runs on the server where its Addendum 8 intact results live (server_a: q7, q14, mis, gemma; server_b: q15, llama, phi, q32), so that regenerated outputs match the archive as closely as possible. Run only on cards that are idle and not on the reserved list.

## Lock
- margins.py md5: a501d882733c96b0b50e2b37753aea67
- analyze11.py md5: 64e3129647d59a4e11e010dc015be3fe

## Pre-collection smoke test and execution notes (2026-10-28, before formal collection)
1. test_margins.py was run on server_b with Qwen2.5-1.5B on 4 tasks × 3 formats × the first 4 items: generation with scoring was verbatim identical to the output of atlas_full.generate in 48/48 cases, every output ended with a stop token, and critical steps were computed as defined (e.g. for MMLU's "C. Left temporal" only step 1 is critical, because letter items judge only the first letter). The margins of these items were not compared with any exponent. The script was not modified because of the smoke test.
2. The recorded margins are taken from generate's output_scores, i.e. the scores after processing by the model's default generation config (e.g. Qwen's repetition_penalty), which are the scores greedy decoding actually uses to choose tokens.
3. Machine assignment (deviation from the "run on the original server" clause): server_a had no idle headroom able to fit Qwen2.5-14B, so q14 was run on server_b instead (torch 2.5.1; the Addendum 8 intact results were generated with torch 2.8.0). The verbatim agreement rate with the archive is reported as usual; the margins describe the decisions the model actually made during regeneration, and correctness is also judged on the regenerated output.
4. 2026-10-28: gemma ran out of GPU memory on server_a card 3 when collecting TWICE (vocabulary of 256k, per-step scores occupy GPU memory); no file was written. Rerun with ATLAS_TOK_BUDGET=8000 (originally 16000) and expandable_segments; only batching changed, script and analysis unchanged, verbatim agreement rate reported as usual.
5. 2026-10-28: on server_b, q14's first launch failed because offline mode could not find the cache (no file written); it was run after switching to online loading. q32 ran on server_b cards 3 and 5 (ATLAS_SPLIT=36, expandable_segments), using the intact results of Addendum 9 (runs/q32/none_0_0) as the archive reference.

## Results (2026-10-28, analyze11.py md5 64e3129647d59a4e11e010dc015be3fe, 7 models; output analysis11.json)
- Verbatim agreement rate with the archive: q15, q7, mis, phi 1.000; q14 0.982; gemma 0.982; llama 0.959.
- M1 median 0.459 (0.39–0.60); M2 median −0.034, baseline B2 (answer margin) 0.796; M3 exactly matching order 3/7.
- Length-only control: M1 0.363. Sensitivity: all steps 0.430, top-2 only 0.408, extreme tasks removed 0.589; none changed the decision.
- Compression methods (secondary): M1 median between 0.06 and 0.75, mostly < 0.5.
- **Decision: STOP.** The margin mechanism does not enter the main line of the paper and is reported in the appendix as a negative result.

Description (does not affect the decision): within format, per-token margins barely predict the task order (M2 ≈ 0), far worse than the answer-level summed log-probability margin (0.80); across formats, they predict JSON as the most fragile (many template tokens, single-step margins not large), whereas in observation JSON is near the middle and TWICE is the most fragile. That is, the model in which "each critical step is independently perturbed by noise" does not hold: template tokens are far more robust under damage than their single-step margins suggest, and the fragility of TWICE does not come from the number of steps.
