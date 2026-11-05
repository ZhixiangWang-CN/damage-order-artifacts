# Addendum 12: Can instruction push + answer margin predict fragility (preregistered)

Written on 2026-10-28, before any push measurement on held-out models. Approved by the investigator on the same day, both design and run ("design it and get the experiment running", translated).

## Disclosure first
- The hypothesis comes from an exploratory pilot (PILOT_PUSH.md): several measures were tried on q7 and llama (push, log-probability without instruction, push per token, log-probability with instruction), and the combination "push + answer margin, z-scores added" was picked (fit-free r: q7 0.78, llama 0.71). The pilot looked at the observed fragility of these two models, so they do not enter the verdict.
- The predicted quantity (the fragility exponent of each measure, Addenda 8, 9) already existed and had been seen; the predictor (push on held-out models) is measured only after this file and the analysis script were locked. The answer margin (intact CHOICE score) already existed, and its predictive power alone was already reported in Addendum 10 (within format B2 0.80); it is not a new finding.
- What this batch tests: on 6 unseen models, whether push orders the formats correctly and adds predictive power beyond the answer margin.

## Hypothesis
For the intact model to answer in the required format, the instruction must push the output away from the model's default way of saying it; damage first weakens this push. The more push a format requires, the earlier it breaks (format term); the task term is determined by the margin of the answer itself.

## Measurement (push.py, run only on intact models, no damage applied)
For each model, 16 tasks, three formats BARE / JSON / TWICE, 300 evaluation items: the summed log-probability of the canonical target output (atlas_full.target + end-of-turn token) under two prompts:
- With instruction: chat(body(it, f)), i.e. the prompt used in Addendum 8;
- Without instruction: chat(item stem, with options for choice items), with the format instruction removed.
The CHOICE format does not apply (no generation instruction) and does not enter this batch.

## Prediction (analyze13.py, no parameters fitted on damaged data)
- push_m = the mean of (with instruction − without instruction) over items judged correct when intact; at least 10 correct items.
- margin_t = −(mean over all evaluation items of CHOICE correct-candidate score − highest distractor score).
- pred_m = z(push) + z(margin), with z computed within the measures of that model that enter the analysis.
- Observed: the mean of the fitted log n of that measure over the four random damages (noise, zero, neuron, ld_rand), at least two (the same as analyze11.observed; data analysis10_out/analysis10_8models.json).

## Models
- Held out (enter the verdict): q15, phi, mis, gemma, q14, q32.
- Pilot (report only, do not enter the verdict): q7, llama.

## Tests and verdict (written before the measurement)
- **P1**: within each held-out model, the Pearson correlation between pred and observed log n; median over the 6 models.
- **P2**: the number of models in which the order of push format means matches the order of observed format means exactly.
- **P3**: the number of models in which the correlation of pred is higher than the correlation using the answer margin alone (whether push adds beyond the answer margin).
- **Verdict**
  - **COMMIT (the two-part explanation enters the main line of the main text)**: P1 ≥ 0.80, and P2 ≥ 5/6, and P3 ≥ 5/6.
  - **PARTIAL (written as a partial two-part explanation)**: P1 ≥ 0.65, and P2 ≥ 4/6, and P3 ≥ 4/6.
  - **STOP (push reported in the appendix as a negative result)**: all other cases.
- **Descriptive (does not enter the verdict, all reported)**: the correlation of push alone; the correlation after replacing push with its format mean (a pure between-format label); the correlation between push and observed within formats. In the pilot the format-mean version had an even higher correlation (q7 0.86, llama 0.73), indicating that the predictive power of push lies mainly between formats and the per-item variation within formats is basically noise; the results are stated honestly on this basis.

## Compute
Forward passes on intact models, 2 × 48 × 300 short sequences per model. server_a: mis, gemma; server_b: q15, phi, q14, q32 (q32 on a single 80 GB card). Only idle cards not on the reserved list are used (on server_b, not cards 2, 6, nor 0, 4 where REVEAL runs). Push depends only on the intact weights and does not need to be bit-identical with the original conditions; numerical differences across machines (on the order of 1e-3) are far smaller than push itself (tens of nats).

## Lock
- push.py md5: 69ab254366c3ed42b716026b65a59e3b (only two changes after the pilot: import atlas_comp to load q32; apply log-softmax only at the positions of the predicted continuation tokens, numerically identical to the full version)
- analyze13.py md5: 673a9c7ccab909201c50c3e5849bb14e (tested on pilot data: reproduces q7 0.778, llama 0.707)

## Run log
1. 2026-10-28 22:50: gemma ran out of memory on card 6 of server_a (full logits over a 256k vocabulary, with another person's process using 19 GB on the card); no file was written. push.py gained the environment variable PUSH_BS (changes only the batch size, values unchanged), md5 became c89335a01b4d54c0559482e6e146a421; gemma was rerun on card 1 with PUSH_BS=8. The other models used the original script; results are unaffected.
2. 2026-10-28 23:14: all 6 held-out models finished (gemma succeeded after the rerun with PUSH_BS=8).

## Results (locked analyze13.py, md5 673a9c7ccab909201c50c3e5849bb14e; output analysis13.json)
- **P1**: median r = 0.671 (q15 0.54, phi 0.63, mis 0.75, gemma 0.55, q14 0.72, q32 0.77). Answer margin alone gives 0.504, push alone 0.467.
- **P2**: format order exactly matching 4/6 (phi, gemma, q14, q32). Not matching: q15 (push JSON 24.8 > TWICE 19.3), mis (JSON 47.4 ≈ TWICE 47.1); although gemma is ordered correctly, its JSON and TWICE push are also almost the same (31.4 vs 31.7). Push stably distinguishes "answering directly" from "template formats", but cannot distinguish JSON from TWICE, whereas in the observations TWICE is far more fragile than JSON.
- **P3**: push adds beyond the answer margin in 6/6.
- **Verdict: PARTIAL.** Written as a partial two-part explanation: the task term comes from the answer margin; the format difference between "answering directly vs template formats" comes from instruction push; why TWICE is the most fragile remains unexplained.
- Descriptive: after replacing push with its format mean (a pure between-format quantity), median r = 0.767, higher than per-measure push; the within-format correlation between push and observed ranges from −0.32 to 0.73, with no consistent direction. All of push's predictive power lies between formats.
- Pilot models (do not enter the verdict): q7 0.778, llama 0.707, both with the order correct.
