# Addendum 11: Do real tasks fall into the same fragility order? (preregistered draft)

Status: Approved by the investigator on 2026-10-28 to run with this design ("go ahead and run it, use more GPUs", translated); script md5s filled in below before the first formal condition. Written on 2026-10-28, before any new-task condition was run. Target journal: Nature Machine Intelligence (the results decide whether to submit to NMI or elsewhere; see "Decision").

## Why
The conclusions of Addenda 8–10 cover only 16 short-answer tasks (output ≤ 32 tokens). The most likely NMI reviewer objection is "what does this have to do with real capabilities?" This batch answers two questions:
1. Do four classes of real tasks, namely reasoning, code, instruction following and function calling, fall into the same fragility order that is consistent across damages?
2. Does "a damaged model first loses the ability to answer in the required format, and only later loses the answer" (Addendum 8: 83% of TWICE errors violate the template, and 79% of those still contain the correct answer) hold for function calling, a real structured output?

## Known priors (from existing data, written here to avoid post-hoc rationalization)
- Post-hoc analysis posthoc_v10 (not preregistered):
  - Predicting post-compression log exponents using only the intact model's answer margin and the format terms of other models: median 0.70 (T10, using same-model random damage, gives 0.81).
  - Using only the first three doses (retaining 95%/85%/75% of scores): the mild-regime order of each cell agrees with its full-range order (median 0.915), and compression methods agree with the mild random reference (0.84); but pairwise agreement between individual random damages drops to 0.63, the pairs involving layer deletion are lowest (0.33–0.35), and only 37 of 56 cells are measurable in the mild regime.
- The exploratory GSM8K chain-of-thought measure of Addendum 8 (100 items, seeds 0/1): in 35 cells it fails after 67% of measures, and in 18 of 19 cells it is more robust than the short-answer four-number addition.

## Design

### Models and conditions (reuse the locked doses, no recalibration)
- Models: Qwen2.5-7B (q7), Llama-3.2-3B (llama), Mistral-7B (mis), Gemma-2-9B (gemma).
- Damages:
  - noise: the 8 doses locked in Addendum 8, seeds 1, 2.
  - wanda: the 8 doses locked in Addendum 9.
- Each model runs on its original Addendum 8/9 server and software stack, so that the damaged weights are bit-identical to the original conditions (llama's noise on server_b, everything else on server_a).
  - Check: after applying damage, each condition first runs a fixed probe (atlas_full.Probe, 64 fixed text passages), whose NLL must differ from the archived condition's probe NLL by no more than 0.1%; non-matching conditions are flagged check_ok = False, do not enter the analysis, and work stops for investigation. (The draft originally said "rerun ARITH2_BARE and SCIQ_CHOICE and require exactly identical accuracy"; changed to the probe: running only two tasks changes batching, and tiny differences in bf16 numerics could flip individual items, so it cannot serve as a criterion for bit-identity; the probe's batching is fixed.)
- Hence the exponents of the new tasks can be projected directly onto the damage strength h(d) already estimated for the original cells, placing them on the same ruler as the original 64 measures, with no need to refit cells.

### New tasks (200 evaluation items each, generated once then frozen with hashes recorded)
| Code | Capability | Source | Answering | Judging |
|---|---|---|---|---|
| GSM_COT | Multi-step reasoning | GSM8K test set (same shuffle as Addendum 8; the first 100 are the original exploratory items) | Brief chain of thought, then write "The answer is N.", at most 384 tokens | Same as Addendum 8's judge_cot |
| CODE | Writing code | HumanEval (all 164 items) + MBPP sanitized to fill up to 200 | Output only one Python function, at most 384 tokens | Run the official unit tests in an isolated subprocess, 10-second timeout; correct if all pass |
| IFEVAL | Instruction following | Of all 541 IFEval items, remove length requirements that cannot be completed within 384 tokens (at least 200+ words, 12+ sentences, 4+ paragraphs), leaving 488; sample 200 | Free answer, at most 384 tokens | Official IFEval checker, strict criterion; correct if all constraints are satisfied |
| FCALL | Function calling (structured output) | BFCL v3 "simple" category (single function), sample 200 | Output only JSON: {"name": ..., "arguments": {...}}, at most 128 tokens | Format: the whole reply (allowed to be wrapped in one code fence) parses into an object with exactly the two keys name and arguments, and all argument names belong to the function schema; content: function name correct, and every argument value belongs to the acceptable answers (types must match, the numbers 1 and 1.0 are equivalent, strings compared under BFCL normalization; arguments whose acceptable answers include the empty string may be omitted); correct if both are satisfied |

- Chance level is 0 for all. The availability rule is the same as Addendum 8 (chance-corrected intact score ≥ 0.4).
- FCALL additionally records a lenient content score: whether the function name and argument values extracted from anywhere in the output are correct, without requiring valid JSON. Used to test "format breaks first".

### Exponents
- For new measure m and cell (model, damage): take the original cell's h(d) (estimated by analyze9.fit_cell on the original 64 measures, unchanged in this batch), and within the window 0.03 < c < 0.97 compute by least squares through the origin n_m = Σ h·(−log c) / Σ h², with at least 2 points.
- Use the same normalization constant as the original measures (the geometric mean of the original cell), so new and old measures can be compared directly.

## Tests and decision (written before running)
- **N1 order consistent across damages (core)**: within each model, the Pearson correlation of log exponents of all measurable measures (those fitted among the original 64 + the 4 new ones) between noise and wanda; the median across the 4 models. Also, the proportion of new measures whose percentile position (percentile among all measures) under the two damages differs by ≤ 25 percentage points.
  - Holds: median ≥ 0.7, and the proportion of new measures with consistent percentile position ≥ 75%.
- **N2 new tasks predictable by two factors**: the additive model of Addendum 8 (task term + format term, fitted on the four random damages) cannot cover the new tasks; instead test whether the new tasks' exponents fall within the range of the original measures (between the 5th and 95th percentiles). Descriptive only; does not enter the decision.
- **N3 format breaks first (function calling)**: at doses where FCALL's relative intact score falls between 0.3 and 0.7 (for each model × damage take the one dose closest to 0.5, noise's two seeds pooled), looking only at items answered correctly when intact and incorrectly when damaged, compute the proportion of format failures (unparseable or schema mismatch) among them, and the proportion of format failures in which the lenient content still appears. Also compare the exponent ratio of FCALL's strict score to the lenient content score.
  - Holds: median share of format failures among errors ≥ 50%, median share of those with correct lenient content ≥ 50%, and median exponent ratio (strict / lenient) ≥ 1.5.
- **Decision**
  - Both N1 and N3 hold: submit to NMI, claim extended to "real tasks are also in the same order, and structured output fails before the answer".
  - Only N1 holds: submit to Nature Communications or NeurIPS, claim being "real tasks are also in the same order", with format-breaks-first holding only on short-answer tasks.
  - N1 does not hold: submit to TMLR, claim narrowed to "a common order for short-answer tasks", with new-task results honestly reported as a scope limitation.

## Compute and execution
- 4 models × (noise 8 doses × 2 seeds + wanda 8 doses + intact) = 100 conditions; each condition about 800 long-generation items. Estimated 10–20 minutes per condition for 7B on one A6000/A100, about 25–35 GPU hours in total, parallelizable by assigning models to cards.
- To download on the servers: GSM8K (already present), HumanEval, MBPP, IFEval and its checker, BFCL simple; code execution runs only in isolated subprocesses, network disabled, time-limited.
- The run script atlas_real.py and the analysis script analyze12.py have their md5 locked before the first formal condition; any modification after the smoke test is recorded item by item as usual.

## Lock
- Item file hashes (items_real/MANIFEST.json, sha256): GSM_COT c074fa39…, CODE cd0a212a…, IFEVAL 105f7906…, FCALL 28ad4c05…
- atlas_real.py md5: a26f5b5422c26175bb2cc397f10fdc5a
- analyze12.py md5: 09617660a80d52f80a35204f6797744b

## Smoke test and pre-run modifications (2026-10-28, before the first formal condition)
1. Judge unit tests (test_real.py): the format/content/lenient judgments for function calling, code execution (including timeout and disabled dangerous calls), and GSM judging all passed.
2. Each of the four intact models ran 4 tasks × 10 smoke items (smoke_real/):
   - q7: GSM 0.8, CODE 0.8, IFEVAL 0.5, FCALL 0.7.
   - llama: GSM 0.6, CODE 0.7, IFEVAL 0.7, FCALL 0.5.
   - mis: GSM 0.5, CODE 0.7, IFEVAL 0.4, FCALL 0.7.
   - gemma: GSM 0.9, CODE 0.6, IFEVAL 0.7, FCALL 0.9.
   All function-calling and code items judged wrong were manually checked: all were genuine model errors (e.g. writing an integer argument as a string, which counts as a type error under BFCL rules).
3. Modification one (judging): the lenient content score originally matched numbers as text, treating 3 and 3.0 as different, causing q7 and mis each to have one item judged correct strictly but wrong leniently; changed so that integer values accept both spellings. After the change, none of the four models has an item that is "strictly correct but leniently wrong".
4. Modification two (scheduling, does not affect results): added two environment variables, REAL_SEEDS and REAL_NO_INTACT, used to split the jobs of the same model across two machines (claim records are not shared between machines).
5. Two implementation clarifications: the HumanEval program for code items is "original signature and docstring + pass + model code + official tests"; when the model gives a complete function it overrides the original function, and when it gives only the function body it is appended within the original function. IFEval depends on nltk 3.9.2 etc., installed in the user directory of the login shell on both servers; workers are always launched via bash -l.

## Run log
1. 2026-10-28: the probe checks of the first batch of conditions all passed; on server_a the difference from the archive was < 1e-4, and llama on server_b was also < 1e-4. q7 noise seed 2 run on server_b (original condition on server_a, torch 2.8) had probe NLL 2.2329 vs archived 2.2310, relative difference 0.085%, within the 0.1% tolerance but not bit-identical; this difference is about one tenth of the difference between two seeds at the same dose (0.017), judged to be the same perturbation with a different numerical implementation. Included in the analysis per the rule; a sensitivity result excluding this batch of conditions is reported separately (does not affect the decision).
2. 2026-10-28: to speed up, q7's 8 wanda conditions were moved to server_b card 1 (placeholder claim records put on server_a; reference probes taken from the Addendum 9 conditions on server_a and copied to server_b's refs_v9/). Likewise checked by probe; non-matching conditions do not enter the analysis and are moved back to server_a.
3. 2026-10-28: the q7 conditions on server_b could not be reproduced bit-identically. The probe relative difference for noise seed 2 grew with dose (0.0003–0.6%), with 4 of 8 doses exceeding the 0.1% tolerance; of the 3 completed wanda doses, 1 exceeded it. Judged to be numerical differences between torch 2.5.1 (server_b) and 2.8.0 (server_a) in perturbation generation or pruning computation. Handling (stricter than the preregistered rule): all q7 conditions on server_b (including those within tolerance) were excluded and moved to runs_real_excluded/q7_server_b, the placeholder claim records on server_a were deleted, and server_a reran all q7 conditions. From then on each model × damage ran only on its original server (llama noise on server_b, everything else on server_a), and all conditions included in the analysis correspond bit-identically to the original conditions. The condition "included per the rule" in item 1 was also excluded, so the sensitivity comparison is no longer needed.
4. 2026-10-28 around 18:40: the worker on server_a card 7 (server_a-r7) exited after 3 consecutive out-of-memory errors once another user's process occupied about 19 GB; on out-of-memory the worker deletes its claim record, leaving no partial output or failure marker. At 18:47 it was restarted on the same card with the same environment (ATLAS_TOK_BUDGET=8000, expandable_segments) as server_a-r7b, claiming only the remaining gemma and mis conditions. Scheduling change; does not affect results.
5. 2026-10-28 around 19:30–20:10: the workers on server_a cards 0, 1 and 3 exited after 3 consecutive out-of-memory errors when switching from finished mis to gemma (other people's processes on the cards used 15–19 GB; when switching models the worker's local references to the old model were not released, which also raised the peak). Likewise only the claim records were deleted, with no failure markers. At 20:14 they were restarted as new processes on the three cards (server_a-g0/g1/g3, claiming only gemma). Scheduling change; does not affect results.
6. 2026-10-28 21:14: all 100 conditions completed, probe checks passed 100/100 (q7, mis, gemma, llama-wanda on server_a/torch 2.8.0; llama intact + noise on server_b/torch 2.5.1, same machine as the original conditions).

## Results (locked analyze12.py, md5 09617660a80d52f80a35204f6797744b; data runs_real/merged; output analysis12.json)
- **N1 does not hold**: median r = 0.778 (q7 0.786, llama 0.769, mis 0.685, gemma 0.843; the correlation criterion passes); but the proportion of new measures whose percentile position under the two damages differs by ≤ 25 points = 10/16 = 62.5% < 75%.
  - The 6 inconsistent ones: GSM_COT is clearly more fragile under pruning in all three models q7/llama/mis (percentile 0.33→0.72, 0.21→0.67, 0.35→0.83; gemma 0.75/0.76 consistent); CODE in llama (0.40→0.70); FCALL in mis (0.91→0.55) and gemma (0.90→0.54). IFEVAL is consistent in all four models.
  - During the run (when gemma was at 19/25) an interim computation was done once with the locked script, which already showed that N1 could not hold (the 3 complete models already had 5 inconsistent ones, upper bound 11/16); the final result agrees with the interim one.
- **N3 holds**: median share of format failures among errors 0.684, median share of format failures with lenient content still correct 0.774, median strict/lenient exponent ratio 2.315 (8 cells).
- **N2 (descriptive)**: see analysis12.json.
- **Decision (per the preregistered rule)**: TMLR. Claim narrowed to "a common order for short-answer tasks"; real tasks reported honestly: instruction following follows the common order, the position of multi-step reasoning shifts systematically with damage type (more fragile under pruning, in 3 of 4 models), and "format breaks first, answer later" is confirmed on function calling.
