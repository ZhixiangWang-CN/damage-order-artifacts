# Addendum 8: Final batch — task-level fragility order and the additive "task + format" structure (preregistered)

Written on 2026-10-22, before any smoke test, calibration or formal result of this batch. For the design rationale see FULL_RUN_DESIGN.md. Code: make_items.py (generates and freezes the items), atlas_full.py (runs), analyze9.py (locked analysis, written together with this protocol, before any results were produced).

## Background
The main conclusion of Addendum 7 (measure exponents are stable across damage types and across models) was taken apart in a post hoc exploration on 2026-10-22: on the "3 tasks × 6 formats" grid, after removing the task main effect and the format main effect, the median cross-damage correlation is only 0.29. So what is stable is more likely two factors: the fragility of the task and the fragility of the format, which add approximately on the log exponent. This reading was arrived at after looking at the data; this batch tests it as a prediction written in advance, and changes the unit of analysis from "measure" to "task".

## Models (7, same as Addendum 7)
Qwen/Qwen2.5-1.5B-Instruct, Qwen/Qwen2.5-7B-Instruct, Qwen/Qwen2.5-14B-Instruct, unsloth/Llama-3.2-3B-Instruct, mistralai/Mistral-7B-Instruct-v0.3, unsloth/gemma-2-9b-it, microsoft/Phi-3.5-mini-instruct. bf16, greedy decoding. Gemma always generates without cache. All conditions for the same model run on the same machine, to avoid numerical differences between different GPUs getting mixed in.

## Tasks (16) and items
Items are generated once by make_items.py and saved to files (with hashes). Formal items, calibration items and smoke items do not overlap with each other.

| Class | Task | Source | Number of candidates |
|---|---|---|---|
| Short answer | ARITH1 / ARITH2 / ARITH3 (adding 2/3/4 two-digit numbers) | Programmatically generated, same generator and same seed as Addendum 7 (first 200 items identical to Addendum 7) | 4 |
| Short answer | FACT (capitals), RETR (retrieval among 12 key-value pairs) | Same as above | 4 |
| Short answer | COPY (repeat a nonsense word) | Same generator as Addendum 6; distractors change one syllable | 4 |
| Short answer | SCIQ (answer ≤3 words), BOOLQ (passage ≤150 words, answer yes/no) | Same filtering and shuffling as Addendum 4 (first 200 items identical to Addendum 4) | 4 / 2 |
| Multiple choice | ARC-Challenge (4 options), CSQA (5 options) | Same filtering and shuffling as Addendum 4 | 4 / 5 |
| Multiple choice | MMLU_CALC (high school mathematics, elementary mathematics, high school physics, college physics), MMLU_MEM (high school world history, nutrition, high school geography, high school psychology) | Stem ≤180 words, mixed then shuffled | 4 |
| Multiple choice | HellaSwag (validation set), PIQA (validation set), WinoGrande-XL (validation set), OpenBookQA (test set) | HuggingFace cache | 4 / 2 / 2 / 4 |

Each task has 300 formal items, 40 calibration items and 30 smoke items. For multiple-choice items, the option order is shuffled with a fixed seed when generating the items, and the answer is a letter.
An additional exploratory measure GSM8K_COT is set up (100 items, reason briefly first, then write "The answer is N.", at most 384 tokens); it does not enter any decision.

## Formats (4, fully crossed with tasks)
| Format | Instruction (the noun for short-answer items is number/name/word/answer; for multiple-choice items, letter; for BOOLQ, "yes or no") | Scoring |
|---|---|---|
| CHOICE | Uses the BARE prompt; compares the summed log probabilities of each candidate as a continuation (with the end-of-turn token appended) | The gold answer gets the highest score |
| BARE | "Answer with just the {n}." | A single line and ≤40 characters, and the content is correct |
| JSON | The F2 instruction of Addendum 6 | Matches `{"answer": ...}`, and the content is correct |
| TWICE | The F5 instruction of Addendum 6 | Matches "X, X", and the content is correct |

Content scoring: for numbers, take the first integer, which must equal the answer; for text items, after normalization (strip accents, lowercase, strip punctuation), the answer must appear as a complete word; for yes/no, take the first yes or no; for multiple-choice items, take the leading letter.
Chance level: for CHOICE, 1/number of candidates; for the generation formats of multiple-choice items and BOOLQ, 1/number of candidates; for the other generation formats, 0.
Measure = task × format, 64 in total.

## Damage
| Damage | Definition | Role | Seeds |
|---|---|---|---|
| noise | Add Gaussian noise to each within-layer weight matrix, standard deviation = r × that matrix's standard deviation | Main analysis | 1, 2, 3 |
| zero | Set a random fraction p of the weights in within-layer weight matrices to zero | Main analysis | 1, 2, 3 |
| neuron | Zero a random fraction p of MLP hidden units (zero the corresponding columns of down_proj) | Main analysis | 1, 2, 3 |
| ld_rand | Randomly delete n layers (first and last layers kept) | Main analysis | 1, 2, 3 |
| ld_late | Delete n consecutive layers before the last layer | Control | none |
| qmix | Randomly select a fraction p of within-layer weight matrices and apply group-wise (group length 64, along the input dimension) symmetric round-to-nearest quantization | Exploratory | 1, 2, 3 |

Bit count for qmix: 3 bits by default. If during the smoke test the mean corrected score of the BARE format for Qwen-7B at p = 1 is still > 0.3, all models switch to 2 bits. This decision is made before Stage 1 starts and is not changed afterward.

## Dose: automatic calibration per model and per damage type (rules written in atlas_full.py)
- Calibration metric: on the calibration items, the mean corrected score of BARE measures whose intact corrected margin is ≥ 0.3. Averaged over seeds 11 and 12 (ld_late only once).
- Continuous doses: proceed along a preset grid from light to heavy, stopping when the mean score falls below 0.05 and at least 3 points have been tested; after monotonization, interpolate and take the doses at which the mean score is 0.95, 0.85, 0.75, 0.60, 0.45, 0.30, 0.18, 0.08. Target points outside the grid range that cannot be interpolated are discarded.
- Layer-count doses: take those among 1, 2, 3, 4, 5, 6, 8, 10, 12, 16 that do not exceed "the number of layers at which the mean score first falls below 0.08".

## Run stages and release rules
0. **Smoke test**: each model runs all measures on the smoke items, plus one dose per damage type. Manually check only one thing: for each "task × format", look at 5 outputs to confirm the scoring rules do not systematically misjudge. After the smoke test, parsing rules and batch size may be modified; any modification is recorded in the "Post-smoke modifications" section at the end of this file.
1. **Intact**: each model runs all 300 formal items. The code locks usable measures by rule: (intact accuracy − chance)/(1 − chance) ≥ 0.4. Only tasks with at least 3 usable formats enter the task-level analysis (the intact data of Addendum 7 show that "write it twice" already fails the bar when intact on the arithmetic items of several models; requiring all 4 to be usable would make the small models lose too many tasks). Locking happens before any damaged condition starts.
2. **Calibration**: according to the rules above.
3. **Formal**: each condition is saved in its own file (per item: whether answered correctly, whether format-compliant, raw output, log probability of the gold answer, all candidate scores), resumable from checkpoints. After each weight restoration, run the damage-degree probe once; the KL from the intact model must be less than 1e-3, otherwise abort.

After Stage 1 starts, code, items, thresholds and dose rules are no longer changed.

## Damage-degree probe (exploratory)
64 SciQ support passages, each truncated to the first 48 tokens, input as raw text without the chat template. For each condition, record: the mean per-position KL divergence of the damaged model relative to the intact model, and the mean negative log-likelihood of the damaged model on these texts.

## Analysis (analyze9.py, locked)
**Exponents**: same as Addendum 7. In each "model × damage" cell, fit a rank-1 model "shared error intensity × per-measure exponent" to the seed-averaged corrected scores, using only points whose corrected score falls between 0.03 and 0.97; only measures with at least 2 such points get an exponent. If a cell has fewer than 12 usable measures with exponents, that cell is recorded as "cannot be measured".

**Task term and format term**: within a cell, take the tasks with exponents in at least 3 formats (hereafter "tabulated tasks") to form a "task × format" table of log exponents, allowing missing cells. Fit by least squares log exponent = grand mean + task term + format term (iterating alternating means until convergence; task terms and format terms are each centered to mean 0). Interaction residual = log exponent − fitted value. For cells with fewer than 8 tabulated tasks, task terms are not computed.

**Correlations**: Pearson correlation throughout; at the task level, ≥ 8 shared tabulated tasks are required, and at the measure level, ≥ 10 shared measures are required; otherwise not computed.

## Predictions (written before the results)
- **T1 within model, task level**: within each model, the correlations of task terms between pairs of the four main-analysis damage types (6 pairs). The median over all "model × damage pair" ≥ 0.7, and at least 6 of the 7 models have a within-model median ≥ 0.6.
- **T2 across models, task level**: for each main-analysis damage type, the median of pairwise between-model task-term correlations (up to 21 pairs) ≥ 0.6, satisfied by all four damage types.
- **T3 additive prediction**: within each model, leave out one main-analysis damage type in turn; use the additive fitted values of each of the other three damage types (grand mean + task term + format term), averaged over the three, to predict the log exponents of the left-out damage type (using only cells with exponents within tabulated tasks), and correlate with the actual values. The median over the 28 "model × left-out damage" ≥ 0.7; and in at least 75% of cases, the correlation of the additive prediction is higher than both the "task term only" and "format term only" predictions.
- **T4 measure-specific component** (reading written in advance, no bet on direction): in the same set of left-out comparisons, add a "per-measure prediction": predict using the mean log exponent of that measure in the other three damage types. Also compute the reliability of the interaction residuals: within the same cell, split the items into two halves by odd/even item number, fit exponents on each, and correlate the interaction residuals of the two halves. Then compute the disattenuated cross-damage correlation of interaction residuals = cross-damage correlation / √(product of the two reliabilities), computed only on damage pairs where both reliabilities are > 0.2.
  - Disattenuated median ≥ 0.6: the measure-specific component is also stable, it was just measured imprecisely before. The paper claims "stable at the measure level too".
  - ≤ 0.3: the measure-specific component is not stable. The paper's claim stops at the two factors.
  - In between: reported as "undecidable".
- **T5 control**: within each model, the task-term correlation between ld_late and noise is lower than that model's T1 median over the 6 pairs. Holds in at least 5 of the 7 models.
- **T6 replication**: for the BARE and CHOICE cells of ARITH1, ARITH2, ARITH3, FACT, RETR, as well as COPY's BARE and the JSON of FACT, ARITH1, COPY (corresponding to Addendum 7's GEN/CHOICE, F0 and F2; for F5 see post-smoke modification 1), in each "model × main-analysis damage" cell, compute the log-exponent correlation between this batch and Addendum 7 (counted only with ≥ 8 shared measures). Median ≥ 0.8.
- **T7 reliability (reported first, used to set the reading)**: in each cell, the odd/even split-half correlation of the task terms, median ≥ 0.8. If not reached, T1 and T2 are decided as usual, but the wording changes to "consistent within the limits of measurement precision", and the disattenuated values are reported alongside.

## What each result means for the paper
- T1, T2, T3 hold: the claim is "across 16 tasks, 7 models and 4 damage types, fragility is determined by two stable factors (task, format) that add approximately".
- T1 holds, T2 fails: the claim is narrowed to "stable within model, each model has its own order".
- T1 fails: the earlier stability came mainly from format differences and those few synthetic tasks; the main conclusion is rewritten.
- T3 fails: drop the "additive" wording, and report only the task-level order.
- T5 fails: the position-related claim is withdrawn.
- T6 fails: before drawing any conclusion, first pin down the source of the differences between the two batches.

## Exploratory analyses (no decision criteria, results labeled separately)
Task-term correlations between the qmix arm and the four main-analysis damage types; the order when a continuous metric (mean log probability of the gold answer) replaces accuracy; comparison of half-decay dose with exponent; the order after aligning doses by probe KL; the relation between the shared error intensity h(d) and KL; whether task terms can be predicted by the intact model's accuracy or log-probability difference; comparison across the three Qwen scales; GSM8K_COT.

## Stopping rules and honesty rules
- If a model has fewer than 8 tabulated tasks: that model does not enter the task-level decisions, and is reported as is.
- Doses are locked by code, not adjusted manually. After seeing the results, the exponent estimation method, bars and thresholds are not changed.
- If a condition fails due to a fault such as running out of GPU memory, it may only be rerun with exactly the same code and settings, and this is recorded.

## Compute
- server_a: use GPUs 0–6. GPUs 0 and 5 are idle and run Qwen-14B and Gemma-9B; GPUs 1–4 and 6 are shared with other people's tasks (about 20GB of GPU memory left on each) and run the smaller models.
- server_b: GPUs 0, 1, 2, 4, 5, 6 (about 20GB of GPU memory left on each), running Qwen-1.5B and Qwen-7B.
- Neither machine uses GPU 7.

## Post-smoke modifications
2026-10-22, after the first round of smoke tests (7 models) and before Stage 1. All changes were made before seeing any formal results.

1. **Wording of the TWICE instruction**. The original instruction "Write the {n} twice" was understood on arithmetic items as "write some number from the question twice" (e.g., "28, 75"); Qwen-7B scored only 0.13 on ARITH2. Changed to "Write the {n} that answers the question twice, separated by a comma and a space, and nothing else."; for yes/no items changed to "Answer yes or no, written twice, separated by a comma and a space, and nothing else.". Because the wording now differs from the F5 of Addendum 6/7, the T6 comparison no longer includes F5 measures (leaving 5 each of GEN/CHOICE, COPY_F0, and 3 F2, 14 in total).
2. **Parsing of TWICE**: first normalize the whitespace on both sides of the comma to ", " before matching ("no , no" and "futeza,futeza" were previously judged as format errors).
3. **Scoring of BARE**: changed to look only at the first non-empty line; if the content of the first line is correct, it counts as correct (models such as Phi often leave a blank line after the answer and write an explanation). This is consistent with how the GEN measure was scored in Addendum 7. The generation limit for BARE is raised from 16 to 32 tokens (the original limit truncated answers).
4. **Bit count of qmix**: per the rule written in advance in this protocol, Qwen-7B at p = 1 and 3 bits has a mean BARE corrected score of 0.73 (> 0.3), so all models switch to 2 bits.
5. **Computation of forced choice**: the question stem is forwarded only once, and all candidates share its KV cache. Compared with a full forward pass per candidate, on 96 items × 3 conditions for Gemma-9B and Qwen-7B, the agreement rate of the selected answer is 100% (intact, noise) and 97.9% (random deletion of 3 layers), and the median difference in candidate scores is 0.05–0.14 nat (half-precision path differences). All models and all conditions use the same computation method.
6. **Log probability of the gold answer**: the value for BARE is taken directly from the score of the correct candidate in CHOICE (same prompt, same continuation); it is no longer computed for JSON and TWICE (used only in exploratory analyses; saves about 30% of compute).
7. **Gemma switched to cached generation**: after deleting layers, the indices and configuration of the retained layers (number of layers, attention type of each layer) are updated in sync, so the cache corresponds to them, and Gemma no longer raises errors. When deleting 3 layers, generation with and without cache is identical word for word: Gemma 93/96, Qwen-7B 87/96; those that differ are all already-damaged outputs (half-precision flips, consistent with the check in Addendum 7). All models and all conditions use cached generation, replacing the original plan's "Gemma always generates without cache".
8. **Calibration grid for qmix**: after switching to 2 bits, in the second round of smoke tests randomly quantizing half the matrices already made all 7 models collapse, and the original grid (starting at 0.1) had too few points at the low end. The grid is changed to 0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0; calibration and locking rules unchanged.

Second round of smoke tests (after the above modifications): 13–15 tabulable tasks per model across the 7 models on smoke items; the estimated time for one intact evaluation of 300 items is 10–28 minutes/condition. Stage 1 starts from this point, with no further modifications afterward.

## In-run modifications (do not change any decision)
1. 2026-10-22, about 3 hours after Stage 1 started: server_a GPUs 0 and 5 (the only two GPUs that can hold Qwen-14B and Gemma-9B) were taken over by other tasks, and each condition for these two models went from about 16 minutes to about 60 minutes, which at that rate would need 4–5 days. Following the reduction order written in advance in the design file (cut the quantization arm first), **Qwen-14B and Gemma-9B do not run the qmix arm**; the other 5 models proceed as usual. qmix was exploratory to begin with and does not enter T1–T7. When this decision was made, these two models had no qmix results yet (it is scheduled last for each model). The other tasks have since been moved to other GPUs by the user.

2. 2026-10-22 evening: after processes were added on the GPUs freed up on server_a, when one process on GPU 6 switched from Qwen-7B to Mistral, the old model's GPU memory was not released, and together with a Gemma process newly added on the same GPU, loading Mistral ran out of GPU memory; the process recorded each failure as an ordinary failure and immediately claimed the next job, marking all 13 remaining Mistral jobs as failed within a few seconds. All failures occurred at the model-loading step, and no condition was written half-computed. The fix involves only scheduling; the evaluation code was not changed: release the old model before switching models; on running out of GPU memory, put the job back in the queue and wait 5 minutes before claiming again, and exit the process after 3 consecutive failures; release the calibration slot if calibration fails midway. Afterward all processes were stopped, unfinished jobs released, and restarted with the new code (one process per GPU on server_a, no longer sharing a GPU). Saved condition files are retained and skipped when resuming; failure records are archived in runs/failed_log. These jobs are "rerun with exactly the same evaluation code and settings" per the preregistered rule.
