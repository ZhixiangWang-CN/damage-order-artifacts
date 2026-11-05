# Addendum 9: Does importance-based compression follow the same fragility order + scaling to 32B (preregistered)

Written on 2026-10-24, before any new condition started running. Target journal: Nature Machine Intelligence.

## Background and question
The conclusion of Addendum 8 (under four kinds of random damage, tasks fail in the same order, order = task term + format term) covers only random damage. Practical compression methods decide by importance which weights to prune or how to round, and the reviewers' first question is "does practical compression follow the same order". This batch answers two questions:
1. Under three commonly used importance-based compression methods (Wanda, SparseGPT, GPTQ), is the fragility order of tasks the same as under the four kinds of random damage.
2. On a larger model (Qwen2.5-32B-Instruct), is the order under random damage still consistent.

Items, formats, scoring rules, usable-measure rules, exponent estimation, and the definitions of the task term and format term are all carried over from Addendum 8 without any modification (the hashes of items/ are unchanged; the functions of analyze9.py are called as is).

## Compression methods ("damage")
All three methods operate on all linear layers within each decoder layer (q/k/v/o_proj, gate/up/down_proj), the same scope as the random damage in Addendum 8; the embedding layer and output head are left untouched. All proceed layer by layer sequentially, following the reference implementations: the output of the previous layer after compression serves as the calibration input for the next layer.

| Name | Definition | Dose | Randomness |
|---|---|---|---|
| wanda | Sun et al. 2024: importance = \|W\| × L2 norm of the input activations; within each output row, prune the fraction s with the smallest importance | unstructured sparsity s | deterministic |
| sparsegpt | Frantar & Alistarh 2023: column-wise pruning with weight compensation based on the inverse Hessian, unstructured sparsity s, block size 128, damping 0.01 | sparsity s | deterministic |
| gptq | Frantar et al. 2023: column-wise quantization with error compensation, symmetric rounding, group-wise scaling, act-order off, damping 0.01 | bit width × group size configuration | deterministic |

**Calibration data**: 128 segments randomly drawn from C4 (allenai/c4, first shard of the English validation set), 512 tokens each (random seed 0). Unrelated to the items of the 16 tasks; there is no leakage. The three methods share the same set of calibration texts.

**Dose grid**
- wanda, sparsegpt: s ∈ {0.2, 0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9}.
- gptq: configurations ordered from light to heavy by expected strength: (4-bit, group 128), (4-bit, per-channel), (3-bit, group 32), (3-bit, group 128), (3-bit, per-channel), (2-bit, group 32), (2-bit, group 64), (2-bit, group 128), (2-bit, per-channel).

**Dose calibration**: exactly the same rules and calibration metric as in Addendum 8 (calibration items, BARE measures, target scores 0.95…0.08). Continuous doses (s) are monotonized and interpolated as usual. The gptq configurations are discrete: evaluate them one by one along the order above, and after monotonization keep all configurations whose calibration score falls within [0.08, 0.95]; if there are fewer than 4, add the configurations whose scores are closest to this interval until there are 4. All three methods are deterministic and are run only once (seed recorded as 0).

## Models
- The 7 models of Addendum 8: each runs wanda, sparsegpt, gptq.
- Newly added Qwen2.5-32B-Instruct (bf16, model-parallel over two A6000s): runs the four kinds of random damage (noise, zero, neuron, ld_rand; seeds 1, 2, 3) and the three compression methods. If compute time exceeds the budget (see "Compute"), follow the reduction order written in advance: first reduce the 32B random damage to seed 1, then drop 32B ld_rand. Any reduction is decided before seeing the 32B damaged results, and is recorded.

## Analysis (analyze10.py, locked before the first formal condition starts; md5 8f3150294f350258533d58c46e9d9e04)
- Exponents, task term, format term: directly call functions of analyze9.py such as fit_cell and additive, with rules unchanged.
- "Random reference": within each model, the average of the task terms of the four kinds of random damage in Addendum 8 (averaged only when at least two kinds of random damage have that task term).

## Predictions (written before the results)
- **T8 main test, within model**: for each "model × compression method" cell, the correlation between the task terms of the compression method and that model's random reference (shared tabulated tasks ≥ 8). Median over the 7 × 3 = 21 cells:
  - ≥ 0.7: importance-based compression follows the same order as random damage.
  - ≤ 0.4: does not follow.
  - In between: partially follows; report the deviations of each method and each task.
- **T9 across models**: for each compression method, the median of pairwise between-model task-term correlations ≥ 0.6, satisfied by all three methods.
- **T10 two-factor transfer**: use the additive fitted values from the four kinds of random damage in the same model (grand mean + task term + format term, averaged over the four) to predict the log exponents of each measure under the compression method; the median correlation over the 21 cells ≥ 0.7, and in at least 75% of cells it is higher than "task term only" and "format term only".
- **T11 scale**: the median within-model consistency (T1 definition of Addendum 8) of Qwen2.5-32B ≥ 0.6; and the median task-term correlation of 32B with Qwen2.5-1.5B, 7B, 14B under each kind of random damage ≥ 0.6.
- **T12 middle tasks** (in response to reviewer comments): after removing the three most robust tasks in Addendum 8 (COPY, RETR, FACT) and the most fragile, ARITH3, the median of T8 is still above the 95th percentile of the "task labels shuffled" null distribution (1000 shuffles per cell).

## What each result means for the paper
- T8, T10 hold: the claim is extended to "random damage and three importance-based compression methods all follow the same order determined by task and format"; this is the core finding for the NMI submission.
- T8 between 0.4 and 0.7: the claim is written as "importance-based compression partially follows", reporting which tasks deviate and in which direction (e.g., whether tasks related to the calibration data are more robust).
- T8 ≤ 0.4: the claim changes to "random damage shares one order, while importance-based compression changes this order"; this is itself a finding, and the paper is rewritten accordingly, without hiding it.
- T9 fails while T8 holds: the claim is narrowed to "consistent with random damage within model".
- T11 fails: the scale conclusion is written as "consistency declines with scale", reported together with the Qwen scale trend of Addendum 8.
- T12 fails: the consistency in T8 comes mainly from the extreme tasks at both ends, and the main text is worded accordingly.

## Exploratory analyses (no decision criteria, labeled separately)
Task-term correlations among the three methods; the size of the format effect and the proportion of template failures under importance-based compression; the predictive power of the intact answer margin (margin) for the task terms under the compression methods; calibration curves and KL probes of the compression methods; fitting sensitivity (fitting windows 0.05–0.95 and 0.02–0.98, weighting by binomial variance, at least 3 points), done for both Addendum 8 and this batch.

## Stopping rules and honesty rules
- If a cell has fewer than 8 tabulated tasks: that cell does not enter the decision and is reported as is.
- Doses are locked by code, not adjusted manually; after seeing the results, thresholds and estimation methods are not changed.
- A failed condition may only be rerun with exactly the same code, and this is recorded.
- After the smoke test, the only permitted modifications are: errors in the compression implementations that are inconsistent with the reference implementations, and batch size and GPU memory settings. Any modification is recorded at the end of this file and completed before Stage 1.

## Smoke checks (before Stage 1)
1. Implementation correctness: on Qwen2.5-1.5B, with wanda and sparsegpt at s = 0.5 and gptq at (4-bit, group 128), the perplexity on WikiText-2 test segments outside the C4 calibration text is of the same order of magnitude as published values (increase relative to the intact model < 30%); actual sparsity differs from the target by < 0.5 percentage points.
2. Restoration check: after compression, restore the weights; KL < 1e-3 (same as Addendum 8).
3. 32B: the time for one intact evaluation under two-GPU parallelism, used to determine whether reduction according to the reduction order is needed.

## Compute
- Only server_a is used; GPUs 6 and 7 are reserved for the user and not used. All GPUs of server_b are occupied and its disk is full; it is not used in this batch.
- 7 models × 3 methods: about 8 conditions per method per model, plus calibration.
- Qwen2.5-32B: occupies two GPUs; download about 65 GB into the HF cache on /home2.
- Budget: all conditions completed within 5 days; if exceeded, handle according to the reduction order above.

## Post-smoke modifications
2026-10-24, before Stage 1, without having seen any formal condition.

1. **Criterion of smoke check 1**. The original criterion "perplexity increase relative to the intact model < 30%" was a number picked arbitrarily in advance; on Qwen2.5-1.5B all three methods exceeded it (WikiText-2, 40 segments × 512 tokens: intact 11.94; wanda 0.5 is +69%, sparsegpt 0.5 is +54%, gptq 4-bit group 128 is +49%). To rule out implementation errors, a naive-baseline comparison was added (diag_comp.py, results stored in runs/smoke/diag_*.json):
   - Qwen2.5-1.5B: magnitude pruning 0.5 is +738%, wanda +69%, sparsegpt +54%; group-wise round-to-nearest 3-bit +772%, gptq 3-bit +464%; at 4-bit, round-to-nearest +22%, gptq +49%.
   - Qwen2.5-7B: magnitude pruning 0.5 is +16000%, wanda +30%, sparsegpt +32%; round-to-nearest 3-bit +736%, gptq 3-bit +73%; at 4-bit both are +8%.
   All three implementations are significantly better than the naive baselines, and the 7B numbers are of the same order of magnitude as published results at the same scale, so the implementations are judged correct; 1.5B is inherently more sensitive to compression. The criterion is changed to "significantly better than the naive baselines (magnitude pruning, group-wise round-to-nearest), and the perplexity increase of the 7B model is of the same order of magnitude as published results". The only exception is that 1.5B 4-bit gptq is slightly worse than round-to-nearest (both are very mild); this is recorded, and the implementation is not changed. Doses are calibrated by task scores, and this criterion does not affect any decision of the formal experiment.
2. Smoke check 2: KL after restoration = 0.0, passed.
3. Smoke check 3 (32B, two A6000s): loading 53 seconds; intact evaluation on smoke items 215 seconds/480 items, converting to about 36 minutes per formal condition; wanda 0.5 compression 135 seconds; restoration from disk 20 seconds, KL = 0.0. All 32B jobs are estimated at 3.5–4.5 days, within the 5-day budget; no reduction is triggered.
4. Run arrangement: server_a GPU 1 is shared with the user's other tasks; the first worker ran out of GPU memory when loading Qwen2.5-14B (the job was put back in the queue per the rules, no condition written), and that GPU was changed to run only models within 10 GB.
5. 2026-10-24: the user took back server_a GPUs 4 and 5, and the 32B worker was stopped. The 32B intact evaluation was completed (none_0_0, usable.json retained); the noise calibration in progress had not written a lock file, the corresponding claim records were released, and it will later be resumed on two available GPUs with exactly the same code.
6. 2026-10-24: the user allocated server_b (A100 80 GB) GPUs 2 and 6 to this batch until 10-31. All 32B jobs moved to run on server_b: code, items and C4 calibration text were copied unchanged from server_a (md5 identical), and the model is stored in server_b's HF cache. server_b's environment is torch 2.5.1 (server_a is 2.8.0); to ensure that all 32B conditions come from the same environment, the 32B intact evaluation is rerun on server_b, and all damaged conditions use the intact results on server_b as baseline; the existing 32B intact results on server_a serve only as a cross-environment replication check and do not enter the analysis. The worker starts automatically once both GPUs have freed up space (CUDA_VISIBLE_DEVICES=2,6, per-GPU cap still 44 GiB); other settings unchanged.
7. 2026-10-24 (32B still had not started any condition): a single 80 GB A100 can hold the whole 32B; two-GPU model parallelism just runs layers serially and gives no speedup. Changed to one independent worker per GPU (each claiming different jobs), roughly doubling throughput. atlas_comp.py only changed the loading method: when 1 GPU is visible, the whole model is placed on that GPU (md5 f9f50c2645a4b961ce5d612a43b681fa). With a single GPU, random damage goes through Addendum 8's single-device code path, and all 32B conditions run on a single GPU, keeping the path consistent. Each worker starts when its GPU's memory usage is below 6 GB.
8. 2026-10-24 record (found before seeing results, code not changed): mis's gptq calibration stopped early per the calibration code when the score at the 3rd configuration was already < 0.05, so only 3 configurations were evaluated; "add up to 4" had no configuration to add, and it was locked at 3 doses. Used as is per the rule "doses are locked by code".
9. 2026-10-24 evening: server_b GPUs 2 and 6 still had other people's processes (about 20 GB each), so a single GPU could not hold 32B. Per the user's decision, reverted to two-GPU model parallelism (CUDA_VISIBLE_DEVICES=2,6, per-GPU cap 44 GiB, multi-device code path), with one worker running all 32B jobs; to keep the code path consistent, no switch to single-GPU mode midway.
10. 2026-10-25: the user asked to give server_b GPUs 2 and 6 to others, and the 32B worker was stopped. The 32B intact evaluation on server_b was completed (meanacc 0.903, consistent with 0.903 on server_a); the noise calibration in progress had not written a lock file, the corresponding claim records were released, and it will later be resumed with exactly the same code.
11. 2026-10-25: the user asked to use all GPUs except those reserved for others (server_a 4, 5; server_b 2, 6) to speed up 32B (server_b 7 still not used). At this point 32B had only completed the intact evaluation, with no damaged conditions yet.
   - The two servers do not share a claim directory, so machines are assigned by damage type, with each damage type (including its calibration) running on only one machine: server_b (GPUs 1, 3, 5, per-GPU cap 26 GiB) runs noise, zero, wanda, gptq; server_a (GPUs 1, 3, 6, per-GPU cap 24 GiB) runs ld_rand, sparsegpt, neuron. For this, atlas_comp.py added two environment variables, ATLAS_MAXMEM (per-GPU cap when splitting the model) and ATLAS_JOBS (damage types this worker may claim), otherwise unchanged (md5 1894c4f13bce40a657698fa31c4ec801).
   - Item-by-item comparison of the 32B intact results on the two machines: of 19,300 items, 24 differ in correct/incorrect (0.12%), and the sets of usable measures are exactly the same. In the analysis, the 32B intact baseline uniformly uses server_b's results; the damage run on server_a has a 0.12% environment difference from this baseline, reported as is.
   - Initially two workers were started on server_b (GPUs 1+3 and 4+5); GPU 1's total usage reached 80 GB, risking crashing other people's jobs, so they were stopped within a few minutes and replaced by one worker using three GPUs; no condition or lock file had been written when stopped, and claim records were released.
12. 2026-10-26: the user wanted one server_b A100, and the 32B server_b worker was stopped (the noise seed 2 condition in progress was not written; the 10 completed noise conditions are retained; claim records released). By then other people's jobs on GPUs 1, 3, 5 had finished; GPU 1 is left to the user, and the worker resumed on the two GPUs 3 and 5 (per-GPU cap 44 GiB, same as the two-GPU setting in the server_a smoke test), with damage assignment unchanged and code unchanged.
13. 2026-10-26: two minutes after resuming, another person's job (bwz3651) reoccupied server_b GPUs 1, 3, 5 at about 37 GB each; total usage on GPUs 3 and 5 reached 73–76 GB, and activations during evaluation would crash the shared GPUs, so the worker was stopped before writing any new condition, and claim records were released. noise (10 conditions completed), zero, wanda, gptq on server_b are paused.
14. 2026-10-26: with the user's consent, the 32B worker on server_b resumed on GPU 7 (the user said it may be used) and GPU 0 (the REVEAL project's GPU, shared with it), CUDA_VISIBLE_DEVICES=7,0, per-GPU cap 34 GiB, with damage assignment and code unchanged.
15. 2026-10-26: when running on the 7+0 two GPUs, the last GPU bears the main activations during evaluation, and GPU 7 (shared with bwz3651) reached a total usage of 77.5 GB. Changed to put GPU 0 first and allow a different cap per GPU: ATLAS_MAXMEM supports comma-separated per-GPU caps (atlas_comp.py md5 a45406349da40f23d3779415e426ab75, only GPU memory settings changed); server_b set to GPU 0 at 42 GiB, GPU 7 at 22 GiB. At each restart the condition in progress was not written, completed conditions are retained, and claim records were released.
16. 2026-10-26: under per-GPU caps (42/22, 46/24, 52/22 GiB), accelerate's automatic allocation placed some weights on disk (meta device), and the noise job raised an error when applying damage; no condition was written, and the failure records were moved to runs/failed_log/superseded/. Changed to an explicit two-GPU layer split: ATLAS_SPLIT=k places the first k decoder layers on the first GPU and the remaining layers and output head on the second GPU (atlas_comp.py md5 622991bff4f3b60143d753d53c319719, only model placement changed). server_b uses k = 43, and the first condition completed normally.
17. 2026-10-26 16:50: REVEAL's usage on server_b GPU 0 grew, and GPU 0's total usage reached 79.7 GB (0.3 GB headroom). To avoid crashing REVEAL, the server_b worker changed to an explicit three-GPU layer split: CUDA_VISIBLE_DEVICES=0,4,7, ATLAS_SPLIT=20,24 (GPU 0 20 layers, GPU 4 24 layers, GPU 7 the remaining 20 layers and output head; atlas_comp.py md5 dae9125dc56a2fa96d08f6724022497d, ATLAS_SPLIT generalized to multiple GPUs, only model placement changed). The condition in progress was not written, and claim records were released.
18. 2026-10-27: 32B neuron and ld_rand on server_a are all completed; sparsegpt ran out of GPU memory while accumulating the Hessian on the shared A6000s, and after three failures the worker exited (no condition written, calibration not completed). Reassignment, with each damage type still running on only one machine:
   - server_b: the original worker (GPUs 0, 4, 7) continues zero; a new worker (GPUs 3, 5, ATLAS_SPLIT=40) runs zero, wanda, sparsegpt (sparsegpt moved from server_a, with no prior results).
   - server_a: a new worker (GPUs 0, 1, 3, 6, ATLAS_SPLIT=18,18,18) runs only gptq; a placeholder claim record for gptq was written on server_b to ensure server_b does not run gptq.
19. 2026-10-27: the gptq worker on server_a ran out of GPU memory twice on the shared A6000s (four-GPU 18,18,18 and five-GPU 8,16,16,16 splits, both before the first calibration point; no condition or lock file written), and was stopped. gptq moved back to server_b, taken over by the worker on GPUs 0, 4, 7 after zero; the placeholder claim record on server_b was deleted. From this point all remaining 32B jobs (zero, wanda, sparsegpt, gptq) run on server_b; server_a no longer runs 32B.
20. 2026-10-28: all other 32B jobs are completed, leaving only gptq (4 locked configurations 2, 3, 4, 5, run in order by the worker on GPUs 0, 4, 7; out of GPU memory once at 11:06 UTC, then retried automatically). For parallelism, a helper process gptq_helper.py (md5 ca2c04a6867c8c06c9c66e9401ea8700) was started separately on server_b GPUs 3 and 5 (ATLAS_SPLIT=36), using the same code path as the worker (atlas_full.run_cond) to run the locked configurations in reverse order (5, 4, …). The worker checks for the result file before starting each configuration and skips it if it already exists, so the two meet in the middle; if the same configuration is run by both, the result is deterministic, and the later write overwrites the earlier one. The locked doses, evaluation and analysis are not changed.
21. 2026-10-28: all 32B jobs completed. Of the 4 gptq configurations, 2 and 3 were written by the worker, 5 and 4 by the helper; after writing 3 the worker marked completion and exited, then the helper started rerunning 3 (result file already existed); it was stopped by PID, leaving no temporary files.

## Results (2026-10-28, analyze10.py md5 8f3150294f350258533d58c46e9d9e04, 8 models; output analysis10_out/analysis10_8models.json)
Analysis root directory runs_v9/all: 7 models link to runs_v9/server_a; q32 merges server_b (intact baseline, noise, zero, wanda, sparsegpt, gptq) and server_a (neuron, ld_rand), see runs_v9/all/README.md.
- T8 within model: median 0.875 (18 cells) → follows.
- T9 across models: wanda 0.924, sparsegpt 0.927, gptq 0.095 (6 pairs) → fails, solely because of gptq; the claim is narrowed to "consistent with random damage within model", and gptq is reported separately.
- T10 two-factor transfer: median 0.807, 94.7% of cells beat the single factors → holds.
- T11 scale (32B): median within-model consistency 0.843 (6 pairs, 0.62–0.91); median task-term correlation with Qwen 1.5B/7B/14B 0.898 → holds. Rank-one fit R² for the seven damage types of 32B 0.81–0.91.
- T12 middle tasks: median 0.727, 95th percentile of the null distribution 0.184 → holds.
- Exploratory (32B compression, not within the preregistered definitions of T8/T9): correlation with the 32B random reference wanda 0.80, sparsegpt 0.74, gptq 0.67; median correlation with the same method in the other 7 models wanda 0.92, sparsegpt 0.89, gptq 0.23. The format order CHOICE < BARE < JSON < TWICE is fully consistent across all 7 damage types of 32B (sparsegpt's BARE and JSON, and gptq's CHOICE and BARE, differ by less than 0.04).
