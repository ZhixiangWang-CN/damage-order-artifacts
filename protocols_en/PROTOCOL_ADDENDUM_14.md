# Addendum 14: Two extension tests (preregistered, existing data only)

Written on 2026-10-29, before the full computation. Approved by the investigator on the same day ("I think directions 1 and 2 can both be done; quickly try the directions", translated). Prompted by two follow-up questions in the reviewer comments: human in the loop (does the model know when it is broken?) and "what does fragility correlate with?". Both use only the existing per-item outputs in runs_v8 / runs_v9 and public corpus counts; no new model inference is run.

## Data seen before this protocol was locked (stated honestly)
- Direction one: before locking, we had seen the mean "write-twice failure rate / probe perplexity / accuracy" over the first 4 noise levels for q7, llama, gemma, phi, mis and q15, as well as the per-level format failure rates of q7 and llama under noise and Wanda; we saw that on gemma and mis the write-twice failure rate moves earlier than perplexity, while on q7 and phi perplexity also moves early. No cross-model predictive quantity had been computed.
- The self-knowledge ability (AUROC, overconfidence) of damaged versus intact models at equal accuracy had already been computed once in full; it is reported as an exploratory result and is not tested in this addendum.
- Direction two: no relation between individual items and frequency had been looked at, no country's count had been queried (only France/Paris was used to test whether the API is reachable), and no per-item cross-damage correlation had been computed.

## Direction one: template early-warning signal (D1)
Question: without looking at the gold answer, only at whether the output format is broken, can we judge how much content capability the model has lost, and do so more accurately and more consistently across models than off-the-shelf label-free signals?

- Conditions: 8 models × 7 damages (noise, zero, neuron, ld_rand, wanda, sparsegpt, gptq), all doses and seeds, one point per condition. The intact reference is runs_v9/all/<model>/none_0_0.
- Target quantity (requires labels, used only for scoring): content loss L = 1 − A/A0. A is the mean chance-corrected accuracy (acc − chance)/(1 − chance) over the usable measures of the CHOICE and BARE formats in that model's usable.json; A0 is the intact value.
- Three label-free signals:
  - S_T: the normalized increase in write-twice format failure rate (B − B0)/(1 − B0), where B is the mean format_ok failure rate of the TWICE measures over the 16 tasks. The format judgment only checks the "X, X" form and does not use the answer.
  - S_ppl: the relative increase in negative log-likelihood on the probe text (nll − nll0)/nll0.
  - S_KL: KL to the intact model on the probe text (requires the intact model as reference).
- Main test: leave-one-model-out cross-validation. For each signal, fit a monotonically increasing mapping (isotonic regression, signal → L) on all points of the other 7 models and predict the held-out model's points; the predictions of the 8 hold-outs are pooled and R² is computed on points with L ≤ 0.5 (the range where early warning is meaningful).
- Decision:
  - COMMIT (enters the main text): R²(S_T) ≥ 0.50, and at least 0.10 higher than max(R²(S_ppl), R²(S_KL));
  - Tie (one sentence in the supplementary material: "the template signal is on par with perplexity"): R²(S_T) ≥ 0.50, but the second condition is not met;
  - STOP: R²(S_T) < 0.50.
- Secondary (reported only, no decision): R² for the three compression damages alone; pooled Spearman of the three signals with L in the early range L ≤ 0.15.

## Direction two: per-item level (D2)

### D2a Frequency effect
- Uses only the capital task (FACT, 300 items, 78 countries). The unit of analysis is the country; items of the same country are pooled.
- Frequency: infini-gram API, index v4_dolma-v1_7_llama (Dolma v1.7), querying the co-occurrence count of "{country} AND {capital}", taking log10(count + 1). Secondary frequencies: the count of the capital name itself; the count of the phrase "capital of {country} is {capital}".
- Survival score: for each model, the mean accuracy on BARE (writing the capital name) over all items of that country across all conditions of the 7 damages. Only countries for which the intact model answers all items correctly on BARE are kept. Secondary: CHOICE.
- Main test: within each model, Spearman(log co-occurrence frequency, survival score).
- Decision:
  - COMMIT: median over the 8 models ≥ 0.30, and at least 7/8 positive;
  - PARTIAL: median ≥ 0.15, and at least 6/8 positive;
  - STOP: otherwise.
- Confound control (reported only): partial Spearman controlling for the token count of the capital name (llama tokenization returned by infini-gram); partial Spearman controlling for the intact model's CHOICE margin (to see whether frequency has explanatory power beyond the margin).

### D2b Shared per-item order (replacing the originally planned Guttman coefficient of reproducibility)
- Reason for replacement (decided before looking at any per-item data): the Guttman coefficient of reproducibility is inflated when items are very easy or very hard, and cannot distinguish "items break in a shared order" from "the difficulty distribution is merely skewed". Instead, directly test whether the same set of items breaks in the same order under different damages.
- Each model × each usable measure × each random damage (noise, zero, neuron, ld_rand) × seeds 1, 2: per-item survival score = mean accuracy over all doses of that damage and seed. Keep only items answered correctly by the intact model, with at least 30 items and survival scores not all identical.
- Within-damage ceiling r_within: Spearman of seed 1 vs seed 2 of the same damage.
- Cross-damage r_cross: damage A seed 1 vs damage B seed 2, and A seed 2 vs B seed 1, averaged; 6 damage pairs.
- For each measure take the median of r_cross over damage pairs and the median of r_within over damages; for each model then take the median over measures; ratio = r_cross / r_within.
- Decision:
  - COMMIT (enters the main text): median ratio over the 8 models ≥ 0.80, and median r_cross ≥ 0.30;
  - PARTIAL (supplementary material): median ratio ≥ 0.60;
  - STOP: otherwise.

## Lock
- analyze15.py md5: ff12b59a1ef7b22d08056d64a8b406c9 (the cache and freq steps only extract data and corpus counts and compute no statistics; the run step produces all decisions)

## Run log
1. 2026-10-29 16:14: cache completed (8 models, 102–123 conditions).
2. The first freq fetch hit infini-gram HTTP 403 (rate limiting) at the 38th country (Japan). Wrote fetch15.py separately: the same queries and output format, with resumable fetching and rate limiting; analyze15.py unchanged (md5 still ff12b59a1ef7b22d08056d64a8b406c9).
3. While waiting for the frequency counts, the original functions d1() and d2b() in the locked script were called via import to produce these two items first (the same computation as the run step).

## Results
4. fetch15.py completed (78 countries); then the locked analyze15.py run was executed once to produce all three items (md5 unchanged), output analysis15.json.

## Results (analysis15.json)
- **D1 template early-warning signal: STOP.** Leave-one-model-out R² (547 points with L ≤ 0.5): S_T 0.383, S_ppl −0.880, S_KL −0.153. S_T did not reach 0.50. Secondary: compression damages only S_T 0.549, S_KL 0.264, S_ppl −0.442; early range (L ≤ 0.15, 254 points) pooled Spearman S_T 0.505, S_ppl 0.460, S_KL 0.590. Reportable side fact: perplexity and KL are completely non-transferable across models (negative R²); format breakage is the only one of the three label-free signals that is partially transferable.
- **D2a frequency effect: COMMIT.** Under BARE, median Spearman(log co-occurrence frequency, survival score) 0.423, 8/8 positive (0.344–0.550). Controlling for capital-name token count 0.318–0.548; controlling for intact model margin 0.261–0.521; intact margin by itself only 0.047–0.453. Secondary frequencies: capital-name count 0.561–0.777; phrase count 0.229–0.528. Under CHOICE, co-occurrence 0.373–0.508.
- **D2b shared per-item order: PARTIAL.** Median ratio 0.898 (0.838–1.120), median r_cross 0.263 (did not reach 0.30); r_within itself is only 0.203–0.335, so the reliability of per-item survival scores is low. Per the rule, goes to the supplementary material.

## Post-hoc check (not preregistered, used only to interpret D2a)
Question: is the frequency effect "common knowledge is retained longer", or "a damaged model guessing answers is biased toward common capital names"?
- CHOICE: when wrong after damage, choosing the most common of the three distractors accounts for 49–62% (random would be about 33%); a guessing bias exists.
- BARE: among wrong answers, only 2–11% write another capital name from the list (the rest are misspellings, other cities, refusals, etc.), and 63–100% of these substitutions are more common than the correct answer (e.g. writing Lagos for Nigeria).
- Conclusion: the frequency effect in the BARE main test cannot be explained by "guessing some common capital" (such guessing is almost never going to hit that country's capital exactly); part of the CHOICE effect comes from guessing bias, so BARE is taken as authoritative. Replacing a rare target with a common neighbor is itself a typical error type in semantic dementia.
