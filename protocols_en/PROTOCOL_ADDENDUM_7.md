# Addendum 7: Is the fragility exponent stable across models and across damage types (preregistered)

Written on 2026-10-12, before any calibration or formal results from this batch were produced. The only thing already done is a smoke test (30 items on each model in the intact state, plus one dose of each of three damage types to confirm the code runs).

## Question
The pooled analysis found: the fragility exponent of the same measure is consistent between noise and random layer deletion (0.85–0.88), and broadly consistent between the two 7B models (0.78–0.81). Does this still hold across more models and more damage types? This batch proposes no new explanation; it only tests the scope of this existing conclusion.

## Models (7, all already cached on the lab server)
- 5 new: Qwen2.5-1.5B-Instruct, Qwen2.5-14B-Instruct, Llama-3.2-3B-Instruct, Gemma-2-9B-it, Phi-3.5-mini-instruct.
- 2 existing: Qwen2.5-7B-Instruct, Mistral-7B-Instruct-v0.3. Their noise and random layer deletion data are carried over from Addendum 6; only the new damage types are run additionally.
The three Qwen sizes form a within-family scale comparison; the rest come from different organizations.

## Measures
The same 29 as in Addendum 6: three tasks × six formats (scoring "both format and content correct"), copying 6 words, and the free generation and forced choice of five reference tasks. Usability threshold as before.
The "end-of-turn token" used by forced choice is now obtained automatically from each model's chat template (for Qwen and Mistral, the result is the same as what was used before).

## Damage
Four position-agnostic types:
- Noise: add Gaussian noise to every weight matrix (as before).
- Random zeroing (new): set a random fraction p of the weights in each layer's weight matrices to zero.
- Random neuron deletion (new): set a random fraction p of the units in the MLP hidden layer to zero (implemented by zeroing the corresponding columns of the output projection matrix).
- Random layer deletion: as before.
One position-specific damage type as a control:
- Late-segment layer deletion: delete n consecutive layers before the last layer.
Random damage types use 3 seeds.

## Doses: automatic calibration, rules hard-coded
Learning from the Mistral noise episode, calibration uses 2 seeds, and each model and each damage type is calibrated separately.
- 40 calibration items (different from the formal items); the metric is the mean corrected score over the usable ones among the 10 reference measures.
- Test along a preset coarse grid from light to heavy, until the mean score falls below 0.05.
- Continuous doses (noise, zeroing, neuron deletion): interpolate the monotonized calibration curve and take the 6 doses at which the mean score is 0.90, 0.75, 0.60, 0.45, 0.30, 0.12.
- Layer-count doses: take those among 1, 2, 3, 4, 5, 6, 8, 10, 12, 16 that do not exceed "the number of layers at which the mean score first falls below 0.08".
Calibration results and locked doses are saved per model.

## Analysis
- The exponent estimation method is exactly the same as in the pooled analysis: on cells between 0.03 and 0.97, fit "shared error intensity × per-measure exponent", fitting each "model × damage" separately. Only measures with at least 2 usable cells get an exponent.
- If a "model × damage" cell has fewer than 12 measures with exponents, that cell is recorded as "not measurable" and does not enter the verdict.
- Stability is measured by the Pearson correlation of log exponents, computed only on measures that have exponents on both sides; not computed if there are fewer than 10 common measures.

## Predictions (written before the results)
- H1 Within the same model, across the four position-agnostic damage types: each model has 6 damage pairs. The median correlation over all "model × damage pair" is ≥ 0.7; and in at least 6 of 7 models the within-model median is ≥ 0.6.
- H2 Across models: for each position-agnostic damage type, the median of the pairwise correlations between models (up to 21 pairs) is ≥ 0.6, satisfied for all four damage types.
- H3 Control: in each model, the correlation "noise vs. late-segment layer deletion" is lower than the median correlation among that model's position-agnostic damage types. Holds in at least 5 of 7 models. This is to show that stability does not automatically hold for any damage.
- H4 Addition ladder: exponents satisfy 4 numbers ≥ 3 numbers ≥ 2 numbers (free generation), holding in at least 80% of the "model × position-agnostic damage" cells where all three have exponents.
- Exploratory: how the exponents change across the three Qwen sizes; whether the two new damage types are more like noise or more like random layer deletion.

## What each outcome means
- H1, H2 and H3 all hold: the fragility exponent is a property of the measure itself, and this statement generalizes to 7 models and 4 damage types.
- H1 holds, H2 fails: the exponent is stable within a model, but each model has its own set. The conclusion must be narrowed to "within model".
- H1 fails: the earlier consistency was only a special case of the noise / random layer deletion pair. The main conclusion is withdrawn.
- H3 fails: late-segment layer deletion also gives the same exponents, showing that the distinction of "position-specific or not" does not matter, and earlier claims about position signatures must be re-examined.

## Honesty rules
As before. Doses are locked by code according to hard-coded rules, with no manual adjustment. After seeing the results, the exponent estimation method, thresholds and cutoffs will not be changed.

## Compute
Cards 2–6 of the lab server server_a, one process per card. Card 7 is not used, and server_b is not used.
