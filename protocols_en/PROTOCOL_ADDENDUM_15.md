# Addendum 15: Extension and controls for the frequency effect (preregistered, existing data only)

Written on 2026-10-29, before querying any new word frequencies or computing any new statistics. Approved by the investigator on the same day ("extend frequency first, then rewrite", translated).

## Motivation
- D2a of Addendum 14 held: on capital items, the more common the co-occurrence of "country AND capital", the later the item is lost after damage (BARE, median 0.423, 8/8 positive). But there is only one task, which cannot support a pillar of the main text.
- Literature collision: Wu et al. 2026 (arXiv 2608.19670, PopQA and Head-to-Tail, 3 models × 11 compressions) report the opposite direction: after compression, the relative retention of long-tail knowledge is higher than that of popular knowledge. Their design differs from ours in three ways: (1) they do not condition on the original model answering correctly, but look at the ratio of within-group accuracies; (2) "long-tail" is defined by the Wikipedia page views of the question subject, not by the answer; (3) they use relative retention, and the long-tail group's baseline accuracy is about 30 points lower. One testable reconciling explanation: damage makes the model collapse onto common outputs, so among long-tail-subject questions, items whose answer happens to be a common value are "guessed correctly" and thus retained. This addendum tests the three quantities needed for this.

## Data seen before this protocol (stated honestly)
- All results of Addendum 14 (including the per-model D2a values and the post hoc guessing-bias check).
- No per-item or frequency relationship for SciQ or RETR has been seen; no counts for any SciQ answer, RETR code or country name have been queried.
- In Addendum 14, "the count of the capital name itself" (0.561–0.777) was stronger than the preregistered co-occurrence count; this is known. In this addendum SciQ and RETR can only use answer-string counts (there is no definable "subject"); this is due to the task structure, not to this known result.

## Data and measures
- Per-item data: analysis15_cache.pkl (extracted for Addendum 14, unchanged).
- Survival score: for each model, the mean accuracy of the item on BARE over all conditions of the 7 damages; only items answered correctly on BARE by the intact model are kept; task-model cells with fewer than 30 items are not counted.
- Frequency: infini-gram, index v4_dolma-v1_7_llama, query the count of the answer string itself, take log10(count + 1); also record the tokenized token count.
- Intact margin: the log-probability difference between the correct candidate and the strongest distractor on the intact model's CHOICE.

## Tests and verdict
- **F1 SciQ (main test, knowledge task)**: within each model, Spearman(log answer count, survival score).
  - COMMIT: median over the 8 models ≥ 0.30, and at least 7/8 positive; PARTIAL: median ≥ 0.15, and at least 6/8 positive; STOP: otherwise.
- **F2 RETR (negative control, answer in context, no reliance on memory)**: the same quantity.
  - Pass: median < 0.10; fail: median ≥ 0.15 (indicating a bias that "common strings are easier to output"); in between: uncertain.
- **F3 splitting subject and answer for capital items (corresponding to the design of Wu et al. 2026, report only)**: country-name count and capital-name count, each with partial Spearman controlling for the other.
- Confounds (report only): partial Spearman controlling for answer token count and for intact margin; the same quantities under CHOICE.

## Main-text wording is determined by the verdict
- Condition for writing the frequency effect into the main text as "knowledge is lost by familiarity": F1 is COMMIT or PARTIAL, and F2 passes.
- F1 is COMMIT or PARTIAL, F2 fails: the main text can only write "frequency predicts survival", cannot interpret it as knowledge retention, and reports the output bias.
- F1 is STOP: the frequency effect is written only for the single capitals task, downgraded to one item in the supplementary material, not as a pillar of the main text.

## Lock
- analyze16.py md5: 6584c59680512033c47cebaa6ed0883b (the fetch step only retrieves counts, computes no statistics)

## Run log
1. 2026-10-29: analyze16.py fetch completed 601 queries (271 SciQ answers, 252 RETR codes, 78 country names), rate-limited at 2 seconds, no errors.
2. Then analyze16.py run (md5 unchanged), output analysis16.json. The token-count control for RETR has no variation (the codes are all three-digit numbers), so the partial correlation equals the original correlation, as expected.

## Results (analysis16.json)
- **F1 SciQ: COMMIT.** Under BARE, the median Spearman(log answer count, survival score) is 0.383, 8/8 positive (0.298–0.449; 132–188 items per model). Controlling for token count, 0.296–0.444; controlling for intact margin, 0.322–0.452; intact margin itself only 0.041–0.273. Weaker under CHOICE (0.176–0.276), and CHOICE survival is mainly predicted by intact margin (0.549–0.658).
- **F2 RETR negative control: pass.** Median 0.079 (< 0.10), 5/8 positive; but three models, llama 0.283, gemma 0.213 and q15 0.164, have residual positive correlations, and the median under CHOICE is about 0.25. This indicates the bias that "common strings are easier to output" exists but is small; the effect on the knowledge task (0.38–0.42) is about 5 times that of the control.
- **F3 splitting subject and answer for capital items:** the capital-name (answer) count controlling for country name gives partial Spearman 0.476–0.697, 8/8 positive; the country-name (subject) count controlling for capital name gives −0.222 to +0.176, median about 0.06. The two counts themselves correlate at 0.66–0.68.
- **Per the preregistered wording rule:** F1 is COMMIT and F2 passes, so the frequency effect is written into the main text as "knowledge is lost by familiarity".
- **Relation to Wu et al. 2026:** what determines what remains after damage is the frequency of the answer, not the popularity of the question subject; after controlling for answer frequency, subject popularity has no independent effect. Grouping by subject popularity (Wu's design) therefore cannot detect "popular items are retained longer", which does not contradict our results. This is an interpretation of F3; F3 itself is report only and not judged.
