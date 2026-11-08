# Reviewer package: The predictable order in which damaged language models lose their capabilities

Project page: https://zhixiangwang-cn.github.io/damage-order-artifacts/ (source in docs/).

- protocols/: the main protocol and every time-stamped amendment in the original Chinese; protocols_en/: English translations of the same files. Each protocol states its predictions, thresholds, the md5 of its locked analysis script, a run log and the result. FREQ_AUDIT documents the re-query of the capital co-occurrence counts.
- code/: damage, compression, evaluation and analysis scripts (analyzeN.py are the locked analyses).
- outputs/: the JSON outputs of every locked analysis.
- data/: frozen item sets (items_v8), pretraining counts (freq15, freq15_fix, freq16) and locked doses.
- CHECKSUMS.md: locked md5 per protocol against the current files.

- runs_v8/, runs_v9/: per-item outputs of every condition (gzipped JSON; host names removed). The derived cache used by the analyses is rebuilt with `python3 code/analyze15.py cache` run from this folder after copying code/*.py and data/* next to runs_v8/ and runs_v9/.
- code/analyze27_explore.py and outputs/analysis27_explore.json: exploratory robustness checks reported in Supplementary Note 9 (not preregistered).
