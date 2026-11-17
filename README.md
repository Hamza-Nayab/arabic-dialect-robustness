# Dialect-Level Generalization of Arabic Sentiment Classifiers: An Exact-Duplicate-Free, Uncertainty-Aware Evaluation of Encoders and LLMs

**Authors:** Malik Mir Hamza Nayab, Syed Ghazi Raza and Baseera Bibi  
**Affiliation:** Department of Computer Science, COMSATS University Islamabad, Abbottabad Campus, Abbottabad, Pakistan  
**Repository:** [https://github.com/Hamza-Nayab/arabic-dialect-robustness](https://github.com/Hamza-Nayab/arabic-dialect-robustness)  
**License:** MIT (Code) / CC BY 4.0 (Manuscript and Derived Split Metadata)

## Overview

This repository contains the sanitized dataset splits, multi-seed model predictions, statistical verification suites and reproduction scripts for the study *"Dialect-Level Generalization of Arabic Sentiment Classifiers: An Exact-Duplicate-Free, Uncertainty-Aware Evaluation of Encoders and LLMs."*

We evaluate how evenly 11 fine-tuned transformer encoders (AraBERTv02, AraBERTv02-Twitter, AraELECTRA, MARBERT, MARBERTv2, ARBERTv2, CAMeLBERT-Mix, CAMeLBERT-MSA, CAMeLBERT-DA, CAMeLBERT-CA and XLM-R-Base) and four instruction-tuned large language models (Qwen2.5-7B-Instruct, ALLaM-7B-Instruct, AceGPT-v2-8B-Chat and Fanar-1-9B-Instruct) generalize across Modern Standard Arabic (MSA), Egyptian, Levantine and Gulf Arabic on an exact-duplicate-free, audited version of ArSarcasm, with a full transfer replication on ArSarcasm-v2.

## Key Contributions and Findings

1. **Data Hygiene and Leakage Audit:** We identify 44 normalized tweets shared between the official ArSarcasm training and test splits (within 118 duplicate or leaking instances) and quantify the resulting score inflation (-0.005 to +0.004 HM-F1 across four core encoders). We release deduplicated split assignments alongside character 4-gram Jaccard near-duplicate audits and automatic dialect-identification (MADAR-6 and NADI) cross-checks.
2. **Subgroup Robustness (RQ1):** Under pooled in-distribution training, MARBERTv2 achieves the highest harmonic-mean macro-F1 (0.870) and worst-dialect F1 (0.833), but differences among the seven strongest Arabic encoders and the best 8-shot LLM (Fanar-1-9B-Instruct at 0.849 HM-F1) lie within two-stage hierarchical bootstrap uncertainty. Levantine is consistently the hardest dialect across all 11 encoders, confirmed by a Bayesian beta-binomial mixed model fitted in NumPyro.
3. **Powered Cross-Dialect Transfer (RQ2):** Using size-matched pooled controls, two one-sided equivalence tests (TOST) and minimum detectable effect (MDE) bounds at 80% power, we show that leaving the target dialect out (LODO) or training on MSA alone costs Arabic encoders no detectable HM-F1 (excluding losses above 0.045 HM-F1, while single-dialect effects below roughly 0.07 to 0.10 F1 lie below the sample resolution limit). Adding in-dialect training tweets yields no statistically detectable gain for Arabic encoders, a finding that replicates on ArSarcasm-v2.
4. **Pretraining Register and Scale (RQ3):** Across the shared-tokenizer CAMeLBERT family, mixed and MSA pretraining significantly outperform dialect-only and classical-only pretraining after Holm-Bonferroni correction, though pretraining corpus scale (ranging from 6 GB for CA to 167 GB for Mix) covaries with register composition.
5. **Critique of Dispersion-Only Robustness Metrics (RQ4):** Evaluating 79 systems, we demonstrate that variance-based dispersion indices on accuracy or F1 rank a degenerate majority-class classifier among the most robust systems (rank 1 of 79 on DRI-F1) because they ignore performance level. We recommend level-aware summaries (HM-F1 and worst-dialect F1) paired with hierarchical bootstrap intervals over tweets and seeds.

## Repository Structure

The complete manuscript PDF and LaTeX source are archived on Zenodo, while this GitHub repository hosts the reproduction pipeline, sanitized splits and multi-seed prediction arrays:

* `experiments/` : End-to-end Python pipeline for data deduplication, encoder fine-tuning, linear probing, LLM prompting, hierarchical bootstrap estimation, NumPyro MCMC mixed modeling and artifact generation.
* `experiments/outputs_v2/` : Sanitized split metadata (internal row indices, split assignments and labels with raw tweet text removed in compliance with data redistribution policies), multi-seed prediction arrays (`.npy`) and aggregated statistical summaries.

## Reproducibility and Verification

To regenerate all tables and figures from the saved prediction arrays and run the automated verification suites offline:

```bash
export HF_HUB_OFFLINE=1
export MPLBACKEND=Agg

python3 experiments/08_generate_paper_artifacts.py
python3 experiments/08b_review_artifacts.py

python3 experiments/verify_all.py          # 168/168 implementation and data checks
python3 experiments/verify_text_numbers.py # 48/48 manuscript prose and constant checks
```
