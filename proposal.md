# Beyond Gender Neutrality: Investigating Cross-Lingual Gender Debiasing and Grammatical Preservation in Urdu Language Models

**By: Group 1**

## Abstract

Existing gender debiasing techniques have primarily been evaluated on English and European languages, with limited research on Urdu. Unlike English, Urdu uses grammatical gender extensively in adjectives and verbs, raising concerns that debiasing may remove important linguistic information alongside harmful stereotypes. This project extends Reusens et al. (2023) by investigating gender debiasing in Urdu, comparing translated and native Urdu datasets, evaluating English-to-Urdu debiasing transfer, and exploring whether grammatical information can be preserved during debiasing.

## 1. Introduction

Multilingual language models such as mBERT can learn gender stereotypes from their training data. Reusens et al. (2023) investigated the transferability of debiasing techniques across English, French, German, and Dutch, finding SentenceDebias to be the most effective overall. However, Urdu presents additional challenges due to its grammatical gender system and culturally specific stereotypes. This project aims to investigate whether existing techniques generalise to Urdu without compromising grammatical understanding.

## 2. Motivation & Problem Statement

Existing debiasing techniques may fail to distinguish between harmful gender stereotypes and grammatical gender information. In Urdu, adjectives and verbs change according to gender, meaning that removing gender-related representations may affect grammatical correctness.

Furthermore, translated English datasets may not accurately capture stereotypes prevalent in Pakistani society. Therefore, it is important to evaluate both the effectiveness and unintended consequences of gender debiasing in Urdu.

## 3. Research Objectives

1. **Cross-Lingual Debiasing:** Evaluate whether gender debiasing techniques transfer effectively from English to Urdu, and compare English-based and Urdu-based debiasing.

2. **Native vs. Translated Urdu:** Compare gender bias measurements on translated CrowS-Pairs examples and naturally written Urdu sentences containing culturally relevant stereotypes.

3. **Grammatical Preservation:** Investigate whether debiasing removes essential grammatical gender information and explore a grammar-aware approach that reduces social bias while preserving grammatical correctness.

4. **Open-Source vs. Commercial LLMs:** Discover how open-source LLMs perform differently from commercial LLMs in their effectiveness in debiasing language and compare their performance with the mBERT model used in the paper.

## 4. Related Work

Reusens et al. (2023), in *Investigating Bias in Multilingual Language Models: Cross-Lingual Transfer of Debiasing Techniques*, evaluated five debiasing techniques, including SentenceDebias, INLP, DensRay, Counterfactual Data Augmentation (CDA), and Dropout Regularisation, using mBERT. Their results demonstrated that debiasing can transfer across languages, with SentenceDebias achieving an average bias reduction of approximately 13%.

The study also identified cases of overcompensation, where debiasing increased measured bias rather than reducing it. The authors suggested that grammatical gender in languages such as French and German may influence debiasing outcomes. However, their experiments were limited to four European languages and relied substantially on translated evaluation datasets.

Our project builds on these limitations by extending the analysis to Urdu and examining whether grammatical gender and culturally specific stereotypes influence debiasing effectiveness.

## 5. Proposed Methodology

### 5.1 Dataset

We will use translated gender-related examples from CrowS-Pairs [2] alongside a small, manually constructed native Urdu dataset containing Pakistani gender stereotypes related to occupations, education, leadership, and household responsibilities.

Additionally, we will create grammatical minimal pairs to evaluate gender agreement in Urdu adjectives and verbs.

### 5.2 Approach

We will reproduce selected experiments from Reusens et al. (2023) using mBERT, with a primary focus on SentenceDebias.

We will:

- Compare English-based and Urdu-based debiasing.
- Evaluate both translated and native Urdu datasets.
- Examine grammatical correctness before and after debiasing.
- Explore grammar-aware SentenceDebias or grammatically consistent counterfactual data augmentation as a potential improvement.
- Analyse the effectiveness of commercial and open-source LLMs in a comparative manner to distinguish their performance in debiasing.

### 5.3 Baselines and Evaluation

The original mBERT model will serve as the baseline, compared against SentenceDebias and our proposed improvement.

Evaluation will include:

- **Stereotype Preference Scores:** Measure the model's preference for stereotypical associations.
- **Grammatical Agreement Accuracy:** Evaluate grammatical gender agreement in Urdu adjectives and verbs.
- **Cross-Lingual Debiasing Effectiveness:** Assess how effectively debiasing transfers from English to Urdu.

We will also analyse examples where reducing social bias negatively affects grammatical correctness.

## 6. Expected Results

We expect to identify differences in debiasing effectiveness between English and Urdu, as well as between translated and native Urdu datasets.

The experiments may reveal whether existing techniques remove useful grammatical information alongside social stereotypes. We aim to determine whether grammar-aware debiasing can reduce this trade-off while maintaining effective bias mitigation.

## 7. Conclusion

This project extends multilingual gender debiasing research to Urdu by examining cross-lingual transfer, cultural representation, and grammatical preservation. The goal is to evaluate existing techniques and explore improvements that reduce harmful stereotypes without compromising essential linguistic information.

## References

[1] Reusens, M., et al. (2023). *Investigating Bias in Multilingual Language Models: Cross-Lingual Transfer of Debiasing Techniques*. EMNLP.

[2] [CrowS-Pairs Dataset — GitHub](https://github.com/nyu-mll/crows-pairs/tree/master)
