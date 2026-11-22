# Member 4: Native Urdu data, grammar check, grammar-aware SentenceDebias

No training happens here. Everything reuses `debiasing/sentence_debias.py` (Member 3), which uses the same scoring code as Member 2's baseline. Our "before" scores match `baseline/` exactly, and our EN/UR "after" scores match `debiasing/results/debias_summary.csv`.

## Run

```bash
uv venv .venv --python 3.11 && uv pip install --python .venv/bin/python torch transformers pandas numpy scikit-learn scipy tqdm
.venv/bin/python grammar/run_grammar.py          # ~35 min on an M-series Mac; results are cached per condition
.venv/bin/python grammar/run_grammar.py --force  # recompute everything
```

## Data (in `data/`)

| file | what |
|---|---|
| `native_urdu_stereotypes.csv` | 91 hand-written Urdu stereotype pairs set in Pakistan, not translated (household, occupation, education, leadership, finance, family, traits, mobility). `sent_more` is the stereotypical sentence. `stereo_gender` says which gender the stereotype is about (44 F / 47 M), so a model that simply prefers male sentences does not look "biased". |
| `urdu_grammar_pairs.csv` | 104 agreement minimal pairs (52 masc. / 52 fem. controller). Each pair differs in **exactly one** word (checked by the loader, which also checks that mBERT's tokens differ). 10 constructions: adjective, copula past, perfective, habitual, progressive, future, modal, attractor (`میری بہن کا بیٹا … گیا`), inanimate (`گاڑی بڑی`), and ergative (`لڑکے نے روٹی کھائی`, where the verb agrees with the object). |

## Method

* **Grammar check.** For each pair we compute the pseudo-log-likelihood (PLL: mask every token, sum log-probs). The model is right when PLL(correct) > PLL(wrong). We also report the CrowS-style score (unchanged tokens only) as a robustness check, and the *margin* PLL(correct) − PLL(wrong), which measures how confident the model is.
* **Grammar-aware SentenceDebias (ours).** We take Member 3's saved Wikipedia sentences and swap **only agreement words** (Member 1's `use_for = grammar` list: تھا↔تھی, گیا↔گئی, …), leaving nouns alone. PCA on these pairs gives a *grammatical gender direction* `g`, using the same recipe as Member 3's `v`. We then debias with `v' = normalise(v − (v·g)g)`. Since `v' ⟂ g`, the grammatical-gender component of every hidden state is left untouched by construction.
* **Stress test.** We recompute 5 bias directions (k=5) from the same Wikipedia sentences, i.e. stronger debiasing.

## Results (`results/headline.csv`)

| condition | grammar acc. | fem.-subject acc. | PLL margin | Urdu CrowS stereo % | Native Urdu stereo % | English CrowS stereo % |
|---|---|---|---|---|---|---|
| original mBERT | 76.0 | 71.2 | 2.63 | 46.7 | 63.7 | 50.4 |
| SentenceDebias, EN dir | 74.0 | 65.4 | 2.00 | 48.3 | 63.7 | 53.4 |
| SentenceDebias, UR dir | 71.2 | **59.6** (p=.03) | 2.05 | 48.8 | 63.7 | 51.9 |
| **grammar-aware, EN dir** | **77.9** | 75.0 | 2.59 | 47.1 | 64.8 | 50.4 |
| **grammar-aware, UR dir** | 74.0 | 67.3 | 2.63 | 46.3 | 61.5 | 50.4 |
| SentenceDebias, EN k=5 | **68.3** (p=.02) | **53.8** (p=.004) | 1.68 | 49.6 | 60.4 | 54.6 |
| SentenceDebias, UR k=5 | 71.2 | 61.5 | 1.96 | 48.8 | 62.6 | 51.5 |
| grammar-aware, EN k=5 | 76.0 | 71.2 | 2.51 | 46.7 | 59.3 | 52.3 |
| grammar-aware, UR k=5 | 76.0 | 69.2 | 2.67 | 47.5 | 61.5 | 50.8 |

p-values are exact McNemar tests against original mBERT, on the same pairs. Full breakdowns are in `results/grammar_summary.csv`, `results/bias_summary.csv` and `results/paired_tests.csv`; pair-level outputs are in `results/pairs/`.

## Findings

1. **About half of the "gender direction" is grammar.** In `results/diagnostics.json`, 52% (EN) and 45% (UR) of Member 3's gender direction lies along the grammatical-agreement direction (|cos| ≈ 0.72 / 0.67).
2. **SentenceDebias damages Urdu agreement, and only for feminine subjects.** Masculine-subject accuracy stays at ~81–83% in every condition. Feminine-subject accuracy falls from 71.2% to 59.6% with the Urdu direction (6 pairs lost, 0 gained, p=0.03) and to 53.8% (≈ chance) with the stronger English k=5 debiasing (p=0.004). The model's confidence on feminine pairs roughly halves (margin 2.22 → 1.22). Removing "gender" pushes the model toward a **masculine default**. Ergative (object agreement) also drops, 75% → 50%.
3. **Grammar-aware debiasing removes this damage.** No grammar-aware condition loses a significant number of pairs, and the PLL margin stays at its original level (2.51–2.67 vs 2.63), for both EN and UR directions and for k=1 and k=5. On bias it is never worse: English stays at 50.4% where plain EN debiasing overshot to 53.4%.
4. **Translated data under-measures bias.** On native Urdu sentences mBERT prefers the stereotype 63.7% of the time (95% CI 53.8–73.6, which **excludes 50**). On translated CrowS it is 46.7% (CI includes 50). The effect is strongest for stereotypes about men (70.2%, e.g. men as earners and decision-makers). No debiasing variant moves the native score significantly.
5. **Caution on Member 3's small Urdu "improvement".** On the 28 CrowS pairs separated only by an agreement suffix (Member 1's `morphological` pairs), plain UR debiasing moves 32.1% → 39.3%, while grammar-aware stays at 28.6%. Part of the apparent move toward 50% may come from the model losing agreement information, not from losing stereotypes (only 28 pairs, so this is suggestive).

## Caveats

* Bias changes are small and none are significant; the translated baseline was already near 50%.
* 104 grammar pairs and 91 native pairs are small. The grammar and native data were written by one author and should get a second native-speaker check.
* `g` is estimated from Wikipedia sentences, which are disjoint from the evaluation pairs, so there is no leakage. Member 1's word list may still change (v1).
