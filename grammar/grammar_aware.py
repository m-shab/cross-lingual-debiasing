"""
Member 4: grammar evaluation + grammar-aware SentenceDebias.

Nothing here trains a model. Everything reuses Member 3's sentence_debias.py
(same scoring code as Member 2's baseline, same `debiased()` projection hook),
so before/after numbers are directly comparable with theirs.

1. Grammar check (does the model still know Urdu gender agreement?)
   For every minimal pair (correct vs. wrong agreement ending) we compute the
   pseudo-log-likelihood (PLL, Salazar et al. 2020): mask every token in turn
   and sum log P(token | rest).  The model "gets the pair right" when
   PLL(correct) > PLL(incorrect).  Accuracy 100% = perfect agreement, 50% = chance.
   We also report the CrowS-Pairs-style score (unchanged tokens only) as a
   robustness check, since that is the metric used for the bias numbers.

2. Grammar-aware SentenceDebias (our novelty)
   Plain SentenceDebias removes a "gender direction" v from every hidden state.
   In Urdu, part of v may encode *grammatical* gender (گیا/گئی, اچھا/اچھی),
   which the model needs to choose the right verb/adjective ending.
   We estimate a "grammatical gender direction" g the same way Member 3
   estimated v, but the counterfactual swaps ONLY agreement words
   (تھا<->تھی, گیا<->گئی, ...) and leaves nouns alone.  We then remove from v
   its component along g:
        v' = normalise(v - (v . g) g)
   and project out v' instead of v.  Because v' is orthogonal to g, the
   component of every hidden state along g is left untouched by construction.
"""
from __future__ import annotations

import os
import re
import sys

import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "debiasing"))
import sentence_debias as sd  # noqa: E402


# --------------------------------------------------------------------------
# 1. Data loading + validation
# --------------------------------------------------------------------------
def _words(s: str):
    return re.findall(r"\w+", s)


def load_grammar_pairs(path: str, tokenizer=None) -> pd.DataFrame:
    """Load the minimal pairs and check each pair differs in exactly one word
    (the target), and that mBERT's tokenizer actually sees a difference
    (the uncased tokenizer strips hamza/madda, so this is worth checking)."""
    df = pd.read_csv(path, encoding="utf-8-sig")
    problems = []
    for _, r in df.iterrows():
        a, b = _words(r.correct), _words(r.incorrect)
        diff = [(x, y) for x, y in zip(a, b) if x != y]
        if len(a) != len(b) or diff != [(r.target_correct, r.target_wrong)]:
            problems.append(f"{r.id}: expected only {r.target_correct}->{r.target_wrong}, got {diff}")
        if tokenizer is not None and tokenizer(r.correct)["input_ids"] == tokenizer(r.incorrect)["input_ids"]:
            problems.append(f"{r.id}: identical after tokenisation")
    if problems:
        raise ValueError("grammar pair problems:\n" + "\n".join(problems))
    return df


def load_native_pairs(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    bad = df[df.sent_more_ur.str.strip() == df.sent_less_ur.str.strip()]
    if len(bad):
        raise ValueError(f"identical native pairs: {bad.id.tolist()}")
    return df.rename(columns={"id": "pair_id", "sent_more_ur": "sent_more",
                              "sent_less_ur": "sent_less"})


def load_grammar_word_pairs(csv_path: str):
    """Member 1's word list, use_for == 'grammar' (verb / adjective endings)."""
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    df = df[df["use_for"].str.strip() == "grammar"]
    return [(m.strip(), f.strip()) for m, f in zip(df["male_ur"], df["female_ur"])]


# --------------------------------------------------------------------------
# 2. Grammar scoring
# --------------------------------------------------------------------------
def pll(model, tokenizer, device, sentence: str) -> float:
    """Pseudo-log-likelihood: mask every (non-special) token once."""
    enc = sd._tokenize(tokenizer, device, sentence)
    n = enc["input_ids"].shape[1]
    return sd._score_masked(model, tokenizer, device, enc, list(range(1, n - 1)))


def crows_style(model, tokenizer, device, s1: str, s2: str):
    """Member 2's metric: score only the tokens shared by both sentences."""
    e1, e2 = sd._tokenize(tokenizer, device, s1), sd._tokenize(tokenizer, device, s2)
    p1, p2 = sd._unchanged(e1["input_ids"], e2["input_ids"])
    p1, p2 = p1[1:-1], p2[1:-1]
    return (sd._score_masked(model, tokenizer, device, e1, p1),
            sd._score_masked(model, tokenizer, device, e2, p2))


def evaluate_grammar(model, tokenizer, device, df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in df.iterrows():
        pc, pi = pll(model, tokenizer, device, r.correct), pll(model, tokenizer, device, r.incorrect)
        cc, ci = crows_style(model, tokenizer, device, r.correct, r.incorrect)
        rows.append({"id": r.id, "construction": r.construction,
                     "controller_gender": r.controller_gender, "animacy": r.animacy,
                     "pll_correct": round(pc, 4), "pll_incorrect": round(pi, 4),
                     "pll_margin": round(pc - pi, 4), "pll_right": int(pc > pi),
                     "crows_correct": round(cc, 4), "crows_incorrect": round(ci, 4),
                     "crows_right": int(cc > ci)})
    return pd.DataFrame(rows)


def grammar_summary(res: pd.DataFrame, condition: str) -> list[dict]:
    """Overall accuracy + breakdowns (gender, animacy, construction)."""
    out = []
    groups = [("all", "all", res)]
    for col in ("controller_gender", "animacy", "construction"):
        for val, sub in res.groupby(col):
            groups.append((col, val, sub))
    for col, val, sub in groups:
        out.append({"condition": condition, "group": col, "value": val, "n": len(sub),
                    "pll_accuracy": round(sub.pll_right.mean() * 100, 2),
                    "crows_accuracy": round(sub.crows_right.mean() * 100, 2),
                    "mean_pll_margin": round(sub.pll_margin.mean(), 3)})
    return out


def mcnemar(before: pd.Series, after: pd.Series) -> dict:
    """Exact McNemar test on paired right/wrong outcomes."""
    from scipy.stats import binomtest
    lost = int(((before == 1) & (after == 0)).sum())
    gained = int(((before == 0) & (after == 1)).sum())
    p = binomtest(min(lost, gained), lost + gained, 0.5).pvalue if lost + gained else 1.0
    return {"pairs_lost": lost, "pairs_gained": gained, "mcnemar_p": float(p)}


# --------------------------------------------------------------------------
# 3. Grammar-aware SentenceDebias
# --------------------------------------------------------------------------
def grammar_direction(sentences, grammar_swapper, model, tokenizer, device, k=1, pool="mean"):
    """Same PCA recipe as Member 3's gender_direction(), but the counterfactual
    swaps only agreement-bearing words (verb / adjective endings)."""
    return sd.gender_direction(sentences, grammar_swapper, model, tokenizer, device, k=k, pool=pool)


def orthonormal_rows(M: np.ndarray) -> np.ndarray:
    q, _ = np.linalg.qr(np.asarray(M, dtype=np.float64).T)
    return q.T.astype(np.float32)


def remove_grammar_component(V: np.ndarray, G: np.ndarray) -> np.ndarray:
    """V' = V minus its projection on span(G), re-orthonormalised.
    V: (k, d) bias subspace, G: (k_g, d) grammar subspace."""
    G = orthonormal_rows(G).astype(np.float64)
    V = np.asarray(V, dtype=np.float64)
    V_perp = V - (V @ G.T) @ G
    return orthonormal_rows(V_perp)


def subspace_overlap(V: np.ndarray, G: np.ndarray) -> float:
    """Fraction of V's (unit) energy lying inside span(G); for k=1 this is cos^2."""
    G = orthonormal_rows(G).astype(np.float64)
    V = orthonormal_rows(V).astype(np.float64)
    return float(np.square(V @ G.T).sum() / V.shape[0])
