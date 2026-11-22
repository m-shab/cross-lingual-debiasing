"""
No training happens anywhere in this file. We
  1. collect Wikipedia sentences that contain a gender word,
  2. make a counterfactual of each sentence (swap every gender word for its
     opposite-gender partner),
  3. embed both versions with mBERT, centre each pair on its own mean, and run
     PCA  ->  top principal component = "gender direction"  (Liang et al. 2020),
  4. at scoring time, subtract every hidden state's projection on that direction
     right before mBERT's language-model head.

The CrowS-Pairs scoring code (section 6) is a line-for-line port of Member 2's
notebook, with the model/tokenizer passed in so we can score a debiased model
with exactly the same procedure as the baseline.
"""
from __future__ import annotations

import itertools
import json
import random
import re
from contextlib import contextmanager
from difflib import SequenceMatcher

import numpy as np
import pandas as pd
import torch

MODEL_NAME = "bert-base-multilingual-uncased"   # same as baseline


# --------------------------------------------------------------------------
# 1. Gender word pairs
# --------------------------------------------------------------------------
EN_PAIRS = [
    ("he", "she"), ("himself", "herself"), ("his", "her"),
    ("man", "woman"), ("men", "women"), ("boy", "girl"), ("boys", "girls"),
    ("male", "female"), ("males", "females"), ("gentleman", "lady"),
    ("gentlemen", "ladies"), ("father", "mother"), ("fathers", "mothers"),
    ("son", "daughter"), ("sons", "daughters"), ("brother", "sister"),
    ("brothers", "sisters"), ("husband", "wife"), ("husbands", "wives"),
    ("grandfather", "grandmother"), ("grandson", "granddaughter"),
    ("uncle", "aunt"), ("nephew", "niece"), ("king", "queen"),
    ("kings", "queens"), ("prince", "princess"), ("groom", "bride"),
    ("actor", "actress"), ("waiter", "waitress"), ("boyfriend", "girlfriend"),
    ("policeman", "policewoman"), ("chairman", "chairwoman"),
    ("businessman", "businesswoman"), ("sir", "madam"), ("dad", "mom"),
    ("stepfather", "stepmother"), ("widower", "widow"),
    # object / possessive-pronoun forms. setdefault() in GenderSwapper keeps the
    # first mapping, so (his, her) still wins for "her" -> "his".
    ("him", "her"), ("hers", "his"),
]
EN_PRONOUN_WORDS = {"he", "she", "himself", "herself", "his", "her"}


# Terms that are mostly names / honorifics / kunyas in Urdu Wikipedia
# (ابو الوفاء, امام مالک, دوست محمد, حاجی ..., صاحب after a name) rather than
# gendered nouns; swapping them corrupts the counterfactual, so they are excluded.
EXCLUDE_UR = {"ابو", "مالک", "صاحب", "حاجی", "جناب", "دوست", "سہیلی"}

# Common Urdu agreement-bearing tokens (verb endings, possessives). A noun-only
# swap leaves these unchanged, giving ungrammatical counterfactuals such as
# "اپنی بیٹا".  Used only if the optional agreement filter is switched on.
UR_AGREEMENT_RX = re.compile(
    r"(?<!\w)(تھا|تھے|تھی|تھیں|گیا|گئے|گئی|گئیں|ہوا|ہوئی|ہوئے|رہا|رہی|رہے|"
    r"آیا|آئی|آئے|کرتا|کرتی|کرتے|اپنا|اپنی|اپنے|کا|کی|کے)(?!\w)"
)


def load_urdu_pairs(csv_path: str, exclude=EXCLUDE_UR):
    """Member 1's file; keep ONLY use_for == 'debias', minus `exclude`."""
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    df = df[df["use_for"].str.strip() == "debias"]
    pairs = [(m.strip(), f.strip()) for m, f in zip(df["male_ur"], df["female_ur"])]
    return [(m, f) for m, f in pairs if m not in exclude and f not in exclude]


class GenderSwapper:
    """Bidirectional whole-word swap.  Longest terms are matched first so that
    multi-word terms (طالب علم) win over their parts."""

    def __init__(self, pairs):
        self.map = {}
        for m, f in pairs:
            self.map.setdefault(m.lower(), f)
            self.map.setdefault(f.lower(), m)
        terms = sorted(self.map, key=len, reverse=True)
        # (?<!\w)/(?!\w) = whole-word match that also works for Urdu script
        self.rx = re.compile(
            r"(?<!\w)(" + "|".join(re.escape(t) for t in terms) + r")(?!\w)",
            flags=re.IGNORECASE,
        )

    def find(self, sent: str):
        return [m.group(0).lower() for m in self.rx.finditer(sent)]

    def swap(self, sent: str) -> str:
        def rep(m):
            src = m.group(0)
            dst = self.map[src.lower()]
            return dst.capitalize() if src[:1].isupper() else dst
        return self.rx.sub(rep, sent)


# --------------------------------------------------------------------------
# 2. Corpus -> sentences that contain a gender word
# --------------------------------------------------------------------------
_SPLIT_EN = re.compile(r"(?<=[.!?])\s+")
_SPLIT_UR = re.compile(r"(?<=[۔.!?؟])\s+")


def split_sentences(text: str, lang: str):
    rx = _SPLIT_UR if lang == "ur" else _SPLIT_EN
    for line in text.split("\n"):
        for s in rx.split(line.strip()):
            s = s.strip()
            if s:
                yield s


def iter_wikipedia(lang: str, max_articles: int = 200_000, seed: int | None = 0):
    """Stream Wikipedia articles (no full download).  Needs internet + `datasets`.
    seed != None -> shuffled (shard order + 10k-article buffer) so the sample is
    not just the first articles of the dump; change the seed to resample."""
    from datasets import load_dataset
    ds = load_dataset("wikimedia/wikipedia", f"20231101.{lang}",
                      split="train", streaming=True)
    if seed is not None:
        ds = ds.shuffle(seed=seed, buffer_size=10_000)
    for i, row in enumerate(ds):
        if i >= max_articles:
            break
        yield row["text"]


def iter_textfile(path: str):
    with open(path, encoding="utf-8") as f:
        yield f.read()


def collect_sentences(texts, lang, swapper, n_target=10_000, max_per_term=400,
                      min_words=6, max_words=50, seed=0, reject_regex=None):
    """Return up to n_target distinct sentences containing >=1 gender term.
    max_per_term stops frequent words ('he', 'his') from dominating the
    direction: a sentence is counted against its FIRST matched term."""
    rng = random.Random(seed)
    per_term, out, seen = {}, [], set()
    for text in texts:
        for s in split_sentences(text, lang):
            n = len(s.split())
            if n < min_words or n > max_words or s in seen:
                continue
            if reject_regex is not None and reject_regex.search(s):
                continue
            hits = swapper.find(s)
            if not hits:
                continue
            t = hits[0]
            if per_term.get(t, 0) >= max_per_term:
                continue
            per_term[t] = per_term.get(t, 0) + 1
            seen.add(s)
            out.append(s)
            if len(out) >= n_target:
                break
        if len(out) >= n_target:
            break
    rng.shuffle(out)
    return out, per_term


# --------------------------------------------------------------------------
# 3. Gender direction (SentenceDebias)
# --------------------------------------------------------------------------
@torch.inference_mode()
def embed(sentences, model, tokenizer, device, pool="mean", batch_size=64,
          max_length=128):
    """Sentence vectors from the LAST hidden layer of the encoder.
    pool='mean' : attention-masked mean over tokens (default)
    pool='cls'  : [CLS] vector
    NOTE: always call with the *un-debiased* model."""
    out = []
    for i in range(0, len(sentences), batch_size):
        enc = tokenizer(sentences[i:i + batch_size], return_tensors="pt",
                        padding=True, truncation=True, max_length=max_length)
        enc = {k: v.to(device) for k, v in enc.items()}
        h = model.bert(**enc).last_hidden_state            # (B, T, 768)
        if pool == "cls":
            v = h[:, 0]
        else:
            m = enc["attention_mask"].unsqueeze(-1).to(h.dtype)
            v = (h * m).sum(1) / m.sum(1)
        v = torch.nn.functional.normalize(v.float(), p=2, dim=-1)
        out.append(v.cpu())
    return torch.cat(out).numpy()


def gender_direction(sentences, swapper, model, tokenizer, device,
                     k=1, pool="mean", batch_size=64):
    """PCA on pair-centred sentence embeddings (Liang et al., 2020).
    Returns dict(direction=(k,d) orthonormal, explained=..., n_pairs=...)."""
    from sklearn.decomposition import PCA
    cf = [swapper.swap(s) for s in sentences]
    keep = [(s, c) for s, c in zip(sentences, cf) if s != c]
    orig = [s for s, _ in keep]
    cfs = [c for _, c in keep]
    a = embed(orig, model, tokenizer, device, pool, batch_size)
    b = embed(cfs, model, tokenizer, device, pool, batch_size)
    mu = (a + b) / 2.0
    X = np.concatenate([a - mu, b - mu], axis=0)            # pair-centred
    if len(keep) < 2 or k < 1 or k > X.shape[1]:
        raise ValueError("Not enough valid sentence pairs for the requested PCA components")
    pca = PCA(n_components=k, svd_solver="full").fit(X)
    return {
        "direction": pca.components_.astype(np.float32),     # unit-norm rows
        "explained": pca.explained_variance_ratio_.tolist(),
        "n_pairs": len(keep),
        "pool": pool,
    }


def save_direction(d, path_prefix, meta=None):
    np.save(path_prefix + ".npy", d["direction"])
    with open(path_prefix + ".json", "w", encoding="utf-8") as f:
        json.dump({"explained": d["explained"], "n_pairs": d["n_pairs"],
                   "pool": d["pool"], **(meta or {})}, f, indent=2)


def load_direction(path_prefix):
    return np.load(path_prefix + ".npy")


# --------------------------------------------------------------------------
# 4. Applying the debiasing (no training: a projection before the LM head)
# --------------------------------------------------------------------------
@contextmanager
def debiased(model, direction):
    """Within the `with` block `model` has the gender subspace projected out of
    every token's final hidden state:   h <- h - sum_i <h,v_i> v_i .
    Use for Member 2's scoring AND Member 4's grammar check."""
    V = torch.as_tensor(np.asarray(direction), dtype=torch.float32)
    V = V.to(next(model.parameters()).device)               # (k, d)

    def pre_hook(module, args, kwargs):
        h = args[0] if args else kwargs["hidden_states"]
        h = h - (h @ V.T.to(h.dtype)) @ V.to(h.dtype)
        if args:
            return (h,) + tuple(args[1:]), kwargs
        kwargs = dict(kwargs); kwargs["hidden_states"] = h
        return args, kwargs

    handle = model.cls.register_forward_pre_hook(pre_hook, with_kwargs=True)
    try:
        yield model
    finally:
        handle.remove()


# --------------------------------------------------------------------------
# 5. Sanity diagnostics (report-friendly numbers)
# --------------------------------------------------------------------------
def cosine(u, v):
    u, v = np.ravel(u), np.ravel(v)
    return float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v)))


def residual_gender_signal(sentences, swapper, model, tokenizer, device,
                           direction, pool="mean"):
    """Mean |<h(s) - h(cf(s)), v>| before vs after removal.
    After removal this is 0 by construction at the *projection* level; the
    useful number is the 'before' value and how it compares between
    directions (e.g. English direction applied to Urdu sentences)."""
    cf = [swapper.swap(s) for s in sentences]
    pairs = [(s, c) for s, c in zip(sentences, cf) if s != c]
    a = embed([p[0] for p in pairs], model, tokenizer, device, pool)
    b = embed([p[1] for p in pairs], model, tokenizer, device, pool)
    v = np.asarray(direction)[0]
    before = np.abs((a - b) @ v).mean()
    total = np.linalg.norm(a - b, axis=1).mean()
    return {"mean_abs_proj_diff": float(before), "mean_norm_diff": float(total),
            "fraction_along_direction": float(before / total) if total else float("nan")}


# --------------------------------------------------------------------------
# 6. CrowS-Pairs scoring 
# --------------------------------------------------------------------------
def _tokenize(tokenizer, device, sentence, max_length=512):
    enc = tokenizer(str(sentence), return_tensors="pt", truncation=True,
                    max_length=max_length)
    return {k: v.to(device) for k, v in enc.items()}


def _unchanged(ids1, ids2):
    t1, t2 = ids1[0].tolist(), ids2[0].tolist()
    p1, p2 = [], []
    for blk in SequenceMatcher(None, t1, t2).get_matching_blocks():
        for o in range(blk.size):
            p1.append(blk.a + o); p2.append(blk.b + o)
    return p1, p2


def _score_masked(model, tokenizer, device, encoded, positions, batch_size=32):
    ids = encoded["input_ids"][0]
    total = 0.0
    for s in range(0, len(positions), batch_size):
        bp = positions[s:s + batch_size]
        n = len(bp)
        masked = ids.unsqueeze(0).repeat(n, 1)
        targets = []
        for i, p in enumerate(bp):
            targets.append(ids[p].item())
            masked[i, p] = tokenizer.mask_token_id
        inp = {"input_ids": masked,
               "attention_mask": encoded["attention_mask"].repeat(n, 1)}
        if "token_type_ids" in encoded:
            inp["token_type_ids"] = encoded["token_type_ids"].repeat(n, 1)
        with torch.inference_mode():
            logits = model(**inp).logits
            rows = torch.arange(n, device=device)
            cols = torch.tensor(bp, device=device)
            tg = torch.tensor(targets, device=device)
            lp = torch.log_softmax(logits[rows, cols], dim=-1)
            total += lp[rows, tg].sum().item()
    return total


def evaluate_pair(model, tokenizer, device, s_more, s_less, label):
    e1, e2 = _tokenize(tokenizer, device, s_more), _tokenize(tokenizer, device, s_less)
    p1, p2 = _unchanged(e1["input_ids"], e2["input_ids"])
    p1, p2 = p1[1:-1], p2[1:-1]           # drop [CLS]/[SEP]
    if not p1:
        return None
    sc1 = round(_score_masked(model, tokenizer, device, e1, p1), 3)
    sc2 = round(_score_masked(model, tokenizer, device, e2, p2), 3)
    label = str(label).strip().lower()
    if label == "stereo":
        st, an = sc1, sc2
    elif label == "antistereo":
        st, an = sc2, sc1
    else:
        raise ValueError(f"Unknown stereotype label: {label}")
    pref = 1 if st > an else (0 if st < an else 0.5)
    return {"stereotypical_score": st, "anti_stereotypical_score": an,
            "shared_tokens": len(p1), "preference": pref}


def evaluate_df(model, tokenizer, device, df, col_more, col_less,
                id_col="pair_id", progress=True):
    rows = df.iterrows()
    if progress:
        try:
            from tqdm import tqdm
            rows = tqdm(rows, total=len(df))
        except ImportError:
            pass
    res = []
    for idx, r in rows:
        out = evaluate_pair(model, tokenizer, device, r[col_more], r[col_less],
                            r["stereo_antistereo"])
        if out is None:
            continue
        res.append({"pair_id": r[id_col] if id_col in r else idx, **out})
    return pd.DataFrame(res)


def summarise(results, name):
    p = results["preference"]
    s = (p == 1).mean() * 100
    return {"dataset": name, "evaluated_pairs": len(results),
            "stereotype_preference_pct": round(s, 2),
            "anti_stereotype_preference_pct": round((p == 0).mean() * 100, 2),
            "ties_pct": round((p == 0.5).mean() * 100, 2),
            "deviation_from_50": round(abs(s - 50), 2)}


def paired_change(before, after, n_boot=2000, seed=0):
    """Paired change in stereotype-win rate, exact two-sided discordance test."""
    from scipy.stats import binomtest
    a = before.set_index("pair_id")["preference"]
    b = after.set_index("pair_id")["preference"]
    paired = pd.concat([a.rename("before"), b.rename("after")], axis=1, join="inner").dropna()
    x = (paired["before"] == 1).to_numpy(dtype=int)
    y = (paired["after"] == 1).to_numpy(dtype=int)
    toward = int(((x == 0) & (y == 1)).sum())
    away = int(((x == 1) & (y == 0)).sum())
    delta = (y - x) * 100.0
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(delta), size=(n_boot, len(delta)))
    lo, hi = np.percentile(delta[idx].mean(axis=1), [2.5, 97.5])
    return {"n_paired": len(paired), "delta_stereotype_pp": round(float(delta.mean()), 2),
            "delta_ci95_lo": round(float(lo), 2), "delta_ci95_hi": round(float(hi), 2),
            "changed_toward_stereotype": toward, "changed_away_from_stereotype": away,
            "paired_p_value": float(binomtest(min(toward, away), toward + away, 0.5).pvalue)
            if toward + away else 1.0}


# --------------------------------------------------------------------------
# 7. Whole experiment (what the notebook calls)
# --------------------------------------------------------------------------
def ci95(results, n_boot=2000, seed=0):
    """Bootstrap 95% CI for the reported stereotype_preference_pct
    (share of pairs where the stereotypical sentence strictly wins)."""
    p = (results["preference"].to_numpy(dtype=float) == 1).astype(float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(p), size=(n_boot, len(p)))
    m = p[idx].mean(1) * 100
    return round(float(np.percentile(m, 2.5)), 1), round(float(np.percentile(m, 97.5)), 1)


def run_experiment(model, tokenizer, device, en_sents, ur_sents, eval_sets,
                   out_dir, en_swapper, ur_swapper, k=1, pool="mean",
                   baseline_summary_csv=None):
    """eval_sets: {name: DataFrame[pair_id, sent_more, sent_less, stereo_antistereo]}
    Returns the summary DataFrame (also saved to out_dir/debias_summary.csv)."""
    import os
    os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame({"sentence": en_sents}).to_csv(f"{out_dir}/wiki_sentences_en.csv", index=False)
    pd.DataFrame({"sentence": ur_sents}).to_csv(f"{out_dir}/wiki_sentences_ur.csv", index=False)

    # -- directions (always from the UN-debiased model) ----------------------
    dirs = {}
    for name, sents, sw in (("en", en_sents, en_swapper), ("ur", ur_sents, ur_swapper)):
        d = gender_direction(sents, sw, model, tokenizer, device, k=k, pool=pool)
        save_direction(d, f"{out_dir}/gender_direction_{name}",
                       {"lang": name, "model": MODEL_NAME, "k": k})
        dirs[name] = d
        print(f"{name} direction: {d['n_pairs']} sentence pairs, explained variance "
              f"{[round(x, 3) for x in d['explained']]}")
    _c = cosine(dirs["en"]["direction"][0], dirs["ur"]["direction"][0])
    diag = {"cosine_en_vs_ur_top_component": _c,
            "abs_cosine_en_vs_ur": abs(_c),   # PCA sign is arbitrary -> report this one
            "explained_en": dirs["en"]["explained"], "explained_ur": dirs["ur"]["explained"],
            "n_pairs_en": dirs["en"]["n_pairs"], "n_pairs_ur": dirs["ur"]["n_pairs"],
            "pool": pool, "k": k}
    # how much of a gender-swap difference lies along each direction, per language
    for dname in ("en", "ur"):
        for sname, sents, sw in (("en", en_sents, en_swapper), ("ur", ur_sents, ur_swapper)):
            r = residual_gender_signal(sents[:2000], sw, model, tokenizer, device,
                                       dirs[dname]["direction"], pool)
            diag[f"swap_diff_along_{dname}_dir__on_{sname}_sentences"] = r["fraction_along_direction"]
    with open(f"{out_dir}/direction_diagnostics.json", "w") as f:
        json.dump(diag, f, indent=2)
    print(json.dumps(diag, indent=2))

    # -- scoring: before / en-direction / ur-direction ------------------------
    # score the largest English set once; the 242 subset is filtered from it
    en_big = max((n for n in eval_sets if n.lower().startswith("english")),
                 key=lambda n: len(eval_sets[n]))
    conds = {"before": ("before (original mBERT)", None),
             "en_dir": ("after: English direction", dirs["en"]["direction"]),
             "ur_dir": ("after: Urdu direction", dirs["ur"]["direction"])}
    rows, pair_level = [], {}
    for ckey, (cname, V) in conds.items():
        cache = {}
        for dname, df in eval_sets.items():
            src = en_big if dname.lower().startswith("english") else dname
            if src not in cache:
                if V is None:
                    cache[src] = evaluate_df(model, tokenizer, device, eval_sets[src],
                                             "sent_more", "sent_less")
                else:
                    with debiased(model, V):
                        cache[src] = evaluate_df(model, tokenizer, device, eval_sets[src],
                                                 "sent_more", "sent_less")
            res = cache[src]
            res = res[res["pair_id"].isin(set(df["pair_id"]))].copy()
            pair_level[(cname, dname)] = res
            res.to_csv(f"{out_dir}/pairs__{ckey}__{dname.replace(' ', '_')}.csv", index=False)
            row = summarise(res, dname); row["condition"] = cname
            row["ci95_lo"], row["ci95_hi"] = ci95(res)
            rows.append(row)
    summ = pd.DataFrame(rows)
    # NOTE: no 'bias reduction %' column on purpose: with baselines this close to
    # 50% (deviation 0-3 pts) the ratio is unstable (e.g. -702%).  Use
    # deviation_from_50, the CIs and paired_comparisons.csv instead.
    summ = summ[["dataset", "condition", "evaluated_pairs", "stereotype_preference_pct",
                 "ci95_lo", "ci95_hi", "anti_stereotype_preference_pct", "ties_pct",
                 "deviation_from_50"]]
    summ.to_csv(f"{out_dir}/debias_summary.csv", index=False)

    comparisons = []
    for dname in eval_sets:
        base = pair_level[(conds["before"][0], dname)]
        for ckey in ("en_dir", "ur_dir"):
            cname = conds[ckey][0]
            comparisons.append({"dataset": dname, "condition": cname,
                                **paired_change(base, pair_level[(cname, dname)])})
    pd.DataFrame(comparisons).to_csv(f"{out_dir}/paired_comparisons.csv", index=False)

    # -- does our 'before' reproduce Member 2's baseline? ---------------------
    if baseline_summary_csv:
        name_map = {"English - 262": "English - all 262 pairs",
                    "English - 242 aligned": "English - 242 aligned pairs",
                    "Urdu - 242": "Urdu - 242 pairs"}
        b = pd.read_csv(baseline_summary_csv).set_index("dataset")["stereotype_preference_pct"]
        mine = summ[summ.condition.str.startswith("before")].set_index("dataset")["stereotype_preference_pct"]
        for d in mine.index:
            if name_map.get(d) in b.index:
                ok = abs(b[name_map[d]] - mine[d]) <= 0.5
                print(f"baseline check  {d:24s} mine={mine[d]:6.2f}  member2={b[name_map[d]]:6.2f}  "
                      + ("OK" if ok else "MISMATCH -- investigate before trusting the 'after' numbers"))
    return summ, dirs, pair_level
