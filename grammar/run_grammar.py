"""
Member 4 experiment runner.

    .venv/bin/python grammar/run_grammar.py            # run everything (cached per condition)
    .venv/bin/python grammar/run_grammar.py --force    # recompute everything

Conditions (all on the same pretrained mBERT, no training):
  before                 original mBERT
  en / ur                Member 3's saved SentenceDebias directions (k=1)
  en+GA / ur+GA          same directions with the grammatical-gender component removed (ours)
  en_k5 / ur_k5 (+GA)    stress test: 5 bias directions recomputed from Member 3's saved
                         Wikipedia sentences, to see whether stronger debiasing hurts grammar
  *_g3                   sensitivity: 3 grammar directions instead of 1

Evaluated on:
  grammar  - 104 Urdu agreement minimal pairs (data/urdu_grammar_pairs.csv)
  bias     - Urdu CrowS 242 (Member 1), native Urdu 91 (ours), English CrowS 262 (Member 2)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
import grammar_aware as ga  # noqa: E402

sd = ga.sd
OUT = os.path.join(HERE, "results")
M3 = os.path.join(ROOT, "debiasing", "results")
DATA = os.path.join(ROOT, "data")


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def main(force=False):
    os.makedirs(OUT, exist_ok=True)
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available() else "cpu")
    from transformers import AutoModelForMaskedLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(sd.MODEL_NAME)
    model = AutoModelForMaskedLM.from_pretrained(sd.MODEL_NAME).to(device).eval()
    log("device", device)

    # ---------------- data ----------------
    grammar = ga.load_grammar_pairs(f"{DATA}/urdu_grammar_pairs.csv", tok)
    native = ga.load_native_pairs(f"{DATA}/native_urdu_stereotypes.csv")
    ur = pd.read_csv(f"{DATA}/crows_pairs_gender_ur_eval.csv", encoding="utf-8-sig").rename(
        columns={"id": "pair_id", "sent_more_ur": "sent_more", "sent_less_ur": "sent_less"})
    en = pd.read_csv(f"{ROOT}/baseline/english_gender_all_results.csv")
    cols = ["pair_id", "sent_more", "sent_less", "stereo_antistereo"]
    bias_sets = {"Urdu CrowS - 242": ur[cols], "Native Urdu - 91": native[cols],
                 "English CrowS - 262": en[cols]}
    wiki_ur = pd.read_csv(f"{M3}/wiki_sentences_ur.csv").sentence.tolist()
    wiki_en = pd.read_csv(f"{M3}/wiki_sentences_en.csv").sentence.tolist()

    # ---------------- directions ----------------
    sw_en = sd.GenderSwapper(sd.EN_PAIRS)
    sw_ur = sd.GenderSwapper(sd.load_urdu_pairs(f"{DATA}/gender_word_pairs_ur.csv"))
    sw_gr = sd.GenderSwapper(ga.load_grammar_word_pairs(f"{DATA}/gender_word_pairs_ur.csv"))

    def cached_direction(name, sents, swapper, k):
        path = f"{OUT}/direction_{name}"
        if os.path.exists(path + ".npy") and not force:
            return np.load(path + ".npy"), json.load(open(path + ".json"))
        log("computing direction", name, f"k={k}")
        d = sd.gender_direction(sents, swapper, model, tok, device, k=k)
        sd.save_direction(d, path, {"k": k})
        return d["direction"], json.load(open(path + ".json"))

    V = {"en": sd.load_direction(f"{M3}/gender_direction_en"),
         "ur": sd.load_direction(f"{M3}/gender_direction_ur")}
    G1, g1_meta = cached_direction("grammar_ur_k1", wiki_ur, sw_gr, 1)
    G3, g3_meta = cached_direction("grammar_ur_k3", wiki_ur, sw_gr, 3)
    ur_k1_re, _ = cached_direction("bias_ur_k1_recomputed", wiki_ur, sw_ur, 1)
    V["en_k5"], en5_meta = cached_direction("bias_en_k5", wiki_en, sw_en, 5)
    V["ur_k5"], ur5_meta = cached_direction("bias_ur_k5", wiki_ur, sw_ur, 5)

    diag = {
        "reproduce_m3_ur_direction_abs_cos": abs(sd.cosine(ur_k1_re[0], V["ur"][0])),
        "grammar_k1_explained": g1_meta["explained"], "grammar_k1_n_pairs": g1_meta["n_pairs"],
        "grammar_k3_explained": g3_meta["explained"],
        "bias_en_k5_explained": en5_meta["explained"], "bias_ur_k5_explained": ur5_meta["explained"],
    }
    for b in ("en", "ur", "en_k5", "ur_k5"):
        diag[f"overlap_{b}_with_grammar_k1"] = ga.subspace_overlap(V[b], G1)
        diag[f"overlap_{b}_with_grammar_k3"] = ga.subspace_overlap(V[b], G3)
    diag["abs_cos_en_ur_bias"] = abs(sd.cosine(V["en"][0], V["ur"][0]))

    conds = {"before": None, "en": V["en"], "ur": V["ur"],
             "en+GA": ga.remove_grammar_component(V["en"], G1),
             "ur+GA": ga.remove_grammar_component(V["ur"], G1),
             "en_k5": V["en_k5"], "ur_k5": V["ur_k5"],
             "en_k5+GA": ga.remove_grammar_component(V["en_k5"], G1),
             "ur_k5+GA": ga.remove_grammar_component(V["ur_k5"], G1),
             "ur+GA_g3": ga.remove_grammar_component(V["ur"], G3),
             "ur_k5+GA_g3": ga.remove_grammar_component(V["ur_k5"], G3)}
    # sanity: the grammar-aware directions really are orthogonal to the grammar direction
    for c, M in conds.items():
        if M is not None and "+GA" in c:
            G = G3 if c.endswith("_g3") else G1
            assert ga.subspace_overlap(M, G) < 1e-6, c
    with open(f"{OUT}/diagnostics.json", "w") as f:
        json.dump(diag, f, indent=2)
    log(json.dumps(diag, indent=2))

    # ---------------- evaluation ----------------
    from contextlib import nullcontext
    grammar_rows, bias_rows, gram_res, bias_res = [], [], {}, {}
    for c, M in conds.items():
        tag = c.replace("+", "_plus_")
        gpath = f"{OUT}/pairs/grammar__{tag}.csv"
        os.makedirs(f"{OUT}/pairs", exist_ok=True)
        ctx = nullcontext() if M is None else sd.debiased(model, M)
        with ctx:
            if os.path.exists(gpath) and not force:
                g = pd.read_csv(gpath)
            else:
                log("grammar  ", c)
                g = ga.evaluate_grammar(model, tok, device, grammar)
                g.to_csv(gpath, index=False)
            gram_res[c] = g
            grammar_rows += ga.grammar_summary(g, c)
            for dname, df in bias_sets.items():
                bpath = f"{OUT}/pairs/bias__{tag}__{dname.split(' -')[0].replace(' ', '_')}.csv"
                if os.path.exists(bpath) and not force:
                    r = pd.read_csv(bpath)
                else:
                    log("bias     ", c, dname)
                    r = sd.evaluate_df(model, tok, device, df, "sent_more", "sent_less", progress=False)
                    r.to_csv(bpath, index=False)
                bias_res[(c, dname)] = r
                row = sd.summarise(r, dname)
                row["condition"] = c
                row["ci95_lo"], row["ci95_hi"] = sd.ci95(r)
                bias_rows.append(row)

    # native set: split by which gender the stereotype is about
    nat_gender = native.set_index("pair_id")["stereo_gender"]
    for c in conds:
        r = bias_res[(c, "Native Urdu - 91")]
        for gval in ("F", "M"):
            sub = r[r.pair_id.map(nat_gender) == gval]
            row = sd.summarise(sub, f"Native Urdu - stereotype about {gval}")
            row["condition"] = c
            row["ci95_lo"], row["ci95_hi"] = sd.ci95(sub)
            bias_rows.append(row)

    gsum = pd.DataFrame(grammar_rows)
    gsum.to_csv(f"{OUT}/grammar_summary.csv", index=False)
    bsum = pd.DataFrame(bias_rows)[["dataset", "condition", "evaluated_pairs", "stereotype_preference_pct",
                                    "ci95_lo", "ci95_hi", "anti_stereotype_preference_pct",
                                    "ties_pct", "deviation_from_50"]]
    bsum.to_csv(f"{OUT}/bias_summary.csv", index=False)

    # paired tests vs. original mBERT
    tests = []
    for c in conds:
        if c == "before":
            continue
        t = ga.mcnemar(gram_res["before"].pll_right, gram_res[c].pll_right)
        tests.append({"test": "grammar (PLL)", "condition": c,
                      "before_acc": round(gram_res["before"].pll_right.mean() * 100, 2),
                      "after_acc": round(gram_res[c].pll_right.mean() * 100, 2),
                      "mean_margin_change": round((gram_res[c].pll_margin - gram_res["before"].pll_margin).mean(), 4),
                      **t})
        for dname in bias_sets:
            pc = sd.paired_change(bias_res[("before", dname)], bias_res[(c, dname)])
            tests.append({"test": f"bias: {dname}", "condition": c, **pc})
    pd.DataFrame(tests).to_csv(f"{OUT}/paired_tests.csv", index=False)

    # one-line-per-condition headline table
    head = []
    for c in conds:
        g = gsum[(gsum.condition == c) & (gsum.group == "all")].iloc[0]
        rec = {"condition": c, "grammar_pll_acc": g.pll_accuracy, "grammar_crows_acc": g.crows_accuracy,
               "grammar_mean_pll_margin": g.mean_pll_margin}
        for dname in bias_sets:
            b = bsum[(bsum.condition == c) & (bsum.dataset == dname)].iloc[0]
            rec[f"stereo% {dname}"] = b.stereotype_preference_pct
        head.append(rec)
    head = pd.DataFrame(head)
    head.to_csv(f"{OUT}/headline.csv", index=False)
    pd.set_option("display.width", 250)
    print("\n", head.to_string(index=False))
    log("done")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    main(**vars(ap.parse_args()))
