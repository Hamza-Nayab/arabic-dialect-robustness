# -*- coding: utf-8 -*-
"""Step 6b: statistics added in response to review.

* Hierarchical (two-stage) bootstrap: tweets resampled within dialect (shared across systems -> paired),
  then the 5 seeds resampled with replacement per system; statistic = mean over the resampled seeds.
  Point estimates are seed means, so every interval matches the number printed next to it (fixes the
  ensemble-vs-mean mismatch). B = 2000.
* Transfer equivalence (TOST): delta = LODO - PID_n1531 and MSA_only - PID_n1905 (paired by tweet).
  Pre-stated margins: +/-0.02 HM-F1 and +/-0.05 per-dialect F1. Equivalent iff the 90% interval lies inside.
  Minimum detectable effect (80% power, two-sided alpha .05): MDE = 2.80 x bootstrap SE of the delta.
* Variance-based DRI on F1 (1/(1+Var(per-dialect F1))) as well as on accuracy; metric validity on seed-mean systems.
* Learning curve (5c), leaky vs clean (5d), ArSarcasm-v2 (5e) and LLM (9) summaries.
Output: outputs_v2/review_stats.json
"""
from __future__ import annotations

import json
from importlib.machinery import SourceFileLoader
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs_v2"
PR = OUT / "raw_preds"
step3 = SourceFileLoader("step3", str(ROOT / "03_model_and_fertility_audit.py")).load_module()
MAIN = ["MSA", "Egyptian", "Levantine", "Gulf"]
SEEDS = [42, 43, 44, 45, 46]
B = 2000
Z80 = 2.80
MARGIN_HM, MARGIN_D = 0.02, 0.05
PREFIX = {"PID": "PID", "PID_weighted": "PIDw", "LODO_matched": "LODO", "MSA_only": "MSAonly",
          "PID_n1531": "PIDn1531", "PID_n1905": "PIDn1905"}


def per_dialect_f1(y, p, d, dialects, idx_rep):
    """F1 per dialect for many bootstrap replicates at once. idx_rep: [B, n] indices. Returns [B, len(dialects)]."""
    yr, pr, dr = y[idx_rep], p[idx_rep], d[idx_rep]
    out = np.zeros((idx_rep.shape[0], len(dialects)))
    for k, g in enumerate(dialects):
        m = dr == g
        f = []
        for c in (0, 1):
            tp = ((pr == c) & (yr == c) & m).sum(1)
            fp = ((pr == c) & (yr != c) & m).sum(1)
            fn = ((pr != c) & (yr == c) & m).sum(1)
            f.append(np.where(tp == 0, 0.0, 2 * tp / np.maximum(2 * tp + fp + fn, 1)))
        out[:, k] = (f[0] + f[1]) / 2
    return out


def pooled_f1(y, p, idx_rep):
    yr, pr = y[idx_rep], p[idx_rep]
    f = []
    for c in (0, 1):
        tp = ((pr == c) & (yr == c)).sum(1)
        fp = ((pr == c) & (yr != c)).sum(1)
        fn = ((pr != c) & (yr == c)).sum(1)
        f.append(np.where(tp == 0, 0.0, 2 * tp / np.maximum(2 * tp + fp + fn, 1)))
    return (f[0] + f[1]) / 2


def summarize(per, pooled):
    """per: [..., n_dialects] -> dict of arrays."""
    hm = per.shape[-1] / np.sum(1.0 / np.maximum(per, 1e-6), axis=-1)
    return {"pooled_macro_f1": pooled, "hm_f1": hm, "wd_f1": per.min(-1), "gap_msa": per[..., 0] - per.min(-1),
            "old_dri_f1": 1.0 / (1.0 + per.var(-1)), **{f"f1_{g}": per[..., k] for k, g in enumerate(DIALECTS_CUR)}}


DIALECTS_CUR = MAIN


class HBoot:
    """Shared two-stage bootstrap over one test set."""

    def __init__(self, y, d, dialects, seed=42):
        self.y, self.d, self.dialects = y, d, dialects
        rng = np.random.default_rng(seed)
        groups = [np.where(d == g)[0] for g in dialects]
        self.idx = np.stack([np.concatenate([rng.choice(g, len(g)) for g in groups]) for _ in range(B)])
        self.full = np.arange(len(y))[None, :]
        self.seed_draw = rng.integers(0, 5, size=(B, 5))  # same seed-index draw for every system -> paired
        self.cache = {}

    def stats(self, key, pred_list):
        """pred_list: list of per-seed prediction arrays. Returns (point dict, replicate dict)."""
        if key in self.cache:
            return self.cache[key]
        global DIALECTS_CUR
        DIALECTS_CUR = self.dialects
        per_s, pool_s, per_pt, pool_pt = [], [], [], []
        for p in pred_list:
            per_s.append(per_dialect_f1(self.y, p, self.d, self.dialects, self.idx))
            pool_s.append(pooled_f1(self.y, p, self.idx))
            per_pt.append(per_dialect_f1(self.y, p, self.d, self.dialects, self.full)[0])
            pool_pt.append(pooled_f1(self.y, p, self.full)[0])
        per_s, pool_s = np.stack(per_s, 1), np.stack(pool_s, 1)  # [B, S, D], [B, S]
        S = per_s.shape[1]
        if S > 1:
            draw = self.seed_draw[:, :S] % S
            rep = {k: v for k, v in summarize(per_s, pool_s).items()}  # per (B, S)
            rep = {k: np.take_along_axis(v, draw, 1).mean(1) for k, v in rep.items()}
        else:
            rep = {k: v[:, 0] for k, v in summarize(per_s, pool_s).items()}
        pt = {k: float(np.mean(v)) for k, v in summarize(np.stack(per_pt), np.array(pool_pt)).items()}
        self.cache[key] = (pt, rep)
        return pt, rep


def ci(rep, lo=0.025, hi=0.975):
    return [float(np.quantile(rep, lo)), float(np.quantile(rep, hi))]


def seed_preds(prefix, model, n=None):
    fs = [PR / f"{prefix}_{model}_seed{s}.npy" for s in SEEDS]
    fs = [f for f in fs if f.exists()]
    return [np.load(f) for f in fs]


def tost(pt_a, rep_a, pt_b, rep_b, key, margin):
    delta = pt_a[key] - pt_b[key]
    r = rep_a[key] - rep_b[key]
    se = float(np.std(r, ddof=1))
    c90, c95 = ci(r, 0.05, 0.95), ci(r)
    return {"delta": float(delta), "ci90": c90, "ci95": c95, "se": se, "mde80": Z80 * se, "margin": margin,
            "equivalent": bool(c90[0] > -margin and c90[1] < margin), "ci95_excludes_zero": bool(c95[0] > 0 or c95[1] < 0)}


def main() -> None:
    res = {"settings": {"B": B, "margin_hm": MARGIN_HM, "margin_dialect": MARGIN_D, "mde_power": 0.8}}
    te = pd.read_csv(OUT / "test_4.csv")
    y, d = te.label.to_numpy(), te.dialect.to_numpy()
    hb = HBoot(y, d, MAIN)
    models = [m for m in step3.MODEL_REGISTRY if seed_preds("PID", m)]
    res["models"] = models

    # ---- 1. seed-mean point + hierarchical 95% CI for every (model, regime) ----
    table = {}
    for m in models:
        for reg, pre in PREFIX.items():
            ps = seed_preds(pre, m)
            if not ps:
                continue
            pt, rep = hb.stats(f"{pre}:{m}", ps)
            table[f"{m}|{reg}"] = {"n_seeds": len(ps), **{k: {"point": pt[k], "ci95": ci(rep[k])} for k in pt}}
    cls = json.loads((OUT / "preds_classical_test4.json").read_text())
    probe = json.loads((OUT / "preds_linear_probe_test4.json").read_text())
    for name, p in [("Majority", cls["Majority"]), ("TF-IDF", cls["TFIDF_LR_balanced"])] + \
                   [(f"Probe:{k}", v) for k, v in probe.items()]:
        pt, rep = hb.stats(name, [np.asarray(p)])
        table[name] = {"n_seeds": 1, **{k: {"point": pt[k], "ci95": ci(rep[k])} for k in pt}}
    res["hier_ci"] = table

    # ---- 2. pairwise PID differences (all encoders + TF-IDF) ----
    pairs = []
    names = [f"PID:{m}" for m in models] + ["TF-IDF"]
    get = lambda n: hb.cache[n] if n in hb.cache else hb.cache[n]
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = hb.cache[names[i]], hb.cache[names[j]]
            for k in ["pooled_macro_f1", "hm_f1", "wd_f1"]:
                r = a[1][k] - b[1][k]
                c = ci(r)
                pairs.append({"a": names[i], "b": names[j], "metric": k, "diff": a[0][k] - b[0][k], "ci95": c,
                              "excludes_zero": bool(c[0] > 0 or c[1] < 0)})
    res["pairwise_pid"] = pairs

    # ---- 3. transfer equivalence + MDE ----
    tr = {}
    for m in models:
        for reg, ctrl in [("LODO", "PIDn1531"), ("MSAonly", "PIDn1905")]:
            if f"{reg}:{m}" not in hb.cache or f"{ctrl}:{m}" not in hb.cache:
                continue
            (pa, ra), (pb, rb) = hb.cache[f"{reg}:{m}"], hb.cache[f"{ctrl}:{m}"]
            e = {"hm_f1": tost(pa, ra, pb, rb, "hm_f1", MARGIN_HM), "wd_f1": tost(pa, ra, pb, rb, "wd_f1", MARGIN_HM)}
            for g in MAIN:
                e[f"f1_{g}"] = tost(pa, ra, pb, rb, f"f1_{g}", MARGIN_D)
            tr[f"{m}|{reg}"] = e
    res["transfer_tost"] = tr

    # ---- 4. HM-F1 variance decomposition: how much does each dialect contribute to HM's bootstrap variance? ----
    var = {}
    for m in models:
        pt, rep = hb.cache[f"PID:{m}"]
        per = np.stack([rep[f"f1_{g}"] for g in MAIN], 1)
        hm = rep["hm_f1"]
        # first-order (delta-method) contribution: dHM/dF1_g = HM^2 / (n * F1_g^2)
        grad = np.array([pt["hm_f1"] ** 2 / (4 * pt[f"f1_{g}"] ** 2) for g in MAIN])
        cov = np.cov(per.T)
        contrib = grad * (cov @ grad)
        var[m] = {"sd_hm": float(hm.std(ddof=1)), "sd_pooled": float(rep["pooled_macro_f1"].std(ddof=1)),
                  "sd_dialect": {g: float(per[:, k].std(ddof=1)) for k, g in enumerate(MAIN)},
                  "share_of_hm_variance": {g: float(c / contrib.sum()) for g, c in zip(MAIN, contrib)}}
    res["hm_variance"] = var

    # ---- 5. metric validity on seed-mean systems, with DRI on accuracy AND on F1 ----
    agg = pd.read_csv(OUT / "aggregated_multiseed_results.csv")
    sysrows = []
    for key, v in table.items():
        sysrows.append({"system": key, "pooled": v["pooled_macro_f1"]["point"], "hm": v["hm_f1"]["point"],
                        "wd": v["wd_f1"]["point"], "dri_f1": v["old_dri_f1"]["point"]})
    sdf = pd.DataFrame(sysrows)
    acc = {}
    for _, r in agg.iterrows():
        acc[f"{r.model}|{r.regime}"] = r.old_dri_acc_mean
    mv = pd.read_csv(OUT / "metric_validity_empirical.csv")
    for _, r in mv.iterrows():
        nm = r.system
        if nm.startswith("Probe:"):
            acc[nm] = r.old_dri_acc
        elif nm == "Classical:Majority":
            acc["Majority"] = r.old_dri_acc
        elif nm == "Classical:TFIDF_LR_balanced":
            acc["TF-IDF"] = r.old_dri_acc
    sdf["dri_acc"] = sdf.system.map(acc)
    sdf = sdf.dropna()
    sdf.to_csv(OUT / "metric_validity_seedmean.csv", index=False)
    rank = lambda col: int(sdf[col].rank(ascending=False, method="min")[sdf.system == "Majority"].iloc[0])
    t = lambda a, b: float(kendalltau(sdf[a], sdf[b])[0])
    res["metric_validity"] = {"n_systems": int(len(sdf)), "majority_rank_dri_acc": rank("dri_acc"),
                              "majority_rank_dri_f1": rank("dri_f1"), "majority_rank_hm": rank("hm"),
                              "tau_hm_dri_acc": t("hm", "dri_acc"), "tau_hm_dri_f1": t("hm", "dri_f1"),
                              "tau_hm_pooled": t("hm", "pooled"), "tau_hm_wd": t("hm", "wd"),
                              "dri_f1_range": [float(sdf.dri_f1.min()), float(sdf.dri_f1.max())],
                              "majority_dri_f1": float(sdf.dri_f1[sdf.system == "Majority"].iloc[0])}

    # ---- 6. learning curve ----
    lc_path = OUT / "lc_runs.jsonl"
    if lc_path.exists():
        lc = pd.DataFrame([json.loads(l) for l in lc_path.read_text().splitlines() if l.strip()])
        out = {}
        for (m, tg), g in lc.groupby(["model", "target"]):
            mask = d == tg
            yt = y[mask]
            hb_t = HBoot(yt, np.full(mask.sum(), tg), [tg], seed=7)
            pts = {}
            lodo = [p[mask] for p in seed_preds("LODO", m)]
            pts["0"] = hb_t.stats("k0", lodo)
            for kt, gg in g.groupby("k_tag"):
                ps = [np.load(PR / f) for f in sorted(sum(gg.pred_files, []))]
                pts[kt] = hb_t.stats(kt, ps)
            ctrl = [p[mask] for p in seed_preds("PIDn1531", m)]
            pid = [p[mask] for p in seed_preds("PID", m)]
            pts["ctrl_n1531"], pts["PID"] = hb_t.stats("ctrl", ctrl), hb_t.stats("pid", pid)
            ks = sorted([k for k in pts if k.isdigit()], key=int) + (["all"] if "all" in pts else [])
            out[f"{m}|{tg}"] = {
                "n_target_train": int(g[g.k_tag == "all"].k.iloc[0]) if (g.k_tag == "all").any() else None,
                "points": {k: {"f1": pts[k][0][f"f1_{tg}"], "ci95": ci(pts[k][1][f"f1_{tg}"])} for k in pts},
                "gain_vs_k0": {k: {"diff": pts[k][0][f"f1_{tg}"] - pts["0"][0][f"f1_{tg}"],
                                   "ci95": ci(pts[k][1][f"f1_{tg}"] - pts["0"][1][f"f1_{tg}"])} for k in ks if k != "0"}}
        res["learning_curve"] = out

    # ---- 7. leaky vs clean ----
    lk_path = OUT / "leaky_runs.jsonl"
    if lk_path.exists():
        lk = pd.DataFrame([json.loads(l) for l in lk_path.read_text().splitlines() if l.strip()])
        out = {}
        for m, g in lk.groupby("model"):
            clean = table.get(f"{m}|PID")
            raw_pool = [r["pooled_macro_f1"] for r in g.raw_test]
            raw_hm = [r["hm_f1"] for r in g.raw_test]
            lc_pool = [r["pooled_macro_f1"] for r in g.clean_test]
            lc_hm = [r["hm_f1"] for r in g.clean_test]
            out[m] = {"n_seeds": int(len(g)), "leaky_on_raw_test": {"pooled": float(np.mean(raw_pool)), "hm": float(np.mean(raw_hm))},
                      "leaky_on_clean_test": {"pooled": float(np.mean(lc_pool)), "hm": float(np.mean(lc_hm))},
                      "clean_on_clean_test": {"pooled": clean["pooled_macro_f1"]["point"], "hm": clean["hm_f1"]["point"]},
                      "inflation_pooled": float(np.mean(raw_pool) - clean["pooled_macro_f1"]["point"]),
                      "inflation_hm": float(np.mean(raw_hm) - clean["hm_f1"]["point"])}
            # paired bootstrap of leaky-model vs clean-model on the clean test set
            pt_l, rep_l = hb.stats(f"LEAKYclean:{m}", [np.load(PR / f"LEAKYclean_{m}_seed{s}.npy") for s in g.seed])
            pt_c, rep_c = hb.cache[f"PID:{m}"]
            for k in ["pooled_macro_f1", "hm_f1"]:
                out[m][f"leaky_minus_clean_on_clean_test_{k}"] = {"diff": pt_l[k] - pt_c[k], "ci95": ci(rep_l[k] - rep_c[k])}
        res["leaky"] = out

    # ---- 8. ArSarcasm-v2 replication ----
    v2_path = OUT / "v2_runs.jsonl"
    if v2_path.exists():
        v2te = pd.read_csv(OUT / "arsarcasm_v2" / "test.csv")
        y2, d2 = v2te.label.to_numpy(), v2te.dialect.to_numpy()
        D3 = ["MSA", "Egyptian", "Gulf"]  # Levantine: 28 test tweets, 1 positive -> descriptive only
        keep = np.isin(d2, D3)
        hb2 = HBoot(y2[keep], d2[keep], D3, seed=11)
        hb2_lev = HBoot(y2, d2, MAIN, seed=12)
        runs = pd.DataFrame([json.loads(l) for l in v2_path.read_text().splitlines() if l.strip()])
        out = {"dialects_for_profile": D3, "table": {}, "tost": {}}
        for (m, reg), g in runs.groupby(["model", "regime"]):
            ps = [np.load(PR / f) for f in sorted(sum(g.pred_files, []))]
            pt, rep = hb2.stats(f"{reg}:{m}", [p[keep] for p in ps])
            pt4, _ = hb2_lev.stats(f"{reg}:{m}", ps)
            out["table"][f"{m}|{reg}"] = {"n_seeds": len(ps), **{k: {"point": pt[k], "ci95": ci(rep[k])} for k in pt},
                                          "f1_Levantine_descriptive": pt4["f1_Levantine"]}
        clsv2 = json.loads((OUT / "arsarcasm_v2" / "preds_classical_v2.json").read_text())
        for nm, p in clsv2.items():
            pt, rep = hb2.stats(nm, [np.asarray(p)[keep]])
            out["table"][nm] = {"n_seeds": 1, **{k: {"point": pt[k], "ci95": ci(rep[k])} for k in pt}}
        for m in runs.model.unique():
            for reg, ctrl in [("LODO_matched", "PID_nLODO"), ("MSA_only", "PID_nMSA")]:
                if f"{reg}:{m}" in hb2.cache and f"{ctrl}:{m}" in hb2.cache:
                    (pa, ra), (pb, rb) = hb2.cache[f"{reg}:{m}"], hb2.cache[f"{ctrl}:{m}"]
                    e = {"hm_f1": tost(pa, ra, pb, rb, "hm_f1", MARGIN_HM)}
                    for g in D3:
                        e[f"f1_{g}"] = tost(pa, ra, pb, rb, f"f1_{g}", MARGIN_D)
                    out["tost"][f"{m}|{reg}"] = e
        res["arsarcasm_v2"] = out

    # ---- 9. LLMs ----
    llm_dir = OUT / "llm"
    if llm_dir.exists():
        out = {}
        for f in sorted(llm_dir.glob("*.json")):
            if f.name.endswith(".failed.json"):
                continue
            r = json.loads(f.read_text())
            name = r["model"]
            z = np.load(llm_dir / f"{name}_preds_test4.npz")
            out[name] = {}
            fs = [z[k] for k in z.files if k.startswith("FS8")]
            for tag, ps in [("ZS", [z["ZS"]]), ("ZScal", [z["ZScal"]]), ("FS8", fs)]:
                if not ps:
                    continue
                pt, rep = hb.stats(f"LLM:{name}:{tag}", ps)
                out[name][tag] = {"n_draws": len(ps), **{k: {"point": pt[k], "ci95": ci(rep[k])} for k in pt}}
            out[name]["maghrebi_acc"] = {k: r["settings"][k]["maghrebi"]["accuracy"] for k in r["settings"]}
            out[name]["v2_zero_shot"] = {k: {"pooled": r["settings"][k]["v2_test"]["pooled_macro_f1"],
                                             "per_dialect_f1": r["settings"][k]["v2_test"]["per_dialect_f1"]}
                                         for k in ("ZS", "ZScal")}
            out[name]["zscal_threshold"] = r["settings"]["ZScal"]["threshold"]
            best = f"PID:{max(models, key=lambda m: hb.cache[f'PID:{m}'][0]['hm_f1'])}"
            for tag in ("ZScal", "FS8"):
                if f"LLM:{name}:{tag}" in hb.cache:
                    a, b = hb.cache[f"LLM:{name}:{tag}"], hb.cache[best]
                    out[name][f"{tag}_minus_best_encoder_hm"] = {"vs": best, "diff": a[0]["hm_f1"] - b[0]["hm_f1"],
                                                               "ci95": ci(a[1]["hm_f1"] - b[1]["hm_f1"])}
                    t_ = hb.cache["TF-IDF"]
                    out[name][f"{tag}_minus_tfidf_hm"] = {"diff": a[0]["hm_f1"] - t_[0]["hm_f1"],
                                                        "ci95": ci(a[1]["hm_f1"] - t_[1]["hm_f1"])}
        res["llm"] = out
        res["llm_failed"] = {f.name: json.loads(f.read_text())["status"] for f in llm_dir.glob("*.failed.json")}

    (OUT / "review_stats.json").write_text(json.dumps(res, indent=1))
    print(f"[6b] models={models}")
    for k, v in res["transfer_tost"].items():
        h = v["hm_f1"]
        print(f"  {k:28s} dHM={h['delta']:+.3f} 90%CI=[{h['ci90'][0]:+.3f},{h['ci90'][1]:+.3f}] "
              f"equiv={h['equivalent']} MDE={h['mde80']:.3f} | MDE GLF={v['f1_Gulf']['mde80']:.3f} LEV={v['f1_Levantine']['mde80']:.3f}")
    print(json.dumps(res["metric_validity"], indent=1))
    print("[STEP 6b PASSED]")


if __name__ == "__main__":
    main()
