# -*- coding: utf-8 -*-
"""Step 6: Seed aggregation, stratified paired bootstrap, Holm-McNemar, mixed-effects GLM, metric validity."""
from __future__ import annotations

import json
from importlib.machinery import SourceFileLoader
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2, kendalltau
from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs_v2"
PREDS_DIR = OUT / "raw_preds"
step2 = SourceFileLoader("step2", str(ROOT / "02_classical_baselines.py")).load_module()
MAIN_DIALECTS = ["MSA", "Egyptian", "Levantine", "Gulf"]
SEEDS = [42, 43, 44, 45, 46]
CORE = ["AraBERTv02", "MARBERT", "CAMeLBERT-Mix", "XLM-R-Base"]
METRICS = ["pooled_macro_f1", "pooled_balanced_acc", "pooled_mcc", "pooled_acc", "dialect_macro_f1", "wd_f1",
           "hm_f1", "max_gap_vs_msa", "mean_retention_vs_msa", "std_f1", "cv_f1", "old_dri_acc"]


def majority_vote_over_seeds(model_name: str, prefix: str) -> np.ndarray | None:
    mats = [np.load(PREDS_DIR / f"{prefix}_{model_name}_seed{s}.npy")
            for s in SEEDS if (PREDS_DIR / f"{prefix}_{model_name}_seed{s}.npy").exists()]
    if not mats:
        return None
    return (np.mean(np.vstack(mats), axis=0) >= 0.5).astype(int)


def fast_profile(y: np.ndarray, p: np.ndarray, d: np.ndarray) -> dict:
    """Vectorised subset of compute_robustness_profile (identical definitions) for bootstrap speed."""
    def macro_f1(yt, yp):
        f = []
        for c in (0, 1):
            tp = np.sum((yp == c) & (yt == c)); fp = np.sum((yp == c) & (yt != c)); fn = np.sum((yp != c) & (yt == c))
            f.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
        return float(np.mean(f))
    per = np.array([macro_f1(y[d == g], p[d == g]) for g in MAIN_DIALECTS])
    return {"pooled_macro_f1": macro_f1(y, p), "dialect_macro_f1": float(per.mean()), "wd_f1": float(per.min()),
            "hm_f1": float(len(per) / np.sum(1.0 / np.maximum(per, 1e-6))), "max_gap_vs_msa": float(per[0] - per.min())}


def stratified_paired_bootstrap(y, d, preds: dict[str, np.ndarray], pair_models: list[str], n_boot=2000, seed=42):
    rng = np.random.default_rng(seed)
    groups = [np.where(d == g)[0] for g in MAIN_DIALECTS]
    boot = {m: [] for m in preds}
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(g, size=len(g), replace=True) for g in groups])
        for m, p in preds.items():
            boot[m].append(fast_profile(y[idx], p[idx], d[idx]))
    keys = ["pooled_macro_f1", "dialect_macro_f1", "wd_f1", "hm_f1", "max_gap_vs_msa"]
    ci = {}
    for m, lst in boot.items():
        point = fast_profile(y, preds[m], d)
        ci[m] = {k: {"point": point[k], "ci95_low": float(np.quantile([b[k] for b in lst], 0.025)),
                     "ci95_high": float(np.quantile([b[k] for b in lst], 0.975))} for k in keys}
    pairs = []
    for i in range(len(pair_models)):
        for j in range(i + 1, len(pair_models)):
            m1, m2 = pair_models[i], pair_models[j]
            for k in ["pooled_macro_f1", "hm_f1", "wd_f1"]:
                diffs = np.array([a[k] - b[k] for a, b in zip(boot[m1], boot[m2])])
                lo, hi = np.quantile(diffs, [0.025, 0.975])
                pairs.append({"comparison": f"{m1} - {m2}", "metric": k, "mean_diff": float(diffs.mean()),
                              "ci95_low": float(lo), "ci95_high": float(hi), "ci_excludes_zero": bool(lo > 0 or hi < 0)})
    return {"ci_summary": ci, "pairwise_diffs": pairs}


def per_dialect_mcnemar_holm(y, d, preds, core_models):
    rows = []
    for g in ["ALL_POOLED"] + MAIN_DIALECTS:
        mask = np.ones(len(y), dtype=bool) if g == "ALL_POOLED" else (d == g)
        yt = y[mask]
        for i in range(len(core_models)):
            for j in range(i + 1, len(core_models)):
                m1, m2 = core_models[i], core_models[j]
                c1, c2 = preds[m1][mask] == yt, preds[m2][mask] == yt
                b, c = int(np.sum(c1 & ~c2)), int(np.sum(~c1 & c2))
                stat = (abs(b - c) - 1.0) ** 2 / max(b + c, 1)
                p_raw = float(1.0 - chi2.cdf(stat, df=1)) if (b + c) > 0 else 1.0
                rows.append({"dialect": g, "comparison": f"{m1} vs {m2}", "b_m1_only": b, "c_m2_only": c,
                             "odds_ratio": round(float((b + 0.5) / (c + 0.5)), 3), "chi2_stat": round(stat, 4),
                             "p_raw": p_raw})
    df = pd.DataFrame(rows).sort_values("p_raw").reset_index(drop=True)
    m_tests, running, adj = len(df), 0.0, []
    for i, r in df.iterrows():
        running = max(running, min(1.0, r["p_raw"] * (m_tests - i)))
        adj.append(running)
    df["p_holm"] = adj
    df["significant_005"] = df["p_holm"] < 0.05
    return df.to_dict(orient="records")


def mixed_effects_glm(te: pd.DataFrame, preds: dict[str, np.ndarray], models: list[str]) -> dict:
    """correct ~ C(model) * C(dialect) + (1 | tweet_id), variational Bayes binomial GLMM."""
    rows = []
    for m in models:
        rows.append(pd.DataFrame({"correct": (preds[m] == te["label"].to_numpy()).astype(int), "model": m,
                                  "dialect": te["dialect"].to_numpy(), "tweet_id": te["tweet_id"].astype(str)}))
    long = pd.concat(rows, ignore_index=True)
    long["model"] = pd.Categorical(long["model"], categories=models)
    long["dialect"] = pd.Categorical(long["dialect"], categories=MAIN_DIALECTS)
    glmm = BinomialBayesMixedGLM.from_formula("correct ~ C(model) * C(dialect)", {"tweet": "0 + C(tweet_id)"}, long)
    res = glmm.fit_vb()
    names = list(glmm.exog_names)
    fe_mean, fe_sd = res.fe_mean, res.fe_sd
    coefs = [{"term": n, "post_mean": float(mu), "post_sd": float(sd),
              "ci95_low": float(mu - 1.96 * sd), "ci95_high": float(mu + 1.96 * sd),
              "excludes_zero": bool(abs(mu) > 1.96 * sd)} for n, mu, sd in zip(names, fe_mean, fe_sd)]
    return {"reference_model": models[0], "reference_dialect": MAIN_DIALECTS[0], "n_obs": int(len(long)),
            "tweet_random_effect_sd": float(np.exp(res.vcp_mean[0])), "fixed_effects": coefs}


def synthetic_metric_validity_audit() -> list[dict]:
    profiles = {
        "Ideal uniform high": [0.85, 0.85, 0.85, 0.85],
        "Strong, slightly uneven": [0.88, 0.84, 0.82, 0.80],
        "MSA-biased drop": [0.90, 0.78, 0.70, 0.62],
        "Uniform mediocre": [0.55, 0.55, 0.55, 0.55],
        "Degenerate majority": [0.402, 0.421, 0.407, 0.408],
    }
    out = []
    for label, vals in profiles.items():
        v = np.array(vals, dtype=float)
        out.append({"profile": label, "per_dialect_f1": vals, "mean_f1": round(float(v.mean()), 4),
                    "wd_f1": round(float(v.min()), 4), "hm_f1": round(float(len(v) / np.sum(1.0 / v)), 4),
                    "max_gap_msa": round(float(v[0] - v.min()), 4),
                    "old_dri": round(float(1.0 / (1.0 + np.var(v, ddof=0))), 4)})
    return out


def main() -> None:
    te = pd.read_csv(OUT / "test_4.csv")
    y, d = te["label"].to_numpy(), te["dialect"].to_numpy()
    runs = [json.loads(l) for l in (OUT / "finetune_runs.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    df_runs = pd.DataFrame(runs)

    # 1. Mean +/- SD across seeds per (model, regime)
    agg = []
    for (model, regime), g in df_runs.groupby(["model", "regime"]):
        row = {"model": model, "regime": regime, "n_seeds": len(g), "lr": float(g["lr"].iloc[0])}
        for col in METRICS:
            row[f"{col}_mean"] = float(g[col].mean())
            row[f"{col}_std"] = float(g[col].std(ddof=1)) if len(g) > 1 else 0.0
        for dd in MAIN_DIALECTS:
            vals = [x[dd] for x in g["per_dialect_f1"]]
            row[f"f1_{dd}_mean"] = float(np.mean(vals))
            row[f"f1_{dd}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        if "maghrebi" in g and g["maghrebi"].notna().all():
            row["mag_acc_mean"] = float(np.mean([x["accuracy"] for x in g["maghrebi"]]))
            row["mag_acc_std"] = float(np.std([x["accuracy"] for x in g["maghrebi"]], ddof=1)) if len(g) > 1 else 0.0
            row["mag_f1_mean"] = float(np.mean([x["macro_f1"] for x in g["maghrebi"]]))
        agg.append(row)
    df_agg = pd.DataFrame(agg)
    df_agg.to_csv(OUT / "aggregated_multiseed_results.csv", index=False)

    # 2. Seed-ensembled (majority-vote) PID predictions + baselines for paired instance-level statistics
    pid_models = [m for m in ["AraBERTv02", "AraBERTv02-Twitter", "MARBERT", "MARBERTv2", "CAMeLBERT-Mix", "CAMeLBERT-MSA", "CAMeLBERT-DA", "XLM-R-Base",
                                "CAMeLBERT-CA", "ARBERTv2", "AraELECTRA"]
                  if majority_vote_over_seeds(m, "PID") is not None]
    preds = {f"PID:{m}": majority_vote_over_seeds(m, "PID") for m in pid_models}
    for m in pid_models:
        for prefix in ["LODO", "MSAonly", "PIDw", "PIDn1531", "PIDn1905"]:
            p = majority_vote_over_seeds(m, prefix)
            if p is not None:
                preds[f"{prefix}:{m}"] = p
    probe = json.load(open(OUT / "preds_linear_probe_test4.json"))
    for m, p in probe.items():
        preds[f"Probe:{m}"] = np.array(p, dtype=int)
    classical = json.load(open(OUT / "preds_classical_test4.json"))
    for m, p in classical.items():
        preds[f"Classical:{m}"] = np.array(p, dtype=int)
    np.savez_compressed(OUT / "ensemble_preds_test4.npz", **preds)

    pair_models = [f"PID:{m}" for m in pid_models] + ["Classical:TFIDF_LR_balanced"]
    boot = stratified_paired_bootstrap(y, d, preds, pair_models)
    core = [f"PID:{m}" for m in CORE if f"PID:{m}" in preds]
    mcn = per_dialect_mcnemar_holm(y, d, preds, core)
    glmm_models = [m for m in CORE if f"PID:{m}" in preds]
    glmm = (mixed_effects_glm(te, {m: preds[f"PID:{m}"] for m in glmm_models}, glmm_models)
            if len(glmm_models) >= 2 else None)

    # 3. Metric rank agreement across every evaluated system (empirical validity check)
    sys_rows = []
    for m, p in preds.items():
        prof = step2.compute_robustness_profile(y, p, d)
        sys_rows.append({"system": m, **{k: prof[k] for k in ["pooled_macro_f1", "hm_f1", "wd_f1", "old_dri_acc",
                                                               "max_gap_vs_msa", "cv_f1"]}})
    sys_df = pd.DataFrame(sys_rows)
    sys_df.to_csv(OUT / "metric_validity_empirical.csv", index=False)
    tau = lambda a, b: float(kendalltau(sys_df[a], sys_df[b])[0])

    bundle = {
        "bootstrap_pid": boot,
        "mcnemar_holm": mcn,
        "mixed_effects_glm": glmm,
        "synthetic_metric_validity": synthetic_metric_validity_audit(),
        "kendall_tau_hm_vs_wd": tau("hm_f1", "wd_f1"),
        "kendall_tau_hm_vs_pooled_f1": tau("hm_f1", "pooled_macro_f1"),
        "kendall_tau_hm_vs_old_dri": tau("hm_f1", "old_dri_acc"),
        "kendall_tau_pooled_f1_vs_old_dri": tau("pooled_macro_f1", "old_dri_acc"),
        "n_systems_for_tau": int(len(sys_df)),
    }
    with open(OUT / "statistical_analysis_bundle.json", "w", encoding="utf-8") as f:
        json.dump(bundle, f, indent=2)
    print(df_agg[["model", "regime", "n_seeds", "pooled_macro_f1_mean", "pooled_macro_f1_std", "hm_f1_mean",
                  "wd_f1_mean"]].round(4).to_string(index=False))
    print({k: v for k, v in bundle.items() if k.startswith("kendall")})
    print("\n[STEP 6 PASSED] Saved aggregated_multiseed_results.csv and statistical_analysis_bundle.json.")


if __name__ == "__main__":
    main()
