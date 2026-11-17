# -*- coding: utf-8 -*-
"""Step 7: Quantitative error analysis by sarcasm, linguistic artifacts, and tokenizer fertility."""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs_v2"
PREDS_DIR = OUT / "raw_preds"
MAIN_DIALECTS = ["MSA", "Egyptian", "Levantine", "Gulf"]
SEEDS = [42, 43, 44, 45, 46]


def strip_platform_tokens(t: str) -> str:
    # URLs, @mentions and RT markers are Twitter artifacts, not code-switching
    return re.sub(r"http\S+|@\w+|\bRT\b", " ", str(t))


def has_elongation(t: str) -> bool:
    # Arabic letter repeated 3+ times or tatweel; repeated punctuation is a separate bucket
    return bool(re.search(r"([\u0621-\u064a])\1{2,}|\u0640", str(t)))


def has_repeated_punct(t: str) -> bool:
    return bool(re.search(r"([!?\u061f.])\1+", str(t)))


def has_emoji(t: str) -> bool:
    return bool(re.search(r"[\U0001F300-\U0001FAFF\u2600-\u27bf]", str(t)))


def has_latin(t: str) -> bool:
    return bool(re.search(r"[A-Za-z]", strip_platform_tokens(t)))


def has_hashtag(t: str) -> bool:
    return "#" in str(t)


def main() -> None:
    te = pd.read_csv(OUT / "test_4.csv")
    te["has_elongation"] = te["text"].apply(has_elongation)
    te["has_latin"] = te["text"].apply(has_latin)
    te["has_hashtag"] = te["text"].apply(has_hashtag)
    te["has_repeated_punct"] = te["text"].apply(has_repeated_punct)
    te["has_emoji"] = te["text"].apply(has_emoji)
    te["word_count"] = te["text"].astype(str).apply(lambda s: len(s.split()))
    te["length_bucket"] = pd.cut(
        te["word_count"], bins=[0, 10, 20, 1000], labels=["Short (<=10w)", "Medium (11-20w)", "Long (>20w)"]
    )

    df_agg = pd.read_csv(OUT / "aggregated_multiseed_results.csv")
    models = sorted(df_agg["model"].unique().tolist())

    sarcasm_rows = []
    sarcasm_effect_rows = []
    artifact_rows = []

    for m in models:
        mats = [
            np.load(PREDS_DIR / f"PID_{m}_seed{s}.npy")
            for s in SEEDS
            if (PREDS_DIR / f"PID_{m}_seed{s}.npy").exists()
        ]
        if not mats:
            continue
        p_ens = (np.mean(np.vstack(mats), axis=0) >= 0.5).astype(int)

        # 1. Sarcasm x Dialect breakdown
        for d in ["ALL"] + MAIN_DIALECTS:
            d_mask = np.ones(len(te), dtype=bool) if d == "ALL" else (te["dialect"].to_numpy() == d)
            for sarc_val, sarc_name in [(False, "Non-Sarcastic"), (True, "Sarcastic")]:
                mask = d_mask & (te["sarcasm"].to_numpy() == sarc_val)
                if np.sum(mask) >= 5:
                    f1 = float(
                        f1_score(
                            te["label"].to_numpy()[mask],
                            p_ens[mask],
                            average="macro",
                            labels=[0, 1],
                            zero_division=0,
                        )
                    )
                    yt, yp = te["label"].to_numpy()[mask], p_ens[mask]
                    # Sarcastic tweets are overwhelmingly negative, so macro-F1 alone conflates
                    # sarcasm with class composition; report per-class recall and accuracy too.
                    sarcasm_rows.append(
                        {
                            "model": m,
                            "dialect": d,
                            "sarcasm": sarc_name,
                            "n": int(np.sum(mask)),
                            "n_pos": int(np.sum(yt == 1)),
                            "macro_f1": round(f1, 4),
                            "accuracy": round(float(np.mean(yt == yp)), 4),
                            "recall_neg": round(float(np.mean(yp[yt == 0] == 0)), 4) if np.any(yt == 0) else np.nan,
                            "recall_pos": round(float(np.mean(yp[yt == 1] == 1)), 4) if np.any(yt == 1) else np.nan,
                        }
                    )

        # 1b. Bootstrap CI of the sarcasm effect on negative-class recall and accuracy (pooled test set)
        rng = np.random.default_rng(42)
        y_all, s_all = te["label"].to_numpy(), te["sarcasm"].to_numpy().astype(bool)
        groups = [np.where(s_all & (y_all == c))[0] for c in (0, 1)] + [np.where(~s_all & (y_all == c))[0] for c in (0, 1)]
        diffs_rneg, diffs_acc = [], []
        for _ in range(2000):
            idx = np.concatenate([rng.choice(g, size=len(g), replace=True) for g in groups])
            yi, pi, si = y_all[idx], p_ens[idx], s_all[idx]
            rneg = lambda msk: np.mean(pi[msk & (yi == 0)] == 0)
            diffs_rneg.append(rneg(~si) - rneg(si))
            diffs_acc.append(np.mean(pi[~si] == yi[~si]) - np.mean(pi[si] == yi[si]))
        sarcasm_effect_rows.append({
            "model": m,
            "neg_recall_gap_nonsarc_minus_sarc": float(np.mean(diffs_rneg)),
            "neg_recall_gap_ci95": [float(np.quantile(diffs_rneg, 0.025)), float(np.quantile(diffs_rneg, 0.975))],
            "acc_gap_nonsarc_minus_sarc": float(np.mean(diffs_acc)),
            "acc_gap_ci95": [float(np.quantile(diffs_acc, 0.025)), float(np.quantile(diffs_acc, 0.975))],
        })

        # 2. Linguistic artifact buckets
        for feat_col in ["has_hashtag", "has_elongation", "has_repeated_punct", "has_emoji", "has_latin", "length_bucket"]:
            for val, g in te.groupby(feat_col, observed=False):
                idx = g.index.to_numpy()
                if len(idx) >= 10:
                    f1 = float(
                        f1_score(
                            te["label"].to_numpy()[idx],
                            p_ens[idx],
                            average="macro",
                            labels=[0, 1],
                            zero_division=0,
                        )
                    )
                    artifact_rows.append(
                        {
                            "model": m,
                            "feature": feat_col,
                            "bucket": str(val),
                            "n": int(len(idx)),
                            "macro_f1": round(f1, 4),
                        }
                    )

    pd.DataFrame(sarcasm_rows).to_csv(OUT / "error_sarcasm_by_dialect.csv", index=False)
    with open(OUT / "sarcasm_effect_bootstrap.json", "w", encoding="utf-8") as f:
        json.dump(sarcasm_effect_rows, f, indent=2)
    pd.DataFrame(artifact_rows).to_csv(OUT / "error_linguistic_artifacts.csv", index=False)

    # 3. Correlation between Tokenizer Fertility and Per-Dialect F1
    fert_df = pd.read_csv(OUT / "tokenizer_fertility.csv")
    fert_main = fert_df[fert_df["dialect"].isin(MAIN_DIALECTS)].copy()
    pid_rows = df_agg[df_agg["regime"] == "PID"]

    merged_points = []
    for _, r in pid_rows.iterrows():
        m = r["model"]
        for d in MAIN_DIALECTS:
            f_row = fert_main[(fert_main["model"] == m) & (fert_main["dialect"] == d)]
            if not f_row.empty:
                merged_points.append(
                    {
                        "model": m,
                        "dialect": d,
                        "fertility": float(f_row["fertility"].values[0]),
                        "unk_rate": float(f_row["unk_rate"].values[0]),
                        "f1_mean": float(r[f"f1_{d}_mean"]),
                    }
                )
    df_corr = pd.DataFrame(merged_points)
    df_corr.to_csv(OUT / "fertility_vs_f1.csv", index=False)
    rho, p_val = spearmanr(df_corr["fertility"], df_corr["f1_mean"])
    with open(OUT / "fertility_correlation.json", "w", encoding="utf-8") as f:
        json.dump({"spearman_rho": float(rho), "p_value": float(p_val), "n_points": len(df_corr)}, f, indent=2)

    print(f"Fertility vs F1 Spearman rho = {rho:.4f} (p = {p_val:.4g})")
    print("\n[STEP 7 PASSED] Quantitative sarcasm, artifact, and fertility analyses saved.")


if __name__ == "__main__":
    main()


