# -*- coding: utf-8 -*-
"""Step 2: Majority & TF-IDF+LR baselines + degenerate-classifier DRI audit."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
)
from sklearn.pipeline import make_pipeline, make_union
from statsmodels.stats.proportion import proportion_confint

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs_v2"
MAIN_DIALECTS = ["MSA", "Egyptian", "Levantine", "Gulf"]


def compute_robustness_profile(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    dialects: np.ndarray,
    ref_dialect: str = "MSA",
) -> dict:
    per_dialect_f1 = {}
    per_dialect_acc = {}
    per_dialect_bal_acc = {}

    for d in MAIN_DIALECTS:
        mask = dialects == d
        yt, yp = y_true[mask], y_pred[mask]
        per_dialect_f1[d] = float(
            f1_score(yt, yp, average="macro", labels=[0, 1], zero_division=0)
        )
        per_dialect_acc[d] = float(accuracy_score(yt, yp))
        per_dialect_bal_acc[d] = float(balanced_accuracy_score(yt, yp))

    f1_vals = np.array([per_dialect_f1[d] for d in MAIN_DIALECTS], dtype=float)
    acc_vals = np.array([per_dialect_acc[d] for d in MAIN_DIALECTS], dtype=float)

    dialect_macro_f1 = float(np.mean(f1_vals))
    wd_f1 = float(np.min(f1_vals))
    hm_f1 = float(len(f1_vals) / np.sum(1.0 / np.maximum(f1_vals, 1e-6)))
    max_gap_msa = float(per_dialect_f1[ref_dialect] - wd_f1)
    non_ref = [per_dialect_f1[d] / max(per_dialect_f1[ref_dialect], 1e-6) for d in MAIN_DIALECTS if d != ref_dialect]
    mean_retention = float(np.mean(non_ref))
    std_f1 = float(np.std(f1_vals, ddof=0))
    cv_f1 = float(std_f1 / max(dialect_macro_f1, 1e-6))
    old_dri_acc = float(1.0 / (1.0 + np.var(acc_vals, ddof=0)))

    return {
        "pooled_macro_f1": float(
            f1_score(y_true, y_pred, average="macro", labels=[0, 1], zero_division=0)
        ),
        "pooled_balanced_acc": float(balanced_accuracy_score(y_true, y_pred)),
        "pooled_mcc": float(matthews_corrcoef(y_true, y_pred)),
        "pooled_acc": float(accuracy_score(y_true, y_pred)),
        "dialect_macro_f1": dialect_macro_f1,
        "wd_f1": wd_f1,
        "hm_f1": hm_f1,
        "max_gap_vs_msa": max_gap_msa,
        "mean_retention_vs_msa": mean_retention,
        "std_f1": std_f1,
        "cv_f1": cv_f1,
        "old_dri_acc": old_dri_acc,
        "per_dialect_f1": per_dialect_f1,
        "per_dialect_acc": per_dialect_acc,
        "per_dialect_bal_acc": per_dialect_bal_acc,
    }


def eval_maghrebi_exploratory(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    k = int(np.sum(y_true == y_pred))
    n = int(len(y_true))
    lo, hi = proportion_confint(k, n, alpha=0.05, method="wilson")
    return {
        "k_correct": k,
        "n_total": n,
        "accuracy": float(k / n),
        "wilson_ci95_low": float(lo),
        "wilson_ci95_high": float(hi),
        "macro_f1": float(
            f1_score(y_true, y_pred, average="macro", labels=[0, 1], zero_division=0)
        ),
    }


def main() -> None:
    tr = pd.read_csv(OUT / "train_4.csv")
    dv = pd.read_csv(OUT / "dev_4.csv")
    te = pd.read_csv(OUT / "test_4.csv")
    mag = pd.read_csv(OUT / "test_maghrebi.csv")

    results = {}
    preds_out = {}

    # 1. Majority-class baseline
    maj = DummyClassifier(strategy="most_frequent")
    maj.fit(tr["text"], tr["label"])
    p_te_maj = maj.predict(te["text"])
    p_mag_maj = maj.predict(mag["text"])
    results["Majority"] = {
        "test_4": compute_robustness_profile(
            te["label"].to_numpy(), p_te_maj, te["dialect"].to_numpy()
        ),
        "maghrebi": eval_maghrebi_exploratory(mag["label"].to_numpy(), p_mag_maj),
    }
    preds_out["Majority"] = p_te_maj.tolist()

    # 2. TF-IDF + Logistic Regression (tune C on dev_4 for unweighted and balanced)
    for mode, cw in [("TFIDF_LR_unweighted", None), ("TFIDF_LR_balanced", "balanced")]:
        best_c, best_dev_f1, best_pipe = 1.0, -1.0, None
        for c_val in [0.25, 0.5, 1.0, 2.0, 4.0]:
            vec = make_union(
                TfidfVectorizer(
                    analyzer="word", ngram_range=(1, 2), min_df=2, sublinear_tf=True
                ),
                TfidfVectorizer(
                    analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True
                ),
            )
            pipe = make_pipeline(
                vec,
                LogisticRegression(
                    max_iter=2000, C=c_val, class_weight=cw, random_state=42
                ),
            )
            pipe.fit(tr["text"], tr["label"])
            dev_pred = pipe.predict(dv["text"])
            dev_f1 = f1_score(dv["label"], dev_pred, average="macro", zero_division=0)
            if dev_f1 > best_dev_f1:
                best_dev_f1, best_c, best_pipe = dev_f1, c_val, pipe

        p_te = best_pipe.predict(te["text"])
        p_mag = best_pipe.predict(mag["text"])
        results[mode] = {
            "best_C": best_c,
            "dev_macro_f1": float(best_dev_f1),
            "test_4": compute_robustness_profile(
                te["label"].to_numpy(), p_te, te["dialect"].to_numpy()
            ),
            "maghrebi": eval_maghrebi_exploratory(mag["label"].to_numpy(), p_mag),
        }
        preds_out[mode] = p_te.tolist()

    with open(OUT / "classical_baselines.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    with open(OUT / "preds_classical_test4.json", "w", encoding="utf-8") as f:
        json.dump(preds_out, f)

    print(json.dumps(results, indent=2))
    assert results["Majority"]["test_4"]["old_dri_acc"] > 0.999
    assert results["Majority"]["test_4"]["hm_f1"] < 0.42
    assert results["TFIDF_LR_balanced"]["test_4"]["pooled_macro_f1"] > 0.75
    print("\n[STEP 2 PASSED] Classical baselines & DRI counter-example verified.")


if __name__ == "__main__":
    main()


