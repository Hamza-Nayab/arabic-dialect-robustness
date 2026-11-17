# -*- coding: utf-8 -*-
"""Step 4: Trained Linear Probe on frozen mean-pooled encoder representations."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs_v2"
from importlib.machinery import SourceFileLoader
step2 = SourceFileLoader("step2", str(ROOT / "02_classical_baselines.py")).load_module()
step3 = SourceFileLoader("step3", str(ROOT / "03_model_and_fertility_audit.py")).load_module()
cache_mod = SourceFileLoader("cache_mod", str(ROOT / "_encode_cache.py")).load_module()

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
)
BATCH_SIZE = 32


@torch.no_grad()
def extract_mean_pooled_embeddings(
    model_ckpt: str, texts: list[str]
) -> np.ndarray:
    tok = AutoTokenizer.from_pretrained(model_ckpt)
    enc_model = AutoModel.from_pretrained(model_ckpt).to(DEVICE)
    enc_model.eval()

    all_embs = []
    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc=f"Embed {model_ckpt}"):
        batch = texts[i : i + BATCH_SIZE]
        inp = tok(
            batch,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt",
        ).to(DEVICE)
        out = enc_model(**inp)
        last_hidden = out.last_hidden_state  # [B, L, H]
        mask = inp["attention_mask"].unsqueeze(-1).float()  # [B, L, 1]
        summed = torch.sum(last_hidden * mask, dim=1)
        counts = torch.clamp(mask.sum(dim=1), min=1e-9)
        mean_pooled = (summed / counts).cpu().numpy()
        all_embs.append(mean_pooled)

    del enc_model, tok
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return np.vstack(all_embs)


def main() -> None:
    tr = pd.read_csv(OUT / "train_4.csv")
    dv = pd.read_csv(OUT / "dev_4.csv")
    te = pd.read_csv(OUT / "test_4.csv")
    mag = pd.read_csv(OUT / "test_maghrebi.csv")

    probe_results = {}
    probe_preds = {}

    for name, ckpt in step3.MODEL_REGISTRY.items():
        # One length-sorted encoder pass per model (also caches top-block inputs for Step 5)
        cache_mod.encode_and_cache(name, ckpt)
        z = np.load(cache_mod.EMB_DIR / f"{name}.npz")
        X_tr, X_dv, X_te, X_mg = z["tr"], z["dv"], z["te"], z["mg"]

        best_c, best_f1, best_clf = 1.0, -1.0, None
        for c_val in [0.0003, 0.001, 0.003, 0.01, 0.1, 1.0, 5.0]:
            clf = make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    max_iter=2000, C=c_val, class_weight="balanced", random_state=42
                ),
            )
            clf.fit(X_tr, tr["label"].to_numpy())
            dv_p = clf.predict(X_dv)
            f1 = f1_score(dv["label"], dv_p, average="macro", zero_division=0)
            if f1 > best_f1:
                best_f1, best_c, best_clf = f1, c_val, clf

        p_te = best_clf.predict(X_te)
        p_mg = best_clf.predict(X_mg)
        probe_results[name] = {
            "best_C": best_c,
            "dev_macro_f1": float(best_f1),
            "test_4": step2.compute_robustness_profile(
                te["label"].to_numpy(), p_te, te["dialect"].to_numpy()
            ),
            "maghrebi": step2.eval_maghrebi_exploratory(mag["label"].to_numpy(), p_mg),
        }
        probe_preds[name] = p_te.tolist()
        print(f"{name}: C={best_c} dev_f1={best_f1:.4f} test pooled_f1={probe_results[name]['test_4']['pooled_macro_f1']:.4f} hm_f1={probe_results[name]['test_4']['hm_f1']:.4f}", flush=True)

    with open(OUT / "linear_probe_results.json", "w", encoding="utf-8") as f:
        json.dump(probe_results, f, indent=2)
    with open(OUT / "preds_linear_probe_test4.json", "w", encoding="utf-8") as f:
        json.dump(probe_preds, f)

    print("\n[STEP 4 PASSED] Saved linear_probe_results.json and predictions.")


if __name__ == "__main__":
    main()


