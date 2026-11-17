# -*- coding: utf-8 -*-
"""Step 4b: layer-wise linear probes (mean-pooled and [CLS]) for every encoder.

For each encoder and each hidden-state layer l = 0..L (0 = embeddings), fit the Step-4 probe recipe
(standardize -> class-balanced L2 logistic regression, C in {3e-4, 1e-3, 3e-3, 1e-2, 1e-1, 1, 5} chosen on dev_4)
on (a) the attention-masked mean of layer-l token states and (b) the layer-l [CLS] (first-token) state.
Saves outputs_v2/layerwise_probe.json (dev/test pooled F1, HM-F1, WD-F1 per layer) and the per-layer
feature cache in outputs_v2/embeddings_layers/ (float16) so the fits can be re-run without re-encoding.
"""
from __future__ import annotations

import json
from importlib.machinery import SourceFileLoader
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs_v2"
CACHE = OUT / "embeddings_layers"
step2 = SourceFileLoader("step2", str(ROOT / "02_classical_baselines.py")).load_module()
step3 = SourceFileLoader("step3", str(ROOT / "03_model_and_fertility_audit.py")).load_module()
C_GRID = [0.0003, 0.001, 0.003, 0.01, 0.1, 1.0, 5.0]


@torch.no_grad()
def encode_layers(name, ckpt, texts):
    path = CACHE / f"{name}.npz"
    if path.exists():
        z = np.load(path)
        return z["mean"], z["cls"]
    tok = AutoTokenizer.from_pretrained(ckpt)
    model = AutoModel.from_pretrained(ckpt).eval()
    enc = tok(texts, truncation=True, max_length=128)
    order = np.argsort([len(x) for x in enc["input_ids"]])
    L = model.config.num_hidden_layers + 1
    H = model.config.hidden_size
    mean = np.zeros((L, len(texts), H), dtype=np.float16)
    cls = np.zeros((L, len(texts), H), dtype=np.float16)
    for i in range(0, len(order), 64):
        b = order[i : i + 64]
        batch = tok.pad({"input_ids": [enc["input_ids"][j] for j in b],
                         "attention_mask": [enc["attention_mask"][j] for j in b]}, return_tensors="pt")
        hs = model(**batch, output_hidden_states=True).hidden_states
        m = batch["attention_mask"].unsqueeze(-1).float()
        for l, h in enumerate(hs):
            mean[l, b] = ((h * m).sum(1) / m.sum(1).clamp(min=1)).numpy().astype(np.float16)
            cls[l, b] = h[:, 0].numpy().astype(np.float16)
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez(path, mean=mean, cls=cls)
    return mean, cls


def fit_probe(X_tr, y_tr, X_dv, y_dv, X_te):
    best = (-1.0, None, None)
    for c in C_GRID:
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=c, class_weight="balanced", max_iter=3000))
        clf.fit(X_tr, y_tr)
        f = f1_score(y_dv, clf.predict(X_dv), average="macro", zero_division=0)
        if f > best[0]:
            best = (f, c, clf)
    return best[0], best[1], best[2].predict(X_te)


def main() -> None:
    tr, dv, te = (pd.read_csv(OUT / f) for f in ["train_4.csv", "dev_4.csv", "test_4.csv"])
    texts = pd.concat([tr.text, dv.text, te.text]).astype(str).tolist()
    a, b = len(tr), len(tr) + len(dv)
    res_path = OUT / "layerwise_probe.json"
    res = json.loads(res_path.read_text()) if res_path.exists() else {}
    for name, ckpt in step3.MODEL_REGISTRY.items():
        if name in res:
            continue
        print(f"[4b] {name}", flush=True)
        mean, cls = encode_layers(name, ckpt, texts)
        jobs = [(pool, l) for pool in ("mean", "cls") for l in range(mean.shape[0])]
        feats = {"mean": mean, "cls": cls}

        def one(pool, l):
            X = feats[pool][l].astype(np.float32)
            f_dv, c, p = fit_probe(X[:a], tr.label, X[a:b], dv.label, X[b:])
            prof = step2.compute_robustness_profile(te.label.to_numpy(), p, te.dialect.to_numpy())
            return pool, l, {"dev_f1": f_dv, "C": c, "pooled_macro_f1": prof["pooled_macro_f1"], "hm_f1": prof["hm_f1"],
                             "wd_f1": prof["wd_f1"], "per_dialect_f1": prof["per_dialect_f1"]}

        out = Parallel(n_jobs=-1, prefer="threads")(delayed(one)(p, l) for p, l in jobs)
        res[name] = {"mean": {}, "cls": {}}
        for pool, l, r in out:
            res[name][pool][str(l)] = r
        for pool in ("mean", "cls"):
            best_l = max(res[name][pool], key=lambda k: res[name][pool][k]["dev_f1"])
            res[name][pool + "_best_layer_by_dev"] = int(best_l)
        res_path.write_text(json.dumps(res, indent=2))
        last = str(mean.shape[0] - 1)
        print(f"   last-layer mean={res[name]['mean'][last]['hm_f1']:.3f} cls={res[name]['cls'][last]['hm_f1']:.3f} "
              f"best-dev mean layer={res[name]['mean_best_layer_by_dev']}", flush=True)
    print("[STEP 4b PASSED]")


if __name__ == "__main__":
    main()
