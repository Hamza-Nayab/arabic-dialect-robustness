# -*- coding: utf-8 -*-
"""Step 3: Verify checkpoints, log HF commit hashes, and compute tokenizer fertility."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from transformers import AutoConfig, AutoTokenizer

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs_v2"
MAIN_DIALECTS = ["MSA", "Egyptian", "Levantine", "Gulf"]

# Core Tier-1 + Controlled CAMeLBERT Pretraining Variants
MODEL_REGISTRY = {
    "AraBERTv02": "aubmindlab/bert-base-arabertv02",
    "AraBERTv02-Twitter": "aubmindlab/bert-base-arabertv02-twitter",
    "MARBERT": "UBC-NLP/MARBERT",
    "MARBERTv2": "UBC-NLP/MARBERTv2",
    "CAMeLBERT-Mix": "CAMeL-Lab/bert-base-arabic-camelbert-mix",
    "CAMeLBERT-MSA": "CAMeL-Lab/bert-base-arabic-camelbert-msa",
    "CAMeLBERT-DA": "CAMeL-Lab/bert-base-arabic-camelbert-da",
    "XLM-R-Base": "FacebookAI/xlm-roberta-base",
    # added in response to review: completes the CAMeLBERT variant study and adds newer Arabic encoders
    "CAMeLBERT-CA": "CAMeL-Lab/bert-base-arabic-camelbert-ca",
    "ARBERTv2": "UBC-NLP/ARBERTv2",
    "AraELECTRA": "aubmindlab/araelectra-base-discriminator",
}
NEW_ENCODERS = ["CAMeLBERT-CA", "ARBERTv2", "AraELECTRA"]


def compute_fertility(tokenizer, texts: list[str]) -> dict:
    total_subwords = 0
    total_words = 0
    total_unk = 0
    unk_tok = tokenizer.unk_token
    for t in texts:
        words = str(t).split()
        toks = tokenizer.tokenize(str(t))
        total_words += max(len(words), 1)
        total_subwords += len(toks)
        if unk_tok is not None:
            total_unk += sum(1 for tok in toks if tok == unk_tok)
    return {
        "fertility": round(total_subwords / max(total_words, 1), 4),
        "unk_rate": round(total_unk / max(total_subwords, 1), 6),
    }


def main() -> None:
    te = pd.read_csv(OUT / "test_4.csv")
    mag = pd.read_csv(OUT / "test_maghrebi.csv")

    audit_rows = []
    fertility_rows = []

    for name, ckpt in MODEL_REGISTRY.items():
        print(f"Auditing {name} ({ckpt})...")
        cfg = AutoConfig.from_pretrained(ckpt)
        tok = AutoTokenizer.from_pretrained(ckpt)
        commit_hash = getattr(cfg, "_commit_hash", "unknown")

        audit_rows.append(
            {
                "model": name,
                "checkpoint": ckpt,
                "commit_hash": commit_hash,
                "vocab_size": int(getattr(cfg, "vocab_size", -1)),
                "hidden_size": int(getattr(cfg, "hidden_size", -1)),
                "num_hidden_layers": int(getattr(cfg, "num_hidden_layers", -1)),
            }
        )

        for d in MAIN_DIALECTS:
            sub_texts = te[te["dialect"] == d]["text"].astype(str).tolist()
            f_stats = compute_fertility(tok, sub_texts)
            fertility_rows.append(
                {"model": name, "dialect": d, **f_stats}
            )

        mag_texts = mag["text"].astype(str).tolist()
        f_mag = compute_fertility(tok, mag_texts)
        fertility_rows.append(
            {"model": name, "dialect": "Maghrebi", **f_mag}
        )

    pd.DataFrame(audit_rows).to_csv(OUT / "model_audit.csv", index=False)
    pd.DataFrame(fertility_rows).to_csv(OUT / "tokenizer_fertility.csv", index=False)
    print("\n[STEP 3 PASSED] Saved model_audit.csv and tokenizer_fertility.csv.")


if __name__ == "__main__":
    main()


