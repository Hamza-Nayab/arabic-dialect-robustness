# -*- coding: utf-8 -*-
"""Step 1: Leak-free deduplication, official train/dev/test splits, and data card."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent
IN_CSV = ROOT / "outputs" / "arsarcasm_binary.csv"
OUT = ROOT / "outputs_v2"
OUT.mkdir(parents=True, exist_ok=True)

MAIN_DIALECTS = ["MSA", "Egyptian", "Levantine", "Gulf"]


def normalize_for_dedup(text: str) -> str:
    t = str(text)
    t = re.sub(r"http\S+|@\w+|\bRT\b", " ", t)
    t = re.sub(r"[\u064B-\u0652\u0640]", "", t)  # strip diacritics and tatweel
    t = re.sub(r"[إأآا]", "ا", t)                 # unify alef variants
    t = re.sub(r"\s+", " ", t).strip().lower()
    return t


def main() -> None:
    raw = pd.read_csv(IN_CSV)
    raw["tweet_id"] = np.arange(len(raw))
    raw["norm_text"] = raw["text"].apply(normalize_for_dedup)
    raw = raw[raw["norm_text"].str.len() > 0].copy()

    # 1. Measure raw duplicates and cross-split leakage before cleaning
    raw_train_norms = set(raw[raw["split"] == "train"]["norm_text"])
    raw_test_norms = set(raw[raw["split"] == "test"]["norm_text"])
    cross_split_overlap = len(raw_train_norms.intersection(raw_test_norms))

    # 2. Deduplicate official test set first (preserve official test distribution)
    test_clean = (
        raw[raw["split"] == "test"]
        .drop_duplicates(subset=["norm_text"], keep="first")
        .copy()
    )

    # 3. Remove any train tweet that leaks into test_clean, then deduplicate train
    train_pool = raw[raw["split"] == "train"].copy()
    train_pool = train_pool[~train_pool["norm_text"].isin(set(test_clean["norm_text"]))]
    train_pool = train_pool.drop_duplicates(subset=["norm_text"], keep="first").copy()

    # 4. Separate Maghrebi into a single held-out zero-shot exploratory test set
    maghrebi_df = pd.concat(
        [
            train_pool[train_pool["dialect"] == "Maghrebi"],
            test_clean[test_clean["dialect"] == "Maghrebi"],
        ],
        ignore_index=True,
    )

    # 5. Filter 4 major dialects for main train/dev/test protocol
    train_4 = train_pool[train_pool["dialect"].isin(MAIN_DIALECTS)].copy()
    test_4 = test_clean[test_clean["dialect"].isin(MAIN_DIALECTS)].copy()

    # 6. Stratified 85/15 train/dev split on (dialect x sentiment label)
    strata = train_4["dialect"].astype(str) + "_" + train_4["label"].astype(str)
    tr_df, dv_df = train_test_split(
        train_4, test_size=0.15, random_state=42, stratify=strata
    )

    tr_df = tr_df.sort_values("tweet_id").reset_index(drop=True)
    dv_df = dv_df.sort_values("tweet_id").reset_index(drop=True)
    test_4 = test_4.sort_values("tweet_id").reset_index(drop=True)
    maghrebi_df = maghrebi_df.sort_values("tweet_id").reset_index(drop=True)

    # Save splits
    tr_df.to_csv(OUT / "train_4.csv", index=False, encoding="utf-8-sig")
    dv_df.to_csv(OUT / "dev_4.csv", index=False, encoding="utf-8-sig")
    test_4.to_csv(OUT / "test_4.csv", index=False, encoding="utf-8-sig")
    maghrebi_df.to_csv(OUT / "test_maghrebi.csv", index=False, encoding="utf-8-sig")

    # Compute LODO training sizes to determine exact size-matched LODO budget
    lodo_sizes = {
        d: int(len(tr_df[tr_df["dialect"] != d])) for d in MAIN_DIALECTS
    }
    n_lodo_matched = min(lodo_sizes.values())

    # Build data card summary
    def split_stats(df_part: pd.DataFrame) -> dict:
        out = {}
        for d, g in df_part.groupby("dialect"):
            n_neg = int((g["label"] == 0).sum())
            n_pos = int((g["label"] == 1).sum())
            n_sarc = int((g["sarcasm"] == True).sum() if "sarcasm" in g else 0)
            out[d] = {
                "total": int(len(g)),
                "neg": n_neg,
                "pos": n_pos,
                "neg_rate": round(n_neg / len(g), 4),
                "sarcasm_rate": round(n_sarc / len(g), 4),
            }
        return out

    card = {
        "raw_total": int(len(raw)),
        "unique_normalized_total": int(len(train_pool) + len(test_clean)),
        "duplicates_or_leaks_removed": int(len(raw) - (len(train_pool) + len(test_clean))),
        "raw_cross_split_leak_texts": int(cross_split_overlap),
        "train_4_total": int(len(tr_df)),
        "dev_4_total": int(len(dv_df)),
        "test_4_total": int(len(test_4)),
        "test_maghrebi_total": int(len(maghrebi_df)),
        "lodo_train_sizes": lodo_sizes,
        "n_lodo_matched": int(n_lodo_matched),
        "splits_by_dialect": {
            "train": split_stats(tr_df),
            "dev": split_stats(dv_df),
            "test": split_stats(test_4),
            "maghrebi_exploratory": split_stats(maghrebi_df),
        },
        "sha256_train_ids": hashlib.sha256(
            ",".join(map(str, tr_df["tweet_id"])).encode()
        ).hexdigest(),
        "sha256_dev_ids": hashlib.sha256(
            ",".join(map(str, dv_df["tweet_id"])).encode()
        ).hexdigest(),
        "sha256_test_ids": hashlib.sha256(
            ",".join(map(str, test_4["tweet_id"])).encode()
        ).hexdigest(),
    }

    with open(OUT / "data_card.json", "w", encoding="utf-8") as f:
        json.dump(card, f, indent=2, ensure_ascii=False)

    print(json.dumps(card, indent=2, ensure_ascii=False))

    # Hard verification assertions
    assert card["raw_total"] == 5207
    assert card["raw_cross_split_leak_texts"] == 44
    assert card["train_4_total"] == 3436
    assert card["dev_4_total"] == 607
    assert card["test_4_total"] == 1025
    assert card["test_maghrebi_total"] == 21
    assert card["n_lodo_matched"] == 1531
    assert set(tr_df["norm_text"]).isdisjoint(set(test_4["norm_text"]))
    assert set(dv_df["norm_text"]).isdisjoint(set(test_4["norm_text"]))
    print("\n[STEP 1 PASSED] All split counts and zero-leakage assertions verified.")


if __name__ == "__main__":
    main()


