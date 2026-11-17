# -*- coding: utf-8 -*-
"""Step 0: Rebuild outputs/arsarcasm_binary.csv from the public HF ArSarcasm release."""
from pathlib import Path

import pandas as pd
from datasets import load_dataset

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

DIALECT_MAP = {"egypt": "Egyptian", "gulf": "Gulf", "levant": "Levantine", "magreb": "Maghrebi", "msa": "MSA"}


def main() -> None:
    ds = load_dataset("iabufarha/ar_sarcasm")
    frames = []
    for split in ["train", "test"]:
        d = ds[split]
        f = d.features
        df = pd.DataFrame({
            "text": d["tweet"],
            "dialect": [DIALECT_MAP[f["dialect"].int2str(x)] for x in d["dialect"]],
            "sentiment": [f["sentiment"].int2str(x) for x in d["sentiment"]],
            "sarcasm": [bool(x) for x in d["sarcasm"]],
            "source": d["source"],
        })
        df["split"] = split
        frames.append(df)
    raw = pd.concat(frames, ignore_index=True)
    print("Raw corpus:", len(raw))
    binary = raw[raw["sentiment"] != "neutral"].copy()
    binary["label"] = (binary["sentiment"] == "positive").astype(int)
    binary = binary[["text", "label", "dialect", "sarcasm", "split", "source", "sentiment"]].reset_index(drop=True)
    binary.to_csv(OUT / "arsarcasm_binary.csv", index=False, encoding="utf-8-sig")
    print("Binary rows:", len(binary))
    print(binary["dialect"].value_counts().to_dict())
    print(binary["label"].value_counts().to_dict())
    print(binary["split"].value_counts().to_dict())


if __name__ == "__main__":
    main()
