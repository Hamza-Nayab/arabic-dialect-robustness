# -*- coding: utf-8 -*-
"""Shared CPU-efficient encoder pass (used by Steps 4 and 5).

One forward pass per checkpoint over every unique text in train/dev/test/Maghrebi.
Saves:
  * embeddings/{model}.npz   mean-pooled last-layer states (Step 4 linear probe)
  * hidden_cache/{model}.pt  frozen layer-(L - TOP_K) token states (Step 5 top-layer fine-tuning)
Texts are length-sorted inside the pass so padding waste is minimal; outputs are
returned in the original order.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs_v2"
EMB_DIR = OUT / "embeddings"
HID_DIR = OUT / "hidden_cache"
MAX_LEN = 128
BATCH_SIZE = 32
CACHE_LAYERS_FROM_TOP = 4  # cache input to the top-4 transformer blocks

torch.set_num_threads(max(1, torch.get_num_threads()))


def load_split_frames() -> dict[str, pd.DataFrame]:
    return {
        "tr": pd.read_csv(OUT / "train_4.csv"),
        "dv": pd.read_csv(OUT / "dev_4.csv"),
        "te": pd.read_csv(OUT / "test_4.csv"),
        "mg": pd.read_csv(OUT / "test_maghrebi.csv"),
    }


@torch.no_grad()
def encode_and_cache(name: str, ckpt: str, force: bool = False) -> None:
    EMB_DIR.mkdir(parents=True, exist_ok=True)
    HID_DIR.mkdir(parents=True, exist_ok=True)
    emb_path, hid_path = EMB_DIR / f"{name}.npz", HID_DIR / f"{name}.pt"
    if emb_path.exists() and hid_path.exists() and not force:
        return

    frames = load_split_frames()
    keys = list(frames.keys())
    texts = pd.concat([frames[k]["text"] for k in keys]).astype(str).tolist()
    bounds = np.cumsum([0] + [len(frames[k]) for k in keys])

    tok = AutoTokenizer.from_pretrained(ckpt)
    model = AutoModel.from_pretrained(ckpt)
    model.eval()
    n_layers = model.config.num_hidden_layers
    cache_layer = n_layers - CACHE_LAYERS_FROM_TOP  # hidden_states index = input to block cache_layer

    enc = tok(texts, truncation=True, max_length=MAX_LEN)
    lengths = np.array([len(x) for x in enc["input_ids"]])
    order = np.argsort(lengths, kind="stable")

    mean_pooled = np.zeros((len(texts), model.config.hidden_size), dtype=np.float32)
    cached_states: list[torch.Tensor | None] = [None] * len(texts)

    for start in range(0, len(order), BATCH_SIZE):
        idx = order[start : start + BATCH_SIZE]
        batch = tok.pad({"input_ids": [enc["input_ids"][i] for i in idx],
                         "attention_mask": [enc["attention_mask"][i] for i in idx]},
                        return_tensors="pt")
        out = model(**batch, output_hidden_states=True)
        mask = batch["attention_mask"].unsqueeze(-1).float()
        pooled = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        mean_pooled[idx] = pooled.numpy()
        h = out.hidden_states[cache_layer]
        for j, i in enumerate(idx):
            L = int(batch["attention_mask"][j].sum())
            cached_states[i] = h[j, :L].to(torch.float16).clone()
        if (start // BATCH_SIZE) % 20 == 0:
            print(f"  [{name}] encoded {start + len(idx)}/{len(texts)}", flush=True)

    np.savez_compressed(
        emb_path, **{k: mean_pooled[bounds[i] : bounds[i + 1]] for i, k in enumerate(keys)}
    )
    torch.save(
        {
            "checkpoint": ckpt,
            "cache_layer": cache_layer,
            "n_layers": n_layers,
            **{k: cached_states[bounds[i] : bounds[i + 1]] for i, k in enumerate(keys)},
        },
        hid_path,
    )
    del model
