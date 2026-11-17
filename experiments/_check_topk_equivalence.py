# -*- coding: utf-8 -*-
"""Sanity check: cached lower-layer states + top-K blocks reproduce the full model's logits; time a training step."""
import sys
import time
from importlib.machinery import SourceFileLoader
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, set_seed

ROOT = Path(__file__).resolve().parent
s5 = SourceFileLoader("s5", str(ROOT / "05_finetune_transfer.py")).load_module()

name = sys.argv[1] if len(sys.argv) > 1 else "AraBERTv02"
ckpt = s5.step3.MODEL_REGISTRY[name]
cache = torch.load(s5.cache_mod.HID_DIR / f"{name}.pt")
dv = pd.read_csv(s5.OUT / "dev_4.csv")

set_seed(0)
full = AutoModelForSequenceClassification.from_pretrained(ckpt, num_labels=2).eval()
tok = AutoTokenizer.from_pretrained(ckpt)
top = s5.TopBlocksClassifier(full, cache["cache_layer"]).eval()
with torch.no_grad():
    texts = dv["text"].astype(str).tolist()[:16]
    enc = tok(texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
    ref = full(**enc).logits
    x, m = s5.pad_batch(cache["dv"][:16])
    got = top(x, m)
print("max |logit diff| =", float((ref - got).abs().max()))

top.train()
opt = torch.optim.AdamW(top.parameters(), lr=5e-5)
states = cache["tr"][:320]
t0 = time.time()
for i in range(0, 320, 16):
    x, m = s5.pad_batch(states[i : i + 16])
    loss = torch.nn.functional.cross_entropy(top(x, m), torch.zeros(len(x), dtype=torch.long))
    loss.backward(); opt.step(); opt.zero_grad()
print("train examples/s =", 320 / (time.time() - t0))
