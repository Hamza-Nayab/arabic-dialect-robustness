# -*- coding: utf-8 -*-
"""Step 9 (GPU): zero- and few-shot LLM baselines on the same leak-free test sets.

Scoring is deterministic and parse-free: after the chat template's generation prompt we read the next-token
distribution and compare P(first token of "positive"/"Positive") with P(first token of "negative"/"Negative").
  zero-shot      : threshold 0.5 on p_pos / (p_pos + p_neg)            ("ZS")
  zero-shot-cal  : threshold chosen on dev_4 to maximize macro-F1      ("ZScal"; corrects label bias)
  8-shot         : 2 exemplars per dialect (1 pos + 1 neg) from train_4, shuffled; 3 exemplar draws ("FS8_s{0,1,2}")
Also zero-shot on the ArSarcasm-v2 test set. Models are loaded in fp16 split over the visible GPUs.

Usage: python 09_llm_prompting.py --models Qwen2.5-7B-Instruct ALLaM-7B-Instruct
"""
from __future__ import annotations

import argparse
import json
import os
import time
from importlib.machinery import SourceFileLoader
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parent
step2 = SourceFileLoader("step2", str(ROOT / "02_classical_baselines.py")).load_module()

LLMS = {
    "Qwen2.5-7B-Instruct": "Qwen/Qwen2.5-7B-Instruct",
    "ALLaM-7B-Instruct": "humain-ai/ALLaM-7B-Instruct-preview",
    "AceGPT-v2-8B-Chat": "FreedomIntelligence/AceGPT-v2-8B-Chat",
    "Fanar-1-9B-Instruct": "QCRI/Fanar-1-9B-Instruct",
}
MAIN = ["MSA", "Egyptian", "Levantine", "Gulf"]
SYSTEM = "You are an expert annotator of sentiment in Arabic social-media posts (MSA and all dialects)."
INSTR = ("Classify the sentiment of the following Arabic tweet as positive or negative. "
         "Answer with exactly one word: positive or negative.\n\nTweet: {text}")
FS_SEEDS = [0, 1, 2]


def build_messages(tokenizer, text, shots):
    msgs = [{"role": "system", "content": SYSTEM}]
    for t, lab in shots:
        msgs += [{"role": "user", "content": INSTR.format(text=t)},
                 {"role": "assistant", "content": "positive" if lab == 1 else "negative"}]
    msgs.append({"role": "user", "content": INSTR.format(text=text)})
    try:
        return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    except Exception:  # templates without a system role (e.g. Gemma-2): fold it into the first user turn
        msgs = msgs[1:]
        msgs[0] = {"role": "user", "content": SYSTEM + "\n\n" + msgs[0]["content"]}
        return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def label_token_ids(tokenizer):
    ids = {}
    for lab, words in [(1, ["positive", "Positive", " positive"]), (0, ["negative", "Negative", " negative"])]:
        s = set()
        for w in words:
            t = tokenizer.encode(w, add_special_tokens=False)
            if t:
                s.add(t[0])
        ids[lab] = sorted(s)
    assert not set(ids[0]) & set(ids[1]), f"label first tokens collide: {ids}"
    return ids


@torch.no_grad()
def score(model, tokenizer, prompts, lab_ids, batch_size):
    order = np.argsort([len(p) for p in prompts])[::-1]
    p_pos = np.zeros(len(prompts))
    dev0 = next(model.parameters()).device
    for i in range(0, len(order), batch_size):
        b = order[i : i + batch_size]
        enc = tokenizer([prompts[j] for j in b], return_tensors="pt", padding=True, add_special_tokens=False).to(dev0)
        logits = model(**enc, logits_to_keep=1).logits[:, -1, :].float()  # only last position (vocab up to 128k)
        lp = torch.log_softmax(logits, -1)
        pos = torch.logsumexp(lp[:, lab_ids[1]], -1)
        neg = torch.logsumexp(lp[:, lab_ids[0]], -1)
        pr = torch.sigmoid(pos - neg).cpu().numpy()
        if not np.isfinite(pr).all():
            raise FloatingPointError("non-finite label probabilities (fp16 overflow?)")
        p_pos[b] = pr
    return p_pos


def pick_shots(tr, seed):
    rng = np.random.default_rng(seed)
    shots = []
    for d in MAIN:
        for lab in (1, 0):
            g = tr[(tr.dialect == d) & (tr.label == lab) & (tr.text.str.len() < 200)]
            r = g.iloc[int(rng.integers(len(g)))]
            shots.append((str(r.text), int(r.label), int(r.tweet_id)))
    order = rng.permutation(len(shots))
    return [shots[i] for i in order]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=list(LLMS))
    ap.add_argument("--out_dir", default=str(ROOT / "outputs_v2" / "llm"))
    ap.add_argument("--smoke_test", action="store_true")
    ap.add_argument("--deadline_hours", type=float, default=None)
    a = ap.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    t_end = None if not a.deadline_hours else time.time() + a.deadline_hours * 3600
    data = ROOT / "outputs_v2"
    tr, dv, te, mg = (pd.read_csv(data / f) for f in ["train_4.csv", "dev_4.csv", "test_4.csv", "test_maghrebi.csv"])
    v2 = pd.read_csv(data / "arsarcasm_v2" / "test.csv")
    out = Path(a.out_dir)
    if a.smoke_test:
        out = Path(os.environ.get("SMOKE_DIR", str(ROOT / "smoke_out"))) / "llm"
        per = lambda df, n: pd.concat([g.head(n) for _, g in df.groupby("dialect")]).reset_index(drop=True)
        dv, te, mg, v2 = per(dv, 2), per(te, 2), mg.head(3), per(v2, 2)
        FS_SEEDS[:] = [0]
    out.mkdir(parents=True, exist_ok=True)
    n_gpu = torch.cuda.device_count()
    for name in a.models:
        res_file = out / f"{name}.json"
        if res_file.exists():
            print(f"[llm] skip {name} (done)", flush=True)
            continue
        if t_end and time.time() > t_end:
            print("[llm] deadline reached", flush=True)
            break
        ckpt = LLMS[name] if not a.smoke_test else os.environ.get("SMOKE_LLM", "sshleifer/tiny-gpt2")
        t0 = time.time()
        rec = {"model": name, "checkpoint": ckpt, "settings": {}}
        try:
            tok = AutoTokenizer.from_pretrained(ckpt)
            tok.padding_side = "left"
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token
            if tok.chat_template is None:
                tok.chat_template = ("{% for m in messages %}{{ m['role'] }}: {{ m['content'] }}\n{% endfor %}"
                                     "{% if add_generation_prompt %}assistant: {% endif %}")
            kw = dict(torch_dtype=torch.float16 if n_gpu else torch.float32)
            if n_gpu:
                kw.update(device_map="auto", max_memory={i: "11GiB" for i in range(n_gpu)})
            model = AutoModelForCausalLM.from_pretrained(ckpt, **kw).eval()
            rec["commit_hash"] = getattr(model.config, "_commit_hash", None)
            lab_ids = label_token_ids(tok)
            rec["label_token_ids"] = {str(k): v for k, v in lab_ids.items()}

            def run(df, shots, bs):
                return score(model, tok, [build_messages(tok, str(t), [(s[0], s[1]) for s in shots]) for t in df.text],
                             lab_ids, bs)

            preds = {}
            # zero-shot (+ dev-calibrated threshold)
            pz_dv, pz_te, pz_mg, pz_v2 = run(dv, [], 16), run(te, [], 16), run(mg, [], 16), run(v2, [], 16)
            grid = np.unique(np.concatenate([[0.5], np.quantile(pz_dv, np.linspace(0.02, 0.98, 49))]))
            thr = float(max(grid, key=lambda t: f1_score(dv.label, (pz_dv >= t).astype(int), average="macro",
                                                         zero_division=0)))
            for tag, t in [("ZS", 0.5), ("ZScal", thr)]:
                p_te = (pz_te >= t).astype(int)
                rec["settings"][tag] = {"threshold": t, **step2.compute_robustness_profile(te.label.to_numpy(), p_te,
                                                                                         te.dialect.to_numpy()),
                                        "maghrebi": step2.eval_maghrebi_exploratory(mg.label.to_numpy(),
                                                                                    (pz_mg >= t).astype(int)),
                                        "v2_test": step2.compute_robustness_profile(v2.label.to_numpy(),
                                                                                    (pz_v2 >= t).astype(int),
                                                                                    v2.dialect.to_numpy())}
                preds[tag] = p_te
            np.savez_compressed(out / f"{name}_probs.npz", zs_dev=pz_dv, zs_test=pz_te, zs_mag=pz_mg, zs_v2=pz_v2)
            # few-shot
            fs_probs = {}
            for s in FS_SEEDS:
                if t_end and time.time() > t_end:
                    break
                shots = pick_shots(tr, s)
                p_te_prob, p_mg_prob = run(te, shots, 4), run(mg, shots, 4)
                fs_probs[f"fs{s}_test"], fs_probs[f"fs{s}_mag"] = p_te_prob, p_mg_prob
                p_te = (p_te_prob >= 0.5).astype(int)
                rec["settings"][f"FS8_s{s}"] = {"threshold": 0.5, "exemplar_tweet_ids": [x[2] for x in shots],
                                                **step2.compute_robustness_profile(te.label.to_numpy(), p_te,
                                                                                 te.dialect.to_numpy()),
                                                "maghrebi": step2.eval_maghrebi_exploratory(
                                                    mg.label.to_numpy(), (p_mg_prob >= 0.5).astype(int))}
                preds[f"FS8_s{s}"] = p_te
                print(f"  [{name}] FS8 seed {s}: pooled={rec['settings'][f'FS8_s{s}']['pooled_macro_f1']:.4f}", flush=True)
            np.savez_compressed(out / f"{name}_fs_probs.npz", **fs_probs)
            np.savez_compressed(out / f"{name}_preds_test4.npz", **preds)
            rec["status"] = "ok"
            del model, run
            import gc; gc.collect()
            torch.cuda.empty_cache()
        except Exception as e:  # record and continue with the next model
            rec["status"] = f"failed: {type(e).__name__}: {e}"
            print(f"[llm] {name} FAILED: {e}", flush=True)
            torch.cuda.empty_cache()
        rec["seconds"] = round(time.time() - t0, 1)
        dst = res_file if rec["status"] == "ok" else out / f"{name}.failed.json"  # failures are retried next time
        dst.with_suffix(".tmp").write_text(json.dumps(rec, indent=2))
        os.replace(dst.with_suffix(".tmp"), dst)
        s = rec["settings"]
        print(f"[llm] {name}: " + ", ".join(f"{k}={v['pooled_macro_f1']:.3f}/{v['hm_f1']:.3f}" for k, v in s.items())
              + f" [{rec['seconds']:.0f}s] {rec['status']}", flush=True)
    print("[STEP 9 DONE]", flush=True)


if __name__ == "__main__":
    main()
