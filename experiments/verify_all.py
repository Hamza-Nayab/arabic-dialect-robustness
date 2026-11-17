# -*- coding: utf-8 -*-
"""Independent audit of every Verification Gate in IMPLEMENTATION_MASTERPLAN.md plus run-level integrity checks."""
from __future__ import annotations

import json
import re
from importlib.machinery import SourceFileLoader
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs_v2"
P = OUT / "raw_preds"
step2 = SourceFileLoader("step2", str(ROOT / "02_classical_baselines.py")).load_module()
D4 = ["MSA", "Egyptian", "Levantine", "Gulf"]
results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, bool(ok), detail))


# ---------------- Step 1 ----------------
card = json.load(open(OUT / "data_card.json", encoding="utf-8"))
tr, dv, te, mg = (pd.read_csv(OUT / f) for f in ["train_4.csv", "dev_4.csv", "test_4.csv", "test_maghrebi.csv"])
exp = {"tr": (tr, 3436, {"MSA": 1905, "Egyptian": 1108, "Levantine": 239, "Gulf": 184}),
       "dv": (dv, 607, {"MSA": 336, "Egyptian": 196, "Levantine": 43, "Gulf": 32}),
       "te": (te, 1025, {"MSA": 551, "Egyptian": 327, "Levantine": 86, "Gulf": 61})}
for k, (df, n, per) in exp.items():
    got = df["dialect"].value_counts().to_dict()
    check(f"S1 {k} size={n} & per-dialect counts", len(df) == n and got == per, f"{len(df)} {got}")
check("S1 Maghrebi 21 (17 neg / 4 pos)", len(mg) == 21 and (mg.label == 0).sum() == 17 and (mg.label == 1).sum() == 4)
check("S1 n_lodo_matched=1531, raw=5207, leaks=44, removed=118",
      card["n_lodo_matched"] == 1531 and card["raw_total"] == 5207 and card["raw_cross_split_leak_texts"] == 44
      and card["duplicates_or_leaks_removed"] == 118)
id_col = "norm_text" if "norm_text" in tr.columns else "tweet_id"
sets = {k: set(df[id_col]) for k, df in [("tr", tr), ("dv", dv), ("te", te), ("mg", mg)]}
check(f"S1 zero overlap train/dev/test/maghrebi ({id_col})",
      all(sets[a].isdisjoint(sets[b]) for a, b in [("tr", "te"), ("dv", "te"), ("tr", "dv"), ("tr", "mg"), ("dv", "mg"), ("te", "mg")]))
check(f"S1 no duplicates within any split", all(df[id_col].is_unique for df in [tr, dv, te, mg]))

# ---------------- Step 2 ----------------
cb = json.load(open(OUT / "classical_baselines.json"))
m = cb["Majority"]["test_4"]
check("S2 Majority old_dri=0.9996, hm=0.4095, wd=0.4024",
      round(m["old_dri_acc"], 4) == 0.9996 and round(m["hm_f1"], 4) == 0.4095 and round(m["wd_f1"], 4) == 0.4024)
t = cb["TFIDF_LR_balanced"]["test_4"]
check("S2 TF-IDF balanced pooled>0.75 & hm>0.74", t["pooled_macro_f1"] > 0.75 and t["hm_f1"] > 0.74,
      f"{t['pooled_macro_f1']:.4f}/{t['hm_f1']:.4f}")
cp = json.load(open(OUT / "preds_classical_test4.json"))
for k in cp:
    prof = step2.compute_robustness_profile(te.label.to_numpy(), np.array(cp[k]), te.dialect.to_numpy())
    check(f"S2 {k} preds reproduce stored metrics", abs(prof["hm_f1"] - cb[k]["test_4"]["hm_f1"]) < 1e-9)

# ---------------- Step 3 ----------------
au = pd.read_csv(OUT / "model_audit.csv")
NENC = 11
check(f"S3 {NENC} distinct checkpoints with commit hashes",
      len(au) == NENC and au.checkpoint.is_unique and au.commit_hash.str.fullmatch(r"[0-9a-f]{40}").all())
fe = pd.read_csv(OUT / "tokenizer_fertility.csv")
check(f"S3 fertility+unk for {NENC} models x 5 varieties", len(fe) == 5 * NENC and fe[["fertility", "unk_rate"]].notna().all().all()
      and set(fe.dialect) == set(D4 + ["Maghrebi"]))

# ---------------- Step 4 ----------------
lp = json.load(open(OUT / "linear_probe_results.json"))
pp = json.load(open(OUT / "preds_linear_probe_test4.json"))
check(f"S4 probe for all {NENC} models", set(lp) == set(au.model))
for k, v in lp.items():
    p = np.array(pp[k]); y = te.label.to_numpy()
    r0, r1 = np.mean(p[y == 0] == 0), np.mean(p[y == 1] == 1)
    prof = step2.compute_robustness_profile(y, p, te.dialect.to_numpy())
    check(f"S4 {k}: pooled F1>0.65, both recalls>0, preds reproduce",
          v["test_4"]["pooled_macro_f1"] > 0.65 and r0 > 0 and r1 > 0 and abs(prof["pooled_macro_f1"] - v["test_4"]["pooled_macro_f1"]) < 1e-9,
          f"F1={v['test_4']['pooled_macro_f1']:.3f} C={v['best_C']} rec0={r0:.2f} rec1={r1:.2f}")
    grid = [0.0003, 0.001, 0.003, 0.01, 0.1, 1.0, 5.0]
    check(f"S4 {k}: selected C not on grid boundary", grid[0] < v["best_C"] < grid[-1], str(v["best_C"]))

# ---------------- Step 5 ----------------
runs = [json.loads(l) for l in (OUT / "finetune_runs.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
keys = [r["run_key"] for r in runs]
check("S5 no duplicate run keys", len(keys) == len(set(keys)), f"{len(keys)} runs")
check("S5 full 12-layer fine-tuning, 4 epochs max, in every run",
      all(r.get("finetune") == "full" and r["epochs_max"] == 4 for r in runs))
sel = {p.stem: json.loads(p.read_text()) for p in (OUT / "selected_lrs").glob("*.json")}
check(f"S5 per-model LR selected on dev over {{1e-5,2e-5,3e-5,5e-5}} for all {NENC} encoders",
      set(sel) == set(au.model)
      and all(sorted(map(float, v["grid"])) == [1e-5, 2e-5, 3e-5, 5e-5] for v in sel.values())
      and all(abs(float(v["selected_lr"]) - float(max(v["grid"], key=lambda k: v["grid"][k]["dev_f1"]))) < 1e-15
              for v in sel.values()))
check("S5 every run uses its model's selected LR", all(abs(r["lr"] - float(sel[r["model"]]["selected_lr"])) < 1e-15 for r in runs))
plan = {m: "full" for m in au.model}
want = {"full": {"PID": 5, "PID_weighted": 5, "LODO_matched": 5, "MSA_only": 5, "PID_n1531": 5, "PID_n1905": 5}}
cnt = pd.Series([(r["model"], r["regime"]) for r in runs]).value_counts().to_dict()
for mdl, kind in plan.items():
    for reg, n in want[kind].items():
        check(f"S5 {mdl} {reg}: {n} seeds", cnt.get((mdl, reg), 0) == n, str(cnt.get((mdl, reg), 0)))
prefix = {"PID": "PID", "PID_weighted": "PIDw", "LODO_matched": "LODO", "MSA_only": "MSAonly",
          "PID_n1531": "PIDn1531", "PID_n1905": "PIDn1905"}
bad = []
for r in runs:
    f = P / f"{prefix[r['regime']]}_{r['model']}_seed{r['seed']}.npy"
    if not f.exists():
        bad.append(f"missing {f.name}")
        continue
    p = np.load(f)
    if len(p) != 1025:
        bad.append(f"len {f.name}")
        continue
    prof = step2.compute_robustness_profile(te.label.to_numpy(), p, te.dialect.to_numpy())
    for k in ["pooled_macro_f1", "hm_f1", "wd_f1"]:
        if abs(prof[k] - r[k]) > 1e-9:
            bad.append(f"{r['run_key']} {k} mismatch")
check("S5 every run has a 1,025-length prediction file that reproduces its logged metrics", not bad, "; ".join(bad[:5]))
check("S5 seeds are exactly 42-46 for every (model, regime)",
      all(sorted(r["seed"] for r in runs if (r["model"], r["regime"]) == k) == [42, 43, 44, 45, 46] for k in cnt))

# ---------------- Step 5b (3-class replication) ----------------
three = json.load(open(OUT / "three_class_replication.json", encoding="utf-8"))
dc3 = three["data_card"]
check("S5b 3-class data: 10,547 raw, leak-free splits, 3 labels",
      dc3["raw_total"] == 10547 and dc3["train"] + dc3["dev"] + dc3["test"] + dc3["maghrebi"] == dc3["unique_total"]
      and sorted(map(int, dc3["train_label_counts"])) == [0, 1, 2])
check("S5b 3-class: 4 core encoders x 5 seeds + Majority + TF-IDF",
      set(three["models"]) == {"AraBERTv02", "MARBERT", "CAMeLBERT-Mix", "XLM-R-Base"}
      and all(v["n_seeds"] == 5 for v in three["models"].values()) and {"Majority", "TFIDF_LR_balanced"} <= set(three["classical"]))
te3 = pd.read_csv(OUT / "three_class" / "test_3cls.csv")
runs3 = []
for pth in (OUT / "three_class").glob("threeclass_runs*.jsonl"):
    runs3 += [json.loads(l) for l in pth.read_text(encoding="utf-8").splitlines() if l.strip()]
runs3 = list({r["run_key"]: r for r in runs3}.values())
bad3 = [r["run_key"] for r in runs3 if len(np.load(OUT / "three_class" / "raw_preds" / r["pred_files"][0])) != len(te3)]
check("S5b every 3-class run has a full-length prediction file", len(runs3) == 20 and not bad3, f"{len(runs3)} runs {bad3[:3]}")
lodo = [r for r in runs if r["regime"] == "LODO_matched"]
check("S5 LODO: 4 target models per run, each trained on 1,531", all(set(r["lodo_targets"]) == set(D4) and
      all(v["train_n"] == 1531 for v in r["lodo_targets"].values()) for r in lodo))
check("S5 MSA-only trained on 1,905", all(r.get("train_n") == 1905 for r in runs if r["regime"] == "MSA_only"))
check("S5 controls trained on 1,531 / 1,905", all(r.get("train_n") == int(r["regime"].split("_n")[1]) for r in runs if "_n" in r["regime"]))
check("S5 Maghrebi evaluated for PID/PIDw/MSA-only runs", all("maghrebi" in r for r in runs if r["regime"] in ("PID", "PID_weighted", "MSA_only")))

# ---------------- Step 6 ----------------
agg = pd.read_csv(OUT / "aggregated_multiseed_results.csv")
check("S6 aggregated has *_mean and *_std for all (model,regime)", len(agg) == len(cnt) and
      all(c in agg for c in ["hm_f1_mean", "hm_f1_std", "wd_f1_mean", "pooled_macro_f1_std"]))
for _, row in agg.iterrows():
    g = [r for r in runs if r["model"] == row.model and r["regime"] == row.regime]
    if abs(np.mean([r["hm_f1"] for r in g]) - row.hm_f1_mean) > 1e-9 or row.n_seeds != len(g):
        check(f"S6 agg mismatch {row.model} {row.regime}", False)
st = json.load(open(OUT / "statistical_analysis_bundle.json", encoding="utf-8"))
check("S6 bundle: bootstrap CIs, Holm-McNemar, GLMM, synthetic validity",
      all(k in st and st[k] for k in ["bootstrap_pid", "mcnemar_holm", "mixed_effects_glm", "synthetic_metric_validity"]))
check("S6 McNemar covers 4 core x pooled+4 dialects (30 tests)", len(st["mcnemar_holm"]) == 30)
ens = np.load(OUT / "ensemble_preds_test4.npz")
for mdl in plan:
    mats = np.vstack([np.load(P / f"PID_{mdl}_seed{s}.npy") for s in [42, 43, 44, 45, 46]])
    ok = np.array_equal((mats.mean(0) >= 0.5).astype(int), ens[f"PID:{mdl}"])
    check(f"S6 ensemble PID:{mdl} = majority vote of its 5 seeds", ok)
# stale-file check: newest prediction must not be newer than the stats bundle
newest = max(f.stat().st_mtime for f in P.glob("*.npy") if "selectedLR" not in f.name)
check("S6 stats newer than every prediction file", (OUT / "statistical_analysis_bundle.json").stat().st_mtime >= newest)

# ---------------- Step 7 ----------------
for f in ["error_sarcasm_by_dialect.csv", "error_linguistic_artifacts.csv", "fertility_vs_f1.csv"]:
    df = pd.read_csv(OUT / f)
    cols = [c for c in df if c not in ("recall_pos", "recall_neg")]
    check(f"S7 {f} populated, no missing values", len(df) > 0 and df[cols].notna().all().all(), f"{len(df)} rows")
check(f"S7 fertility correlation uses {NENC} models x 4 dialects", json.load(open(OUT / "fertility_correlation.json"))["n_points"] == 4 * NENC)
check("S7 newer than stats", (OUT / "fertility_correlation.json").stat().st_mtime >= (OUT / "aggregated_multiseed_results.csv").stat().st_mtime)

# ---------------- Step 8 ----------------
FIG, TAB = ROOT.parent / "figures_v2", ROOT.parent / "tables_v2"
figs = ["fig1_per_dialect_forest", "fig2_transfer_drop_heatmap", "fig3_sarcasm_and_fertility", "fig4_metric_validity",
        "fig5_learning_curve"]
check("S8 5 figures as PDF + PNG", all((FIG / f"{f}.{e}").exists() for f in figs for e in ["pdf", "png"]))
src8 = (ROOT / "08_generate_paper_artifacts.py").read_text(encoding="utf-8")
check("S8 no 'zero-shot' in figure titles", "zero-shot" not in src8.lower())
tabs = ["tab_data_card", "tab_models", "tab_main_results", "tab_transfer", "tab_metric_validity", "tab_sarcasm", "tab_mcnemar",
        "tab_lr_sensitivity", "tab_maghrebi", "tab_three_class", "tab_tost", "tab_hm_variance", "tab_learning_curve", "tab_leaky",
        "tab_v2", "tab_llm", "tab_probe_layers", "tab_audit_rescore", "tab_did", "tab_glmm", "tab_pidw", "macros_review"]
check("S8 all LaTeX tables generated", all((TAB / f"{t}.tex").exists() for t in tabs),
      str([t for t in tabs if not (TAB / f"{t}.tex").exists()]))
check("S8 artifacts newer than stats", min((FIG / f"{f}.pdf").stat().st_mtime for f in figs) >= (OUT / "statistical_analysis_bundle.json").stat().st_mtime)
check("S8 review artifacts newer than review stats and GLMM",
      min((TAB / f"{t}.tex").stat().st_mtime for t in ["tab_tost", "tab_glmm", "macros_review"])
      >= max((OUT / "review_stats.json").stat().st_mtime, (OUT / "glmm_mcmc.json").stat().st_mtime))
check("S8 main table has no Maghrebi column", "Maghrebi" not in (TAB / "tab_main_results.tex").read_text(encoding="utf-8")
      and "Maghrebi" not in (TAB / "tab_data_card.tex").read_text(encoding="utf-8"))
mt = (TAB / "tab_main_results.tex").read_text(encoding="utf-8")
ncol = len(re.search(r"\\begin\{tabular\}\{([^}]*)\}", mt).group(1))
check("S8 main table column spec matches header", ncol == mt.splitlines()[2].count("&") + 1, str(ncol))

# ---------------- Step 9 (paper) ----------------
tex = (ROOT.parent / "main.tex").read_text(encoding="utf-8")
bibtex = (ROOT.parent / "references.tex").read_text(encoding="utf-8")
check("S9 title states dialect-level generalization", "Dialect-Level Generalization" in tex)
check("S9 canonical citations present", all(k in tex for k in ["antoun2020arabert", "abdulmageed2021arbert", "inoue2021interplay",
      "abufarha2020arsarcasm", "abufarha2021arsarcasmv2", "conneau2020xlmr", "sagawa2020groupdro", "dror2018hitchhiker", "abdelali2024larabench"]))
cited = set(c.strip() for c in sum([c.split(",") for c in re.findall(r"\\cite\{([^}]*)\}", tex)], []))
bib = set(re.findall(r"\\bibitem\{([^}]*)\}", bibtex))
check("S9 every \\cite has a bibitem and every bibitem is cited", cited <= bib and bib <= cited, f"uncited={bib - cited} missing={cited - bib}")
check("S9 40-60 references", 40 <= len(bib) <= 60, str(len(bib)))
labels = set(re.findall(r"\\label\{([^}]*)\}", tex))
refs = set(re.findall(r"\\ref\{([^}]*)\}", tex))
check("S9 every \\ref resolves", refs <= labels, str(refs - labels))
inputs = re.findall(r"\\input\{([^}]*)\}", tex) + re.findall(r"\\includegraphics\[[^]]*\]\{([^}]*)\}", tex)
check("S9 every input/figure file exists", all((ROOT.parent / i).exists() for i in inputs), str([i for i in inputs if not (ROOT.parent / i).exists()]))
check("S9 old random-head results / 120-200 sample caps not reported", "frozen\\_probe" not in tex and "0.9955" not in tex and "351 train" not in tex)
check("S9 old figures/ dir not referenced", "{figures/" not in tex)
check("S9 no 'earlier version' / 'previous version' / 'Old DRI' framing",
      not re.search(r"earlier version|previous version|original study|Old DRI|former index|this revision", tex, re.I))
check("S9 no TBD placeholders left", "\\TBD{" not in tex.split("\\begin{document}")[1]
      and "\\TBD{" not in (TAB / "macros_review.tex").read_text(encoding="utf-8"))
abstract = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", tex, re.S).group(1)
words = len(re.sub(r"\\[a-zA-Z]+|[{}$~]", " ", abstract).split())
check("S9 abstract 200-250 words (macros counted as one word)", 200 <= words <= 250, str(words))
check("S9 Maghrebi only in the appendix (exploratory)", "app:maghrebi" in tex and "exploratory" in tex.lower())
check("S9 Arabic quotes are literal \" in the example table", "\\textarabic{ضحك على الذقون}" in tex and "\\textarabic{بـ}" in tex and "full tweet text omitted" in tex)
check("S9 numbered contributions, RQ conclusion, Discussion, Declarations",
      "\\begin{enumerate}" in tex and "\\section{Discussion}" in tex and "\\section*{Declarations}" in tex
      and all(f"\\textbf{{RQ{i}:}}" in tex for i in range(1, 5)))
check("S9 consistent naming AraBERTv02-Twitter", not re.search(r"AraBERT[- ]?Twitter|AraBERTv0?2 Twitter|XLM-R-base", tex))
check("S9 compiled PDF newer than main.tex and tables",
      (ROOT.parent / "main.pdf").stat().st_mtime >= max((ROOT.parent / "main.tex").stat().st_mtime, *[(TAB / f"{t}.tex").stat().st_mtime for t in tabs]))

# ---------------- Step 10 (review-round experiments) ----------------
def rd(p):
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()] if p.exists() else []


rs = json.load(open(OUT / "review_stats.json"))
check("S10 hierarchical CIs for every encoder x 6 regimes", all(f"{m}|{r}" in rs["hier_ci"] for m in plan for r in want["full"]))
check("S10 TOST + MDE for every encoder (LODO and MSA-only)", all(f"{m}|{r}" in rs["transfer_tost"] for m in plan for r in ("LODO", "MSAonly")))
llm_ok = {p.stem for p in (OUT / "llm").glob("*.json") if not p.name.endswith(".failed.json")}
check("S10 4 LLMs evaluated (ZS, ZScal, FS8 x3)", llm_ok == {"Qwen2.5-7B-Instruct", "ALLaM-7B-Instruct", "AceGPT-v2-8B-Chat", "Fanar-1-9B-Instruct"}
      and all(len([k for k in np.load(OUT / "llm" / f"{n}_preds_test4.npz").files if k.startswith("FS8")]) == 3 for n in llm_ok))
lc = rd(OUT / "lc_runs.jsonl")
lc_cnt = pd.Series([(r["model"], r["target"], r["k_tag"]) for r in lc]).value_counts() if lc else pd.Series(dtype=int)
check("S10 learning curve: 4 encoders x 3 targets, 5 seeds per k", len(lc) > 0 and lc_cnt.min() == 5 and
      len({(r["model"], r["target"]) for r in lc}) == 12, f"{len(lc)} runs")
lk = rd(OUT / "leaky_runs.jsonl")
check("S10 leaky splits: 4 encoders x 5 seeds", len(lk) == 20 and len({r["model"] for r in lk}) == 4, f"{len(lk)} runs")
v2 = rd(OUT / "v2_runs.jsonl")
check("S10 ArSarcasm-v2: 5 encoders x 5 regimes x 5 seeds", len(v2) == 125, f"{len(v2)} runs")
G = json.load(open(OUT / "glmm_mcmc.json"))
check(f"S10 GLMM (MCMC) over all {NENC} encoders: 4 chains x 1000, 0 divergences, R-hat<1.01",
      len(G["models"]) == NENC and G["chains"] == 4 and G["draws"] == 1000 and G["divergences"] == 0 and G["max_r_hat"] < 1.01,
      f"{len(G['models'])} models, div={G['divergences']}, rhat={G['max_r_hat']:.3f}")
LP = json.load(open(OUT / "layerwise_probe.json"))
check(f"S10 layer-wise mean+CLS probes for {NENC} encoders", len(LP) == NENC and all("cls" in v and "mean" in v for v in LP.values()))
DA = json.load(open(OUT / "data_audit.json", encoding="utf-8"))
check("S10 near-duplicate and dialect-ID audits present", "near_duplicates" in DA and "dialect_id" in DA
      and len(DA["pid_full_test"]) == NENC)

npass = sum(ok for _, ok, _ in results)
for name, ok, det in results:
    print(("PASS " if ok else "FAIL ") + name + (f"  [{det}]" if det and not ok else ""))
print(f"\n{npass}/{len(results)} checks passed")
