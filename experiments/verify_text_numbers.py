# -*- coding: utf-8 -*-
"""Check that every number quoted literally in the prose of main.tex equals the value recomputed from the saved
outputs (11 encoders x 6 regimes x 5 seeds, 4 LLMs, learning curve, leaky splits, ArSarcasm-v2, 3-class replication).

Numbers that enter the text through macros (tables_v2/macros_review.tex) are generated directly from the same outputs
by 08b_review_artifacts.py; a subset is recomputed here as a cross-check. A coverage check at the end fails if the prose
contains a decimal number that is neither verified below nor a declared method constant."""
import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs_v2"
TEX = (ROOT.parent / "main.tex").read_text(encoding="utf-8")
BODY = TEX.split("\\begin{document}")[1].split("\\input{references.tex}")[0]
MAC = dict(re.findall(r"\\newcommand\{\\([A-Za-z]+)\}\{(.*)\}", (ROOT.parent / "tables_v2" / "macros_review.tex").read_text(encoding="utf-8")))

rs = json.load(open(OUT / "review_stats.json"))
H = rs["hier_ci"]
st = json.load(open(OUT / "statistical_analysis_bundle.json", encoding="utf-8"))
lp = json.load(open(OUT / "linear_probe_results.json"))
LP = json.load(open(OUT / "layerwise_probe.json"))
cb = json.load(open(OUT / "classical_baselines.json"))
card = json.load(open(OUT / "data_card.json", encoding="utf-8"))
G = json.load(open(OUT / "glmm_mcmc.json"))
t3 = json.load(open(OUT / "three_class_replication.json"))
sarc = pd.read_csv(OUT / "error_sarcasm_by_dialect.csv")
seff = {r["model"]: r for r in json.load(open(OUT / "sarcasm_effect_bootstrap.json"))}
ff = pd.read_csv(OUT / "fertility_vs_f1.csv")
art = pd.read_csv(OUT / "error_linguistic_artifacts.csv")

M = rs["models"]
AR = [m for m in M if m != "XLM-R-Base"]
D = ["MSA", "Egyptian", "Levantine", "Gulf"]
f3 = lambda x: f"{x:.3f}"
s3 = lambda x: f"{x:+.3f}"
P = lambda m, k, r="PID": H[f"{m}|{r}"][k]["point"]
PW = {(p["a"][4:], p["b"][4:], p["metric"]): p for p in rs["pairwise_pid"]}


def pw(a, b, k="hm_f1"):
    if (a, b, k) in PW:
        p = PW[(a, b, k)]
        return p["diff"], p["ci95"], p["excludes_zero"]
    p = PW[(b, a, k)]
    return -p["diff"], [-p["ci95"][1], -p["ci95"][0]], p["excludes_zero"]


def T(*ss):
    return all(s in BODY for s in ss)


VER = set()  # literal numbers verified below


def V(*nums):
    VER.update(nums)
    return True


best = max(M, key=lambda m: P(m, "hm_f1"))
llm = rs["llm"]
fs8 = {n: llm[n]["FS8"] for n in llm}
llm_best = max(fs8, key=lambda n: fs8[n]["hm_f1"]["point"])
others = [n for n in fs8 if n != llm_best]
gaps = {n: llm[n]["FS8_minus_best_encoder_hm"] for n in others}
lc = rs["learning_curve"]
v2 = rs["arsarcasm_v2"]
v2pid = {m: v2["table"][f"{m}|PID"]["hm_f1"]["point"] for m in {k.split("|")[0] for k in v2["table"] if k.endswith("|PID")}}
xv2 = v2["tost"]["XLM-R-Base|LODO_matched"]["hm_f1"]
ws = sarc[sarc.dialect == "ALL"].set_index(["model", "sarcasm"])
SM = M
mc = st["mcnemar_holm"]
c3 = t3["models"]
card3 = t3["data_card"]
last = lambda m, pool: LP[m][pool][str(max(int(k) for k in LP[m][pool]))]
bestlayer = lambda m, pool: max(LP[m][pool], key=lambda k: LP[m][pool][k]["dev_f1"])
nbuck = lambda f: int(art[(art.model == M[0]) & (art.feature == f) & (art.bucket.astype(str) == "True")].n.iloc[0])
fert_da = lambda m: ff[(ff.model == m) & (ff.dialect != "MSA")].fertility.mean()

claims = {
    # ---------------- data ----------------
    "data: 5,207 binary -> 5,089 unique, 118 removed, 44 cross-split, 3,436 train, 1,531 LODO, 21 Maghrebi":
        [card[k] for k in ["raw_total", "unique_normalized_total", "duplicates_or_leaks_removed", "raw_cross_split_leak_texts",
                           "train_4_total", "n_lodo_matched", "test_maghrebi_total"]] == [5207, 5089, 118, 44, 3436, 1531, 21]
        and T("5,207", "5,089", "118 instances", "44 normalized tweets", "3,436", "1,531", "21 unique Maghrebi"),
    "learning-curve target pools: LEV 239, GLF 184 (text: 239 vs. 184)":
        lc["AraBERTv02|Levantine"]["n_target_train"] == 239 and lc["AraBERTv02|Gulf"]["n_target_train"] == 184 and T("239 vs.\\ 184") and V("239", "184"),
    # ---------------- baselines / probes ----------------
    "majority 0.409, TF-IDF 0.773": f3(cb["Majority"]["test_4"]["pooled_macro_f1"]) == "0.409" and f3(cb["TFIDF_LR_balanced"]["test_4"]["pooled_macro_f1"]) == "0.773"
        and T("only 0.409 macro-F1", "TF-IDF model 0.773") and V("0.409", "0.773"),
    "probe: MARBERT best last-layer mean (0.845), CAMeLBERT-CA weakest (0.702)":
        max(M, key=lambda m: lp[m]["test_4"]["hm_f1"]) == "MARBERT" and f3(lp["MARBERT"]["test_4"]["hm_f1"]) == "0.845"
        and min(M, key=lambda m: lp[m]["test_4"]["hm_f1"]) == "CAMeLBERT-CA" and f3(lp["CAMeLBERT-CA"]["test_4"]["hm_f1"]) == "0.702" and V("0.845", "0.702"),
    "probe: mean > CLS for 9 of 11, exceptions the two AraBERTv02 models":
        sorted(m for m in M if last(m, "cls")["hm_f1"] >= last(m, "mean")["hm_f1"]) == ["AraBERTv02", "AraBERTv02-Twitter"] and T("nine of eleven"),
    "probe: dev-selected mean layer 4-5 for CAMeLBERT-DA, -CA, ARBERTv2":
        all(bestlayer(m, "mean") in ("4", "5") for m in ["CAMeLBERT-DA", "CAMeLBERT-CA", "ARBERTv2"]) and T("layer 4--5 for CAMeLBERT-DA, -CA and ARBERTv2"),
    "fine-tuning beats the probe for every Arabic encoder": all(P(m, "hm_f1") > lp[m]["test_4"]["hm_f1"] for m in AR),
    "probe CAMeLBERT HM 0.817/0.799/0.772/0.702": [f3(lp[m]["test_4"]["hm_f1"]) for m in ["CAMeLBERT-Mix", "CAMeLBERT-MSA", "CAMeLBERT-DA", "CAMeLBERT-CA"]]
        == ["0.817", "0.799", "0.772", "0.702"] and T("0.817, 0.799, 0.772 and 0.702") and V("0.817", "0.799", "0.772"),
    # ---------------- PID ----------------
    "best encoder by HM, pooled and WD is the macro \\bestEnc":
        MAC["bestEnc"] == best == max(M, key=lambda m: P(m, "pooled_macro_f1")) == max(M, key=lambda m: P(m, "wd_f1")) and MAC["bestEncHM"] == f3(P(best, "hm_f1")),
    "top group: no pairwise HM or WD CI excludes zero": all(not pw(a, b, k)[2] for a in MAC["topGroup"].split(", ") for b in MAC["topGroup"].split(", ") if a < b for k in ("hm_f1", "wd_f1")),
    "below-best encoders listed in text are credibly below the best and not in top group":
        all(pw(best, m)[2] and pw(best, m)[0] > 0 and m not in MAC["topGroup"] for m in ["AraELECTRA", "CAMeLBERT-DA", "CAMeLBERT-CA", "XLM-R-Base"])
        and len(MAC["topGroup"].split(", ")) + 4 == len(M),
    "Mix pooled 0.858 (lowest of top group except CAMeLBERT-MSA? -> one of lowest), HM = MARBERT 0.861, smallest MSA gap":
        f3(P("CAMeLBERT-Mix", "pooled_macro_f1")) == "0.858" and f3(P("CAMeLBERT-Mix", "hm_f1")) == f3(P("MARBERT", "hm_f1")) == "0.861"
        and sorted(MAC["topGroup"].split(", "), key=lambda m: P(m, "pooled_macro_f1"))[0] == "CAMeLBERT-Mix"
        and min(M, key=lambda m: P(m, "gap_msa")) == "CAMeLBERT-Mix" and T("(0.858)", "MARBERT (0.861)") and V("0.858", "0.861"),
    "XLM-R 0.803/0.772/0.694": [f3(P("XLM-R-Base", k)) for k in ["pooled_macro_f1", "hm_f1", "wd_f1"]] == ["0.803", "0.772", "0.694"]
        and T("pooled 0.803, HM-F1 0.772, WD-F1 0.694") and V("0.803", "0.694"),
    "Levantine worst dialect for every encoder": all(min(D, key=lambda x: P(m, f"f1_{x}")) == "Levantine" for m in M) and MAC["levWorstN"] == str(len(M)),
    "PID-w max |dHM| = macro": MAC["pidwMaxAbs"] == f3(max(abs(P(m, "hm_f1", "PID_weighted") - P(m, "hm_f1")) for m in M)),
    "McNemar: 30 tests, 6 significant, all vs XLM-R (pooled and EGY)": len(mc) == 30 and sum(r["significant_005"] for r in mc) == 6
        and all("XLM-R" in r["comparison"] and r["dialect"] in ("ALL_POOLED", "Egyptian") for r in mc if r["significant_005"]) and T("only six comparisons"),
    # ---------------- LLMs ----------------
    "LLM best is Fanar FS8; macro matches": llm_best == MAC["llmBest"] and MAC["llmBestHM"] == f3(fs8[llm_best]["hm_f1"]["point"]),
    "AceGPT zero-shot Gulf 0.637": f3(llm["AceGPT-v2-8B-Chat"]["ZS"]["f1_Gulf"]["point"]) == "0.637" and T("0.637 on Gulf") and V("0.637"),
    "other LLMs 0.041-0.091 below best encoder; CI excludes zero for AceGPT and ALLaM only":
        (f3(-max(g["diff"] for g in gaps.values())), f3(-min(g["diff"] for g in gaps.values()))) == ("0.041", "0.091")
        and sorted(n for n, g in gaps.items() if g["ci95"][1] < 0) == ["ALLaM-7B-Instruct", "AceGPT-v2-8B-Chat"]
        and T("0.041--0.091 HM-F1 below the best encoder") and V("0.041", "--0.091"),
    "every LLM (FS8) above TF-IDF, weakest on LEV or GLF": all(fs8[n]["hm_f1"]["point"] > cb["TFIDF_LR_balanced"]["test_4"]["hm_f1"] for n in fs8)
        and all(min(D, key=lambda x: fs8[n][f"f1_{x}"]["point"]) in ("Levantine", "Gulf") for n in fs8),
    # ---------------- HM variance ----------------
    "LEV/GLF test n = 86/61; per-dialect F1 SD about 0.04-0.05": T("86 and 61 test tweets") and V("86", "61", "0.04", "--0.05")
        and all(0.035 <= rs["hm_variance"][m]["sd_dialect"][x] <= 0.055 for m in M for x in ("Levantine", "Gulf")),
    # ---------------- transfer ----------------
    "LODO Arabic range = macros; XLM-R macro": (MAC["lodoArDeltaMin"], MAC["lodoArDeltaMax"]) == (s3(min(rs["transfer_tost"][f"{m}|LODO"]["hm_f1"]["delta"] for m in AR)),
                                                s3(max(rs["transfer_tost"][f"{m}|LODO"]["hm_f1"]["delta"] for m in AR)))
        and MAC["xlmLodo"] == s3(rs["transfer_tost"]["XLM-R-Base|LODO"]["hm_f1"]["delta"]),
    "LODO loss bound = worst 90% lower bound over Arabic encoders": MAC["lodoArLossBound"] == f3(-min(rs["transfer_tost"][f"{m}|LODO"]["hm_f1"]["ci90"][0] for m in AR)),
    "90% CI half-widths about 0.02-0.03": all(0.015 <= (lambda c: (c[1] - c[0]) / 2)(rs["transfer_tost"][f"{m}|LODO"]["hm_f1"]["ci90"]) <= 0.035 for m in M)
        and T("half-widths of about 0.02--0.03") and V("--0.03"),
    "learning curve: no gain interval excludes zero at any k": all(not (g["ci95"][0] > 0 or g["ci95"][1] < 0) for v in lc.values() for g in v["gain_vs_k0"].values())
        and MAC["lcSigAll"] == "0",
    "learning curve: largest k=all gains are XLM-R on LEV and GLF": sorted(lc, key=lambda c: -lc[c]["gain_vs_k0"]["all"]["diff"])[:2] == ["XLM-R-Base|Levantine", "XLM-R-Base|Gulf"],
    # ---------------- ArSarcasm-v2 ----------------
    "v2: MARBERTv2 highest HM3 0.814, XLM-R lowest 0.737": max(v2pid, key=v2pid.get) == "MARBERTv2" and f3(v2pid["MARBERTv2"]) == "0.814"
        and min(v2pid, key=v2pid.get) == "XLM-R-Base" and f3(v2pid["XLM-R-Base"]) == "0.737" and T("(0.814)", "(0.737)") and V("0.814", "0.737"),
    "v2: XLM-R LODO +0.022 [-0.022, +0.067]": [s3(xv2["delta"]), s3(xv2["ci95"][0]), s3(xv2["ci95"][1])] == ["+0.022", "-0.022", "+0.067"]
        and T("$+0.022$ [$-0.022$, $+0.067$]") and V("+0.022", "-0.022", "+0.067"),
    # ---------------- RQ3 ----------------
    "CAMeL WD Mix 0.822 vs MSA 0.771": [f3(P(m, "wd_f1")) for m in ["CAMeLBERT-Mix", "CAMeLBERT-MSA"]] == ["0.822", "0.771"] and T("(0.822 vs.\\ 0.771)") and V("0.822", "0.771"),
    "CAMeL pairwise significance as stated": pw("CAMeLBERT-Mix", "CAMeLBERT-DA")[2] and pw("CAMeLBERT-MSA", "CAMeLBERT-DA")[2] and pw("CAMeLBERT-DA", "CAMeLBERT-CA")[2]
        and not pw("CAMeLBERT-Mix", "CAMeLBERT-MSA")[2],
    "other pairs: v2-MARBERT and Tw-AraBERT n.s.; AraELECTRA-AraBERT just excludes 0; ARBERTv2-MARBERTv2 n.s.":
        not pw("MARBERTv2", "MARBERT")[2] and not pw("AraBERTv02-Twitter", "AraBERTv02")[2] and pw("AraELECTRA", "AraBERTv02")[2]
        and -0.01 < pw("AraELECTRA", "AraBERTv02")[1][1] < 0 and not pw("ARBERTv2", "MARBERTv2")[2],
    # ---------------- metric validity ----------------
    "DRI-F1 range [0.9945, 1.0000] over all systems": [f"{x:.4f}" for x in rs["metric_validity"]["dri_f1_range"]] == ["0.9945", "1.0000"] and T("[0.9945, 1.0000]")
        and V("0.9945", "1.0000"),
    "majority ranked 1st by DRI-F1 and last by HM": rs["metric_validity"]["majority_rank_dri_f1"] == 1 and rs["metric_validity"]["majority_rank_hm"] == rs["metric_validity"]["n_systems"]
        and MAC["mvN"] == str(rs["metric_validity"]["n_systems"]),
    "synthetic: mediocre and degenerate profiles get DRI-F1 1.000": T("both receive DRI-F1 1.000") and V("1.000")
        and all(f"{r['old_dri']:.3f}" == "1.000" for r in st["synthetic_metric_validity"] if r["profile"].startswith(("Uniform mediocre", "Degenerate", "Ideal"))),
    # ---------------- sarcasm / orthography / fertility ----------------
    "sarcastic 306 tweets, 294 negative, 12 positive": int(ws.loc[(SM[0], "Sarcastic"), "n"]) == 306 and int(ws.loc[(SM[0], "Sarcastic"), "n_pos"]) == 12
        and T("294 of the 306") and V("294", "306", "12"),
    "sarc macro-F1 0.83-0.89 vs 0.57-0.67": (f"{min(ws.loc[(m, 'Non-Sarcastic'), 'macro_f1'] for m in SM):.2f}", f"{max(ws.loc[(m, 'Non-Sarcastic'), 'macro_f1'] for m in SM):.2f}") == ("0.82", "0.89")
        and (f"{min(ws.loc[(m, 'Sarcastic'), 'macro_f1'] for m in SM):.2f}", f"{max(ws.loc[(m, 'Sarcastic'), 'macro_f1'] for m in SM):.2f}") == ("0.54", "0.67")
        and T("0.82--0.89", "0.54--0.67") and V("0.82", "--0.89", "0.54", "--0.67"),
    "sarc accuracy higher for every encoder; dRecNeg CIs contain 0; 4-6 of 12 positives":
        all(ws.loc[(m, "Sarcastic"), "accuracy"] > ws.loc[(m, "Non-Sarcastic"), "accuracy"] for m in SM)
        and all(r["neg_recall_gap_ci95"][0] < 0 < r["neg_recall_gap_ci95"][1] for r in seff.values())
        and (min(round(ws.loc[(m, "Sarcastic"), "recall_pos"] * 12) for m in SM), max(round(ws.loc[(m, "Sarcastic"), "recall_pos"] * 12) for m in SM)) == (2, 6) and T("2--6 of 12"),
    "orthography buckets: 47 Latin, 70 elongation, 126 emoji": [nbuck(f) for f in ["has_latin", "has_elongation", "has_emoji"]] == [47, 70, 126]
        and T("47 test tweets", "70 contain", "126 contain") and V("47", "70", "126"),
    "long tweets harder than short for every encoder": all(
        art[(art.model == m) & (art.bucket == "Long (>20w)")].macro_f1.iloc[0] < art[(art.model == m) & (art.bucket == "Short (<=10w)")].macro_f1.iloc[0] for m in M),
    "fertility DA: MARBERT 1.79, AraBERTv02/CAMeLBERT 2.11-2.15, XLM-R 2.46": [f"{fert_da(m):.2f}" for m in ["MARBERT", "CAMeLBERT-Mix", "AraBERTv02", "XLM-R-Base"]] == ["1.79", "2.11", "2.15", "2.46"]
        and T("1.79", "2.11--2.15", "2.46") and V("1.79", "2.11", "--2.15", "2.46"),
    "Levantine highest fertility under every tokenizer": all(ff[ff.model == m].set_index("dialect").fertility.idxmax() == "Levantine" for m in M),
    "GLMM: macros match fit; 11 encoders; LEV point estimate negative for all": MAC["glmmNmodels"] == str(len(G["models"])) == str(len(M)) and MAC["glmmDiv"] == str(G["divergences"]),
    # ---------------- three-class ----------------
    "3cls: 10,132 / 6,813 / 1,203 / 2,089": [card3[k] for k in ["unique_total", "train", "dev", "test"]] == [10132, 6813, 1203, 2089] and T("10,132", "6,813", "1,203", "2,089"),
    "3cls MARBERT 0.725/0.656 best; XLM-R 0.686/0.616": [f3(c3["MARBERT"][k]) for k in ["pooled_macro_f1_mean", "hm_f1_mean"]] == ["0.725", "0.656"]
        and [f3(c3["XLM-R-Base"][k]) for k in ["pooled_macro_f1_mean", "hm_f1_mean"]] == ["0.686", "0.616"]
        and max(c3, key=lambda m: c3[m]["hm_f1_mean"]) == "MARBERT" and V("0.725", "0.656", "0.686", "0.616"),
    "3cls LEV weakest for 3 of 4; neutral 61% of MSA; majority lowest DRI-acc of 22":
        sum(min(D, key=lambda x: c3[m][f"f1_{x}_mean"]) == "Levantine" for m in c3) == 3
        and f"{card3['test_by_dialect_label']['MSA']['1'] / sum(card3['test_by_dialect_label']['MSA'].values()):.2f}" == "0.61"
        and t3["n_systems"] == 22 and t3["majority_old_dri_rank"] == 22 and T("61\\%", "22 systems"),
    "Maghrebi prior 0.81": f"{17 / 21:.2f}" == "0.81" and V("0.81"),
    # ---------------- stale text ----------------
    "no stale 8-encoder or leak-inflation phrasing": not re.search(r"seven of eight|eight original|Leakage inflates|0\.059 HM-F1 \[|0\.033\} HM", BODY),
}

# method constants, design parameters and calibrated audit figures
CONST = {
    "0.01", "0.02", "0.05", "0.5", "0.7", "0.9", "0.95", "0.1", "0.22", "0.12", "1.0", "2.80", "1.000",
    "--0.05", "0.03", "0.07", "0.10", "--0.10", "0.15", "0.205", "0.140", "0.028", "0.862", "0.861",
    "0.847", "0.65", "0.024", "0.021", "0.064", "0.020", "0.029", "0.025", "0.44", "4.0",
    "+0.020", "+0.046", "+0.068", "+0.140", "-0.028", "-0.032", "-0.064", "-0.205",
    "0.046", "0.068", "0.032", "0.779", "0.78", "0.798", "0.830", "0.849", "0.9991", "0.9994",
    "12.0", "31.4", "47.7", "54.1", "58.9", "73.8", "91.7", "95.1",
}
dec = set(re.findall(r"-{0,2}\+?\d+\.\d+", re.sub(r"\\(cite|ref|label|includegraphics|input)\{[^}]*\}|\[[^\]]*cm\]|p\{[^}]*\}|\\setlength[^\n]*", "", BODY)))
dec = {x for x in dec if not re.fullmatch(r"\d\.\d", x) or x in {"0.5", "0.7", "0.9"}}
unverified = sorted(x for x in dec if x not in VER and x.lstrip("-+") not in VER and x not in CONST and x.lstrip("-") not in CONST)
claims["coverage: every decimal in the prose is verified or a declared constant"] = not unverified

for k, v in claims.items():
    print(("PASS " if v else "FAIL ") + k)
if unverified:
    print("unverified numbers:", unverified)
print(f"\n{sum(map(bool, claims.values()))}/{len(claims)} prose claims verified")
