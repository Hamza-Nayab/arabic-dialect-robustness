# -*- coding: utf-8 -*-
"""Step 8b: LaTeX tables and figures for the analyses added in the review round (no hand-typed numbers).

Reads review_stats.json (06b), glmm_mcmc.json (06c), data_audit.json (07b), layerwise_probe.json (04b),
model_audit.csv / tokenizer_fertility.csv (03). Writes tables_v2/tab_*.tex, figures_v2/fig5_*.pdf and
tables_v2/macros_review.tex.
"""
from __future__ import annotations

import json
from importlib.machinery import SourceFileLoader
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs_v2"
FIG = ROOT.parent / "figures_v2"
TAB = ROOT.parent / "tables_v2"
s8 = SourceFileLoader("step8", str(ROOT / "08_generate_paper_artifacts.py")).load_module()
DISPLAY, FT_ORDER, MAIN, DSHORT = s8.DISPLAY, s8.FT_ORDER, s8.MAIN_DIALECTS, s8.DSHORT
MODEL_COLOR = s8.MODEL_COLOR
macros = {}


def tab(name, cols, header, rows):
    (TAB / f"{name}.tex").write_text(f"\\begin{{tabular}}{{{cols}}}\n\\toprule\n{header} \\\\\n\\midrule\n"
                                     + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n", encoding="utf-8")


def c2(v, nd=3):
    return f"[{v[0]:+.{nd}f}, {v[1]:+.{nd}f}]"


def mac(k, v):
    macros[k] = v


def main():
    rs = json.load(open(OUT / "review_stats.json"))
    H = rs["hier_ci"]
    models = [m for m in FT_ORDER if f"{m}|PID" in H]

    # ---------------- Transfer: TOST + MDE ----------------
    rows = []
    T = rs["transfer_tost"]
    for m in models:
        cells = []
        for reg in ("LODO", "MSAonly"):
            e = T.get(f"{m}|{reg}")
            if not e:
                cells += ["--"] * 4
                continue
            h = e["hm_f1"]
            eq = "\\cmark" if h["equivalent"] else ""
            cells += [f"{h['delta']:+.3f}", f"{{\\scriptsize{c2(h['ci90'])}}}", eq, f"{h['mde80']:.3f}"]
        lo = T.get(f"{m}|LODO")
        cells += [f"{lo['f1_Gulf']['mde80']:.3f}", f"{lo['f1_Levantine']['mde80']:.3f}"] if lo else ["--", "--"]
        rows.append(f"{DISPLAY[m]} & " + " & ".join(cells) + " \\\\")
    tab("tab_tost", "lcccc|cccc|cc",
        " & \\multicolumn{4}{c|}{\\textbf{LODO $-$ Pooled(1,531)}} & \\multicolumn{4}{c|}{\\textbf{MSA-only $-$ Pooled(1,905)}}"
        " & \\multicolumn{2}{c}{\\textbf{MDE (LODO)}} \\\\\n\\textbf{Encoder} & $\\Delta$HM & 90\\% CI & Eq. & MDE & $\\Delta$HM & "
        "90\\% CI & Eq. & MDE & GLF & LEV", rows)
    lodo = {m: T[f"{m}|LODO"] for m in models if f"{m}|LODO" in T}
    msa = {m: T[f"{m}|MSAonly"] for m in models if f"{m}|MSAonly" in T}
    mac("nLodoModels", str(len(lodo)))
    mac("nLodoEquiv", str(sum(v["hm_f1"]["equivalent"] for v in lodo.values())))
    mac("nLodoSig", str(sum(v["hm_f1"]["ci95_excludes_zero"] for v in lodo.values())))
    mac("lodoMdeMin", f"{min(v['hm_f1']['mde80'] for v in lodo.values()):.3f}")
    mac("lodoMdeMax", f"{max(v['hm_f1']['mde80'] for v in lodo.values()):.3f}")
    mac("lodoMdeGlfMin", f"{min(v['f1_Gulf']['mde80'] for v in lodo.values()):.3f}")
    mac("lodoMdeGlfMax", f"{max(v['f1_Gulf']['mde80'] for v in lodo.values()):.3f}")
    mac("lodoMdeLevMin", f"{min(v['f1_Levantine']['mde80'] for v in lodo.values()):.3f}")
    mac("lodoMdeLevMax", f"{max(v['f1_Levantine']['mde80'] for v in lodo.values()):.3f}")
    mac("lodoDeltaMin", f"{min(v['hm_f1']['delta'] for v in lodo.values()):+.3f}")
    mac("lodoDeltaMax", f"{max(v['hm_f1']['delta'] for v in lodo.values()):+.3f}")
    mac("nMsaSig", str(sum(v["hm_f1"]["ci95_excludes_zero"] for v in msa.values())))
    mac("msaDeltaMin", f"{min(v['hm_f1']['delta'] for v in msa.values()):+.3f}")
    mac("msaDeltaMax", f"{max(v['hm_f1']['delta'] for v in msa.values()):+.3f}")

    # ---------------- HM variance ----------------
    V = rs["hm_variance"]
    rows = [f"{DISPLAY[m]} & {V[m]['sd_pooled']:.3f} & {V[m]['sd_hm']:.3f} & "
            + " & ".join(f"{V[m]['sd_dialect'][g]:.3f}" for g in MAIN) + " & "
            + " & ".join(f"{100 * V[m]['share_of_hm_variance'][g]:.0f}" for g in MAIN) + " \\\\" for m in models]
    tab("tab_hm_variance", "lcc|cccc|cccc", " & & & \\multicolumn{4}{c|}{\\textbf{Bootstrap SD of dialect F1}} & "
        "\\multicolumn{4}{c}{\\textbf{Share of Var(HM) (\\%)}} \\\\\n\\textbf{Encoder (PID)} & SD pooled & SD HM & "
        + " & ".join(DSHORT[g] for g in MAIN) + " & " + " & ".join(DSHORT[g] for g in MAIN), rows)
    mac("hmSdMin", f"{min(V[m]['sd_hm'] for m in models):.3f}")
    mac("hmSdMax", f"{max(V[m]['sd_hm'] for m in models):.3f}")
    mac("pooledSdMax", f"{max(V[m]['sd_pooled'] for m in models):.3f}")
    lev_share = [V[m]["share_of_hm_variance"]["Levantine"] + V[m]["share_of_hm_variance"]["Gulf"] for m in models]
    mac("levGlfShareMin", f"{100 * min(lev_share):.0f}")
    mac("levGlfShareMax", f"{100 * max(lev_share):.0f}")

    # ---------------- Learning curve ----------------
    LC = rs.get("learning_curve", {})
    if LC:
        targets = ["Egyptian", "Levantine", "Gulf"]
        lc_models = [m for m in FT_ORDER if any(f"{m}|{t}" in LC for t in targets)]
        fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5), sharey=False)
        for ax, tg in zip(axes, targets):
            for m in lc_models:
                e = LC.get(f"{m}|{tg}")
                if not e:
                    continue
                n_all = e["n_target_train"]
                ks = sorted([k for k in e["points"] if k.isdigit()], key=int) + ["all"]
                xs = [int(k) if k != "all" else n_all for k in ks]
                ys = [e["points"][k]["f1"] for k in ks]
                lo = [e["points"][k]["f1"] - e["points"][k]["ci95"][0] for k in ks]
                hi = [e["points"][k]["ci95"][1] - e["points"][k]["f1"] for k in ks]
                ax.errorbar(np.array(xs) + 1, ys, yerr=[lo, hi], marker="o", ms=3, capsize=2, lw=1,
                            color=MODEL_COLOR[m], label=DISPLAY[m])
            ax.set_xscale("log")
            ax.set_title(f"{tg} (target)")
            ax.grid(ls=":", alpha=0.6)
        axes[1].set_xlabel(r"Target-dialect training tweets $k$ ($+1$, log)")
        axes[0].set_ylabel("Target-dialect macro-F1")
        axes[0].legend(frameon=False, fontsize=6.5)
        plt.tight_layout(pad=1.2)
        s8.save(fig, "fig5_learning_curve")
        ks_all = ["25", "50", "100", "200", "all"]
        rows = []
        for tg in targets:
            for m in lc_models:
                e = LC.get(f"{m}|{tg}")
                if not e:
                    continue
                g = e["gain_vs_k0"]
                rows.append(f"{DSHORT[tg]} & {DISPLAY[m]} & {e['points']['0']['f1']:.3f} & "
                            + " & ".join((f"{g[k]['diff']:+.3f}" + ("$^*$" if g[k]["ci95"][0] > 0 or g[k]["ci95"][1] < 0 else ""))
                                         if k in g else "--" for k in ks_all)
                            + f" & {e['n_target_train']} \\\\")
        tab("tab_learning_curve", "llcccccc|c", "\\textbf{Target} & \\textbf{Encoder} & $k{=}0$ F1 & "
            + " & ".join(f"$\\Delta k{{=}}{k}$" for k in ks_all) + " & $N_{\\text{all}}$", rows)
        gains_all = {k: g["gain_vs_k0"]["all"]["diff"] for k, g in LC.items() if "all" in g["gain_vs_k0"]}
        sig_all = {k: (g["gain_vs_k0"]["all"]["ci95"][0] > 0) for k, g in LC.items() if "all" in g["gain_vs_k0"]}
        mac("lcCells", str(len(gains_all)))
        mac("lcSigAll", str(sum(sig_all.values())))
        mac("lcGainAllMin", f"{min(gains_all.values()):+.3f}")
        mac("lcGainAllMax", f"{max(gains_all.values()):+.3f}")

    # ---------------- Leaky vs clean ----------------
    LK = rs.get("leaky", {})
    if LK:
        rows = []
        for m in [m for m in FT_ORDER if m in LK]:
            v = LK[m]
            d = v["leaky_minus_clean_on_clean_test_hm_f1"]
            rows.append(f"{DISPLAY[m]} & {v['leaky_on_raw_test']['pooled']:.3f} & {v['leaky_on_raw_test']['hm']:.3f} & "
                        f"{v['leaky_on_clean_test']['pooled']:.3f} & {v['leaky_on_clean_test']['hm']:.3f} & "
                        f"{v['clean_on_clean_test']['pooled']:.3f} & {v['clean_on_clean_test']['hm']:.3f} & "
                        f"{v['inflation_hm']:+.3f} & {d['diff']:+.3f} {{\\scriptsize{c2(d['ci95'])}}} \\\\")
        tab("tab_leaky", "lcc|cc|cc|cc", " & \\multicolumn{2}{c|}{\\textbf{Leaky train, raw test}} & "
            "\\multicolumn{2}{c|}{\\textbf{Leaky train, clean test}} & \\multicolumn{2}{c|}{\\textbf{Clean train, clean test}} & "
            "\\multicolumn{2}{c}{\\textbf{Differences (HM)}} \\\\\n\\textbf{Encoder} & Pooled & HM & Pooled & HM & Pooled & HM & "
            "Reported$-$true & Leaky$-$clean (clean test)", rows)
        mac("leakInflMin", f"{min(v['inflation_hm'] for v in LK.values()):+.3f}")
        mac("leakInflMax", f"{max(v['inflation_hm'] for v in LK.values()):+.3f}")
        mac("leakInflPooledMin", f"{min(v['inflation_pooled'] for v in LK.values()):+.3f}")
        mac("leakInflPooledMax", f"{max(v['inflation_pooled'] for v in LK.values()):+.3f}")

    # ---------------- ArSarcasm-v2 ----------------
    V2 = rs.get("arsarcasm_v2")
    if V2 and V2["table"]:
        Tb, To = V2["table"], V2["tost"]
        d3 = V2["dialects_for_profile"]
        rows = []
        for nm, lab in [("Majority", "Majority"), ("TFIDF_LR_balanced", "TF-IDF+LR (bal.)")]:
            if nm in Tb:
                v = Tb[nm]
                rows.append(f"{lab} & {v['pooled_macro_f1']['point']:.3f} & " + " & ".join(f"{v[f'f1_{g}']['point']:.3f}" for g in d3)
                            + f" & {v['hm_f1']['point']:.3f} & -- & -- & -- \\\\")
        rows.append("\\midrule")
        v2models = [m for m in FT_ORDER if f"{m}|PID" in Tb]
        for m in v2models:
            v = Tb[f"{m}|PID"]
            cells = [f"{v['pooled_macro_f1']['point']:.3f}"] + [f"{v[f'f1_{g}']['point']:.3f}" for g in d3] + \
                    [f"{v['hm_f1']['point']:.3f} {{\\scriptsize[{v['hm_f1']['ci95'][0]:.2f},{v['hm_f1']['ci95'][1]:.2f}]}}",
                     f"{v['f1_Levantine_descriptive']:.2f}"]
            for reg in ("LODO_matched", "MSA_only"):
                e = To.get(f"{m}|{reg}")
                cells.append(f"{e['hm_f1']['delta']:+.3f} {{\\scriptsize{c2(e['hm_f1']['ci95'])}}}" if e else "--")
            rows.append(f"{DISPLAY[m]} & " + " & ".join(cells) + " \\\\")
        tab("tab_v2", "lcccccc|cc", "\\textbf{System} & Pooled & " + " & ".join(DSHORT[g] for g in d3)
            + " & HM$_3$ [95\\% CI] & LEV$^\\ddagger$ & $\\Delta$HM LODO & $\\Delta$HM MSA-only", rows)
        lo2 = [To[f"{m}|LODO_matched"]["hm_f1"] for m in v2models if f"{m}|LODO_matched" in To]
        ms2 = [To[f"{m}|MSA_only"]["hm_f1"] for m in v2models if f"{m}|MSA_only" in To]
        if lo2:
            mac("vLodoDeltaMin", f"{min(e['delta'] for e in lo2):+.3f}")
            mac("vLodoDeltaMax", f"{max(e['delta'] for e in lo2):+.3f}")
            mac("vLodoSig", str(sum(e["ci95_excludes_zero"] for e in lo2)))
            mac("vModels", str(len(lo2)))
        if ms2:
            mac("vMsaDeltaMin", f"{min(e['delta'] for e in ms2):+.3f}")
            mac("vMsaDeltaMax", f"{max(e['delta'] for e in ms2):+.3f}")
            mac("vMsaSig", str(sum(e["ci95_excludes_zero"] for e in ms2)))

    # ---------------- LLMs ----------------
    L = rs.get("llm", {})
    if L:
        rows = []
        for name, v in L.items():
            for tag, lab in [("ZS", "zero-shot"), ("ZScal", "zero-shot, dev threshold"), ("FS8", "8-shot (3 draws)")]:
                if tag not in v:
                    continue
                s = v[tag]
                cells = [f"{s['pooled_macro_f1']['point']:.3f}"] + [f"{s[f'f1_{g}']['point']:.3f}" for g in MAIN] + \
                        [f"{s['hm_f1']['point']:.3f} {{\\scriptsize[{s['hm_f1']['ci95'][0]:.2f},{s['hm_f1']['ci95'][1]:.2f}]}}",
                         f"{s['wd_f1']['point']:.3f}"]
                dd = v.get(f"{tag}_minus_best_encoder_hm")
                cells.append(f"{dd['diff']:+.3f} {{\\scriptsize{c2(dd['ci95'])}}}" if dd else "--")
                rows.append(f"{name} & {lab} & " + " & ".join(cells) + " \\\\")
            rows.append("\\midrule")
        rows = rows[:-1]
        tab("tab_llm", "llccccccc|c", "\\textbf{LLM} & \\textbf{Setting} & Pooled & " + " & ".join(DSHORT[g] for g in MAIN)
            + " & HM-F1 [95\\% CI] & WD & $\\Delta$HM vs best encoder", rows)
        fs = {n: v["FS8"]["hm_f1"]["point"] for n, v in L.items() if "FS8" in v}
        best_llm = max(fs, key=fs.get)
        mac("llmBest", best_llm)
        mac("llmBestHM", f"{fs[best_llm]:.3f}")
        mac("llmBestGap", f"{L[best_llm]['FS8_minus_best_encoder_hm']['diff']:+.3f}")
        mac("llmN", str(len(L)))
        mac("llmFailed", ", ".join(sorted(k.replace(".failed.json", "") for k in rs.get("llm_failed", {}))) or "none")

    # ---------------- Probes: final layer vs best layer, mean vs CLS ----------------
    lp_path = OUT / "layerwise_probe.json"
    if lp_path.exists():
        LP = json.load(open(lp_path))
        rows = []
        gain = []
        for m in [m for m in FT_ORDER if m in LP]:
            e = LP[m]
            lm, lc = e["mean"], e["cls"]
            last = str(max(int(k) for k in lm))
            bm, bc = str(e["mean_best_layer_by_dev"]), str(e["cls_best_layer_by_dev"])
            pid = H[f"{m}|PID"]["hm_f1"]["point"] if f"{m}|PID" in H else float("nan")
            pid_str = "--" if pid != pid else f"{pid:.3f}"
            rows.append(f"{DISPLAY[m]} & {lm[last]['hm_f1']:.3f} & {lc[last]['hm_f1']:.3f} & "
                        f"{bm} & {lm[bm]['hm_f1']:.3f} & {bc} & {lc[bc]['hm_f1']:.3f} & {pid_str} \\\\")
            gain.append(lm[bm]["hm_f1"] - lm[last]["hm_f1"])
        tab("tab_probe_layers", "lcc|cc|cc|c", " & \\multicolumn{2}{c|}{\\textbf{Final layer}} & "
            "\\multicolumn{2}{c|}{\\textbf{Best layer (dev), mean}} & \\multicolumn{2}{c|}{\\textbf{Best layer (dev), CLS}} & \\\\\n"
            "\\textbf{Encoder} & Mean & CLS & Layer & HM & Layer & HM & FT PID HM", rows)
        mac("probeLayerGainMin", f"{min(gain):+.3f}")
        mac("probeLayerGainMax", f"{max(gain):+.3f}")
        s4b = SourceFileLoader("step4b", str(ROOT / "04b_layerwise_probe.py")).load_module()
        mac("probeCgrid", ", ".join(f"{c:g}" for c in s4b.C_GRID))
        mac("probeNenc", str(len(rows)))

    # ---------------- Data audit ----------------
    da_path = OUT / "data_audit.json"
    if da_path.exists():
        DA = json.load(open(da_path, encoding="utf-8"))
        nd, did = DA["near_duplicates"], DA["dialect_id"]
        full = DA["pid_full_test"]
        rows = []
        for m in [m for m in FT_ORDER if m in full]:
            a = nd["pid_rescored_without_ge_0.7"][m]
            b = did["pid_rescored_on_madar6_agreeing_test"][m]
            rows.append(f"{DISPLAY[m]} & {full[m]['hm_f1']:.3f} & {a['hm_f1']:.3f} & {a['hm_f1'] - full[m]['hm_f1']:+.3f} & "
                        f"{b['hm_f1']:.3f} & {b['hm_f1'] - full[m]['hm_f1']:+.3f} \\\\")
        tab("tab_audit_rescore", "lc|cc|cc", " & \\textbf{Full test} & \\multicolumn{2}{c|}{\\textbf{No near-dup. ($J{\\ge}0.7$)}} & "
            "\\multicolumn{2}{c}{\\textbf{DID-agreeing subset}} \\\\\n\\textbf{Encoder (PID)} & HM & HM & $\\Delta$ & HM & $\\Delta$", rows)
        m6, nr = did["did_madar6"], did["did_nadi_region"]
        rows = [f"MADAR-6 (6 classes) & {m6['agreement']:.3f} & {m6['cohen_kappa']:.3f} & "
                + " & ".join(f"{m6['per_gold_dialect_agreement'][g]:.2f}" for g in MAIN) + " \\\\",
                f"NADI$\\rightarrow$region (no MSA) & {nr['agreement']:.3f} & {nr['cohen_kappa']:.3f} & -- & "
                + " & ".join(f"{nr['per_gold_dialect_agreement'][g]:.2f}" for g in MAIN[1:]) + " \\\\"]
        tab("tab_did", "lcc|cccc", "\\textbf{Dialect-ID model} & Agree. & $\\kappa$ & " + " & ".join(DSHORT[g] for g in MAIN), rows)
        mac("ndSeven", str(nd["test_with_neighbour_ge_0.7"]))
        mac("ndFive", str(nd["test_with_neighbour_ge_0.5"]))
        mac("ndNine", str(nd["test_with_neighbour_ge_0.9"]))
        mac("didAgree", f"{m6['agreement']:.2f}")
        mac("didKappa", f"{m6['cohen_kappa']:.2f}")
        mac("nadiAgree", f"{nr['agreement']:.2f}")
        mac("nadiKappa", f"{nr['cohen_kappa']:.2f}")
        mac("didLev", f"{m6['per_gold_dialect_agreement']['Levantine']:.2f}")
        mac("didMSA", f"{m6['per_gold_dialect_agreement']['MSA']:.2f}")
        dlt = [nd["pid_rescored_without_ge_0.7"][m]["hm_f1"] - full[m]["hm_f1"] for m in full]
        mac("ndDeltaMaxAbs", f"{max(abs(x) for x in dlt):.3f}")
        dlt2 = [did["pid_rescored_on_madar6_agreeing_test"][m]["hm_f1"] - full[m]["hm_f1"] for m in full]
        mac("didDeltaMin", f"{min(dlt2):+.3f}")
        mac("didDeltaMax", f"{max(dlt2):+.3f}")
        cnt = did["test_agreeing_counts"]
        mac("didTestCounts", ", ".join(f"{DSHORT[g]} {cnt[g]}" for g in MAIN))

    # ---------------- GLMM (MCMC) ----------------
    gp = OUT / "glmm_mcmc.json"
    if gp.exists():
        G = json.load(open(gp))
        rows = []
        for m in [m for m in FT_ORDER if m in G["dialect_effect_by_model"]]:
            e = G["dialect_effect_by_model"][m]
            rows.append(f"{DISPLAY[m]} & " + " & ".join(
                f"{e[g]['mean']:+.2f} {{\\scriptsize[{e[g]['ci95'][0]:+.2f},{e[g]['ci95'][1]:+.2f}]}}" + ("$^*$" if e[g]["excludes_zero"] else "")
                for g in MAIN[1:]) + " \\\\")
        tt = {t["term"]: t for t in G["terms"]}
        f_ = tt["fertility (per SD)"]
        rows.append("\\midrule")
        rows.append(f"\\multicolumn{{4}}{{l}}{{Fertility (per SD, within encoder$\\times$dialect): {f_['post_mean']:+.3f} "
                    f"[{f_['ci95'][0]:+.3f}, {f_['ci95'][1]:+.3f}];\\ $\\sigma_{{\\text{{tweet}}}}$ = {tt['sigma_tweet']['post_mean']:.2f}}} \\\\")
        rows.append(f"\\multicolumn{{4}}{{l}}{{{G['chains']} chains $\\times$ {G['draws']} draws; max $\\hat R$ = {G['max_r_hat']:.3f}; "
                    f"min bulk ESS = {G['min_ess_bulk']:.0f}; divergences = {G['divergences']}}} \\\\")
        tab("tab_glmm", "lccc", "\\textbf{Encoder (PID)} & " + " & ".join(f"{DSHORT[g]} vs MSA" for g in MAIN[1:]), rows)
        mac("glmmFert", f"{f_['post_mean']:+.3f}")
        mac("glmmFertLo", f"{f_['ci95'][0]:+.3f}")
        mac("glmmFertHi", f"{f_['ci95'][1]:+.3f}")
        mac("glmmRhat", f"{G['max_r_hat']:.3f}")
        mac("glmmEss", f"{G['min_ess_bulk']:.0f}")
        mac("glmmDiv", str(G["divergences"]))
        mac("glmmNmodels", str(len(G["models"])))
        mac("glmmNobs", f"{G['n_obs']:,}")
        for g in MAIN[1:]:
            mac(f"glmmNeg{DSHORT[g]}", str(sum(1 for m in G["dialect_effect_by_model"].values()
                                              if m[g]["excludes_zero"] and m[g]["mean"] < 0)))

    # ---------------- PID headline macros (seed mean + hierarchical CI) ----------------
    best = max(models, key=lambda m: H[f"{m}|PID"]["hm_f1"]["point"])
    mac("bestEnc", DISPLAY[best])
    mac("bestEncHM", f"{H[f'{best}|PID']['hm_f1']['point']:.3f}")
    mac("bestEncHMlo", f"{H[f'{best}|PID']['hm_f1']['ci95'][0]:.3f}")
    mac("bestEncHMhi", f"{H[f'{best}|PID']['hm_f1']['ci95'][1]:.3f}")
    mac("tfidfHM", f"{H['TF-IDF']['hm_f1']['point']:.3f}")
    mac("nEnc", str(len(models)))
    best_pool = max(models, key=lambda m: H[f"{m}|PID"]["pooled_macro_f1"]["point"])
    best_wd = max(models, key=lambda m: H[f"{m}|PID"]["wd_f1"]["point"])
    mac("bestPoolEnc", DISPLAY[best_pool])
    mac("bestPoolF", f"{H[f'{best_pool}|PID']['pooled_macro_f1']['point']:.3f}")
    mac("bestWdEnc", DISPLAY[best_wd])
    mac("bestWdF", f"{H[f'{best_wd}|PID']['wd_f1']['point']:.3f}")
    worst = {m: min(MAIN, key=lambda g: H[f"{m}|PID"][f"f1_{g}"]["point"]) for m in models}
    mac("levWorstN", str(sum(1 for g in worst.values() if g == "Levantine")))
    mac("levWorstExcept", ", ".join(f"{DISPLAY[m]} ({DSHORT[g]})" for m, g in worst.items() if g != "Levantine") or "none")

    # ---------------- Arabic-only transfer summaries ----------------
    arab = [m for m in lodo if m != "XLM-R-Base"]
    mac("lodoArDeltaMin", f"{min(lodo[m]['hm_f1']['delta'] for m in arab):+.3f}")
    mac("lodoArDeltaMax", f"{max(lodo[m]['hm_f1']['delta'] for m in arab):+.3f}")
    mac("lodoArSig", str(sum(lodo[m]["hm_f1"]["ci95_excludes_zero"] for m in arab)))
    mac("lodoArLossBound", f"{max(-lodo[m]['hm_f1']['ci90'][0] for m in arab):.3f}")
    mac("msaArLossBound", f"{max(-msa[m]['hm_f1']['ci90'][0] for m in arab if m in msa):.3f}")
    mac("nArab", str(len(arab)))
    xl = lodo.get("XLM-R-Base")
    if xl:
        mac("xlmLodo", f"{xl['hm_f1']['delta']:+.3f}")
        mac("xlmLodoLo", f"{xl['hm_f1']['ci90'][0]:+.3f}")
        mac("xlmLodoHi", f"{xl['hm_f1']['ci90'][1]:+.3f}")
    glf_loss = sum(1 for m in msa if msa[m]["f1_Gulf"]["delta"] < 0)
    mac("msaGlfLossN", str(glf_loss))

    # ---------------- PID-w ablation ----------------
    rows, dd = [], []
    for m in models:
        if f"{m}|PID_weighted" not in H:
            continue
        a, b = H[f"{m}|PID_weighted"], H[f"{m}|PID"]
        cells = [f"{a[k]['point'] - b[k]['point']:+.3f}" for k in ("pooled_macro_f1", "hm_f1", "wd_f1")]
        dd.append(abs(a["hm_f1"]["point"] - b["hm_f1"]["point"]))
        rows.append(f"{DISPLAY[m]} & {b['hm_f1']['point']:.3f} & {a['hm_f1']['point']:.3f} & " + " & ".join(cells) + " \\\\")
    tab("tab_pidw", "lcc|ccc", "\\textbf{Encoder} & HM (PID) & HM (PID-w) & $\\Delta$Pooled & $\\Delta$HM & $\\Delta$WD", rows)
    mac("pidwMaxAbs", f"{max(dd):.3f}")

    # ---------------- pairwise PID differences used in the text ----------------
    PW = {(p["a"], p["b"], p["metric"]): p for p in rs["pairwise_pid"]}

    def pdiff(a, b, metric="hm_f1"):
        k = (f"PID:{a}", f"PID:{b}", metric)
        if k in PW:
            p = PW[k]
            return p["diff"], p["ci95"], p["excludes_zero"]
        k = (f"PID:{b}", f"PID:{a}", metric)
        if k in PW:
            p = PW[k]
            return -p["diff"], [-p["ci95"][1], -p["ci95"][0]], p["excludes_zero"]
        return None

    clean = lambda s: "".join(ch for ch in s if ch.isalpha())
    pairs = [("CAMeLBERT-Mix", "CAMeLBERT-DA"), ("CAMeLBERT-MSA", "CAMeLBERT-DA"), ("CAMeLBERT-Mix", "CAMeLBERT-MSA"),
             ("CAMeLBERT-DA", "CAMeLBERT-CA"), ("CAMeLBERT-MSA", "CAMeLBERT-CA"), ("CAMeLBERT-Mix", "CAMeLBERT-CA"),
             ("MARBERTv2", "MARBERT"), ("AraBERTv02-Twitter", "AraBERTv02"), ("AraELECTRA", "AraBERTv02"),
             ("ARBERTv2", "MARBERTv2"), ("ARBERTv2", "MARBERT")]
    for a, b in pairs:
        for metric, tag in (("hm_f1", "HM"), ("wd_f1", "WD")):
            r = pdiff(a, b, metric)
            if r:
                mac(f"pd{clean(a)}{clean(b)}{tag}", f"{r[0]:+.3f} [{r[1][0]:+.3f}, {r[1][1]:+.3f}]")
    # top group: encoders whose HM-F1 is not credibly below the best
    top = [m for m in models if m == best or not (pdiff(best, m) or (0, [0, 0], False))[2]]
    mac("topGroup", ", ".join(DISPLAY[m] for m in top))
    mac("topGroupN", str(len(top)))
    for m in models:
        mac(f"pidHM{clean(m)}", f"{H[f'{m}|PID']['hm_f1']['point']:.3f}")
        mac(f"pidF{clean(m)}", f"{H[f'{m}|PID']['pooled_macro_f1']['point']:.3f}")
        mac(f"pidWD{clean(m)}", f"{H[f'{m}|PID']['wd_f1']['point']:.3f}")
        mac(f"pidGap{clean(m)}", f"{H[f'{m}|PID']['gap_msa']['point']:.3f}")
    tf = H["TF-IDF"]["hm_f1"]["point"]
    above_tf = [m for m in models if not pdiff(m, "TF-IDF") or True]
    tfd = []
    for m in models:
        k = (f"PID:{m}", "TF-IDF", "hm_f1")
        if k in PW:
            tfd.append((m, PW[k]["diff"], PW[k]["excludes_zero"]))
    mac("nAboveTfidf", str(sum(1 for _, d_, e_ in tfd if e_ and d_ > 0)))
    mac("notAboveTfidf", ", ".join(DISPLAY[m] for m, d_, e_ in tfd if not (e_ and d_ > 0)) or "none")

    # ---------------- metric validity ----------------
    mv = rs["metric_validity"]
    mac("mvN", str(mv["n_systems"]))
    mac("mvRankAcc", str(mv["majority_rank_dri_acc"]))
    mac("mvRankFone", str(mv["majority_rank_dri_f1"]))
    mac("mvTauAcc", f"{mv['tau_hm_dri_acc']:.2f}")
    mac("mvTauFone", f"{mv['tau_hm_dri_f1']:.2f}")
    mac("mvTauWd", f"{mv['tau_hm_wd']:.2f}")
    mac("mvTauPooled", f"{mv['tau_hm_pooled']:.2f}")
    mac("mvDriFoneMin", f"{mv['dri_f1_range'][0]:.4f}")

    # placeholders for analyses whose inputs are not available yet (so the paper still compiles)
    for k in ["lcCells", "lcSigAll", "lcGainAllMin", "lcGainAllMax", "leakInflMin", "leakInflMax", "leakInflPooledMin",
              "leakInflPooledMax", "vLodoDeltaMin", "vLodoDeltaMax", "vLodoSig", "vModels", "vMsaDeltaMin",
              "vMsaDeltaMax", "vMsaSig"]:
        macros.setdefault(k, "\\TBD{??}")
    (TAB / "macros_review.tex").write_text("\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in macros.items()) + "\n",
                                          encoding="utf-8")
    print(f"[STEP 8b PASSED] {len(macros)} macros")


if __name__ == "__main__":
    main()
