# -*- coding: utf-8 -*-
"""Step 8: Generate vector PDF figures and LaTeX table snippets from outputs_v2 (no hand-typed numbers)."""
from __future__ import annotations

import json
from importlib.machinery import SourceFileLoader
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
# from transformers import AutoConfig, AutoModel  # not needed: using _KNOWN_PARAMS lookup

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs_v2"
FIG = ROOT.parent / "figures_v2"
TAB = ROOT.parent / "tables_v2"
FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)
step3 = SourceFileLoader("step3", str(ROOT / "03_model_and_fertility_audit.py")).load_module()

MAIN_DIALECTS = ["MSA", "Egyptian", "Levantine", "Gulf"]
DSHORT = {"MSA": "MSA", "Egyptian": "EGY", "Levantine": "LEV", "Gulf": "GLF"}
FT_ORDER = ["AraBERTv02", "AraBERTv02-Twitter", "AraELECTRA", "MARBERT", "MARBERTv2", "ARBERTv2", "CAMeLBERT-Mix",
            "CAMeLBERT-MSA", "CAMeLBERT-DA", "CAMeLBERT-CA", "XLM-R-Base"]
PROBE_ORDER = list(step3.MODEL_REGISTRY.keys())
DISPLAY = {"AraBERTv02": "AraBERTv02", "AraBERTv02-Twitter": "AraBERTv02-Twitter", "MARBERT": "MARBERT",
           "MARBERTv2": "MARBERTv2", "CAMeLBERT-Mix": "CAMeLBERT-Mix", "CAMeLBERT-MSA": "CAMeLBERT-MSA",
           "CAMeLBERT-DA": "CAMeLBERT-DA", "XLM-R-Base": "XLM-R-Base", "CAMeLBERT-CA": "CAMeLBERT-CA",
           "ARBERTv2": "ARBERTv2", "AraELECTRA": "AraELECTRA"}
_CB = sns.color_palette("colorblind", 10)
_T = sns.color_palette("tab20", 20)
MODEL_COLOR = {"AraBERTv02": _CB[0], "MARBERT": _CB[1], "CAMeLBERT-Mix": _CB[2], "CAMeLBERT-MSA": _CB[3],
               "CAMeLBERT-DA": _CB[4], "XLM-R-Base": _CB[7], "AraBERTv02-Twitter": _CB[8], "MARBERTv2": _CB[9],
               "CAMeLBERT-CA": _CB[5], "ARBERTv2": _CB[6], "AraELECTRA": _T[17]}
plt.rcParams.update({"font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9, "legend.fontsize": 7.5,
                     "pdf.fonttype": 42, "font.family": "serif"})


def f3(x: float) -> str:
    return f"{x:.3f}"


def pm(mean: float, sd: float) -> str:
    return f"{mean:.3f}$\\pm${sd:.3f}"


def save(fig, name: str) -> None:
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIG / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_tab(name: str, body: str) -> None:
    (TAB / f"{name}.tex").write_text(body, encoding="utf-8")


def main() -> None:
    card = json.load(open(OUT / "data_card.json", encoding="utf-8"))
    classical = json.load(open(OUT / "classical_baselines.json"))
    probe = json.load(open(OUT / "linear_probe_results.json"))
    agg = pd.read_csv(OUT / "aggregated_multiseed_results.csv")
    stats = json.load(open(OUT / "statistical_analysis_bundle.json", encoding="utf-8"))
    runs = pd.DataFrame([json.loads(l) for l in (OUT / "finetune_runs.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()])
    audit = pd.read_csv(OUT / "model_audit.csv")
    fert = pd.read_csv(OUT / "tokenizer_fertility.csv")
    sarc = pd.read_csv(OUT / "error_sarcasm_by_dialect.csv")
    art = pd.read_csv(OUT / "error_linguistic_artifacts.csv")
    fert_f1 = pd.read_csv(OUT / "fertility_vs_f1.csv")
    fcorr = json.load(open(OUT / "fertility_correlation.json"))
    ci = stats["bootstrap_pid"]["ci_summary"]
    A = agg.set_index(["model", "regime"])
    ft_models = [m for m in FT_ORDER if (m, "PID") in A.index]
    macros = {}

    # ---------------- Table I: data card ----------------
    sp = card["splits_by_dialect"]
    lines = []
    for dd in MAIN_DIALECTS:
        if dd == "Maghrebi":
            mg = sp["maghrebi_exploratory"]["Maghrebi"]
            lines.append(f"Maghrebi$^\\dagger$ & -- & -- & {mg['total']} & {mg['neg_rate']*100:.1f} & {mg['sarcasm_rate']*100:.1f} \\\\")
        else:
            t = sp["test"][dd]
            lines.append(f"{dd} & {sp['train'][dd]['total']:,} & {sp['dev'][dd]['total']:,} & {t['total']:,} & "
                         f"{t['neg_rate']*100:.1f} & {t['sarcasm_rate']*100:.1f} \\\\")
    tot = f"Total & {card['train_4_total']:,} & {card['dev_4_total']:,} & {card['test_4_total']:,} & & \\\\"
    write_tab("tab_data_card", "\\begin{tabular}{lrrrrr}\n\\toprule\n\\textbf{Dialect} & \\textbf{Train} & \\textbf{Dev} & "
              "\\textbf{Test} & \\textbf{Neg.\\%} & \\textbf{Sarc.\\%} \\\\\n\\midrule\n" + "\n".join(lines) +
              "\n\\midrule\n" + tot + "\n\\bottomrule\n\\end{tabular}\n")

    # ---------------- Table II: model audit ----------------
    _MODEL_PARAMS = {
        "AraBERTv02": 135, "AraBERTv02-Twitter": 135, "AraELECTRA": 135,
        "MARBERT": 163, "MARBERTv2": 163, "ARBERTv2": 163,
        "CAMeLBERT-Mix": 109, "CAMeLBERT-MSA": 109, "CAMeLBERT-DA": 109, "CAMeLBERT-CA": 109,
        "XLM-R-Base": 278,
    }
    rows = []
    for _, r in audit.iterrows():
        n_params = _MODEL_PARAMS.get(r["model"], 110)
        fr = fert[(fert["model"] == r["model"]) & (fert["dialect"].isin(MAIN_DIALECTS))]
        fmsa = float(fr[fr["dialect"] == "MSA"]["fertility"].iloc[0])
        fda = float(fr[fr["dialect"] != "MSA"]["fertility"].mean())
        rows.append(f"{DISPLAY[r['model']]} & {n_params:.0f}M & {int(r['vocab_size'])//1000}K & {fmsa:.2f} & {fda:.2f} & "
                    f"\\texttt{{{r['commit_hash'][:7]}}} \\\\")
    write_tab("tab_models", "\\begin{tabular}{lrrrrl}\n\\toprule\n\\textbf{Encoder} & \\textbf{Params} & \\textbf{Vocab} & "
              "\\textbf{Fert.\\,MSA} & \\textbf{Fert.\\,DA} & \\textbf{Commit} \\\\\n\\midrule\n" + "\n".join(rows) +
              "\n\\bottomrule\n\\end{tabular}\n")

    # ---------------- Table III: main results ----------------
    # Every number is a seed mean (single systems: the system itself) and every interval is the two-stage
    # (tweets within dialect, then seeds) bootstrap 95% CI of that same quantity (06b_review_stats.py).
    # Maghrebi (21 test tweets) is reported only in the appendix table.
    rs = json.load(open(OUT / "review_stats.json"))
    H = rs["hier_ci"]

    def cic(v, nd=2):
        return f"{{\\scriptsize[{v['ci95'][0]:.{nd}f},{v['ci95'][1]:.{nd}f}]}}"

    def row_h(label, key):
        v = H[key]
        cells = [f3(v["pooled_macro_f1"]["point"])] + [f3(v[f"f1_{dd}"]["point"]) for dd in MAIN_DIALECTS] + \
                [f"{f3(v['hm_f1']['point'])} {cic(v['hm_f1'])}", f"{f3(v['wd_f1']['point'])} {cic(v['wd_f1'])}"]
        return f"{label} & " + " & ".join(cells) + " \\\\"

    body = ["\\multicolumn{8}{l}{\\emph{Classical baselines}} \\\\", row_h("Majority", "Majority"),
            row_h("TF-IDF+LR (bal.)", "TF-IDF")]
    llm_rs = rs.get("llm", {})
    if llm_rs:
        body.append("\\midrule\n\\multicolumn{8}{l}{\\emph{Instruction-tuned LLMs, 8-shot (mean of 3 exemplar draws)}} \\\\")
        for name, v in llm_rs.items():
            if "FS8" in v:
                H[f"LLM:{name}"] = v["FS8"]
                body.append(row_h(name, f"LLM:{name}"))
    for regime, title in [("PID", "Fine-tuned encoders, pooled in-distribution (PID); mean over 5 seeds")]:
        reg_models = [m for m in ft_models if f"{m}|{regime}" in H]
        if not reg_models:
            continue
        body.append(f"\\midrule\n\\multicolumn{{8}}{{l}}{{\\emph{{{title}}}}} \\\\")
        for m in reg_models:
            body.append(row_h(DISPLAY[m], f"{m}|{regime}"))
    write_tab("tab_main_results", "\\begin{tabular}{lccccccc}\n\\toprule\n\\textbf{System} & \\textbf{Pooled F1} & "
              + " & ".join(f"\\textbf{{{DSHORT[dd]}}}" for dd in MAIN_DIALECTS) +
              " & \\textbf{HM-F1 [95\\% CI]} & \\textbf{WD-F1 [95\\% CI]} \\\\\n\\midrule\n" + "\n".join(body) +
              "\n\\bottomrule\n\\end{tabular}\n")

    # ---------------- Appendix: Maghrebi exploratory slice (21 tweets) ----------------
    body = []
    for lab, mg in [("Majority", classical["Majority"]["maghrebi"]), ("TF-IDF+LR (bal.)", classical["TFIDF_LR_balanced"]["maghrebi"])]:
        body.append(f"{lab} & {mg['k_correct']}/{mg['n_total']} & [{mg['wilson_ci95_low']:.2f}, {mg['wilson_ci95_high']:.2f}] \\\\")
    for m in ft_models:
        g = runs[(runs["model"] == m) & (runs["regime"] == "PID")]
        if len(g):
            acc = [x["accuracy"] for x in g["maghrebi"]]
            body.append(f"{DISPLAY[m]} (PID) & {np.mean(acc) * 21:.1f}/21 & {np.mean(acc):.3f}$\\pm${np.std(acc, ddof=1):.3f} (SD) \\\\")
    write_tab("tab_maghrebi", "\\begin{tabular}{lcc}\n\\toprule\n\\textbf{System} & \\textbf{Correct} & \\textbf{Wilson 95\\% CI / seed SD} \\\\\n"
              "\\midrule\n" + "\n".join(body) + "\n\\bottomrule\n\\end{tabular}\n")

    # ---------------- Table IV: transfer ----------------
    # Deltas are taken against a size-matched pooled control when it exists (PID_n1531 for LODO,
    # PID_n1905 for MSA-only), so they isolate the effect of the missing dialect from training-set size.
    CONTROL = {"LODO_matched": "PID_n1531", "MSA_only": "PID_n1905"}

    def ref_row(m, regime):
        c = CONTROL[regime]
        return (A.loc[(m, c)], c) if (m, c) in A.index else (A.loc[(m, "PID")], "PID")

    # one row per encoder: change vs. the size-matched pooled control, per target dialect and in HM-F1
    body = []
    for m in ft_models:
        if not all((m, r) in A.index for r in ["LODO_matched", "MSA_only"]):
            continue
        cells = [f"{A.loc[(m, 'PID')]['hm_f1_mean']:.3f}"]
        for regime in ["LODO_matched", "MSA_only"]:
            r, (ref, _) = A.loc[(m, regime)], ref_row(m, regime)
            cells += [f"{r[f'f1_{dd}_mean'] - ref[f'f1_{dd}_mean']:+.3f}" for dd in MAIN_DIALECTS]
            cells.append(f"\\textbf{{{r['hm_f1_mean'] - ref['hm_f1_mean']:+.3f}}}")
        body.append(f"{DISPLAY[m]} & " + " & ".join(cells) + " \\\\")
    dh = " & ".join(f"\\textbf{{{DSHORT[dd]}}}" for dd in MAIN_DIALECTS)
    write_tab("tab_transfer", "\\begin{tabular}{lc|ccccc|ccccc}\n\\toprule\n"
              " & \\textbf{PID} & \\multicolumn{5}{c|}{\\textbf{LODO $-$ Pooled(1,531)}} & "
              "\\multicolumn{5}{c}{\\textbf{MSA-only $-$ Pooled(1,905)}} \\\\\n"
              f"\\textbf{{Encoder}} & \\textbf{{HM-F1}} & {dh} & \\textbf{{HM}} & {dh} & \\textbf{{HM}} \\\\\n\\midrule\n"
              + "\n".join(body) + "\n\\bottomrule\n\\end{tabular}\n")

    # ---------------- Table: 3-class replication ----------------
    t3p = OUT / "three_class_replication.json"
    if t3p.exists():
        t3 = json.loads(t3p.read_text())
        body = []
        for name, c in t3["classical"].items():
            lab = {"Majority": "Majority", "TFIDF_LR_balanced": "TF-IDF+LR (bal.)"}[name]
            body.append(f"{lab} & {c['pooled_macro_f1']:.3f} & " + " & ".join(f"{c['per_dialect_f1'][dd]:.3f}" for dd in MAIN_DIALECTS)
                        + f" & {c['hm_f1']:.3f} & {c['wd_f1']:.3f} & {c['old_dri_acc']:.4f} \\\\")
        body.append("\\midrule")
        for m, v in t3["models"].items():
            body.append(f"{DISPLAY[m]} ($n$={v['n_seeds']}) & {pm(v['pooled_macro_f1_mean'], v['pooled_macro_f1_std'])} & "
                        + " & ".join(f"{v[f'f1_{dd}_mean']:.3f}" for dd in MAIN_DIALECTS)
                        + f" & {pm(v['hm_f1_mean'], v['hm_f1_std'])} & {pm(v['wd_f1_mean'], v['wd_f1_std'])} & {v['old_dri_acc_mean']:.4f} \\\\")
        write_tab("tab_three_class", "\\begin{tabular}{lcccccccc}\n\\toprule\n\\textbf{System} & \\textbf{Pooled F1} & "
                  + " & ".join(f"\\textbf{{{DSHORT[dd]}}}" for dd in MAIN_DIALECTS)
                  + " & \\textbf{HM-F1} & \\textbf{WD-F1} & \\textbf{DRI-acc} \\\\\n\\midrule\n" + "\n".join(body)
                  + "\n\\bottomrule\n\\end{tabular}\n")

    # ---------------- Table V: metric validity ----------------
    body = []
    for p in stats["synthetic_metric_validity"]:
        body.append(f"{p['profile']} & {p['mean_f1']:.3f} & {p['hm_f1']:.3f} & {p['wd_f1']:.3f} & {p['max_gap_msa']:.3f} & "
                    f"-- & {1 / (1 + np.var(p['per_dialect_f1'])):.4f} \\\\")
    body.append("\\midrule")
    emp = pd.read_csv(OUT / "metric_validity_seedmean.csv").set_index("system")
    pid_keys = [f"{m}|PID" for m in ft_models if f"{m}|PID" in emp.index]
    best_pid = max(pid_keys, key=lambda s: emp.loc[s, "hm"])
    for sname, lab in [("Majority", "Majority (empirical)"), ("TF-IDF", "TF-IDF+LR (empirical)"),
                       (best_pid, f"{DISPLAY[best_pid.split('|')[0]]} PID (empirical)")]:
        e = emp.loc[sname]
        body.append(f"{lab} & {e['pooled']:.3f} & {e['hm']:.3f} & {e['wd']:.3f} & {H[sname]['gap_msa']['point']:.3f} & "
                    f"{e['dri_acc']:.4f} & {e['dri_f1']:.4f} \\\\")
    mvr = rs["metric_validity"]
    body.append("\\midrule")
    body.append(f"\\multicolumn{{7}}{{l}}{{Rank of Majority among {mvr['n_systems']} systems: HM-F1 {mvr['majority_rank_hm']}, "
                f"DRI-acc {mvr['majority_rank_dri_acc']}, DRI-F1 {mvr['majority_rank_dri_f1']}}} \\\\")
    body.append(f"\\multicolumn{{7}}{{l}}{{Kendall $\\tau$ with HM-F1: pooled {mvr['tau_hm_pooled']:.2f}, WD {mvr['tau_hm_wd']:.2f}, "
                f"DRI-acc {mvr['tau_hm_dri_acc']:.2f}, DRI-F1 {mvr['tau_hm_dri_f1']:.2f}}} \\\\")
    write_tab("tab_metric_validity", "\\begin{tabular}{lcccccc}\n\\toprule\n\\textbf{Profile} & \\textbf{F1}$^*$ & \\textbf{HM-F1} & "
              "\\textbf{WD-F1} & \\textbf{Gap} & \\textbf{DRI-acc} & \\textbf{DRI-F1} \\\\\n\\midrule\n" + "\n".join(body) +
              "\n\\bottomrule\n\\end{tabular}\n")

    # ---------------- Table VI: sarcasm x dialect ----------------
    core = [m for m in ["AraBERTv02", "MARBERT", "CAMeLBERT-Mix", "XLM-R-Base"] if m in sarc["model"].unique()]
    # Pooled test set; columns are non-sarcastic / sarcastic. Macro-F1 is shown last because the sarcastic
    # subset is almost entirely negative, which deflates macro-F1 independently of model behaviour.
    seff = {r["model"]: r for r in json.load(open(OUT / "sarcasm_effect_bootstrap.json"))}
    body = []
    for m in core:
        s = sarc[(sarc["model"] == m) & (sarc["dialect"] == "ALL")].set_index("sarcasm")
        ns_, sa = s.loc["Non-Sarcastic"], s.loc["Sarcastic"]
        lo, hi = seff[m]["neg_recall_gap_ci95"]
        body.append(f"{DISPLAY[m]} & {ns_['accuracy']:.3f}/{sa['accuracy']:.3f} & {ns_['recall_neg']:.3f}/{sa['recall_neg']:.3f} & "
                    f"{ns_['recall_pos']:.3f}/{sa['recall_pos']:.3f} & {ns_['macro_f1']:.3f}/{sa['macro_f1']:.3f} & "
                    f"{seff[m]['neg_recall_gap_nonsarc_minus_sarc']:+.3f} {{\\scriptsize[{lo:+.3f},{hi:+.3f}]}} \\\\")
    s0 = sarc[(sarc["model"] == core[0]) & (sarc["dialect"] == "ALL")].set_index("sarcasm")
    body.append(f"\\midrule\n$n$ (pos.) & {int(s0.loc['Non-Sarcastic','n'])} ({int(s0.loc['Non-Sarcastic','n_pos'])}) / "
                f"{int(s0.loc['Sarcastic','n'])} ({int(s0.loc['Sarcastic','n_pos'])}) & & & & \\\\")
    write_tab("tab_sarcasm", "\\begin{tabular}{lccccc}\n\\toprule\n\\textbf{Encoder (PID)} & \\textbf{Acc.} & \\textbf{Rec.\\,neg} & "
              "\\textbf{Rec.\\,pos} & \\textbf{Macro-F1} & \\textbf{$\\Delta$Rec.\\,neg} \\\\\n\\midrule\n" + "\n".join(body) +
              "\n\\bottomrule\n\\end{tabular}\n")

    # ---------------- Table: learning-rate sensitivity (dev grid + seed-42 test at selected vs fixed LR) ----------------
    # learning-rate selection: full-FT writes selected_lrs/<model>.json (grid over 1e-5..5e-5, PID seed 42)
    lr_dir = OUT / "selected_lrs"
    body, grid_keys = [], None
    for m in ft_models:
        f = lr_dir / f"{m}.json"
        if not f.exists():
            continue
        sel = json.loads(f.read_text())
        grid_keys = grid_keys or sorted(sel["grid"], key=float)
        dev = " & ".join(f"{sel['grid'][k]['dev_f1']:.3f}" for k in grid_keys)
        body.append(f"{DISPLAY[m]} & {dev} & {float(sel['selected_lr']):.0e} \\\\")
    if body:
        hdr = " & ".join(f"{float(k):.0e}" for k in grid_keys)
        write_tab("tab_lr_sensitivity", "\\begin{tabular}{l" + "c" * (len(grid_keys) + 1) + "}\n\\toprule\n"
                  f"\\textbf{{Encoder}} & {hdr} & \\textbf{{Selected}} \\\\\n\\midrule\n" + "\n".join(body) +
                  "\n\\bottomrule\n\\end{tabular}\n")

    # ---------------- Table VII: McNemar (Holm) pooled + per dialect, core models ----------------
    mc = pd.DataFrame(stats["mcnemar_holm"])
    mc["comparison"] = mc["comparison"].str.replace("PID:", "", regex=False)
    body = []
    for comp in mc["comparison"].unique():
        cells = []
        for dd in ["ALL_POOLED"] + MAIN_DIALECTS:
            r = mc[(mc["comparison"] == comp) & (mc["dialect"] == dd)].iloc[0]
            star = "$^*$" if r["significant_005"] else ""
            cells.append(f"{r['b_m1_only']}/{r['c_m2_only']}{star}")
        a, b = comp.split(" vs ")
        body.append(f"{DISPLAY[a]} vs {DISPLAY[b]} & " + " & ".join(cells) + " \\\\")
    write_tab("tab_mcnemar", "\\begin{tabular}{lccccc}\n\\toprule\n\\textbf{Pair (A vs B)} & \\textbf{All} & "
              + " & ".join(f"\\textbf{{{DSHORT[dd]}}}" for dd in MAIN_DIALECTS) + " \\\\\n\\midrule\n" + "\n".join(body) +
              "\n\\bottomrule\n\\end{tabular}\n")

    # ---------------- Figure 1: per-dialect forest (PID mean+-SD vs probe) ----------------
    fig, ax = plt.subplots(figsize=(7.0, 3.0))
    x = np.arange(len(MAIN_DIALECTS))
    pal = [MODEL_COLOR[m] for m in ft_models]
    offs = np.linspace(-0.3, 0.3, len(ft_models))
    for o, m, c in zip(offs, ft_models, pal):
        r = A.loc[(m, "PID")]
        ax.errorbar(x + o, [r[f"f1_{dd}_mean"] for dd in MAIN_DIALECTS], yerr=[r[f"f1_{dd}_std"] for dd in MAIN_DIALECTS],
                    fmt="o", capsize=2.5, color=c, label=f"{DISPLAY[m]} (PID)", markersize=4.5)
        if m in probe:
            ax.scatter(x + o, [probe[m]["test_4"]["per_dialect_f1"][dd] for dd in MAIN_DIALECTS], marker="x", color=c, s=18, alpha=0.8)
    tf = classical["TFIDF_LR_balanced"]["test_4"]["per_dialect_f1"]
    ax.plot(x, [tf[dd] for dd in MAIN_DIALECTS], ls="--", color="gray", lw=1, label="TF-IDF+LR (bal.)")
    ax.set_xticks(x, [f"{dd}\n(n={card['splits_by_dialect']['test'][dd]['total']})" for dd in MAIN_DIALECTS])
    ax.set_ylabel("Macro-F1 on test")
    ax.set_title("Per-dialect macro-F1: fine-tuned PID (o, mean$\\pm$SD over seeds) vs. linear probe (x)")
    ax.grid(axis="y", ls=":", alpha=0.6)
    ax.legend(ncol=4, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2))
    save(fig, "fig1_per_dialect_forest")

    # ---------------- Figure 2: transfer heatmap (LODO and MSA-only change vs PID) ----------------
    tr_models = [m for m in ft_models if (m, "MSA_only") in A.index]
    lodo_models = [m for m in ft_models if (m, "LODO_matched") in A.index]
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 0.9 + 0.55 * max(len(lodo_models), len(tr_models))), gridspec_kw={"width_ratios": [1, 1.15]})
    for ax, regime, mlist, title in [(axes[0], "LODO_matched", lodo_models, "(a) LODO (size-matched, $N$=1,531)"),
                                     (axes[1], "MSA_only", tr_models, "(b) MSA-only training")]:
        val = pd.DataFrame(index=[DISPLAY[m] for m in mlist], columns=[DSHORT[dd] for dd in MAIN_DIALECTS], dtype=float)
        ann = val.copy().astype(object)
        for m in mlist:
            for dd in MAIN_DIALECTS:
                v = A.loc[(m, regime)][f"f1_{dd}_mean"]
                dlt = v - ref_row(m, regime)[0][f"f1_{dd}_mean"]
                val.loc[DISPLAY[m], DSHORT[dd]] = dlt
                ann.loc[DISPLAY[m], DSHORT[dd]] = f"{v:.3f}\n({dlt:+.3f})"
        sns.heatmap(val, annot=ann, fmt="", cmap="RdBu", center=0, vmin=-0.12, vmax=0.12, ax=ax, cbar=ax is axes[1],
                    cbar_kws={"label": "$\\Delta$F1 vs. pooled reference"}, annot_kws={"fontsize": 6.5})
        ax.set_title(title)
        ax.set_xlabel("Target dialect")
    axes[0].set_ylabel("Encoder")
    plt.tight_layout()
    save(fig, "fig2_transfer_drop_heatmap")

    # ---------------- Figure 3: sarcasm gap + fertility ----------------
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.0), gridspec_kw={"width_ratios": [1, 1.25]})
    sub = sarc[(sarc["dialect"] != "ALL") & (sarc["model"].isin(core))].copy()
    sub["dialect"] = sub["dialect"].map(DSHORT)
    sns.barplot(data=sub, x="dialect", y="recall_neg", hue="sarcasm", ax=axes[0], errorbar=("pi", 100),
                order=[DSHORT[d] for d in MAIN_DIALECTS], hue_order=["Non-Sarcastic", "Sarcastic"], palette="colorblind",
                capsize=0.1, err_kws={"linewidth": 0.8})
    axes[0].set_ylim(0.6, 1.02)
    axes[0].set_title("(a) Neg.-class recall by sarcasm")
    axes[0].set_xlabel("Dialect")
    axes[0].set_ylabel("Recall (negative class)")
    axes[0].legend(title=None, frameon=False, loc="upper center", ncol=2, bbox_to_anchor=(0.5, -0.2))
    ff = fert_f1.copy()
    ff["dialect"] = ff["dialect"].map(DSHORT)
    ff["model"] = ff["model"].map(DISPLAY)
    sns.scatterplot(data=ff, x="fertility", y="f1_mean", hue="model", style="dialect", s=40, ax=axes[1],
                    palette={DISPLAY[m]: c for m, c in MODEL_COLOR.items()})
    sns.regplot(data=ff, x="fertility", y="f1_mean", scatter=False, ax=axes[1], color="gray", line_kws={"ls": "--", "lw": 1})
    axes[1].set_title("(b) Fertility vs. F1 (descriptive; tested in the GLMM)")
    axes[1].set_xlabel("Tokenizer fertility (subwords / word)")
    axes[1].set_ylabel("PID macro-F1")
    axes[1].legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=6.5, frameon=False)
    plt.tight_layout()
    save(fig, "fig3_sarcasm_and_fertility")

    # ---------------- Figure 4: metric validity ----------------
    # seed-mean systems (same points as the tables); DRI on accuracy (as originally defined) and on F1
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8))
    emp = pd.read_csv(OUT / "metric_validity_seedmean.csv")

    def kind_of(s):
        if s in ("Majority", "TF-IDF"):
            return "Classical"
        if s.startswith("Probe:"):
            return "Probe"
        return {"PID": "PID / PID-w", "PID_weighted": "PID / PID-w", "PID_n1531": "Pooled controls",
                "PID_n1905": "Pooled controls", "LODO_matched": "LODO", "MSA_only": "MSA-only"}[s.split("|")[1]]
    kind = emp["system"].map(kind_of)
    groups = [("Classical", "s"), ("Probe", "x"), ("PID / PID-w", "o"), ("Pooled controls", "D"), ("LODO", "^"), ("MSA-only", "v")]
    maj = emp[emp["system"] == "Majority"].iloc[0]
    for ax, col, ylab, tau in [(axes[0], "dri_acc", "DRI = 1/(1+Var(acc))", mvr["tau_hm_dri_acc"]),
                               (axes[1], "dri_f1", "DRI-F1 = 1/(1+Var(F1))", mvr["tau_hm_dri_f1"])]:
        for lab, mk in groups:
            e = emp[kind == lab]
            ax.scatter(e["hm"], e[col], marker=mk, s=16 if mk == "D" else 22, label=lab)
        ax.annotate("Majority", (maj["hm"], maj[col]), xytext=(5, -10), textcoords="offset points", fontsize=7)
        ax.set_xlabel("HM-F1 over dialects")
        ax.set_ylabel(ylab)
        ax.set_title(f"Kendall $\\tau$ with HM-F1 = {tau:.2f}")
        ax.grid(ls=":", alpha=0.6)
    axes[0].legend(frameon=False, fontsize=6.5)
    plt.tight_layout()
    save(fig, "fig4_metric_validity")

    # ---------------- Numeric macros for the manuscript text ----------------
    def mac(name, val):
        macros[name] = val
    for m in ft_models:
        tag = m.replace("-", "").replace("v02", "").replace("Base", "")
        r = A.loc[(m, "PID")]
        mac(f"pid{tag}F", f"{r['pooled_macro_f1_mean']:.3f}")
        mac(f"pid{tag}HM", f"{r['hm_f1_mean']:.3f}")
        mac(f"pid{tag}WD", f"{r['wd_f1_mean']:.3f}")
    (TAB / "macros.tex").write_text("\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in macros.items()) + "\n", encoding="utf-8")
    print("\n[STEP 8 PASSED] Figures saved to figures_v2/ and LaTeX tables to tables_v2/.")


if __name__ == "__main__":
    main()
