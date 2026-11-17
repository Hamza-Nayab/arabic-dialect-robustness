# -*- coding: utf-8 -*-
"""Step 6c: Bayesian mixed-effects model fitted with MCMC (NumPyro NUTS), all fine-tuned encoders (PID).

Data: one row per (encoder m, test tweet t): k = number of the 5 seeds that classify t correctly.
Likelihood: k ~ BetaBinomial(5, mu*phi, (1-mu)*phi)       (phi absorbs seed-to-seed correlation)
            logit(mu) = a + b_model[m] + b_dial[d(t)] + g[m, d(t)] + c_fert * z(fertility_{m,t}) + u_t
            u_t ~ Normal(0, sigma_u)                          (tweet random intercept)
Reference levels: model = AraBERTv02, dialect = MSA (treatment coding; b/g of reference levels fixed at 0).
Priors: a ~ N(0, 2.5); b_model, b_dial, g ~ N(0, 1.5); c_fert ~ N(0, 1); sigma_u ~ HalfNormal(2.5);
        phi ~ Gamma(2, 0.1).
fertility_{m,t} = subword tokens / whitespace words of tweet t under encoder m's tokenizer, standardized within each
(encoder, dialect) cell, so c_fert is a within-cell effect (cell means are absorbed by g). This replaces a Spearman
correlation over non-independent (model, dialect) cells, which is kept only as a descriptive figure.
Numerical stability: mu is clipped to [1e-4, 1 - 1e-4] (GLMM_LIK=binomial_olre gives an observation-level-random-effect
binomial alternative). Env: GLMM_ITERS="warmup,draws,chains", GLMM_OUT=<output file name>.
Sampler: 4 chains x (1000 warm-up + 1000 draws), target_accept 0.95; centred tweet effects. Reports 95% equal-tailed credible
intervals, R-hat, bulk ESS and divergences. Output: outputs_v2/glmm_mcmc.json
"""
from __future__ import annotations

import json
import os
from importlib.machinery import SourceFileLoader
from pathlib import Path

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")
import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import pandas as pd
from numpyro.diagnostics import summary
from numpyro.infer import MCMC, NUTS
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs_v2"
step3 = SourceFileLoader("step3", str(ROOT / "03_model_and_fertility_audit.py")).load_module()
MAIN = ["MSA", "Egyptian", "Levantine", "Gulf"]
SEEDS = [42, 43, 44, 45, 46]
numpyro.set_host_device_count(4)


def tweet_fertility(models, texts):
    path = OUT / "tweet_fertility.csv"
    if path.exists():
        f = pd.read_csv(path)
        if set(models) <= set(f.columns):
            return f
    cols = {}
    for m in models:
        tok = AutoTokenizer.from_pretrained(step3.MODEL_REGISTRY[m])
        cols[m] = [len(tok.tokenize(str(t))) / max(len(str(t).split()), 1) for t in texts]
    f = pd.DataFrame(cols)
    f.to_csv(path, index=False)
    return f


LIK = os.environ.get("GLMM_LIK", "betabinomial")  # mu clipped to [1e-4, 1-1e-4]: removes divergences at near-certain tweets


def model_fn(mi, di, ti, fz, k, n_models, n_tweets, n_trials=5):
    a = numpyro.sample("a", dist.Normal(0, 2.5))
    bm = numpyro.sample("b_model_raw", dist.Normal(0, 1.5).expand([n_models - 1]))
    bd = numpyro.sample("b_dial_raw", dist.Normal(0, 1.5).expand([3]))
    g = numpyro.sample("g_raw", dist.Normal(0, 1.5).expand([n_models - 1, 3]))
    c = numpyro.sample("c_fert", dist.Normal(0, 1))
    su = numpyro.sample("sigma_u", dist.HalfNormal(2.5))
    with numpyro.plate("tweets", n_tweets):  # centred: each tweet has 5 x n_models observations (data-rich)
        u = numpyro.sample("u", dist.Normal(0, su))
    BM = jnp.concatenate([jnp.zeros(1), bm])
    BD = jnp.concatenate([jnp.zeros(1), bd])
    G = jnp.zeros((n_models, 4)).at[1:, 1:].set(g)
    eta = a + BM[mi] + BD[di] + G[mi, di] + c * fz + u[ti]
    if LIK == "betabinomial":
        phi = numpyro.sample("phi", dist.Gamma(2.0, 0.1))
        mu = jnp.clip(jax.nn.sigmoid(eta), 1e-4, 1 - 1e-4)
        with numpyro.plate("obs", len(k)):
            numpyro.sample("k", dist.BetaBinomial(mu * phi, (1 - mu) * phi, n_trials), obs=k)
    else:
        # observation-level (encoder x tweet) random effect absorbs seed-to-seed correlation; non-centred
        # because each encoder x tweet cell has only 5 trials
        sv = numpyro.sample("sigma_v", dist.HalfNormal(2.5))
        with numpyro.plate("obs", len(k)):
            vz = numpyro.sample("v_z", dist.Normal(0, 1))
            numpyro.sample("k", dist.BinomialLogits(eta + sv * vz, total_count=n_trials), obs=k)


def main() -> None:
    te = pd.read_csv(OUT / "test_4.csv")
    y = te.label.to_numpy()
    models = [m for m in step3.MODEL_REGISTRY if all((OUT / "raw_preds" / f"PID_{m}_seed{s}.npy").exists() for s in SEEDS)]
    fert = tweet_fertility(models, te.text.tolist())
    rows = []
    for i, m in enumerate(models):
        correct = np.stack([np.load(OUT / "raw_preds" / f"PID_{m}_seed{s}.npy") == y for s in SEEDS]).sum(0)
        rows.append(pd.DataFrame({"mi": i, "di": te.dialect.map(MAIN.index).to_numpy(), "ti": np.arange(len(te)),
                                  "fert": fert[m].to_numpy(), "k": correct}))
    df = pd.concat(rows, ignore_index=True)
    # standardized within encoder x dialect: cell means are absorbed by g[m, d], so c_fert is a within-cell effect
    df["fz"] = df.groupby(["mi", "di"]).fert.transform(lambda x: (x - x.mean()) / x.std(ddof=0) if x.std(ddof=0) > 0 else 0.0)
    kern = NUTS(model_fn, target_accept_prob=0.95)
    NW, NS, NC = (int(x) for x in os.environ.get("GLMM_ITERS", "1000,1000,4").split(","))
    mcmc = MCMC(kern, num_warmup=NW, num_samples=NS, num_chains=NC, chain_method="parallel", progress_bar=False)
    mcmc.run(jax.random.PRNGKey(0), jnp.array(df.mi), jnp.array(df.di), jnp.array(df.ti), jnp.array(df.fz, dtype=jnp.float32),
             jnp.array(df.k), len(models), len(te), extra_fields=("diverging",))
    samples = mcmc.get_samples(group_by_chain=True)
    keep = {k: v for k, v in samples.items() if k not in ("u", "v_z")}
    summ = summary(keep, prob=0.95, group_by_chain=True)
    div = int(np.asarray(mcmc.get_extra_fields()["diverging"]).sum())
    flat = {k: np.asarray(v).reshape(-1, *np.asarray(v).shape[2:]) for k, v in keep.items()}

    def term(name, draws, sm):
        q = np.quantile(draws, [0.025, 0.975])
        return {"term": name, "post_mean": float(draws.mean()), "post_sd": float(draws.std()),
                "ci95": [float(q[0]), float(q[1])], "excludes_zero": bool(q[0] > 0 or q[1] < 0),
                "r_hat": float(sm["r_hat"]), "ess_bulk": float(sm["n_eff"])}

    terms = [term("Intercept", flat["a"], summ["a"]),
             term("fertility (per SD)", flat["c_fert"], summ["c_fert"]),
             term("sigma_tweet", flat["sigma_u"], summ["sigma_u"])]
    for extra, lab in [("phi", "phi"), ("sigma_v", "sigma_model_x_tweet")]:
        if extra in flat:
            terms.append(term(lab, flat[extra], summ[extra]))
    for j, m in enumerate(models[1:]):
        terms.append(term(f"model[{m}]", flat["b_model_raw"][:, j], {k: v[j] for k, v in summ["b_model_raw"].items()}))
    for j, dd in enumerate(MAIN[1:]):
        terms.append(term(f"dialect[{dd}]", flat["b_dial_raw"][:, j], {k: v[j] for k, v in summ["b_dial_raw"].items()}))
    for j, m in enumerate(models[1:]):
        for l, dd in enumerate(MAIN[1:]):
            terms.append(term(f"{m}:{dd}", flat["g_raw"][:, j, l], {k: v[j, l] for k, v in summ["g_raw"].items()}))
    # per-encoder dialect effect (log-odds vs MSA within that encoder)
    per_model = {}
    for i, m in enumerate(models):
        per_model[m] = {}
        for l, dd in enumerate(MAIN[1:]):
            dr = flat["b_dial_raw"][:, l] + (flat["g_raw"][:, i - 1, l] if i > 0 else 0)
            q = np.quantile(dr, [0.025, 0.975])
            per_model[m][dd] = {"mean": float(dr.mean()), "ci95": [float(q[0]), float(q[1])],
                                "excludes_zero": bool(q[0] > 0 or q[1] < 0)}
    allr = [t["r_hat"] for t in terms]
    out = {"software": f"NumPyro {numpyro.__version__} / JAX {jax.__version__} (NUTS)", "chains": NC, "warmup": NW,
           "draws": NS, "target_accept": 0.95, "divergences": div, "max_r_hat": float(max(allr)),
           "min_ess_bulk": float(min(t["ess_bulk"] for t in terms)),
           "priors": {"a": "N(0,2.5)", "b_model,b_dial,g": "N(0,1.5)", "c_fert": "N(0,1)", "sigma_u": "HalfNormal(2.5)",
                      **({"phi": "Gamma(2,0.1)"} if LIK == "betabinomial" else {"sigma_v": "HalfNormal(2.5)"})},
           "likelihood": ("BetaBinomial(5) over seeds, mu clipped to [1e-4, 1-1e-4]" if LIK == "betabinomial"
                          else "Binomial(5) over seeds, logit-normal encoder x tweet effect"), "reference_model": models[0], "reference_dialect": "MSA",
           "models": models, "n_obs": int(len(df)), "n_tweets": int(len(te)),
           "fertility_mean_sd": [float(df.fert.mean()), float(df.fert.std())], "fertility_standardization": "within encoder x dialect", "terms": terms, "dialect_effect_by_model": per_model}
    (OUT / os.environ.get("GLMM_OUT", "glmm_mcmc.json")).write_text(json.dumps(out, indent=1))
    for t in terms:
        if t["term"].startswith(("dialect", "fertility", "sigma")) or t["excludes_zero"]:
            print(f"  {t['term']:32s} {t['post_mean']:+.3f} [{t['ci95'][0]:+.3f},{t['ci95'][1]:+.3f}] rhat={t['r_hat']:.3f}")
    print(f"divergences={div} max_rhat={out['max_r_hat']:.3f} min_ess={out['min_ess_bulk']:.0f}")
    print("[STEP 6c PASSED]")


if __name__ == "__main__":
    main()
