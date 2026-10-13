"""Variants of the estimator: GCV penalty, positive-part output, spread weights.

Monte Carlo at sd = 0.01 (implied-volatility noise) and under a heteroskedastic
price-noise design, and the listed chains with inverse-squared-half-spread
weights. Results go to python/results/tables/variants.json; tables via joe_tables.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

PY = Path(__file__).resolve().parent
sys.path.insert(0, str(PY))

import replicate as joe  # noqa: E402
from replicate import CHAINS, OUT, RES, _errors, _folds, _rms, density_grid, density_stats, ise, iv_noise, models, projected_calls  # noqa: E402
from rnd import estimate_rnd  # noqa: E402
from src import black_scholes as bs  # noqa: E402
from src.spx import load_slice  # noqa: E402

VARIANTS = {
    "rule": dict(penalty="closed"),
    "gcv": dict(penalty="gcv"),
}


def positive_part(q, s):
    """Clip at zero and renormalize to unit mass on the evaluation grid ``s``."""
    pos = np.maximum(np.nan_to_num(np.asarray(q, float)), 0.0)
    mass = float(np.trapezoid(pos, s))
    return pos / mass if mass > 0 else pos


def price_noise(K, C, S0, r, T, q, F, rng, a=0.01, b=0.02):
    """Heteroskedastic noise in the out-of-the-money premium: sd_i = a + b * premium_i."""
    K = np.asarray(K, float)
    disc = np.exp(-r * T)
    stock = S0 * np.exp(-q * T)
    put = np.maximum(C - stock + K * disc, 0.0)
    prem = np.where(K <= F, put, C)
    sd = a + b * prem
    noisy = np.maximum(prem + rng.normal(0.0, sd), 0.0)
    C_obs = np.where(K <= F, noisy + stock - K * disc, noisy)
    return np.maximum(C_obs, 0.0), sd


def mc_one(model, chain, rep, design):
    p, calls, dens = models()[model]
    K = CHAINS[chain]
    s_full = density_grid(p.forward)
    K_eval = np.linspace(60.0, 150.0, 401)
    q_true = dens(K_eval)
    rng = np.random.default_rng([4242, rep, {"heston": 1, "vg": 2, "bates": 3}[model],
                                 {"dense": 1, "sparse": 2, "listed": 3}[chain], {"iv": 1, "price": 2}[design]])
    C_true = np.maximum(calls(K), 0.0)
    if design == "iv":
        C_obs = iv_noise(K, C_true, p.S0, p.r, p.T, p.q, rng, 0.01)
        sd = None
    else:
        C_obs, sd = price_noise(K, C_true, p.S0, p.r, p.T, p.q, p.forward, rng)
    Cp = projected_calls(K, C_obs, p.r, p.T, p.forward)
    out = {}
    specs = [("rule", dict(penalty="closed"), None), ("gcv", dict(penalty="gcv"), None)]
    if sd is not None:
        w = 1.0 / sd**2
        specs += [("rule_w", dict(penalty="closed"), w), ("gcv_w", dict(penalty="gcv"), w)]
    for name, kw, w in specs:
        fit = estimate_rnd(K, Cp, p.S0, p.r, p.T, p.q, K_eval=s_full, obs_weights=w, **kw)
        q_eval = np.interp(K_eval, s_full, fit["q"])
        out[name] = dict(ise=ise(K_eval, q_eval, q_true), c=fit["c"], h=fit["h"])
        q_pos = np.interp(K_eval, s_full, positive_part(fit["q"], s_full))
        out[name + "_pos"] = dict(ise=ise(K_eval, q_pos, q_true))
    return out


def run_mc(n_reps=100, workers=2):
    from concurrent.futures import ProcessPoolExecutor

    path = OUT / "variants.json"
    result = json.loads(path.read_text()) if path.exists() else {}
    for design in ("iv", "price"):
        for model in ("heston", "vg", "bates"):
            for chain in ("dense", "sparse", "listed"):
                tag = f"{design}/{model}/{chain}"
                if len(result.get(tag, {}).get("reps", [])) >= n_reps:
                    continue
                t0 = time.time()
                with ProcessPoolExecutor(max_workers=workers) as ex:
                    reps = list(ex.map(mc_one, [model] * n_reps, [chain] * n_reps, range(n_reps),
                                       [design] * n_reps, chunksize=4))
                result[tag] = dict(reps=reps)
                path.write_text(json.dumps(result))
                print(f"{tag}: {n_reps} reps in {time.time()-t0:.0f}s", flush=True)
                for name in reps[0]:
                    v = np.array([r[name]["ise"] for r in reps], float)
                    print(f"   {name:10} mean ISE {v.mean():.3e} se {v.std(ddof=1)/np.sqrt(v.size):.1e}")
    return result


def run_listed():
    path = OUT / "variants_listed.json"
    result = {}
    for csv, title in joe.LISTED:
        sl = load_slice(RES / csv)
        w_all = 1.0 / (0.5 * sl.spread) ** 2
        result[title] = {}
        for name, kw, weighted in (("rule", dict(penalty="closed"), False), ("gcv", dict(penalty="gcv"), False),
                                   ("rule_w", dict(penalty="closed"), True), ("gcv_w", dict(penalty="gcv"), True)):
            errs = []
            for train, test in _folds(len(sl.K)):
                Kt = sl.K[train]
                Ct = projected_calls(Kt, sl.C[train], sl.r, sl.T, sl.F)
                fit = estimate_rnd(Kt, Ct, sl.S0, sl.r, sl.T, sl.q, K_eval=Kt[:1], K_price=sl.K[test],
                                   obs_weights=(w_all[train] if weighted else None), **kw)
                e, es, ev = _errors(sl, sl.K[test], fit["C"])
                errs.append((_rms(e), _rms(es), _rms(ev)))
            C = projected_calls(sl.K, sl.C, sl.r, sl.T, sl.F)
            s = density_grid(sl.F)
            full = estimate_rnd(sl.K, C, sl.S0, sl.r, sl.T, sl.q, K_eval=s, K_price=sl.K,
                                obs_weights=(w_all if weighted else None), **kw)
            e, es, ev = _errors(sl, sl.K, full["C"])
            rec = dict(ho_pts=float(np.mean([x[0] for x in errs])), ho_spread=float(np.mean([x[1] for x in errs])),
                       ho_iv=float(np.mean([x[2] for x in errs])), otm_spread=_rms(es), c=full["c"], h=full["h"])
            rec.update(density_stats(sl.F, full["q"], s))
            result[title][name] = rec
            print(f"{title} {name:7} HO {rec['ho_pts']:.3f} pts {rec['ho_spread']:.3f} hs {1e4*rec['ho_iv']:.0f} bp "
                  f"c={rec['c']:.2e} peaks {rec['peaks']} tv {rec['tv']:.3f} neg {rec['neg']:.3f}", flush=True)
    path.write_text(json.dumps(result, indent=1))
    return result


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "listed"
    if what == "listed":
        run_listed()
    else:
        run_mc()
