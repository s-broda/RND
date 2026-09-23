#!/usr/bin/env python3
"""Replication for the v2 manuscript.

Keeps ``python/replicate.py`` as the v1 entry point. This script

- regenerates the exact-quote Heston and variance-gamma ablation,
- checks that the signed estimator integrates to zero and has first moment
  equal to the forward,
- runs a noisy irregular-grid experiment with retuned benchmarks,
- rescores the four listed slices with those benchmarks and with SVI.

Run from the repository root::

    python3 python/replicate_v2.py

Writes ``figures_v2/`` and ``python/results/v2_summary.json``.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PY = Path(__file__).resolve().parent
ROOT = PY.parent
import sys

sys.path.insert(0, str(PY))

import replicate as v1
from rnd import estimate_rnd
from src import black_scholes as bs
from src.competitors import priestley_chao_cubic
from src.competitors_v2 import (
    asd_cv_bandwidth,
    asd_fit,
    diagnostics,
    integrate_density,
    otm_rmse,
    pca_cv_bandwidth,
    pca_fit,
    priestley_cv_bandwidth,
    rmse,
    svi_fit,
    yatchew_cv_lambda,
    yatchew_fit,
)
from src.heston import BCC97, carr_madan_calls, heston_spot_density
from src.spx import load_slice
from src.vg import CM99, vg_calls, vg_spot_density

FIG = ROOT / "figures_v2"
RES = PY / "results"
FIG.mkdir(parents=True, exist_ok=True)

N_REPS = 30
IV_NOISE = 0.01
SEED = 20260923


def _style():
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 10,
            "axes.labelsize": 11,
            "legend.fontsize": 8,
            "figure.dpi": 140,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linestyle": "--",
        }
    )


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


def _pc_prices(K_obs, C_obs, K_eval, h, S0, r, T, q):
    _, _, C_hat = priestley_chao_cubic(
        K_obs, C_obs, S0, r, T, K_eval, q=q, h=h, return_call=True
    )
    return C_hat


def _pc_density(K_obs, C_obs, K_eval, h, S0, r, T, q):
    q_hat, _, _ = priestley_chao_cubic(
        K_obs, C_obs, S0, r, T, K_eval, q=q, h=h, return_call=True
    )
    return q_hat


def _pc_rule_h(K, C, S0, r, T, q, F):
    """Second-derivative bandwidth, the v1 Priestley--Chao rule."""
    n = max(len(K), 8)
    iv = bs.implied_vol(C, S0, K, r, T, q)
    atm = np.nanmedian(iv[np.abs(np.asarray(K) - F) <= 0.03 * F])
    if not np.isfinite(atm):
        atm = np.nanmedian(iv)
    if not np.isfinite(atm):
        atm = 0.2
    return float(1.06 * F * float(atm) * np.sqrt(T) * n ** (-1.0 / 9.0))


def _ours_h_grid(K, C, S0, r, T, q, F):
    delta = float(np.median(np.diff(np.sort(K))))
    iv = bs.implied_vol(C, S0, K, r, T, q)
    atm = np.nanmedian(iv[np.abs(K - F) <= 0.03 * F])
    if not np.isfinite(atm):
        atm = np.nanmedian(iv)
    if not np.isfinite(atm):
        atm = 0.2
    s = F * float(atm) * np.sqrt(T)
    n = max(len(K), 8)
    raw = np.concatenate(
        [
            delta * np.array([0.8, 1.2, 2.0, 4.0]),
            1.06 * s * n ** (-0.2) * np.array([0.35, 0.7, 1.0, 1.6, 2.5]),
        ]
    )
    return np.unique(np.clip(raw, max(0.5 * delta, 1e-3), max(s, delta)))


def ours_cv_bandwidth(K, C, S0, r, T, q, F):
    grid = _ours_h_grid(K, C, S0, r, T, q, F)
    best_h, best = float(grid[0]), np.inf
    idx = np.arange(len(K))
    for h in grid:
        errs = []
        for train, test in ((idx % 2 == 0, idx % 2 == 1), (idx % 2 == 1, idx % 2 == 0)):
            if train.sum() < 4 or test.sum() < 2:
                continue
            out = estimate_rnd(
                K[train], C[train], S0, r, T, q, h=float(h), twice=True
            )
            _, C_te = out["interpolant"](K[test])
            errs.append(otm_rmse(K[test], C_te, C[test], S0, r, T, q, F))
        if errs and np.mean(errs) < best:
            best = float(np.mean(errs))
            best_h = float(h)
    return best_h


def check_bs_second_derivative():
    """Numerical Breeden--Litzenberger on a flat smile against the lognormal."""
    S0, r, q, T, sig = 100.0, 0.05, 0.0, 0.25, 0.20
    F = S0 * np.exp((r - q) * T)
    K = np.linspace(40.0, 180.0, 2801)
    C = bs.call_price(S0, K, r, T, sig, q)
    d2 = np.gradient(np.gradient(C, K), K)
    q_hat = np.exp(r * T) * d2
    mu = np.log(F) - 0.5 * sig**2 * T
    s = sig * np.sqrt(T)
    q_true = np.exp(-0.5 * ((np.log(K) - mu) / s) ** 2) / (K * s * np.sqrt(2.0 * np.pi))
    window = (K >= 60.0) & (K <= 150.0)
    ise = float(np.trapezoid((q_hat[window] - q_true[window]) ** 2, K[window]))
    return ise


def check_integral_identity():
    """Wide-grid check of ∫ f̂ = 0 and ∫ K f̂ = F on an exact Heston strip."""
    p = BCC97
    K = np.linspace(30.0, 220.0, 128)
    C = carr_madan_calls(K, p)
    grid = np.linspace(-2.0 * p.forward, 8.0 * p.forward, 6001)
    out = estimate_rnd(K, C, p.S0, p.r, p.T, p.q, K_eval=grid, h="mesh", twice=True)
    diag = diagnostics(grid, out["q"], p.forward)
    diag["jump_sum"] = float(np.sum(out["dp"]))
    diag["h"] = float(out["h"])
    return diag


def _irregular_strikes(F):
    left = np.geomspace(0.45 * F, 0.90 * F, 22, endpoint=False)
    atm = np.linspace(0.90 * F, 1.10 * F, 28, endpoint=False)
    right = np.geomspace(1.10 * F, 1.40 * F, 22)
    return np.unique(np.concatenate([left, atm, right]))


def _add_iv_noise(K, C, S0, r, T, q, rng, sd=IV_NOISE):
    iv = bs.implied_vol(C, S0, K, r, T, q)
    iv = np.where(np.isfinite(iv), iv, np.nanmedian(iv))
    noisy = np.clip(iv + rng.normal(0.0, sd, size=iv.shape), 0.02, 2.5)
    return bs.call_price(S0, K, r, T, noisy, q), noisy


def _ise(K, qh, qt):
    return float(np.trapezoid((qh - qt) ** 2, K))


def _fit_all(K, C, S0, r, T, q, F, K_price, K_dens):
    """Fit every v2 method. Densities are on ``K_dens``; calls on ``K_price``."""
    out = {}
    est = estimate_rnd(K, C, S0, r, T, q, K_eval=K_dens, h="density", twice=True)
    _, C_den = est["interpolant"](K_price)
    out["ours_den"] = {
        "C": C_den,
        "q": est["q"],
        "h": float(est["h"]),
        "label": "Ours, density rule",
    }
    h_cv = ours_cv_bandwidth(K, C, S0, r, T, q, F)
    est_cv = estimate_rnd(K, C, S0, r, T, q, K_eval=K_dens, h=h_cv, twice=True)
    _, C_cv = est_cv["interpolant"](K_price)
    out["ours_cv"] = {
        "C": C_cv,
        "q": est_cv["q"],
        "h": float(h_cv),
        "label": "Ours, pricing CV",
    }

    lam = yatchew_cv_lambda(K, C, S0, r, T, q, F)
    C_yh, _ = yatchew_fit(K, C, S0, r, T, q, F, K_price, lam)
    out["yh"] = {"C": C_yh, "q": None, "lam": float(lam), "label": "Yatchew--Haerdle CV"}

    h_price, h_dens, _ = asd_cv_bandwidth(K, C, S0, r, T, q, F)
    C_asd, q_asd, _ = asd_fit(
        K, C, S0, r, T, q, F, K_price, K_dens, h_price, h_dens
    )
    out["asd"] = {
        "C": C_asd,
        "q": q_asd,
        "h_price": float(h_price),
        "h_dens": float(h_dens),
        "label": "Ait-Sahalia--Duarte CV",
    }

    h_pca = pca_cv_bandwidth(K, C, S0, r, T, q, F)
    C_pca, q_pca, a, _ = pca_fit(K, C, r, T, F, h_pca, K_price, K_dens)
    out["pca"] = {
        "C": C_pca,
        "q": q_pca,
        "h": float(h_pca),
        "weight_sum": float(np.sum(a)),
        "label": "PCA CV",
    }

    def _prices(Ko, Co, Ke, h):
        return _pc_prices(Ko, Co, Ke, h, S0, r, T, q)

    h_pc = priestley_cv_bandwidth(K, C, S0, r, T, q, F, _prices)
    C_pc = _pc_prices(K, C, K_price, h_pc, S0, r, T, q)
    q_pc = _pc_density(K, C, K_dens, h_pc, S0, r, T, q)
    out["pc"] = {"C": C_pc, "q": q_pc, "h": float(h_pc), "label": "Priestley--Chao CV"}
    h_rule = _pc_rule_h(K, C, S0, r, T, q, F)
    out["pc_rule"] = {
        "C": _pc_prices(K, C, K_price, h_rule, S0, r, T, q),
        "q": _pc_density(K, C, K_dens, h_rule, S0, r, T, q),
        "h": float(h_rule),
        "label": "Priestley--Chao n^{-1/9}",
    }

    C_svi, q_svi, params = svi_fit(K, C, S0, r, T, q, F, K_price, K_dens)
    out["svi"] = {"C": C_svi, "q": q_svi, "params": params, "label": "SVI"}
    return out


def _score_fit(fit, K_obs, C_obs, C_true, S0, r, T, q, F, K_ise, q_true):
    row = {
        "rmse_true": otm_rmse(K_obs, fit["C"], C_true, S0, r, T, q, F),
        "rmse_noisy": otm_rmse(K_obs, fit["C"], C_obs, S0, r, T, q, F),
    }
    if fit["q"] is None:
        row["ise"] = None
        row["signed"] = None
        row["positive"] = None
        row["negative"] = None
        row["forward_error"] = None
    else:
        row["ise"] = _ise(K_ise, fit["q"], q_true)
        diag = diagnostics(K_ise, fit["q"], F)
        row["signed"] = diag["signed"]
        row["positive"] = diag["positive"]
        row["negative"] = diag["negative"]
        row["forward_error"] = diag["forward_error"]
    for key in ("h", "lam", "h_price", "h_dens", "weight_sum"):
        if key in fit:
            row[key] = fit[key]
    if "params" in fit:
        row["g_min"] = fit["params"]["g_min"]
    return row


def _mean_std(rows, key):
    vals = [r[key] for r in rows if r.get(key) is not None and np.isfinite(r[key])]
    if not vals:
        return None, None
    return float(np.mean(vals)), float(np.std(vals, ddof=1))


def run_noise(label, call_fn, dens_fn, params):
    rng = np.random.default_rng(SEED)
    F = params.forward
    K_obs = _irregular_strikes(F)
    C_true = np.maximum(call_fn(K_obs), 0.0)
    K_ise = np.linspace(60.0, 140.0, 321)
    q_true = dens_fn(K_ise)
    methods = ("ours_den", "ours_cv", "yh", "asd", "pca", "pc", "pc_rule", "svi")
    bag = {m: [] for m in methods}
    example = None
    example_dist = np.inf
    print(f"\n== noise {label}  strikes={len(K_obs)}  reps={N_REPS}  iv sd={IV_NOISE}")
    for rep in range(N_REPS):
        C_obs, _ = _add_iv_noise(K_obs, C_true, params.S0, params.r, params.T, params.q, rng)
        fit = _fit_all(
            K_obs, C_obs, params.S0, params.r, params.T, params.q, F, K_obs, K_ise
        )
        for m in methods:
            bag[m].append(
                _score_fit(
                    fit[m], K_obs, C_obs, C_true, params.S0, params.r, params.T,
                    params.q, F, K_ise, q_true,
                )
            )
        ise = bag["ours_den"][-1]["ise"]
        if ise is not None and abs(ise - np.nanmedian([r["ise"] for r in bag["ours_den"]])) <= example_dist:
            # Keep updating until the end; final selection is below.
            pass
        if rep == 0 or (rep + 1) % 5 == 0:
            print(f"  rep {rep + 1}/{N_REPS}  ours ISE={ise:.4e}")
    ises = [r["ise"] for r in bag["ours_den"]]
    med = float(np.median(ises))
    # Refit the median-ISE draw for the figure. The RNG is exhausted, so
    # regenerate that draw from a fresh stream advanced to the same index.
    which = int(np.argmin(np.abs(np.asarray(ises) - med)))
    rng2 = np.random.default_rng(SEED)
    C_show = None
    for i in range(which + 1):
        C_show, _ = _add_iv_noise(K_obs, C_true, params.S0, params.r, params.T, params.q, rng2)
    example = _fit_all(
        K_obs, C_show, params.S0, params.r, params.T, params.q, F, K_obs, K_ise
    )
    example_dist = which
    summary = {}
    for m in methods:
        summary[m] = {}
        for key in ("ise", "rmse_true", "rmse_noisy", "signed", "positive", "negative", "forward_error"):
            mu, sd = _mean_std(bag[m], key)
            summary[m][key] = mu
            summary[m][key + "_sd"] = sd
        print(
            f"  {m:10s}  ISE={summary[m]['ise']}  "
            f"RMSE(true)={summary[m]['rmse_true']}  "
            f"RMSE(noisy)={summary[m]['rmse_noisy']}"
        )
    summary["median_rep"] = example_dist
    summary["n_strikes"] = int(len(K_obs))
    return summary, K_ise, q_true, example, K_obs


def _listed_holdout(sl, method):
    """Nested even/odd hold-out. Tuning uses only the training strikes."""
    K, C = sl.K, sl.C
    idx = np.arange(len(K))
    errs = []
    for train, test in ((idx % 2 == 0, idx % 2 == 1), (idx % 2 == 1, idx % 2 == 0)):
        Kt, Ct = K[train], C[train]
        if method == "ours_den":
            est = estimate_rnd(Kt, Ct, sl.S0, sl.r, sl.T, sl.q, twice=True)
            _, C_te = est["interpolant"](K[test])
        elif method == "ours_cv":
            h = ours_cv_bandwidth(Kt, Ct, sl.S0, sl.r, sl.T, sl.q, sl.F)
            est = estimate_rnd(Kt, Ct, sl.S0, sl.r, sl.T, sl.q, h=h, twice=True)
            _, C_te = est["interpolant"](K[test])
        elif method == "yh":
            lam = yatchew_cv_lambda(Kt, Ct, sl.S0, sl.r, sl.T, sl.q, sl.F)
            C_te, _ = yatchew_fit(Kt, Ct, sl.S0, sl.r, sl.T, sl.q, sl.F, K[test], lam)
        elif method == "asd":
            h_price, _, _ = asd_cv_bandwidth(Kt, Ct, sl.S0, sl.r, sl.T, sl.q, sl.F)
            C_te, _, _ = asd_fit(
                Kt, Ct, sl.S0, sl.r, sl.T, sl.q, sl.F, K[test], K[test], h_price, h_price
            )
        elif method == "pca":
            h = pca_cv_bandwidth(Kt, Ct, sl.S0, sl.r, sl.T, sl.q, sl.F)
            C_te, _, _, _ = pca_fit(Kt, Ct, sl.r, sl.T, sl.F, h, K[test], K[test])
        elif method == "pc":
            def _prices(Ko, Co, Ke, h):
                return _pc_prices(Ko, Co, Ke, h, sl.S0, sl.r, sl.T, sl.q)
            h = priestley_cv_bandwidth(Kt, Ct, sl.S0, sl.r, sl.T, sl.q, sl.F, _prices)
            C_te = _pc_prices(Kt, Ct, K[test], h, sl.S0, sl.r, sl.T, sl.q)
        elif method == "pc_rule":
            h = _pc_rule_h(Kt, Ct, sl.S0, sl.r, sl.T, sl.q, sl.F)
            C_te = _pc_prices(Kt, Ct, K[test], h, sl.S0, sl.r, sl.T, sl.q)
        elif method == "svi":
            C_te, _, _ = svi_fit(
                Kt, Ct, sl.S0, sl.r, sl.T, sl.q, sl.F, K[test], K[test]
            )
        else:
            raise KeyError(method)
        errs.append(otm_rmse(K[test], C_te, C[test], sl.S0, sl.r, sl.T, sl.q, sl.F))
    return float(np.mean(errs))


def score_listed(sl, title):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)
    K_win = np.linspace(max(1.0, 0.15 * sl.F), 2.60 * sl.F, 1601)
    wide = np.linspace(-1.0 * sl.F, 8.0 * sl.F, 4001)
    fit = _fit_all(sl.K, sl.C, sl.S0, sl.r, sl.T, sl.q, sl.F, sl.K, K_win)
    # Wide-grid identity for the proposed estimator.
    est_wide = estimate_rnd(
        sl.K, sl.C, sl.S0, sl.r, sl.T, sl.q, K_eval=wide, h="density", twice=True
    )
    wide_diag = diagnostics(wide, est_wide["q"], sl.F)
    rows = {}
    for key, spec in fit.items():
        row = {
            "rmse": otm_rmse(sl.K, spec["C"], sl.C, sl.S0, sl.r, sl.T, sl.q, sl.F),
        }
        if spec["q"] is None:
            row.update(signed=None, positive=None, negative=None, forward_error=None)
        else:
            diag = diagnostics(K_win, spec["q"], sl.F)
            row["signed"] = diag["signed"]
            row["positive"] = diag["positive"]
            row["negative"] = diag["negative"]
            row["forward_error"] = diag["forward_error"]
        for extra in ("h", "lam", "h_price", "h_dens", "weight_sum"):
            if extra in spec:
                row[extra] = spec[extra]
        if "params" in spec:
            row["g_min"] = spec["params"]["g_min"]
            row["svi"] = spec["params"]
        print(f"  tuning {key} done, scoring hold-out")
        row["holdout"] = _listed_holdout(sl, key)
        rows[key] = row
        print(
            f"  {key:10s}  RMSE={row['rmse']:.3f}  hold-out={row['holdout']:.3f}  "
            f"signed={row['signed']}  pos={row['positive']}  "
            f"fwd={row['forward_error']}"
        )
    rows["ours_wide"] = wide_diag
    # IV series for the figure, from the in-sample call fits.
    iv = {}
    iv["market"] = bs.implied_vol(
        np.where(sl.K <= sl.F, sl.P, sl.C),
        sl.S0,
        sl.K,
        sl.r,
        sl.T,
        sl.q,
        False,
    )
    # OTM IV: puts below F are passed as puts. implied_vol with is_call False
    # expects a put price. Build it strike by strike via the call curve.
    call_iv = bs.implied_vol(sl.C, sl.S0, sl.K, sl.r, sl.T, sl.q, True)
    iv["market"] = call_iv
    for key in fit:
        iv[key] = bs.implied_vol(fit[key]["C"], sl.S0, sl.K, sl.r, sl.T, sl.q, True)
    return {
        "title": title,
        "sl": sl,
        "K_win": K_win,
        "fit": fit,
        "rows": rows,
        "iv": iv,
    }


def _plot_noise(path, title, K_ise, q_true, example):
    _style()
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.plot(K_ise, q_true, color="black", lw=2.0, label="True")
    styles = {
        "ours_den": ("#1f77b4", "-", "Ours, density rule"),
        "pca": ("#2ca02c", "-.", "PCA"),
        "asd": ("#8c564b", "--", "Aït-Sahalia–Duarte"),
        "pc_rule": ("#9467bd", ":", "Priestley–Chao"),
        "svi": ("#d62728", "-", "SVI"),
    }
    for key, (color, ls, lab) in styles.items():
        q = example[key]["q"]
        ax.plot(K_ise, np.maximum(q, 0.0), color=color, lw=1.25, ls=ls, label=lab)
    ax.set_xlim(60.0, 140.0)
    ax.set_ylim(bottom=0.0)
    ax.set_xlabel(r"Strike $K$")
    ax.set_ylabel(r"$f_{\mathbb{Q}}(K)$")
    ax.set_title(title)
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    print(f"  wrote {path.name}")


def _plot_listed(rows, path):
    _style()
    fig, axes = plt.subplots(len(rows), 2, figsize=(9.6, 3.15 * len(rows)))
    if len(rows) == 1:
        axes = np.array([axes])
    q_styles = (
        ("ours_den", "#1f77b4", "-", "Ours"),
        ("asd", "#8c564b", "--", "Aït-Sahalia–Duarte"),
        ("pca", "#2ca02c", "-.", "PCA"),
        ("svi", "#d62728", "-", "SVI"),
    )
    iv_styles = (
        ("ours_den", "#1f77b4", "-", "Ours"),
        ("yh", "#ff7f0e", ":", "Yatchew–Härdle"),
        ("pca", "#2ca02c", "-.", "PCA"),
        ("svi", "#d62728", "-", "SVI"),
    )
    for axrow, rec in zip(axes, rows):
        axq, axiv = axrow
        sl = rec["sl"]
        s = rec["K_win"]
        lo, hi = 0.55 * sl.F, 1.40 * sl.F
        for key, color, ls, lab in q_styles:
            q = np.maximum(rec["fit"][key]["q"], 0.0)
            axq.plot(s, q, color=color, lw=1.25, ls=ls, label=lab)
        core = (s >= 0.65 * sl.F) & (s <= 1.30 * sl.F)
        ymax = 1.15 * max(float(np.max(np.maximum(rec["fit"][k]["q"][core], 0.0))) for k, *_ in q_styles)
        axq.axvline(sl.F, color="0.5", ls="--", lw=0.8)
        axq.set_xlim(lo, hi)
        axq.set_ylim(0.0, max(ymax, 1e-8))
        axq.set_ylabel(r"$f_{\mathbb{Q}}(K)$")
        axq.set_xlabel(r"Strike $K$")
        axq.set_title(rec["title"])
        axq.legend(frameon=False, loc="upper right")

        show = np.isfinite(rec["iv"]["market"]) & (sl.K >= lo) & (sl.K <= hi)
        axiv.plot(sl.K[show], rec["iv"]["market"][show], "k.", ms=3, alpha=0.35, label="Market")
        for key, color, ls, lab in iv_styles:
            axiv.plot(sl.K[show], rec["iv"][key][show], color=color, lw=1.15, ls=ls, label=lab)
        ivs = rec["iv"]["market"][show]
        y1 = float(np.nanpercentile(ivs, 2)) - 0.02
        y2 = float(np.nanpercentile(ivs, 98)) + 0.03
        axiv.set_xlim(lo, hi)
        axiv.set_ylim(max(0.05, y1), min(0.70, y2))
        axiv.axvline(sl.F, color="0.5", ls="--", lw=0.8)
        axiv.set_ylabel("OTM implied vol")
        axiv.set_xlabel(r"Strike $K$")
        axiv.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    print(f"  wrote {path.name}")


def _strip_listed(rec):
    """Drop arrays before writing JSON."""
    rows = {}
    for key, row in rec["rows"].items():
        if key == "ours_wide":
            rows[key] = {k: row[k] for k in row}
            continue
        keep = {
            k: row[k]
            for k in row
            if k != "svi"
        }
        rows[key] = keep
    sl = rec["sl"]
    return {
        "title": rec["title"],
        "S0": sl.S0,
        "F": sl.F,
        "r": sl.r,
        "q": sl.q,
        "T": sl.T,
        "asof": sl.asof,
        "n": int(len(sl.K)),
        "rows": rows,
    }


def exact_ablation():
    v1.FIG = FIG
    print("\n== exact-quote ablation")
    heston = v1.heston_ise()
    vg = v1.vg_ise()
    return {"heston": heston, "vg": vg}


def main():
    summary = {}
    ise_bs = check_bs_second_derivative()
    ident = check_integral_identity()
    print(f"BS second-derivative ISE on [60,150] = {ise_bs:.3e}")
    print(
        "Integral identity (exact Heston, mesh h, wide grid): "
        f"signed={ident['signed']:.4f}  moment={ident['moment']:.4f}  "
        f"F={ident['F']:.4f}  jump_sum={ident['jump_sum']:.4f}"
    )
    summary["bs_check_ise"] = ise_bs
    summary["integral_identity"] = ident

    summary["exact"] = exact_ablation()

    def heston_calls(K):
        return carr_madan_calls(K, BCC97)

    def heston_dens(K):
        return heston_spot_density(K, BCC97)

    def vg_call(K):
        return vg_calls(K, CM99)

    def vg_dens(K):
        return vg_spot_density(K, CM99)

    h_sum, K_h, q_h, ex_h, _ = run_noise("Heston", heston_calls, heston_dens, BCC97)
    v_sum, K_v, q_v, ex_v, _ = run_noise("VG", vg_call, vg_dens, CM99)
    summary["noise"] = {"heston": h_sum, "vg": v_sum}
    _plot_noise(
        FIG / "noise_heston.pdf",
        "Heston, one draw at the median density-rule ISE",
        K_h, q_h, ex_h,
    )
    _plot_noise(
        FIG / "noise_vg.pdf",
        "Variance gamma, one draw at the median density-rule ISE",
        K_v, q_v, ex_v,
    )

    slices = [
        ("spx_20261218.csv", "SPX 18 Dec 2026"),
        ("spx_20270319.csv", "SPX 19 Mar 2027"),
        ("ndx_20261218.csv", "NDX 18 Dec 2026"),
        ("rut_20261218.csv", "RUT 18 Dec 2026"),
    ]
    listed = []
    for name, title in slices:
        listed.append(score_listed(load_slice(RES / name), title))
    summary["listed"] = [_strip_listed(rec) for rec in listed]
    _plot_listed(
        [rec for rec in listed if "Mar" not in rec["title"]],
        FIG / "ccdf_listed.pdf",
    )

    out = RES / "v2_summary.json"
    out.write_text(json.dumps(_jsonable(summary), indent=2))
    print(f"\nwrote {out}")
    return summary


if __name__ == "__main__":
    main()
