"""Replication script for the paper.

Every estimator with a bandwidth is tuned by the same criterion, even/odd
out-of-the-money pricing error on a common geometric grid, and the rule
based choices of the source papers are reported next to the tuned ones.
Hold-out scores are nested: the tuning is repeated on the training half.
Pricing errors are reported in index points, in units of the quoted
half-spread, and in Black--Scholes implied volatility.

Run from the repository root::

    python3 python/replicate.py listed      # four main chains, all methods
    python3 python/replicate.py extra       # all remaining monthly chains, all methods
    python3 python/replicate.py mc          # Monte Carlo (Heston, VG, Bates; three noise levels)
    python3 python/replicate.py exact       # exact designs, oracle and rule bandwidths
    python3 python/replicate.py sens        # sensitivity to the bandwidth constant and penalty
    python3 python/replicate.py figures     # the figures in the paper

Results are written as JSON to ``python/results/tables/`` and the LaTeX
tables are produced by ``python/tables.py``. ``replicate.py oracle`` scores every
method at its oracle bandwidth and ``replicate.py transfer`` the competitors at ours.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

PY = Path(__file__).resolve().parent
ROOT = PY.parent
sys.path.insert(0, str(PY))

from rnd import TV_CAP, estimate_rnd, _implied_vol, clipped_variation  # noqa: E402
from src import black_scholes as bs  # noqa: E402
from src.competitors import (  # noqa: E402
    _iv_on_calls,
    asd_bandwidth,
    asd_fit,
    asl_bandwidth,
    asl_fit,
    ghs_call_fit,
    ghs_iv_fit,
    ghs_loo_bandwidth,
    pca_fit,
    shape_fit_calls,
)
from src.spx import load_slice  # noqa: E402

RES = ROOT / "python" / "results"
OUT = RES / "tables"
OUT.mkdir(parents=True, exist_ok=True)

GRID_LO, GRID_HI, GRID_N = 0.04, 3.0, 19
WINDOW_LO, WINDOW_HI = 0.60, 1.30


# ----------------------------------------------------------------------
# Shared pieces
# ----------------------------------------------------------------------


def projected_calls(K, C, r, T, F):
    """Euclidean projection onto decreasing convex calls (shared first step)."""
    K = np.asarray(K, dtype=float)
    C = np.asarray(C, dtype=float)
    disc = float(np.exp(-float(r) * float(T)))
    intrinsic = disc * np.maximum(float(F) - K, 0.0)
    m = shape_fit_calls(K, C, disc, lam=0.0, intrinsic=intrinsic)
    return np.maximum(np.asarray(m, dtype=float), 0.0)


def atm_sigma(K, C, S0, r, T, q, F):
    iv = np.asarray(bs.implied_vol(C, S0, K, r, T, q), dtype=float)
    atm = np.nanmedian(iv[np.abs(K - F) <= 0.03 * F])
    if not np.isfinite(atm):
        atm = np.nanmedian(iv)
    return float(atm) if np.isfinite(atm) and atm > 0 else 0.2


def price_scale(K, C, S0, r, T, q, F):
    """F sigma_ATM sqrt(T): the natural strike scale of the chain."""
    return float(F) * atm_sigma(K, C, S0, r, T, q, F) * np.sqrt(float(T))


def common_grid(K, C, S0, r, T, q, F):
    """One geometric bandwidth grid for every method, in strike units."""
    s = price_scale(K, C, S0, r, T, q, F)
    K = np.asarray(K, dtype=float)
    delta = float(np.median(np.diff(K))) if K.size > 1 else s / 20.0
    grid = s * np.geomspace(GRID_LO, GRID_HI, GRID_N)
    return grid[grid >= 1.5 * delta] if np.any(grid >= 1.5 * delta) else grid[-4:]


def otm_premiums(K, C_hat, S0, r, T, q, F):
    """Floored out-of-the-money premium from a call curve."""
    K = np.asarray(K, dtype=float)
    disc = float(np.exp(-r * T))
    stock = float(S0) * float(np.exp(-q * T))
    C_hat = np.maximum(np.asarray(C_hat, dtype=float), 0.0)
    P_hat = np.maximum(C_hat - stock + K * disc, 0.0)
    return np.where(K <= F, P_hat, C_hat)


def otm_rmse(K, C_hat, C_ref, S0, r, T, q, F):
    a = otm_premiums(K, C_hat, S0, r, T, q, F)
    b = otm_premiums(K, C_ref, S0, r, T, q, F)
    ok = np.isfinite(a) & np.isfinite(b)
    if not np.any(ok):
        return float("inf")
    d = a[ok] - b[ok]
    return float(np.sqrt(np.mean(d * d)))


def _folds(n):
    idx = np.arange(n)
    return ((idx % 2 == 0, idx % 2 == 1), (idx % 2 == 1, idx % 2 == 0))


# ----------------------------------------------------------------------
# Estimators: fit(K, C, S0, r, T, q, F, K_price, K_dens, theta) -> (C_hat, q_hat)
# ----------------------------------------------------------------------


def fit_ours(K, C, S0, r, T, q, F, K_price, K_dens, theta):
    """The paper's estimator. ``theta`` is None for the rule, or a bandwidth."""
    if theta is None:
        out = estimate_rnd(K, C, S0, r, T, q, K_eval=K_dens, K_price=K_price)
    else:
        out = estimate_rnd(K, C, S0, r, T, q, K_eval=K_dens, K_price=K_price, h=float(theta))
    return out["C"], out["q"], out


def fit_pca(K, C, S0, r, T, q, F, K_price, K_dens, theta):
    C_hat, q_hat, *_ = pca_fit(K, C, r, T, F, float(theta), K_price, K_dens)
    return C_hat, q_hat, None


def fit_asl(K, C, S0, r, T, q, F, K_price, K_dens, theta):
    """theta is in strike units; the moneyness bandwidth is theta / F."""
    C_hat, q_hat = asl_fit(K, C, S0, r, T, q, F, K_price, K_dens, float(theta) / float(F))
    return C_hat, q_hat, None


def fit_asd(K, C, S0, r, T, q, F, K_price, K_dens, theta):
    C_hat, q_hat, _ = asd_fit(K, C, S0, r, T, q, F, K_price, K_dens, float(theta))
    return C_hat, q_hat, None


def fit_ghs_iv(K, C, S0, r, T, q, F, K_price, K_dens, theta):
    C_hat, q_hat = ghs_iv_fit(K, C, S0, r, T, q, F, K_price, K_dens, float(theta))
    return C_hat, q_hat, None


def fit_ghs_call(K, C, S0, r, T, q, F, K_price, K_dens, theta):
    C_hat, q_hat = ghs_call_fit(K, C, S0, r, T, q, F, K_price, K_dens, float(theta))
    return C_hat, q_hat, None


FITS = {
    "ours": fit_ours,
    "pca": fit_pca,
    "asl": fit_asl,
    "asd": fit_asd,
    "ghs_iv": fit_ghs_iv,
    "ghs_call": fit_ghs_call,
}


# ----------------------------------------------------------------------
# Tuning: tune(K, C, S0, r, T, q, F) -> theta
# ----------------------------------------------------------------------


def tune_cv(fit, floor=None):
    """Even/odd out-of-the-money pricing error on the common grid.

    ``floor(K)`` is a method-specific lower bound on the bandwidth below which
    the method is not defined as its authors intend it; PCA needs a kernel at
    least as wide as its node spacing, which is the strike range over sixty
    once the node count is capped.
    """

    def _tune(K, C, S0, r, T, q, F):
        K = np.asarray(K, dtype=float)
        C = np.asarray(C, dtype=float)
        grid = common_grid(K, C, S0, r, T, q, F)
        if floor is not None:
            lo = float(floor(K))
            grid = grid[grid >= lo] if np.any(grid >= lo) else np.array([lo])
        best_h, best = float(grid[0]), np.inf
        for h in grid:
            errs = []
            for train, test in _folds(len(K)):
                if int(train.sum()) < 6 or int(test.sum()) < 2:
                    continue
                Ct = projected_calls(K[train], C[train], r, T, F)
                try:
                    C_te, _, _ = fit(K[train], Ct, S0, r, T, q, F, K[test], K[test][:1], float(h))
                except Exception:  # a width the method cannot use
                    errs.append(np.inf)
                    continue
                errs.append(otm_rmse(K[test], C_te, C[test], S0, r, T, q, F))
            if errs and float(np.mean(errs)) < best:
                best = float(np.mean(errs))
                best_h = float(h)
        return best_h

    return _tune


def tune_rule_ours(K, C, S0, r, T, q, F):
    return None


def tune_asd_plugin(K, C, S0, r, T, q, F):
    return asd_bandwidth(K, C, r, T, F)


def tune_asl_appendix(K, C, S0, r, T, q, F):
    return asl_bandwidth(K, F) * float(F)


def tune_ghs_loo_iv(mult=1.0):
    def _tune(K, C, S0, r, T, q, F):
        sig = _iv_on_calls(np.asarray(K, float), np.asarray(C, float), S0, r, T, q)
        return mult * ghs_loo_bandwidth(K, sig)

    return _tune


def tune_ghs_loo_call(mult=1.0):
    def _tune(K, C, S0, r, T, q, F):
        return mult * ghs_loo_bandwidth(K, C)

    return _tune


def tune_rule_h(scale=1.0):
    """The rule bandwidth (16) of our estimator, handed to another method.

    ``scale`` converts a Gaussian standard deviation into the method's own
    bandwidth unit: the quartic kernel on [-h, h] has standard deviation
    h / sqrt(7), so its half-width at equal spread is sqrt(7) h.
    """

    def _tune(K, C, S0, r, T, q, F):
        return scale * estimate_rnd(K, C, S0, r, T, q, K_eval=np.asarray(K, float)[:1])["h"]

    return _tune


# name -> (fit key, tune, label, group)
METHODS = {
    "ours_rule": ("ours", tune_rule_ours, "Ours, rule", "main"),
    "ours_cv": ("ours", tune_cv(fit_ours), "Ours, CV", "main"),
    "pca_cv": ("pca", tune_cv(fit_pca, floor=lambda K: (K[-1] - K[0]) / 60.0), "PCA, CV", "main"),
    "asl_cv": ("asl", tune_cv(fit_asl), "AS--Lo, CV", "main"),
    "asd_cv": ("asd", tune_cv(fit_asd), "AS--Duarte, CV", "main"),
    "ghs_iv_cv": ("ghs_iv", tune_cv(fit_ghs_iv), "GHS IV, CV", "main"),
    "ghs_call_cv": ("ghs_call", tune_cv(fit_ghs_call), "GHS call, CV", "main"),
    "asl_app": ("asl", tune_asl_appendix, "AS--Lo, App.~A", "rule"),
    "asd_plug": ("asd", tune_asd_plugin, "AS--Duarte, plug-in", "rule"),
    "ghs_iv_loo": ("ghs_iv", tune_ghs_loo_iv(1.0), "GHS IV, LOO", "rule"),
    "ghs_iv_loo2": ("ghs_iv", tune_ghs_loo_iv(2.0), "GHS IV, 2$\\times$LOO", "rule"),
    "ghs_call_loo": ("ghs_call", tune_ghs_loo_call(1.0), "GHS call, LOO", "rule"),
    "ghs_call_loo2": ("ghs_call", tune_ghs_loo_call(2.0), "GHS call, 2$\\times$LOO", "rule"),
    "pca_ruleh": ("pca", tune_rule_h(1.0), "PCA, at (16)", "transfer"),
    "asl_ruleh": ("asl", tune_rule_h(1.0), "AS--Lo, at (16)", "transfer"),
    "asd_ruleh": ("asd", tune_rule_h(1.0), "AS--Duarte, at (16)", "transfer"),
    "ghs_iv_ruleh": ("ghs_iv", tune_rule_h(np.sqrt(7.0)), "GHS IV, at (16)", "transfer"),
    "ghs_call_ruleh": ("ghs_call", tune_rule_h(np.sqrt(7.0)), "GHS call, at (16)", "transfer"),
}
TRANSFER = [k for k, v in METHODS.items() if v[3] == "transfer"]
MAIN = [k for k, v in METHODS.items() if v[3] == "main"]
RULE = [k for k, v in METHODS.items() if v[3] == "rule"]


# ----------------------------------------------------------------------
# Density diagnostics
# ----------------------------------------------------------------------


def density_grid(F):
    return np.linspace(max(50.0, 0.2 * float(F)), 2.4 * float(F), 1601)


def density_stats(F, q, s):
    q = np.nan_to_num(np.asarray(q, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    pos = np.maximum(q, 0.0)
    mass = float(np.trapezoid(pos, s))
    neg = float(-np.trapezoid(np.minimum(q, 0.0), s))
    core = (s >= WINDOW_LO * F) & (s <= WINDOW_HI * F)
    y = pos[core]
    peaks = 0
    if y.size > 4 and y.max() > 0:
        peaks = int(np.sum((y[1:-1] > y[:-2]) & (y[1:-1] > y[2:]) & (y[1:-1] > 0.2 * y.max())))
    tv = clipped_variation(y)
    mean = float(np.trapezoid(s * pos, s) / mass) if mass > 0 else float("nan")
    return dict(mass=mass, neg=neg, peaks=peaks, tv=float(tv), mean=mean, min=float(q[core].min()))


# ----------------------------------------------------------------------
# Listed chains
# ----------------------------------------------------------------------


def _errors(sl, K, C_hat, test=None):
    """Point, half-spread and implied-volatility errors on the strikes ``K``."""
    idx = np.searchsorted(sl.K, K)
    assert np.allclose(sl.K[idx], K)
    hat = otm_premiums(K, C_hat, sl.S0, sl.r, sl.T, sl.q, sl.F)
    mkt = np.where(K <= sl.F, sl.P[idx], sl.C[idx])
    e = hat - mkt
    half = 0.5 * sl.spread[idx] if sl.spread is not None else np.full(K.shape, np.nan)
    is_c = K > sl.F
    iv_hat = np.empty_like(K)
    iv_mkt = np.empty_like(K)
    iv_hat[is_c] = bs.implied_vol(hat[is_c], sl.S0, K[is_c], sl.r, sl.T, sl.q, True)
    iv_hat[~is_c] = bs.implied_vol(hat[~is_c], sl.S0, K[~is_c], sl.r, sl.T, sl.q, False)
    iv_mkt[is_c] = bs.implied_vol(mkt[is_c], sl.S0, K[is_c], sl.r, sl.T, sl.q, True)
    iv_mkt[~is_c] = bs.implied_vol(mkt[~is_c], sl.S0, K[~is_c], sl.r, sl.T, sl.q, False)
    div = np.where(np.isfinite(iv_hat) & np.isfinite(iv_mkt), iv_hat - iv_mkt, np.nan)
    # A fitted premium so far off that no implied volatility exists counts as the cap.
    div = np.where(np.isfinite(div), div, np.where(np.isfinite(iv_mkt), 1.0, np.nan))
    return e, e / half, div


def _rms(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return float(np.sqrt(np.mean(x * x))) if x.size else float("nan")


def holdout(sl, name):
    """Nested even/odd hold-out: tune on the training half, score the other."""
    key, tune, _, _ = METHODS[name]
    fit = FITS[key]
    out = {"pts": [], "spread": [], "iv": [], "pts_int": [], "spread_int": [], "iv_int": [], "theta": []}
    for train, test in _folds(len(sl.K)):
        Kt = sl.K[train]
        Ct = projected_calls(Kt, sl.C[train], sl.r, sl.T, sl.F)
        theta = tune(Kt, Ct, sl.S0, sl.r, sl.T, sl.q, sl.F)
        Ke = sl.K[test]
        C_te, _, _ = fit(Kt, Ct, sl.S0, sl.r, sl.T, sl.q, sl.F, Ke, Ke[:1], theta)
        e, es, ev = _errors(sl, Ke, C_te)
        interior = (Ke > Kt.min()) & (Ke < Kt.max())
        out["pts"].append(_rms(e))
        out["spread"].append(_rms(es))
        out["iv"].append(_rms(ev))
        out["pts_int"].append(_rms(e[interior]))
        out["spread_int"].append(_rms(es[interior]))
        out["iv_int"].append(_rms(ev[interior]))
        out["theta"].append(None if theta is None else float(theta))
    return {k: (float(np.mean(v)) if k != "theta" else v) for k, v in out.items()}


def score_chain(sl, names, want_curves=False):
    C = projected_calls(sl.K, sl.C, sl.r, sl.T, sl.F)
    s = density_grid(sl.F)
    rows = {}
    curves = {}
    for name in names:
        t0 = time.time()
        key, tune, label, _ = METHODS[name]
        fit = FITS[key]
        theta = tune(sl.K, C, sl.S0, sl.r, sl.T, sl.q, sl.F)
        C_hat, q_hat, extra = fit(sl.K, C, sl.S0, sl.r, sl.T, sl.q, sl.F, sl.K, s, theta)
        e, es, ev = _errors(sl, sl.K, C_hat)
        row = dict(
            label=label,
            theta=None if theta is None else float(theta),
            h=float(extra["h"]) if extra is not None else (None if theta is None else float(theta)),
            c=float(extra["c"]) if extra is not None else None,
            otm_pts=_rms(e),
            otm_spread=_rms(es),
            otm_iv=_rms(ev),
            puts_pts=_rms(e[sl.K <= sl.F]),
            calls_pts=_rms(e[sl.K > sl.F]),
        )
        row.update(density_stats(sl.F, q_hat, s))
        row.update({"ho_" + k: v for k, v in holdout(sl, name).items()})
        row["seconds"] = time.time() - t0
        rows[name] = row
        if want_curves:
            curves[name] = dict(C=np.asarray(C_hat, float), q=np.asarray(q_hat, float))
        print(
            f"  {label:22} h={row['h'] if row['h'] is not None else float('nan'):9.4g} "
            f"OTM {row['otm_pts']:.3f} ({row['otm_spread']:.2f} hs, {1e4*row['otm_iv']:.0f} bp) "
            f"HO {row['ho_pts']:.3f} ({row['ho_spread']:.2f} hs, {1e4*row['ho_iv']:.0f} bp) "
            f"int {row['ho_spread_int']:.2f} hs mass {row['mass']:.2f} neg {row['neg']:.3f} "
            f"peaks {row['peaks']} tv {row['tv']:.2f} [{row['seconds']:.0f}s]",
            flush=True,
        )
    meta = dict(
        root=sl.root, expiry=sl.expiry.isoformat(), asof=sl.asof, S0=sl.S0, F=sl.F, T=sl.T, q=sl.q,
        n=int(len(sl.K)), puts=int(np.sum(sl.K <= sl.F)), sigma=atm_sigma(sl.K, C, sl.S0, sl.r, sl.T, sl.q, sl.F),
        median_half_spread=float(np.median(0.5 * sl.spread)) if sl.spread is not None else None,
        K_lo=float(sl.K.min() / sl.F), K_hi=float(sl.K.max() / sl.F),
    )
    if want_curves:
        return meta, rows, s, curves
    return meta, rows


LISTED = (
    ("spx_20261218.csv", "SPX Dec 2026"),
    ("spx_20270319.csv", "SPX Mar 2027"),
    ("ndx_20261218.csv", "NDX Dec 2026"),
    ("rut_20261218.csv", "RUT Dec 2026"),
)


def run_listed(names=None, out="listed.json"):
    names = list(METHODS) if names is None else names
    result = {}
    for csv, title in LISTED:
        sl = load_slice(RES / csv)
        print(f"\n{title}  n={len(sl.K)}  F={sl.F:.1f}")
        meta, rows = score_chain(sl, names)
        result[title] = dict(meta=meta, rows=rows)
        (OUT / out).write_text(json.dumps(result, indent=1))
    return result


def _all_monthly_chains():
    """Every remaining standard-monthly chain in the four snapshots."""
    from datetime import date

    from src.spx import asof_from_raw, build_otm_slice, fetch_cboe

    paper = {("SPX", date(2026, 12, 18)), ("SPX", date(2027, 3, 19)),
             ("NDX", date(2026, 12, 18)), ("RUT", date(2026, 12, 18))}
    chains = []
    for symbol, name in (("SPX", "cboe_spx.json"), ("NDX", "cboe_ndx.json"),
                         ("RUT", "cboe_rut.json"), ("DJX", "cboe_djx.json")):
        raw = fetch_cboe(RES / name, symbol)
        asof = asof_from_raw(raw)
        expiries = set()
        for opt in raw["data"]["options"]:
            occ = opt["option"]
            if len(occ) <= len(symbol) or not occ.startswith(symbol):
                continue
            if not occ[len(symbol)].isdigit():
                continue
            yy = int(occ[len(symbol):len(symbol) + 2])
            mm = int(occ[len(symbol) + 2:len(symbol) + 4])
            dd = int(occ[len(symbol) + 4:len(symbol) + 6])
            expiries.add(date(2000 + yy, mm, dd))
        for exp in sorted(expiries):
            if (symbol, exp) in paper:
                continue
            if (exp - asof).days < 14 or (exp - asof).days / 365.25 > 2.0:
                continue
            try:
                sl = build_otm_slice(raw, exp, r=0.04, root=symbol, asof=asof)
            except RuntimeError:
                continue
            if len(sl.K) < 40:
                continue
            chains.append(sl)
    return chains


def run_extra(names=None):
    names = MAIN if names is None else names
    chains = _all_monthly_chains()
    print(f"{len(chains)} further chains")
    result = {}
    for sl in chains:
        title = f"{sl.root} {sl.expiry.isoformat()}"
        print(f"\n{title}  n={len(sl.K)}  F={sl.F:.1f}  T={sl.T:.3f}")
        meta, rows = score_chain(sl, names)
        result[title] = dict(meta=meta, rows=rows)
        (OUT / "extra.json").write_text(json.dumps(result, indent=1))
    return result


# ----------------------------------------------------------------------
# Simulation designs
# ----------------------------------------------------------------------


def models():
    from src.bates import BCC97_SVJ, bates_calls, bates_spot_density
    from src.heston import BCC97, carr_madan_calls, heston_spot_density
    from src.vg import CM99, vg_calls, vg_spot_density

    return {
        "heston": (BCC97, lambda K: carr_madan_calls(K, BCC97), lambda K: heston_spot_density(K, BCC97)),
        "vg": (CM99, lambda K: vg_calls(K, CM99), lambda K: vg_spot_density(K, CM99)),
        "bates": (BCC97_SVJ, lambda K: bates_calls(K, BCC97_SVJ), lambda K: bates_spot_density(K, BCC97_SVJ)),
    }


def listed_mesh(F=100.0):
    """The SPX December mesh rescaled to a forward of 100.

    Strikes every 0.5 percent of the forward near the money, coarser in the
    wings, as on the listed chain: 5 points at F = 7836 is 0.064 percent, but
    the quoted mesh widens quickly, and this mesh keeps the shape rather than
    the count.
    """
    sl = load_slice(RES / "spx_20261218.csv")
    K = sl.K / sl.F * F
    K = K[(K >= 40.0) & (K <= 170.0)]
    # Thin to roughly 120 strikes while keeping the relative spacing pattern.
    step = max(1, int(round(len(K) / 120.0)))
    return np.unique(np.round(K[::step], 4))


CHAINS = {
    "dense": np.linspace(30.0, 220.0, 256),
    "sparse": np.linspace(70.0, 140.0, 32),
    "listed": listed_mesh(),
}


def iv_noise(K, C, S0, r, T, q, rng, sd):
    iv = np.asarray(bs.implied_vol(C, S0, K, r, T, q, True), float)
    med = np.nanmedian(iv[np.isfinite(iv)]) if np.any(np.isfinite(iv)) else 0.2
    iv = np.where(np.isfinite(iv), iv, med)
    iv = np.clip(iv + rng.normal(0.0, sd, size=iv.shape), 0.02, 2.5)
    return bs.call_price(S0, K, r, T, iv, q)


def ise(x, q, qt):
    return float(np.trapezoid((np.nan_to_num(q) - qt) ** 2, x))


def run_exact(names=None):
    """Exact quotes: ISE of the nested estimators at the oracle and rule bandwidths, and of the competitors."""
    names = list(METHODS) if names is None else names
    K_eval = np.linspace(60.0, 150.0, 401)
    result = {}
    for model, (p, calls, dens) in models().items():
        q_true = dens(K_eval)
        for chain, K in CHAINS.items():
            C = np.maximum(calls(K), 0.0)
            F = p.forward
            tag = f"{model}/{chain}"
            print(f"\n{tag}  n={len(K)}")
            rec = {}
            delta = float(np.median(np.diff(K)))
            hs = np.geomspace(max(0.2 * delta, 1e-3), max(30.0 * delta, 40.0), 36)
            for name, tails, kernel in (("quoted", False, "plain"), ("tails", True, "plain"),
                                        ("twice", True, "twice"), ("thrice", True, "thrice")):
                def _q(h):
                    return estimate_rnd(K, C, p.S0, p.r, p.T, p.q, K_eval=K_eval, h=float(h),
                                        tails=tails, kernel=kernel, c=0)["q"]
                best, best_h = np.inf, hs[0]
                for h in hs:
                    err = ise(K_eval, _q(h), q_true)
                    if err < best:
                        best, best_h = err, float(h)
                h9 = estimate_rnd(K, C, p.S0, p.r, p.T, p.q, K_eval=K_eval[:1], tails=tails, kernel=kernel, c=0)["h"]
                rec[name] = dict(h_star=best_h, ise_star=best, h_9=h9, ise_9=ise(K_eval, _q(h9), q_true))
                print(f"  {name:8} oracle h={best_h:.3f} ISE={best:.3e}   rule h={h9:.3f} ISE={rec[name]['ise_9']:.3e}")
            Cp = projected_calls(K, C, p.r, p.T, F)
            for name in names:
                key, tune, label, _ = METHODS[name]
                theta = tune(K, Cp, p.S0, p.r, p.T, p.q, F)
                _, q_hat, extra = FITS[key](K, Cp, p.S0, p.r, p.T, p.q, F, K[:1], K_eval, theta)
                h = extra["h"] if extra is not None else theta
                rec[name] = dict(h=h, ise=ise(K_eval, q_hat, q_true), c=(extra["c"] if extra else None))
                print(f"  {label:22} h={h:.3f} ISE={rec[name]['ise']:.3e}")
            result[tag] = rec
            (OUT / "exact.json").write_text(json.dumps(result, indent=1))
    return result


def mc_one(model, chain, sd, rep, names):
    """One replication. Returns ISE by method, plus the diagnostics of ours."""
    p, calls, dens = models()[model]
    K = CHAINS[chain]
    K_eval = np.linspace(60.0, 150.0, 401)
    q_true = dens(K_eval)
    rng = np.random.default_rng([20260923, rep, {"heston": 1, "vg": 2, "bates": 3}[model],
                                 {"dense": 1, "sparse": 2, "listed": 3}[chain], int(round(sd * 1e4))])
    C_true = np.maximum(calls(K), 0.0)
    C_obs = iv_noise(K, C_true, p.S0, p.r, p.T, p.q, rng, sd)
    Cp = projected_calls(K, C_obs, p.r, p.T, p.forward)
    out = {}
    for name in names:
        key, tune, label, _ = METHODS[name]
        try:
            theta = tune(K, Cp, p.S0, p.r, p.T, p.q, p.forward)
            _, q_hat, extra = FITS[key](K, Cp, p.S0, p.r, p.T, p.q, p.forward, K[:1], K_eval, theta)
            rec = dict(ise=ise(K_eval, q_hat, q_true), h=(extra["h"] if extra else float(theta)))
            if extra is not None:
                rec["c"] = extra["c"]
                rec["neg"] = float(-np.trapezoid(np.minimum(q_hat, 0.0), K_eval))
        except Exception as exc:  # record the failure rather than stop the study
            rec = dict(ise=float("nan"), h=float("nan"), error=repr(exc))
        out[name] = rec
    return out


def run_mc(n_reps=200, sds=(0.005, 0.01, 0.02), names=None, workers=2, chains=("dense", "sparse", "listed"),
           model_names=("heston", "vg", "bates")):
    from concurrent.futures import ProcessPoolExecutor

    names = MAIN + ["asl_app", "asd_plug", "ghs_iv_loo", "ghs_call_loo"] if names is None else names
    path = OUT / "mc.json"
    result = json.loads(path.read_text()) if path.exists() else {}
    for model in model_names:
        for chain in chains:
            for sd in sds:
                tag = f"{model}/{chain}/{sd}"
                have = result.get(tag, {}).get("reps", [])
                missing = [n for n in names if have and n not in have[0]]
                if len(have) >= n_reps and not missing:
                    print(f"{tag}: done")
                    continue
                t0 = time.time()
                if missing and have:
                    # Same seeds, so the new methods see the same draws as the old ones.
                    todo = list(range(min(len(have), n_reps)))
                    with ProcessPoolExecutor(max_workers=workers) as ex:
                        extra = list(ex.map(mc_one, [model] * len(todo), [chain] * len(todo), [sd] * len(todo), todo,
                                            [missing] * len(todo), chunksize=4))
                    for old, new in zip(have, extra):
                        old.update(new)
                    names_all = list(have[0])
                else:
                    names_all = names
                todo = list(range(len(have), n_reps))
                with ProcessPoolExecutor(max_workers=workers) as ex:
                    reps = list(ex.map(mc_one, [model] * len(todo), [chain] * len(todo), [sd] * len(todo), todo,
                                       [names_all] * len(todo), chunksize=4))
                have = have + reps
                result[tag] = dict(model=model, chain=chain, sd=sd, reps=have)
                path.write_text(json.dumps(result))
                print(f"{tag}: {len(have)} reps in {time.time()-t0:.0f}s", flush=True)
                for name in names:
                    v = np.array([r[name]["ise"] for r in have if name in r], float)
                    v = v[np.isfinite(v)]
                    print(f"   {METHODS[name][2]:22} mean ISE {v.mean():.3e}  se {v.std(ddof=1)/np.sqrt(v.size):.1e}  median {np.median(v):.3e}")
    return result


def mc_oracle_one(model, chain, sd, rep, keys):
    """Infeasible oracle: for each method, the grid bandwidth that minimizes the ISE on this draw."""
    p, calls, dens = models()[model]
    K = CHAINS[chain]
    K_eval = np.linspace(60.0, 150.0, 401)
    q_true = dens(K_eval)
    rng = np.random.default_rng([20260923, rep, {"heston": 1, "vg": 2, "bates": 3}[model],
                                 {"dense": 1, "sparse": 2, "listed": 3}[chain], int(round(sd * 1e4))])
    C_true = np.maximum(calls(K), 0.0)
    C_obs = iv_noise(K, C_true, p.S0, p.r, p.T, p.q, rng, sd)
    Cp = projected_calls(K, C_obs, p.r, p.T, p.forward)
    grid = common_grid(K, Cp, p.S0, p.r, p.T, p.q, p.forward)
    out = {}
    for key in keys:
        fit = FITS[key]
        best, best_h = np.inf, float("nan")
        for h in grid:
            try:
                _, q_hat, _ = fit(K, Cp, p.S0, p.r, p.T, p.q, p.forward, K[:1], K_eval, float(h))
            except Exception:
                continue
            err = ise(K_eval, q_hat, q_true)
            if err < best:
                best, best_h = err, float(h)
        out[key] = dict(ise=best, h=best_h)
    return out


def run_mc_oracle(n_reps=100, sd=0.01, workers=2, chains=("dense", "sparse", "listed"),
                  model_names=("heston", "vg", "bates")):
    from concurrent.futures import ProcessPoolExecutor

    keys = list(FITS)
    path = OUT / "mc_oracle.json"
    result = json.loads(path.read_text()) if path.exists() else {}
    for model in model_names:
        for chain in chains:
            tag = f"{model}/{chain}/{sd}"
            if len(result.get(tag, {}).get("reps", [])) >= n_reps:
                continue
            t0 = time.time()
            with ProcessPoolExecutor(max_workers=workers) as ex:
                reps = list(ex.map(mc_oracle_one, [model] * n_reps, [chain] * n_reps, [sd] * n_reps,
                                   range(n_reps), [keys] * n_reps, chunksize=4))
            result[tag] = dict(model=model, chain=chain, sd=sd, reps=reps)
            path.write_text(json.dumps(result))
            print(f"{tag}: {n_reps} oracle reps in {time.time()-t0:.0f}s", flush=True)
            for key in keys:
                v = np.array([r[key]["ise"] for r in reps], float)
                v = v[np.isfinite(v)]
                print(f"   {key:10} oracle mean ISE {v.mean():.3e}  se {v.std(ddof=1)/np.sqrt(v.size):.1e}")
    return result


# ----------------------------------------------------------------------
# Sensitivity of the rule
# ----------------------------------------------------------------------


def run_sensitivity(consts=(0.20, 0.27, 0.34, 0.45, 0.60), n_reps=100, sd=0.01):
    """Hold-out on the listed chains and MC ISE as the bandwidth constant moves; penalty on/off."""
    result = {"listed": {}, "mc": {}}
    for csv, title in LISTED:
        sl = load_slice(RES / csv)
        result["listed"][title] = {}
        for const in consts:
            for pen in ("rule", "zero"):
                def tune(K, C, S0, r, T, q, F, const=const):
                    return estimate_rnd(K, C, S0, r, T, q, K_eval=K[:1], deriv_c=const)["h"]

                def fit(K, C, S0, r, T, q, F, K_price, K_dens, theta, pen=pen):
                    out = estimate_rnd(K, C, S0, r, T, q, K_eval=K_dens, K_price=K_price, h=float(theta),
                                       c=("rule" if pen == "rule" else 0))
                    return out["C"], out["q"], out

                errs = []
                for train, test in _folds(len(sl.K)):
                    Kt = sl.K[train]
                    Ct = projected_calls(Kt, sl.C[train], sl.r, sl.T, sl.F)
                    theta = tune(Kt, Ct, sl.S0, sl.r, sl.T, sl.q, sl.F)
                    C_te, _, _ = fit(Kt, Ct, sl.S0, sl.r, sl.T, sl.q, sl.F, sl.K[test], sl.K[test][:1], theta)
                    e, es, ev = _errors(sl, sl.K[test], C_te)
                    errs.append((_rms(e), _rms(es), _rms(ev)))
                C = projected_calls(sl.K, sl.C, sl.r, sl.T, sl.F)
                s = density_grid(sl.F)
                full = estimate_rnd(sl.K, C, sl.S0, sl.r, sl.T, sl.q, K_eval=s, K_price=sl.K, deriv_c=const,
                                    c=("rule" if pen == "rule" else 0))
                stats = density_stats(sl.F, full["q"], s)
                key = f"{const}/{pen}"
                result["listed"][title][key] = dict(
                    ho_pts=float(np.mean([x[0] for x in errs])), ho_spread=float(np.mean([x[1] for x in errs])),
                    ho_iv=float(np.mean([x[2] for x in errs])), h=full["h"], c=full["c"], **stats,
                )
                print(f"{title} c0={const} {pen}: HO {result['listed'][title][key]['ho_spread']:.3f} hs "
                      f"peaks {stats['peaks']} tv {stats['tv']:.3f} neg {stats['neg']:.3f}", flush=True)
    K_eval = np.linspace(60.0, 150.0, 401)
    for model, (p, calls, dens) in models().items():
        q_true = dens(K_eval)
        for chain in ("dense", "sparse", "listed"):
            K = CHAINS[chain]
            C_true = np.maximum(calls(K), 0.0)
            acc = {}
            for rep in range(n_reps):
                rng = np.random.default_rng([777, rep, hash(model) % 1000, hash(chain) % 1000])
                Cp = projected_calls(K, iv_noise(K, C_true, p.S0, p.r, p.T, p.q, rng, sd), p.r, p.T, p.forward)
                for const in consts:
                    for pen in ("rule", "zero"):
                        out = estimate_rnd(K, Cp, p.S0, p.r, p.T, p.q, K_eval=K_eval, deriv_c=const,
                                           c=("rule" if pen == "rule" else 0))
                        acc.setdefault(f"{const}/{pen}", []).append(ise(K_eval, out["q"], q_true))
            tag = f"{model}/{chain}"
            result["mc"][tag] = {k: dict(mean=float(np.mean(v)), se=float(np.std(v, ddof=1) / np.sqrt(len(v))))
                                 for k, v in acc.items()}
            print(tag, {k: f"{v['mean']:.2e}" for k, v in result["mc"][tag].items()}, flush=True)
    (OUT / "sensitivity.json").write_text(json.dumps(result, indent=1))
    return result


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "listed"
    if what == "listed":
        run_listed()
    elif what == "extra":
        run_extra()
    elif what == "exact":
        run_exact()
    elif what == "mc":
        run_mc()
    elif what == "sens":
        run_sensitivity()
    elif what == "oracle":
        run_mc_oracle()
    elif what == "transfer":
        run_listed(names=TRANSFER, out="listed_transfer.json")
        run_mc(n_reps=100, sds=(0.01,), names=TRANSFER, chains=("dense", "sparse", "listed"))
    elif what == "figures":
        pass
    else:
        raise SystemExit(f"unknown target {what}")


# ----------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------


def _exact_figure(path, model, title):
    import matplotlib.pyplot as plt

    p, calls, dens = models()[model]
    K_eval = np.linspace(60.0, 150.0, 401)
    q_true = dens(K_eval)
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6), sharey=True)
    for ax, chain in zip(axes, ("dense", "sparse")):
        K = CHAINS[chain]
        C = np.maximum(calls(K), 0.0)
        ax.plot(K_eval, q_true, color="black", lw=1.8, label=title, zorder=2)
        for name, tails, kernel, color, ls, lw in (
            ("Quoted spline", False, "plain", "#7f7f7f", ":", 1.15),
            ("Tails", True, "plain", "#d62728", "--", 1.15),
            ("Tails+Twice", True, "twice", "#ff7f0e", "-.", 1.2),
            ("Tails+Thrice", True, "thrice", "#1f77b4", "-", 1.35),
        ):
            q = estimate_rnd(K, C, p.S0, p.r, p.T, p.q, K_eval=K_eval, tails=tails, kernel=kernel, c=0)["q"]
            ax.plot(K_eval, np.maximum(q, 0), color=color, lw=lw, ls=ls, label=name)
        if chain == "sparse":
            ax.plot(K, np.interp(K, K_eval, q_true), linestyle="none", marker="o", ms=3.4,
                    mfc="white", mec="black", mew=0.7, zorder=7)
        ax.set_xlim(60, 150)
        ax.set_ylim(0.0, 1.18 * float(np.nanmax(q_true)))
        ax.set_xlabel(r"Strike $K$")
        ax.set_title("Dense, $m=256$" if chain == "dense" else "Sparse, $m=32$")
        ax.legend(frameon=False)
    axes[0].set_ylabel(r"$f_{\mathbb{Q}}(K)$")
    fig.tight_layout(w_pad=2.4)
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    print("wrote", path.name)


def run_figures():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "serif", "font.size": 10, "axes.labelsize": 11, "legend.fontsize": 7,
        "figure.dpi": 140, "savefig.dpi": 300, "savefig.bbox": "tight",
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
        "grid.alpha": 0.25, "grid.linestyle": "--",
    })
    FIG = ROOT / "figures"
    FIG.mkdir(exist_ok=True)
    show = [
        ("ours_rule", "Ours, rule", "#1f77b4", "-", 1.5),
        ("pca_cv", "PCA, CV", "#2ca02c", "-.", 1.1),
        ("ghs_iv_cv", "GHS IV, CV", "#17becf", (0, (4, 1.5)), 1.1),
        ("asd_plug", "AS--Duarte, plug-in", "#8c564b", "--", 1.1),
        ("ghs_iv_loo2", "GHS IV, 2$\\times$LOO", "#d62728", ":", 1.0),
    ]
    fig, axes = plt.subplots(3, 2, figsize=(9.6, 8.4))
    for row, (csv, title) in enumerate((("spx_20261218.csv", "SPX Dec"), ("ndx_20261218.csv", "NDX Dec"),
                                        ("rut_20261218.csv", "RUT Dec"))):
        sl = load_slice(RES / csv)
        C = projected_calls(sl.K, sl.C, sl.r, sl.T, sl.F)
        s = density_grid(sl.F)
        curves = {}
        for name, *_ in show:
            key, tune, _, _ = METHODS[name]
            theta = tune(sl.K, C, sl.S0, sl.r, sl.T, sl.q, sl.F)
            C_hat, q_hat, _ = FITS[key](sl.K, C, sl.S0, sl.r, sl.T, sl.q, sl.F, sl.K, s, theta)
            curves[name] = (np.asarray(C_hat, float), np.asarray(q_hat, float))
        ax = axes[row, 0]
        core = (s >= 0.65 * sl.F) & (s <= 1.30 * sl.F)
        ymax = 1.15 * max(float(np.nanmax(np.maximum(curves[n][1][core], 0))) for n in ("ours_rule", "pca_cv"))
        for name, label, color, ls, lw in show:
            ax.plot(s, np.maximum(curves[name][1], 0), color=color, ls=ls, lw=lw, label=label.replace("--", "–"))
        ax.axvline(sl.F, color="0.45", ls="--", lw=0.8)
        ax.set_xlim(0.55 * sl.F, 1.40 * sl.F)
        ax.set_ylim(0, ymax)
        ax.set_title(title)
        ax.set_xlabel(r"Strike $K$")
        ax.legend(frameon=False, loc="upper right")
        if row == 0:
            ax.set_ylabel(r"$f_{\mathbb{Q}}(K)$")
        ax = axes[row, 1]
        is_c = sl.K > sl.F
        mkt = np.where(is_c, sl.C, sl.P)

        def iv_of(prem):
            iv = np.empty_like(sl.K)
            iv[is_c] = bs.implied_vol(prem[is_c], sl.S0, sl.K[is_c], sl.r, sl.T, sl.q, True)
            iv[~is_c] = bs.implied_vol(prem[~is_c], sl.S0, sl.K[~is_c], sl.r, sl.T, sl.q, False)
            return iv

        iv_m = iv_of(mkt)
        lo, hi = 0.55 * sl.F, 1.40 * sl.F
        sel = np.isfinite(iv_m) & (sl.K >= lo) & (sl.K <= hi)
        ax.plot(sl.K[sel], iv_m[sel], "k.", ms=2.6, alpha=0.4, label="Market", zorder=2)
        for name, label, color, ls, lw in show:
            prem = otm_premiums(sl.K, curves[name][0], sl.S0, sl.r, sl.T, sl.q, sl.F)
            iv = iv_of(prem)
            ax.plot(sl.K[sel], iv[sel], color=color, ls=ls, lw=lw, label=label.replace("--", "–"))
        y = iv_m[sel]
        ax.set_ylim(max(0.02, np.percentile(y, 1) - 0.05), np.percentile(y, 99) + 0.08)
        ax.set_xlim(lo, hi)
        ax.axvline(sl.F, color="0.45", ls="--", lw=0.8)
        ax.set_xlabel(r"Strike $K$")
        ax.legend(frameon=False, loc="upper right")
        if row == 0:
            ax.set_ylabel("OTM implied vol")
    fig.tight_layout()
    fig.savefig(FIG / "ccdf_listed.pdf", facecolor="white")
    plt.close(fig)
    print("wrote ccdf_listed.pdf")
    _exact_figure(FIG / "ccdf_heston_rnd.pdf", "heston", "Heston")
    _exact_figure(FIG / "ccdf_vg_rnd.pdf", "vg", "Variance gamma")


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "figures":
    run_figures()
