"""Replication for the paper.

Run from the repository root::

    python3 python/replicate.py

Prints the Heston and variance-gamma ISE tables and the listed pricing table,
and writes figures/ccdf_heston_rnd.pdf, figures/ccdf_vg_rnd.pdf,
figures/ccdf_listed.pdf, and figures/ccdf_listed_kern.pdf.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PY = Path(__file__).resolve().parent
ROOT = PY.parent
sys.path.insert(0, str(PY))

from rnd import estimate_rnd
from src import black_scholes as bs
from src.competitors import (
    asd_cv_bandwidth,
    asd_fit,
    asl_fit,
    ghs_call_fit,
    ghs_iv_fit,
    kernel_price_cv,
    pca_cv_bandwidth,
    _second_diff_q,
    pca_fit,
    priestley_chao_cubic,
    shape_fit_calls,
    yatchew_cv_lambda,
    yatchew_fit,
)
from src.heston import BCC97, carr_madan_puts, heston_spot_density
from src.spx import load_slice
from src.vg import CM99, vg_calls, vg_spot_density

RES = ROOT / "python" / "results"
FIG = ROOT / "figures"
FIG.mkdir(parents=True, exist_ok=True)


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


def _projected_calls(K, C, r, T, F):
    """λ=0 decreasing convex call. Shared input of every listed estimator."""
    K = np.asarray(K, dtype=float)
    C = np.asarray(C, dtype=float)
    disc = float(np.exp(-float(r) * float(T)))
    intrinsic = disc * np.maximum(float(F) - K, 0.0)
    m = shape_fit_calls(K, C, disc, lam=0.0, intrinsic=intrinsic)
    return np.maximum(np.asarray(m, dtype=float), 0.0)


def _parity(sl, C):
    """Floored call and the parity put, also floored."""
    C = np.maximum(np.asarray(C, dtype=float), 0.0)
    P = np.maximum(C - sl.S0 * np.exp(-sl.q * sl.T) + sl.K * sl.disc, 0.0)
    return P, C


def _otm_iv(sl, P, C):
    is_c = sl.K > sl.F
    iv = np.empty_like(sl.K, dtype=float)
    iv[is_c] = bs.implied_vol(C[is_c], sl.S0, sl.K[is_c], sl.r, sl.T, sl.q, True)
    iv[~is_c] = bs.implied_vol(P[~is_c], sl.S0, sl.K[~is_c], sl.r, sl.T, sl.q, False)
    return iv


def _ise(x, q, qt):
    return float(np.trapezoid((np.nan_to_num(q) - qt) ** 2, x))


def _exact(label, p, calls, q_true, K_eval):
    print(f"\n{label}")
    rec = {}
    plot = {}
    for chain, K in (
        ("dense", np.linspace(30.0, 220.0, 256)),
        ("sparse", np.linspace(70.0, 140.0, 32)),
    ):
        C = np.maximum(calls(K), 0.0)
        delta = float(np.median(np.diff(K)))
        hs = np.geomspace(max(0.2 * delta, 1e-3), max(30.0 * delta, 40.0), 36)
        print(f"  -- {chain}")
        rec[chain] = {}
        specs = (
            ("Quoted spline", False, False),
            ("Tails", True, False),
            ("Tails+Thrice", True, True),
        )
        for name, tails, higher in specs:
            best, best_h = np.inf, hs[len(hs) // 2]
            for h in hs:
                q = estimate_rnd(
                    K, C, p.S0, p.r, p.T, p.q, K_eval=K_eval,
                    h=float(h), tails=tails, higher=higher,
                )["q"]
                err = _ise(K_eval, q, q_true)
                if err < best:
                    best, best_h = err, float(h)
            fit_m = estimate_rnd(
                K, C, p.S0, p.r, p.T, p.q, K_eval=K_eval,
                h="mesh", tails=tails, higher=higher,
            )
            fit_9 = estimate_rnd(
                K, C, p.S0, p.r, p.T, p.q, K_eval=K_eval,
                h="deriv", tails=tails, higher=higher,
            )
            q_m, q_9 = fit_m["q"], fit_9["q"]
            rec[chain][name] = dict(
                h_star=best_h, ise_star=best,
                h_mesh=fit_m["h"], ise_mesh=_ise(K_eval, q_m, q_true),
                h_9=fit_9["h"], ise_9=_ise(K_eval, q_9, q_true),
            )
            row = rec[chain][name]
            print(
                f"    {name:16} oracle {row['h_star']:.4g} {row['ise_star']:.4e}  "
                f"mesh {row['h_mesh']:.4g} {row['ise_mesh']:.4e}  "
                f"n^-1/9 {row['h_9']:.4g} {row['ise_9']:.4e}"
            )
            plot.setdefault(chain, {})[{"Quoted spline": "quoted", "Tails": "tails", "Tails+Thrice": "higher"}[name]] = q_m
            plot[chain]["K"] = K
    return rec, plot, K_eval, q_true


def _figure_exact(path, K_eval, q_true, plot, title):
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6), sharey=True)
    for ax, chain in zip(axes, ("dense", "sparse")):
        ax.plot(K_eval, q_true, color="black", lw=1.8, label=title, zorder=2)
        ax.plot(K_eval, np.maximum(plot[chain]["quoted"], 0), color="#7f7f7f", lw=1.15, ls=":", label="Quoted spline", zorder=3)
        ax.plot(K_eval, np.maximum(plot[chain]["tails"], 0), color="#d62728", lw=1.15, ls="--", label="Tails", zorder=4)
        ax.plot(K_eval, np.maximum(plot[chain]["higher"], 0), color="#1f77b4", lw=1.35, label="Tails+Thrice", zorder=5)
        if chain == "sparse":
            ax.plot(
                plot[chain]["K"], np.interp(plot[chain]["K"], K_eval, q_true),
                linestyle="none", marker="o", ms=3.4, mfc="white", mec="black", mew=0.7, zorder=6,
            )
        ax.set_xlim(60, 150)
        ax.set_ylim(0.0, 1.18 * float(np.nanmax(q_true)))
        ax.set_xlabel(r"Strike $K$")
        ax.set_title("Dense, $m=256$" if chain == "dense" else "Sparse, $m=32$")
        ax.legend(frameon=False)
    axes[0].set_ylabel(r"$f_{\mathbb{Q}}(K)$")
    fig.tight_layout(w_pad=2.4)
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    print(" wrote", path.name)


def _otm(sl, C_hat):
    C_hat = np.maximum(np.asarray(C_hat, float), 0.0)
    stock = sl.S0 * np.exp(-sl.q * sl.T)
    disc = np.exp(-sl.r * sl.T)
    P = np.maximum(C_hat - stock + sl.K * disc, 0.0)
    mkt = np.where(sl.K <= sl.F, sl.P, sl.C)
    hat = np.where(sl.K <= sl.F, P, C_hat)
    e = hat - mkt
    left, right = sl.K <= sl.F, sl.K > sl.F

    def r(m):
        return float(np.sqrt(np.mean(e[m] ** 2)))

    return r(np.ones(len(e), bool)), r(left), r(right)


def _variation(F, q, s):
    """Total variation of max(q, 0) on [0.55F, 1.40F], divided by twice the maximum.

    A unimodal curve scores 1. Scaling by the forward cancels, so the score is
    the same in strike units and in moneyness.
    """
    m = (s >= 0.55 * F) & (s <= 1.40 * F)
    p = np.maximum(np.nan_to_num(np.asarray(q, float)[m]), 0.0)
    if p.size < 3 or float(p.max()) <= 0.0:
        return float("nan")
    return float(np.sum(np.abs(np.diff(p))) / (2.0 * float(p.max())))


def _mass_peaks(F, q, s):
    qq = np.maximum(np.nan_to_num(q), 0.0)
    mass = float(np.trapezoid(qq, s))
    core = (s >= 0.55 * F) & (s <= 1.40 * F)
    y = qq[core]
    peaks = 0
    if y.size > 4 and np.nanmax(y) > 0:
        peaks = int(np.sum((y[1:-1] > y[:-2]) & (y[1:-1] > y[2:]) & (y[1:-1] > 0.2 * y.max())))
    return mass, peaks


def _folds(n):
    idx = np.arange(n)
    return ((idx % 2 == 0, idx % 2 == 1), (idx % 2 == 1, idx % 2 == 0))


def _otm_rmse_slice(sl, C_hat, test):
    C_hat = np.maximum(np.asarray(C_hat, float), 0.0)
    P = np.maximum(C_hat - sl.S0 * np.exp(-sl.q * sl.T) + sl.K[test] * sl.disc, 0.0)
    hat = np.where(sl.K[test] <= sl.F, P, C_hat)
    mkt = np.where(sl.K[test] <= sl.F, sl.P[test], sl.C[test])
    e = hat - mkt
    return float(np.sqrt(np.mean(e ** 2)))


def _holdout_method(sl, method):
    """Even/odd hold-out. Tuning uses only the training strikes."""
    errs = []
    for train, test in _folds(len(sl.K)):
        Kt, Ct = sl.K[train], _projected_calls(sl.K[train], sl.C[train], sl.r, sl.T, sl.F)
        Ke = sl.K[test]
        if method == "ours":
            C_te = estimate_rnd(
                Kt, Ct, sl.S0, sl.r, sl.T, sl.q, K_price=Ke, h="deriv",
                tails=True, higher=True,
            )["C"]
        elif method == "yh":
            lam = yatchew_cv_lambda(Kt, Ct, sl.S0, sl.r, sl.T, sl.q, sl.F)
            C_te, _ = yatchew_fit(Kt, Ct, sl.S0, sl.r, sl.T, sl.q, sl.F, Ke, lam)
        elif method == "asd":
            h_p, h_d = asd_cv_bandwidth(Kt, Ct, sl.S0, sl.r, sl.T, sl.q, sl.F)
            C_te, _, _ = asd_fit(Kt, Ct, sl.S0, sl.r, sl.T, sl.q, sl.F, Ke, Ke[:1], h_p, h_d)
        elif method == "pca":
            h = pca_cv_bandwidth(Kt, Ct, sl.S0, sl.r, sl.T, sl.q, sl.F)
            C_te, _, _, _, _ = pca_fit(Kt, Ct, sl.r, sl.T, sl.F, h, Ke, Ke[:1])
        elif method == "pc":
            h = _pc_cubic_cv(sl, Kt, Ct)
            _, _, C_te = priestley_chao_cubic(
                Kt, Ct, sl.S0, sl.r, sl.T, Ke, q=sl.q, h=h, return_call=True
            )
        elif method in ("asl", "ghs_call", "ghs_iv"):
            fit = {"asl": asl_fit, "ghs_call": ghs_call_fit, "ghs_iv": ghs_iv_fit}[method]
            h = kernel_price_cv(method, Kt, Ct, sl.S0, sl.r, sl.T, sl.q, sl.F)
            C_te, _ = fit(Kt, Ct, sl.S0, sl.r, sl.T, sl.q, sl.F, Ke, Ke[:1], h)
        else:
            raise ValueError(method)
        errs.append(_otm_rmse_slice(sl, C_te, test))
    return float(np.mean(errs))


def _pc_cubic_cv(sl, K, C):
    """Even/odd pricing bandwidth for the Priestley--Chao cubic.

    The grid starts at the median strike gap. On these chains the minimum is selected.
    """
    from src.competitors import _pc_h_grid

    grid = _pc_h_grid(K, C, sl.S0, sl.r, sl.T, sl.q, sl.F)
    best_h, best = float(grid[0]), np.inf
    idx = np.arange(len(K))
    for h in grid:
        errs = []
        for tr, te in ((idx % 2 == 0, idx % 2 == 1), (idx % 2 == 1, idx % 2 == 0)):
            if int(tr.sum()) < 6 or int(te.sum()) < 2:
                continue
            _, _, C_te = priestley_chao_cubic(
                K[tr], C[tr], sl.S0, sl.r, sl.T, K[te], q=sl.q, h=float(h), return_call=True
            )
            C_te = np.maximum(C_te, 0.0)
            P_te = np.maximum(C_te - sl.S0 * np.exp(-sl.q * sl.T) + K[te] * sl.disc, 0.0)
            hat = np.where(K[te] <= sl.F, P_te, C_te)
            # Quoted mids live on the full slice; training knots are a subset.
            pos = np.searchsorted(sl.K, K[te])
            mkt = np.where(K[te] <= sl.F, sl.P[pos], sl.C[pos])
            errs.append(float(np.sqrt(np.mean((hat - mkt) ** 2))))
        if errs and float(np.mean(errs)) < best:
            best = float(np.mean(errs))
            best_h = float(h)
    return best_h


def _row_metrics(sl, C_hat, q, s, ho):
    o, pu, ca = _otm(sl, C_hat)
    mass, _ = _mass_peaks(sl.F, q, s)
    return np.array([o, pu, ca, ho, mass], dtype=float)


_METHODS = (
    "Ours",
    "Aït-Sahalia–Duarte",
    "Aït-Sahalia–Lo",
    "GHS call",
    "GHS IV",
    "PCA",
)


def _print_metrics(name, m):
    print(
        f"  {name:22} OTM {m[0]:.3f} puts {m[1]:.3f} calls {m[2]:.3f} "
        f"hold {m[3]:.3f} mass {m[4]:.2f}"
    )


def _chain_scores(sl):
    """In-sample prices, densities, and even/odd hold-out for Table 4."""
    C = _projected_calls(sl.K, sl.C, sl.r, sl.T, sl.F)
    s = np.linspace(max(50.0, 0.2 * sl.F), 2.4 * sl.F, 1601)
    fit = estimate_rnd(
        sl.K, C, sl.S0, sl.r, sl.T, sl.q, K_eval=s, K_price=sl.K, h="deriv",
        tails=True, higher=True,
    )
    h = fit["h"]
    h_asd, h_asd_d = asd_cv_bandwidth(sl.K, C, sl.S0, sl.r, sl.T, sl.q, sl.F)
    C_asd, q_asd, _ = asd_fit(
        sl.K, C, sl.S0, sl.r, sl.T, sl.q, sl.F, sl.K, s, h_asd, h_asd_d
    )
    h_pca = pca_cv_bandwidth(sl.K, C, sl.S0, sl.r, sl.T, sl.q, sl.F)
    C_pca, q_pca, _, _, _ = pca_fit(sl.K, C, sl.r, sl.T, sl.F, h_pca, sl.K, s)
    h_asl = kernel_price_cv("asl", sl.K, C, sl.S0, sl.r, sl.T, sl.q, sl.F)
    C_asl, q_asl = asl_fit(sl.K, C, sl.S0, sl.r, sl.T, sl.q, sl.F, sl.K, s, h_asl)
    h_ghs_c = kernel_price_cv("ghs_call", sl.K, C, sl.S0, sl.r, sl.T, sl.q, sl.F)
    C_ghs_c, q_ghs_c = ghs_call_fit(sl.K, C, sl.S0, sl.r, sl.T, sl.q, sl.F, sl.K, s, h_ghs_c)
    h_ghs_i = kernel_price_cv("ghs_iv", sl.K, C, sl.S0, sl.r, sl.T, sl.q, sl.F)
    C_ghs_i, q_ghs_i = ghs_iv_fit(sl.K, C, sl.S0, sl.r, sl.T, sl.q, sl.F, sl.K, s, h_ghs_i)
    calls = {
        "Ours": fit["C"],
        "Aït-Sahalia–Duarte": C_asd,
        "Aït-Sahalia–Lo": C_asl,
        "GHS call": C_ghs_c,
        "GHS IV": C_ghs_i,
        "PCA": C_pca,
    }
    dens = {
        "Ours": fit["q"],
        "Aït-Sahalia–Duarte": q_asd,
        "Aït-Sahalia–Lo": q_asl,
        "GHS call": q_ghs_c,
        "GHS IV": q_ghs_i,
        "PCA": q_pca,
    }
    keys = {
        "Ours": "ours",
        "Aït-Sahalia–Duarte": "asd",
        "Aït-Sahalia–Lo": "asl",
        "GHS call": "ghs_call",
        "GHS IV": "ghs_iv",
        "PCA": "pca",
    }
    metrics = {
        name: _row_metrics(sl, calls[name], dens[name], s, _holdout_method(sl, keys[name]))
        for name in _METHODS
    }
    peaks = {name: _mass_peaks(sl.F, dens[name], s)[1] for name in _METHODS}
    tv = {name: _variation(sl.F, dens[name], s) for name in _METHODS}
    bundle = dict(
        C=C, fit=fit, h=h, s=s, q_asd=q_asd, q_pca=q_pca,
        q_asl=q_asl, q_ghs_i=q_ghs_i, q_ghs_c=q_ghs_c,
        C_pca=C_pca, C_asl=C_asl, C_ghs_c=C_ghs_c, C_ghs_i=C_ghs_i,
        h_asd_d=h_asd_d, h_pca=h_pca,
        h_asl=h_asl, h_ghs_c=h_ghs_c, h_ghs_i=h_ghs_i, peaks=peaks, tv=tv,
    )
    return metrics, bundle


def _average_rows(chain_metrics, forwards):
    names = _METHODS
    raw, inv = {}, {}
    w = (1.0 / forwards) / np.sum(1.0 / forwards)
    for name in names:
        stack = np.vstack([chain_metrics[title][name] for title in chain_metrics])
        raw[name] = stack.mean(axis=0)
        inv[name] = (stack * w[:, None]).sum(axis=0)
    return raw, inv, w


def listed():
    print("\nListed")
    chain_metrics = {}
    forwards = []
    titles = []
    scored = []
    for csv, title in (
        ("spx_20261218.csv", "SPX Dec"),
        ("spx_20270319.csv", "SPX Mar"),
        ("ndx_20261218.csv", "NDX Dec"),
        ("rut_20261218.csv", "RUT Dec"),
    ):
        sl = load_slice(RES / csv)
        metrics, bundle = _chain_scores(sl)
        chain_metrics[title] = metrics
        forwards.append(sl.F)
        titles.append(title)
        print(
            f"  {title:8} F={sl.F:.1f} h={bundle['h']:.2f} "
            f"ASL {bundle['h_asl']:.4g} GHSc {bundle['h_ghs_c']:.4g} "
            f"GHSi {bundle['h_ghs_i']:.4g}"
        )
        for name in _METHODS:
            peaks = bundle["peaks"][name]
            tv = bundle["tv"][name]
            m = metrics[name]
            print(
                f"  {name:22} OTM {m[0]:.3f} puts {m[1]:.3f} calls {m[2]:.3f} "
                f"hold {m[3]:.3f} mass {m[4]:.2f} peaks {peaks} tv {tv:.2f}"
            )
        scored.append((title, sl, bundle))
    raw, inv, w = _average_rows(chain_metrics, np.asarray(forwards, float))
    print("  weights 1/F " + " ".join(f"{t} {wi:.3f}" for t, wi in zip(titles, w)))
    print("  Average")
    for name in _METHODS:
        _print_metrics(name, raw[name])
    print("  Weighed av.")
    for name in _METHODS:
        _print_metrics(name, inv[name])
    _plot_listed(scored)
    return chain_metrics, raw, inv


def _iv_ylim(columns):
    cols = []
    for v in columns:
        v = np.asarray(v, float)
        v = v[np.isfinite(v)]
        if v.size:
            cols.append(v)
    if not cols:
        return 0.08, 0.55
    lo = float(np.percentile(cols[0], 1))
    hi = float(np.percentile(cols[0], 99))
    span = max(hi - lo, 0.04)
    lo -= 0.08 * span
    hi += 0.12 * span
    for v in cols[1:]:
        a = float(np.percentile(v, 5))
        b = float(np.percentile(v, 95))
        lo = min(lo, max(a, lo - 0.2 * span))
        hi = max(hi, min(b, hi + 0.25 * span))
    return max(0.02, lo), hi


def _plot_listed(scored):
    want = {"SPX Dec", "NDX Dec", "RUT Dec"}
    _style()
    fig, axes = plt.subplots(3, 2, figsize=(9.6, 8.4))
    for row, (title, sl, bundle) in enumerate(x for x in scored if x[0] in want):
        fit, h, s = bundle["fit"], bundle["h"], bundle["s"]
        q_asd, q_pca, q_ghs_c = bundle["q_asd"], bundle["q_pca"], bundle["q_ghs_c"]
        C_pca, C_ghs_c = bundle["C_pca"], bundle["C_ghs_c"]
        C_asl = bundle["C_asl"]
        print(
            f"  fig {title}: ASD h={bundle['h_asd_d']:.1f}  PCA h={bundle['h_pca']:.4f}  "
            f"ASL h={bundle['h_asl']:.1f}  GHS IV h={bundle['h_ghs_i']:.1f}"
        )
        ax = axes[row, 0]
        core = (s >= 0.65 * sl.F) & (s <= 1.30 * sl.F)
        ymax = 1.15 * np.nanmax(np.maximum(fit["q"][core], 0))
        h_ours, = ax.plot(s, np.maximum(fit["q"], 0), color="#1f77b4", lw=1.5, label="Ours", zorder=5)
        h_pca, = ax.plot(s, np.maximum(q_pca, 0), color="#2ca02c", lw=1.15, ls="-.", label="PCA", zorder=4)
        h_ghs_c, = ax.plot(s, np.maximum(q_ghs_c, 0), color="#ff7f0e", lw=1.15, label="GHS, call", zorder=4)
        h_asd, = ax.plot(s, np.maximum(q_asd, 0), color="#8c564b", lw=1.15, ls="--", label="Aït-Sahalia–Duarte", zorder=3)
        ax.axvline(sl.F, color="0.45", ls="--", lw=0.8)
        ax.set_xlim(0.55 * sl.F, 1.40 * sl.F)
        ax.set_ylim(0, ymax)
        ax.set_title(title)
        ax.set_xlabel(r"Strike $K$")
        ax.legend(
            [h_ours, h_pca, h_ghs_c, h_asd],
            ["Ours", "PCA", "GHS, call", "Aït-Sahalia–Duarte"],
            frameon=False, fontsize=7.5, loc="upper right",
        )
        if row == 0:
            ax.set_ylabel(r"$f_{\mathbb{Q}}(K)$")

        P_c, C_c = _parity(sl, fit["C"])
        P_pca, C_pca = _parity(sl, C_pca)
        P_asl, C_asl = _parity(sl, C_asl)
        P_ghs_c, C_ghs_c = _parity(sl, C_ghs_c)
        iv = _otm_iv(sl, P_c, C_c)
        iv_pca = _otm_iv(sl, P_pca, C_pca)
        iv_asl = _otm_iv(sl, P_asl, C_asl)
        iv_ghs_c = _otm_iv(sl, P_ghs_c, C_ghs_c)
        iv_m = _otm_iv(sl, sl.P, sl.C)
        lo, hi = 0.55 * sl.F, 1.40 * sl.F
        show = np.isfinite(iv_m) & (sl.K >= lo) & (sl.K <= hi)
        ax = axes[row, 1]
        ax.set_xlim(lo, hi)
        y1, y2 = _iv_ylim(
            [iv_m[show], iv[show], iv_pca[show], iv_asl[show], iv_ghs_c[show]]
        )
        ax.set_ylim(y1, y2)
        h_mkt, = ax.plot(sl.K[show], iv_m[show], "k.", ms=2.6, alpha=0.40, label="Market", zorder=2)
        h_asl, = ax.plot(sl.K[show], iv_asl[show], color="#d62728", lw=1.0, ls="--", label="Aït-Sahalia–Lo", zorder=4)
        h_ghs_c, = ax.plot(sl.K[show], iv_ghs_c[show], color="#ff7f0e", lw=1.15, label="GHS, call", zorder=5)
        h_pca, = ax.plot(sl.K[show], iv_pca[show], color="#2ca02c", lw=1.05, ls="-.", label="PCA", zorder=5)
        h_ours, = ax.plot(sl.K[show], iv[show], color="#1f77b4", lw=1.4, label="Ours", zorder=6)
        ax.axvline(sl.F, color="0.45", ls="--", lw=0.8)
        ax.set_xlabel(r"Strike $K$")
        ax.legend(
            [h_mkt, h_ours, h_ghs_c, h_pca, h_asl],
            ["Market", "Ours", "GHS, call", "PCA", "Aït-Sahalia–Lo"],
            frameon=False, fontsize=6.5, loc="upper right",
        )
        if row == 0:
            ax.set_ylabel("OTM implied vol")
    fig.tight_layout()
    fig.savefig(FIG / "ccdf_listed.pdf", facecolor="white")
    plt.close(fig)
    print(" wrote ccdf_listed.pdf")

    fig, axes = plt.subplots(3, 2, figsize=(9.6, 8.4))
    for row, (title, sl, bundle) in enumerate(x for x in scored if x[0] in want):
        s = bundle["s"]
        fit = bundle["fit"]
        core = (s >= 0.65 * sl.F) & (s <= 1.30 * sl.F)
        ymax = 1.15 * np.nanmax(np.maximum(fit["q"][core], 0))
        ax = axes[row, 0]
        ax.plot(s, np.maximum(bundle["q_asl"], 0), color="#d62728", lw=1.0, ls="--", label="Aït-Sahalia–Lo")
        ax.plot(s, np.maximum(bundle["q_ghs_i"], 0), color="#17becf", lw=1.05, ls="-.", label="GHS, IV")
        ax.axvline(sl.F, color="0.45", ls="--", lw=0.8)
        ax.set_xlim(0.55 * sl.F, 1.40 * sl.F)
        ax.set_ylim(0, ymax)
        ax.set_title(title)
        ax.set_xlabel(r"Strike $K$")
        ax.legend(frameon=False, fontsize=7.5, loc="upper right")
        if row == 0:
            ax.set_ylabel(r"$f_{\mathbb{Q}}(K)$")

        P_asl, C_asl = _parity(sl, bundle["C_asl"])
        P_ghs_i, C_ghs_i = _parity(sl, bundle["C_ghs_i"])
        iv_asl = _otm_iv(sl, P_asl, C_asl)
        iv_ghs_i = _otm_iv(sl, P_ghs_i, C_ghs_i)
        iv_m = _otm_iv(sl, sl.P, sl.C)
        lo, hi = 0.55 * sl.F, 1.40 * sl.F
        show = np.isfinite(iv_m) & (sl.K >= lo) & (sl.K <= hi)
        ax = axes[row, 1]
        y1, y2 = _iv_ylim([iv_m[show], iv_asl[show], iv_ghs_i[show]])
        ax.set_xlim(lo, hi)
        ax.set_ylim(y1, y2)
        ax.plot(sl.K[show], iv_m[show], "k.", ms=2.6, alpha=0.40, label="Market", zorder=2)
        ax.plot(sl.K[show], iv_asl[show], color="#d62728", lw=1.05, ls="--", label="Aït-Sahalia–Lo", zorder=4)
        ax.plot(sl.K[show], iv_ghs_i[show], color="#17becf", lw=1.15, ls="-.", label="GHS, IV", zorder=5)
        ax.axvline(sl.F, color="0.45", ls="--", lw=0.8)
        ax.set_xlabel(r"Strike $K$")
        ax.legend(frameon=False, fontsize=7.5, loc="upper right")
        if row == 0:
            ax.set_ylabel("OTM implied vol")
    fig.tight_layout()
    fig.savefig(FIG / "ccdf_listed_kern.pdf", facecolor="white")
    plt.close(fig)
    print(" wrote ccdf_listed_kern.pdf")


def _iv_noise(K, C, S0, r, T, q, rng, sd=0.01):
    iv = np.asarray(bs.implied_vol(C, S0, K, r, T, q, True), float)
    med = np.nanmedian(iv[np.isfinite(iv)]) if np.any(np.isfinite(iv)) else 0.2
    iv = np.where(np.isfinite(iv), iv, med)
    iv = np.clip(iv + rng.normal(0.0, sd, size=iv.shape), 0.02, 2.5)
    return bs.call_price(S0, K, r, T, iv, q)


def noisy_heston(n_reps=30, seed=20260923, sd=0.01):
    """Mean ISE on Heston quotes with N(0, sd^2) noise in implied volatility."""
    print(f"\nNoisy Heston  reps={n_reps}  iv sd={sd}  seed={seed}")
    rng = np.random.default_rng(seed)
    p = BCC97
    K_eval = np.linspace(60.0, 150.0, 401)
    q_true = heston_spot_density(K_eval, p)
    calls = lambda K: carr_madan_puts(K, p) + p.S0 * np.exp(-p.q * p.T) - K * p.disc
    specs = (
        ("Quoted spline", False, False),
        ("Tails", True, False),
        ("Tails+Thrice", True, True),
    )
    rules = ("mesh", "deriv")
    for chain, K in (
        ("dense", np.linspace(30.0, 220.0, 256)),
        ("sparse", np.linspace(70.0, 140.0, 32)),
    ):
        C_true = np.maximum(calls(K), 0.0)
        acc = {name: {rule: [] for rule in rules} for name, _, _ in specs}
        hs = {name: {rule: [] for rule in rules} for name, _, _ in specs}
        for _rep in range(n_reps):
            C_obs = _iv_noise(K, C_true, p.S0, p.r, p.T, p.q, rng, sd=sd)
            C_proj = _projected_calls(K, C_obs, p.r, p.T, p.forward)
            for name, tails, higher in specs:
                for rule in rules:
                    fit = estimate_rnd(
                        K, C_proj, p.S0, p.r, p.T, p.q, K_eval=K_eval,
                        h=rule, tails=tails, higher=higher,
                    )
                    acc[name][rule].append(_ise(K_eval, fit["q"], q_true))
                    hs[name][rule].append(fit["h"])
        print(f"  -- {chain}")
        for name, _, _ in specs:
            bits = []
            for rule in rules:
                bits.append(
                    f"{rule} h={np.mean(hs[name][rule]):.3g} "
                    f"ISE={np.mean(acc[name][rule]):.4e}"
                )
            print(f"    {name:16} " + "  ".join(bits))


def _kernel_stats(x, q):
    q = np.nan_to_num(np.asarray(q, float), nan=0.0, posinf=0.0, neginf=0.0)
    neg = float(np.trapezoid(np.minimum(q, 0.0), x))
    y = np.maximum(q, 0.0)
    peaks = 0
    if y.size > 4 and np.nanmax(y) > 0:
        peaks = int(np.sum((y[1:-1] > y[:-2]) & (y[1:-1] > y[2:]) & (y[1:-1] > 0.2 * y.max())))
    return neg, peaks


def literature():
    """Aït-Sahalia–Lo, Aït-Sahalia–Duarte, and Grith–Härdle–Schienle.

    The density bandwidth is ``0.9 F σ_ATM √T n^{-1/9}`` (the ASD local-cubic
    rule). The quartic kernel is scaled to the same weight standard deviation.
    Pricing CV is even/odd OTM error on a fixed grid. Priestley–Chao is not
    in this comparison.
    """
    from src.competitors import (
        asl_fit,
        ghs_call_fit,
        ghs_iv_fit,
        kernel_price_cv,
        second_derivative_h,
    )

    fits = {
        "AS-Lo": asl_fit,
        "GHS call": ghs_call_fit,
        "GHS IV": ghs_iv_fit,
    }
    kinds = {"AS-Lo": "asl", "GHS call": "ghs_call", "GHS IV": "ghs_iv"}

    def _one(K, C, S0, r, T, q, F, K_eval, q_true=None, label=""):
        h_rule = second_derivative_h(K, C, S0, r, T, q, F, c=0.9)
        print(f"  {label} n={len(K)} h_rule={h_rule:.4g}")
        rows = {}
        for name, fit in fits.items():
            t0 = time.time()
            h_cv = kernel_price_cv(kinds[name], K, C, S0, r, T, q, F)
            C_cv, q_cv = fit(K, C, S0, r, T, q, F, K, K_eval, h_cv)
            C_ru, q_ru = fit(K, C, S0, r, T, q, F, K, K_eval, h_rule)
            otm_cv = _otm_rmse(K, C_cv, C, S0, r, T, q, F)
            otm_ru = _otm_rmse(K, C_ru, C, S0, r, T, q, F)
            neg_cv, pk_cv = _kernel_stats(K_eval, q_cv)
            neg_ru, pk_ru = _kernel_stats(K_eval, q_ru)
            ise_cv = _ise(K_eval, q_cv, q_true) if q_true is not None else float("nan")
            ise_ru = _ise(K_eval, q_ru, q_true) if q_true is not None else float("nan")
            print(
                f"    {name:8} CV h={h_cv:.4g} OTM {otm_cv:.4g} ISE {ise_cv:.4e} "
                f"peaks {pk_cv} neg {neg_cv:.3e} | "
                f"rule OTM {otm_ru:.4g} ISE {ise_ru:.4e} peaks {pk_ru} neg {neg_ru:.3e} "
                f"[{time.time()-t0:.1f}s]"
            )
            rows[name] = (h_cv, q_cv, h_rule, q_ru, ise_cv, ise_ru)
        # ASD: price at its CV local linear, density at the same rule.
        h_p, h_d = asd_cv_bandwidth(K, C, S0, r, T, q, F)
        C_asd, q_asd, _ = asd_fit(K, C, S0, r, T, q, F, K, K_eval, h_p, h_d)
        otm_asd = _otm_rmse(K, C_asd, C, S0, r, T, q, F)
        neg_a, pk_a = _kernel_stats(K_eval, q_asd)
        ise_a = _ise(K_eval, q_asd, q_true) if q_true is not None else float("nan")
        print(
            f"    {'ASD':8} price h={h_p:.4g} dens h={h_d:.4g} OTM {otm_asd:.4g} "
            f"ISE {ise_a:.4e} peaks {pk_a} neg {neg_a:.3e}"
        )
        for rule in ("mesh", "deriv"):
            fit = estimate_rnd(
                K, C, S0, r, T, q, K_eval=K_eval, K_price=K, h=rule, tails=True, higher=True,
            )
            otm_o = _otm_rmse(K, fit["C"], C, S0, r, T, q, F)
            neg_o, pk_o = _kernel_stats(K_eval, fit["q"])
            ise_o = _ise(K_eval, fit["q"], q_true) if q_true is not None else float("nan")
            print(
                f"    {'Ours':8} {rule:5} h={fit['h']:.4g} OTM {otm_o:.4g} "
                f"ISE {ise_o:.4e} peaks {pk_o} neg {neg_o:.3e}"
            )
        return rows

    import time
    from src.competitors import _otm_rmse

    K_eval = np.linspace(60.0, 150.0, 401)
    print("\nExact")
    for label, model, calls, q_true in (
        ("Heston", BCC97, lambda K: carr_madan_puts(K, BCC97) + BCC97.S0 * np.exp(-BCC97.q * BCC97.T) - K * BCC97.disc, heston_spot_density(K_eval, BCC97)),
        ("VG", CM99, lambda K: vg_calls(K, CM99), vg_spot_density(K_eval, CM99)),
    ):
        print(label)
        q_true = np.asarray(q_true, float)
        for chain, K in (
            ("dense", np.linspace(30.0, 220.0, 256)),
            ("sparse", np.linspace(70.0, 140.0, 32)),
        ):
            C = np.maximum(calls(K), 0.0)
            F = float(model.forward)
            _one(K, C, model.S0, model.r, model.T, model.q, F, K_eval, q_true, chain)

    print("\nNoisy Heston  reps=30  iv sd=0.01  seed=20260923")
    rng = np.random.default_rng(20260923)
    p = BCC97
    q_true = heston_spot_density(K_eval, p)
    calls = lambda K: carr_madan_puts(K, p) + p.S0 * np.exp(-p.q * p.T) - K * p.disc
    names = ("AS-Lo", "GHS call", "GHS IV", "ASD", "Ours mesh", "Ours deriv")
    for chain, K in (
        ("dense", np.linspace(30.0, 220.0, 256)),
        ("sparse", np.linspace(70.0, 140.0, 32)),
    ):
        C_true = np.maximum(calls(K), 0.0)
        acc = {name: [] for name in names}
        acc_cv = {name: [] for name in ("AS-Lo", "GHS call", "GHS IV")}
        for rep in range(30):
            C_obs = _iv_noise(K, C_true, p.S0, p.r, p.T, p.q, rng, sd=0.01)
            C_proj = _projected_calls(K, C_obs, p.r, p.T, p.forward)
            h_rule = second_derivative_h(K, C_proj, p.S0, p.r, p.T, p.q, p.forward, c=0.9)
            for name, fit in fits.items():
                h_cv = kernel_price_cv(kinds[name], K, C_proj, p.S0, p.r, p.T, p.q, p.forward)
                _, q_cv = fit(K, C_proj, p.S0, p.r, p.T, p.q, p.forward, K[:1], K_eval, h_cv)
                _, q_ru = fit(K, C_proj, p.S0, p.r, p.T, p.q, p.forward, K[:1], K_eval, h_rule)
                acc[name].append(_ise(K_eval, q_ru, q_true))
                acc_cv[name].append(_ise(K_eval, q_cv, q_true))
            h_p, h_d = asd_cv_bandwidth(K, C_proj, p.S0, p.r, p.T, p.q, p.forward)
            _, q_asd, _ = asd_fit(K, C_proj, p.S0, p.r, p.T, p.q, p.forward, K[:1], K_eval, h_p, h_d)
            acc["ASD"].append(_ise(K_eval, q_asd, q_true))
            for rule, key in (("mesh", "Ours mesh"), ("deriv", "Ours deriv")):
                fit_o = estimate_rnd(
                    K, C_proj, p.S0, p.r, p.T, p.q, K_eval=K_eval, h=rule, tails=True, higher=True,
                )
                acc[key].append(_ise(K_eval, fit_o["q"], q_true))
            print(f"  {chain} rep {rep+1}/30", flush=True)
        print(f"  -- {chain} mean ISE")
        for name in ("AS-Lo", "GHS call", "GHS IV"):
            print(f"    {name:8} rule {np.mean(acc[name]):.4e}  CV {np.mean(acc_cv[name]):.4e}")
        print(f"    {'ASD':8} {np.mean(acc['ASD']):.4e}")
        print(f"    {'Ours mesh':8} {np.mean(acc['Ours mesh']):.4e}")
        print(f"    {'Ours deriv':8} {np.mean(acc['Ours deriv']):.4e}")

    print("\nListed")
    for csv, title in (
        ("spx_20261218.csv", "SPX Dec"),
        ("spx_20270319.csv", "SPX Mar"),
        ("ndx_20261218.csv", "NDX Dec"),
        ("rut_20261218.csv", "RUT Dec"),
    ):
        sl = load_slice(RES / csv)
        C = _projected_calls(sl.K, sl.C, sl.r, sl.T, sl.F)
        s = np.linspace(max(50.0, 0.2 * sl.F), 2.4 * sl.F, 801)
        h_rule = second_derivative_h(sl.K, C, sl.S0, sl.r, sl.T, sl.q, sl.F, c=0.9)
        print(f"  {title} n={len(sl.K)} h_rule={h_rule:.4g}")

        def _show(tag, h, C_hat, q, hold):
            o, pu, ca = _otm(sl, C_hat)
            mass, peaks = _mass_peaks(sl.F, q, s)
            neg = _kernel_stats(s, q)[0]
            print(
                f"    {tag:16} h={h:.4g} OTM {o:.3f} puts {pu:.3f} calls {ca:.3f} "
                f"hold {hold:.3f} mass {mass:.2f} peaks {peaks} neg {neg:.3e}"
            )

        for name, fit in fits.items():
            h_cv = kernel_price_cv(kinds[name], sl.K, C, sl.S0, sl.r, sl.T, sl.q, sl.F)
            C_cv, q_cv = fit(sl.K, C, sl.S0, sl.r, sl.T, sl.q, sl.F, sl.K, s, h_cv)
            C_ru, q_ru = fit(sl.K, C, sl.S0, sl.r, sl.T, sl.q, sl.F, sl.K, s, h_rule)
            _show(name + " CV", h_cv, C_cv, q_cv, _holdout_kernel(sl, kinds[name], "cv"))
            _show(name + " rule", h_rule, C_ru, q_ru, _holdout_kernel(sl, kinds[name], "rule"))
        h_p, h_d = asd_cv_bandwidth(sl.K, C, sl.S0, sl.r, sl.T, sl.q, sl.F)
        C_asd, q_asd, _ = asd_fit(sl.K, C, sl.S0, sl.r, sl.T, sl.q, sl.F, sl.K, s, h_p, h_d)
        o, pu, ca = _otm(sl, C_asd)
        mass, peaks = _mass_peaks(sl.F, q_asd, s)
        neg = _kernel_stats(s, q_asd)[0]
        ho = _holdout_method(sl, "asd")
        print(
            f"    {'ASD':16} h={h_p:.4g}/{h_d:.4g} OTM {o:.3f} puts {pu:.3f} calls {ca:.3f} "
            f"hold {ho:.3f} mass {mass:.2f} peaks {peaks} neg {neg:.3e}"
        )
        fit_o = estimate_rnd(
            sl.K, C, sl.S0, sl.r, sl.T, sl.q, K_eval=s, K_price=sl.K, h="deriv",
            tails=True, higher=True,
        )
        o, pu, ca = _otm(sl, fit_o["C"])
        mass, peaks = _mass_peaks(sl.F, fit_o["q"], s)
        neg = _kernel_stats(s, fit_o["q"])[0]
        ho = _holdout_method(sl, "ours")
        print(
            f"    {'Ours':16} h={fit_o['h']:.4g} OTM {o:.3f} puts {pu:.3f} calls {ca:.3f} "
            f"hold {ho:.3f} mass {mass:.2f} peaks {peaks} neg {neg:.3e}"
        )


def _holdout_kernel(sl, kind, which):
    from src.competitors import asl_fit, ghs_call_fit, ghs_iv_fit, kernel_price_cv, second_derivative_h

    fit = {"asl": asl_fit, "ghs_call": ghs_call_fit, "ghs_iv": ghs_iv_fit}[kind]
    errs = []
    for train, test in _folds(len(sl.K)):
        Kt = sl.K[train]
        Ct = _projected_calls(sl.K[train], sl.C[train], sl.r, sl.T, sl.F)
        Ke = sl.K[test]
        if which == "cv":
            h = kernel_price_cv(kind, Kt, Ct, sl.S0, sl.r, sl.T, sl.q, sl.F)
        else:
            h = second_derivative_h(Kt, Ct, sl.S0, sl.r, sl.T, sl.q, sl.F, c=0.9)
        C_te, _ = fit(Kt, Ct, sl.S0, sl.r, sl.T, sl.q, sl.F, Ke, Ke[:1], h)
        errs.append(_otm_rmse_slice(sl, C_te, test))
    return float(np.mean(errs))


def main():
    K_eval = np.linspace(60.0, 150.0, 401)
    p = BCC97
    # carr_madan_puts returns puts; put-call parity recovers the calls.
    rec_h, plot_h, _, qh = _exact(
        "Heston", p, lambda K: carr_madan_puts(K, p) + p.S0 * np.exp(-p.q * p.T) - K * p.disc,
        heston_spot_density(K_eval, p), K_eval,
    )
    _figure_exact(FIG / "ccdf_heston_rnd.pdf", K_eval, qh, plot_h, "Heston")
    v = CM99
    rec_v, plot_v, _, qv = _exact("VG", v, lambda K: vg_calls(K, v), vg_spot_density(K_eval, v), K_eval)
    _figure_exact(FIG / "ccdf_vg_rnd.pdf", K_eval, qv, plot_v, "Variance gamma")
    listed()
    print("DONE")


if __name__ == "__main__":
    main()
