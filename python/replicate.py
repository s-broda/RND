"""Replication for the paper.

Run from the repository root::

    python3 python/replicate.py

Prints the Heston and variance-gamma ISE tables, including the listed
competitors and a noisy Heston comparison, the listed pricing table, and the
wall-clock times of the three convolutions at penalty zero (raw seconds
and the paper's half-up seconds). The appendix chains are not part of this
run: the quotes cannot be redistributed. ``estimate_rnd(..., method=)``
selects the convolution: ``"naive"`` sums the closed-form cell integrals,
``"fft"`` is one real FFT of the sampled spline, and ``"fast"`` (the
default) is one FFT of the order-eight box moments in ``density_closed.c``,
with thricing as one multiplier. If that file cannot be compiled,
``"fast"`` uses the sampled-spline FFT. The script writes
figures/ccdf_heston_rnd.pdf, figures/ccdf_vg_rnd.pdf,
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

from rnd import (
    _Phi,
    _convolve_slope,
    _knots,
    _right_wing,
    _smooth,
    clipped_variation,
    estimate_rnd,
    TV_CAP,
)
from src import black_scholes as bs
from src.competitors import (
    asd_bandwidth,
    asd_fit,
    asl_bandwidth,
    asl_fit,
    ghs_call_fit,
    ghs_iv_fit,
    ghs_loo_bandwidth,
    _iv_on_calls,
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
            ("Tails+Twice", True, "twice"),
            ("Tails+Thrice", True, True),
        )
        for name, tails, higher in specs:
            def _q(h, tails=tails, higher=higher):
                if higher == "twice":
                    q1 = estimate_rnd(
                        K, C, p.S0, p.r, p.T, p.q, K_eval=K_eval,
                        h=float(h), tails=True, higher=False, c=0,
                    )["q"]
                    q2 = estimate_rnd(
                        K, C, p.S0, p.r, p.T, p.q, K_eval=K_eval,
                        h=float(h) * np.sqrt(2.0), tails=True, higher=False, c=0,
                    )["q"]
                    return 2.0 * q1 - q2
                return estimate_rnd(
                    K, C, p.S0, p.r, p.T, p.q, K_eval=K_eval,
                    h=h, tails=tails, higher=higher, c=0,
                )["q"]
            best, best_h = np.inf, hs[len(hs) // 2]
            for h in hs:
                err = _ise(K_eval, _q(float(h)), q_true)
                if err < best:
                    best, best_h = err, float(h)
            h_9 = estimate_rnd(
                K, C, p.S0, p.r, p.T, p.q, K_eval=K_eval[:1],
                h="deriv", tails=tails, higher=False, c=0,
            )["h"]
            q_9 = _q(h_9)
            rec[chain][name] = dict(h_star=best_h, ise_star=best, h_9=h_9, ise_9=_ise(K_eval, q_9, q_true))
            row = rec[chain][name]
            print(
                f"    {name:16} oracle {row['h_star']:.4g} {row['ise_star']:.4e}  "
                f"n^-1/9 {row['h_9']:.4g} {row['ise_9']:.4e}"
            )
            if name == "Tails+Thrice":
                ruled = estimate_rnd(K, C, p.S0, p.r, p.T, p.q, K_eval=K_eval, h=h_9)
                print(
                    f"    {'rule':16} c={ruled['c']:.3e} "
                    f"ISE={_ise(K_eval, ruled['q'], q_true):.4e}"
                )
            key = {
                "Quoted spline": "quoted",
                "Tails": "tails",
                "Tails+Twice": "twice",
                "Tails+Thrice": "higher",
            }.get(name)
            if key is not None:
                plot.setdefault(chain, {})[key] = q_9
            plot.setdefault(chain, {})["K"] = K
        for name, h, q in _competitor_rows(K, C, p.S0, p.r, p.T, p.q, p.forward, K_eval):
            print(f"    {name:22} h={h:.4g} ISE={_ise(K_eval, q, q_true):.4e}")
    return rec, plot, K_eval, q_true


_ASL_H = np.geomspace(0.01, 0.40, 21)
# Reported total variation and peak counts. The penalty in rnd._shape_scores
# uses this same interval. The Aït-Sahalia–Lo screen stays wider: on SPX March
# the reported interval selects a smaller bandwidth than the published 0.030.
_REPORT_LO, _REPORT_HI = 0.60, 1.30
_ASL_SCREEN_LO, _ASL_SCREEN_HI = 0.55, 1.40


def asl_shape_bandwidth(K, C, S0, r, T, q, F):
    """Smallest moneyness bandwidth that leaves one peak and TV at most 1.05.

    Not the Aït-Sahalia–Lo Appendix A rule. That rule is about one moneyness
    standard deviation on these chains and flattens the smile. The screen is
    scored on [0.55F, 1.40F], not on the interval the tables report.
    """
    K = np.asarray(K, float)
    s = np.linspace(max(50.0, 0.2 * float(F)), 2.4 * float(F), 501)
    chosen = float(_ASL_H[-1])
    for h in _ASL_H:
        _, qhat = asl_fit(K, C, S0, r, T, q, F, K[:1], s, float(h))
        _, peaks = _mass_peaks(F, qhat, s, _ASL_SCREEN_LO, _ASL_SCREEN_HI)
        tv = _variation(F, qhat, s, _ASL_SCREEN_LO, _ASL_SCREEN_HI)
        if peaks <= 1 and tv <= 1.05:
            chosen = float(h)
            break
    return chosen


def _competitor_rows(K, C, S0, r, T, q, F, K_eval):
    """Listed competitors on one chain, each at the bandwidth its paper specifies."""
    C = _projected_calls(K, np.maximum(np.asarray(C, float), 0.0), r, T, F)
    rows = []
    h = pca_cv_bandwidth(K, C, S0, r, T, q, F)
    _, qhat, *_ = pca_fit(K, C, r, T, F, h, K[:1], K_eval)
    rows.append(("PCA", h, qhat))
    sig = _iv_on_calls(K, C, S0, r, T, q)
    h_i = ghs_loo_bandwidth(K, sig)
    for name, hh in (("GHS IV, 2×", 2.0 * h_i), ("GHS IV, CV", h_i)):
        _, qhat = ghs_iv_fit(K, C, S0, r, T, q, F, K[:1], K_eval, hh)
        rows.append((name, hh, qhat))
    h = asd_bandwidth(K, C, r, T, F)
    _, qhat, _ = asd_fit(K, C, S0, r, T, q, F, K[:1], K_eval, h)
    rows.append(("Aït-Sahalia–Duarte", h, qhat))
    h_c = ghs_loo_bandwidth(K, C)
    for name, hh in (("GHS call, 2×", 2.0 * h_c), ("GHS call, CV", h_c)):
        _, qhat = ghs_call_fit(K, C, S0, r, T, q, F, K[:1], K_eval, hh)
        rows.append((name, hh, qhat))
    h = asl_shape_bandwidth(K, C, S0, r, T, q, F)
    _, qhat = asl_fit(K, C, S0, r, T, q, F, K[:1], K_eval, h)
    rows.append(("Aït-Sahalia–Lo", h, qhat))
    return rows


def _figure_exact(path, K_eval, q_true, plot, title):
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6), sharey=True)
    for ax, chain in zip(axes, ("dense", "sparse")):
        ax.plot(K_eval, q_true, color="black", lw=1.8, label=title, zorder=2)
        ax.plot(K_eval, np.maximum(plot[chain]["quoted"], 0), color="#7f7f7f", lw=1.15, ls=":", label="Quoted spline", zorder=3)
        ax.plot(K_eval, np.maximum(plot[chain]["tails"], 0), color="#d62728", lw=1.15, ls="--", label="Tails", zorder=4)
        ax.plot(K_eval, np.maximum(plot[chain]["twice"], 0), color="#ff7f0e", lw=1.2, ls="-.", label="Tails+Twice", zorder=5)
        ax.plot(K_eval, np.maximum(plot[chain]["higher"], 0), color="#1f77b4", lw=1.35, label="Tails+Thrice", zorder=6)
        if chain == "sparse":
            ax.plot(
                plot[chain]["K"], np.interp(plot[chain]["K"], K_eval, q_true),
                linestyle="none", marker="o", ms=3.4, mfc="white", mec="black", mew=0.7, zorder=7,
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


def _variation(F, q, s, lo=_REPORT_LO, hi=_REPORT_HI):
    """Total variation of max(q, 0) on [lo*F, hi*F], over twice the maximum.

    The default interval is [0.60F, 1.30F], the same window as the penalty
    and as the tables. The curve is completed to zero at the endpoints, so a
    unimodal curve scores 1 and the score is at least 1. Scaling by the
    forward cancels, so the score is the same in strike units and in moneyness.
    """
    m = (s >= lo * F) & (s <= hi * F)
    return clipped_variation(np.asarray(q, float)[m])


def _mass_peaks(F, q, s, lo=_REPORT_LO, hi=_REPORT_HI):
    qq = np.maximum(np.nan_to_num(q), 0.0)
    mass = float(np.trapezoid(qq, s))
    core = (s >= lo * F) & (s <= hi * F)
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
            C_te = smoothing_spline_fit(
                Kt, Ct, sl.S0, sl.r, sl.T, sl.q, K_price=Ke,
            )["C"]
        elif method == "yh":
            lam = yatchew_cv_lambda(Kt, Ct, sl.S0, sl.r, sl.T, sl.q, sl.F)
            C_te, _ = yatchew_fit(Kt, Ct, sl.S0, sl.r, sl.T, sl.q, sl.F, Ke, lam)
        elif method == "asd":
            h = asd_bandwidth(Kt, Ct, sl.r, sl.T, sl.F)
            C_te, _, _ = asd_fit(Kt, Ct, sl.S0, sl.r, sl.T, sl.q, sl.F, Ke, Ke[:1], h)
        elif method == "pca":
            h = pca_cv_bandwidth(Kt, Ct, sl.S0, sl.r, sl.T, sl.q, sl.F)
            C_te, _, _, _, _ = pca_fit(Kt, Ct, sl.r, sl.T, sl.F, h, Ke, Ke[:1])
        elif method == "pc":
            h = _pc_cubic_cv(sl, Kt, Ct)
            _, _, C_te = priestley_chao_cubic(
                Kt, Ct, sl.S0, sl.r, sl.T, Ke, q=sl.q, h=h, return_call=True
            )
        elif method == "asl":
            h = asl_shape_bandwidth(Kt, Ct, sl.S0, sl.r, sl.T, sl.q, sl.F)
            C_te, _ = asl_fit(Kt, Ct, sl.S0, sl.r, sl.T, sl.q, sl.F, Ke, Ke[:1], h)
        elif method in ("ghs_call_cv", "ghs_call_2", "ghs_iv_cv", "ghs_iv_2"):
            if method.startswith("ghs_call"):
                h = ghs_loo_bandwidth(Kt, Ct)
                if method.endswith("_2"):
                    h *= 2.0
                C_te, _ = ghs_call_fit(Kt, Ct, sl.S0, sl.r, sl.T, sl.q, sl.F, Ke, Ke[:1], h)
            else:
                sig = _iv_on_calls(Kt, Ct, sl.S0, sl.r, sl.T, sl.q)
                h = ghs_loo_bandwidth(Kt, sig)
                if method.endswith("_2"):
                    h *= 2.0
                C_te, _ = ghs_iv_fit(Kt, Ct, sl.S0, sl.r, sl.T, sl.q, sl.F, Ke, Ke[:1], h)
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
    "PCA",
    "GHS IV, 2×",
    "GHS IV, CV",
    "Aït-Sahalia–Duarte",
    "GHS call, 2×",
    "GHS call, CV",
    "Aït-Sahalia–Lo",
)


def _print_metrics(name, m):
    print(
        f"  {name:22} OTM {m[0]:.3f} puts {m[1]:.3f} calls {m[2]:.3f} "
        f"hold {m[3]:.3f} mass {m[4]:.1f}"
    )


def _chain_scores(sl):
    """In-sample prices, densities, and even/odd hold-out for Table 4."""
    C = _projected_calls(sl.K, sl.C, sl.r, sl.T, sl.F)
    s = np.linspace(max(50.0, 0.2 * sl.F), 2.4 * sl.F, 1601)
    fit = smoothing_spline_fit(
        sl.K, C, sl.S0, sl.r, sl.T, sl.q, K_eval=s, K_price=sl.K,
    )
    h = fit["h"]
    h_asd = asd_bandwidth(sl.K, C, sl.r, sl.T, sl.F)
    C_asd, q_asd, _ = asd_fit(
        sl.K, C, sl.S0, sl.r, sl.T, sl.q, sl.F, sl.K, s, h_asd
    )
    h_pca = pca_cv_bandwidth(sl.K, C, sl.S0, sl.r, sl.T, sl.q, sl.F)
    C_pca, q_pca, _, _, _ = pca_fit(sl.K, C, sl.r, sl.T, sl.F, h_pca, sl.K, s)
    h_asl = asl_shape_bandwidth(sl.K, C, sl.S0, sl.r, sl.T, sl.q, sl.F)
    C_asl, q_asl = asl_fit(sl.K, C, sl.S0, sl.r, sl.T, sl.q, sl.F, sl.K, s, h_asl)
    h_ghs_c = ghs_loo_bandwidth(sl.K, C)
    C_ghs_cv, q_ghs_cv = ghs_call_fit(sl.K, C, sl.S0, sl.r, sl.T, sl.q, sl.F, sl.K, s, h_ghs_c)
    C_ghs_c, q_ghs_c = ghs_call_fit(sl.K, C, sl.S0, sl.r, sl.T, sl.q, sl.F, sl.K, s, 2.0 * h_ghs_c)
    sig = _iv_on_calls(sl.K, C, sl.S0, sl.r, sl.T, sl.q)
    h_ghs_i = ghs_loo_bandwidth(sl.K, sig)
    C_ghs_iv_cv, q_ghs_iv_cv = ghs_iv_fit(sl.K, C, sl.S0, sl.r, sl.T, sl.q, sl.F, sl.K, s, h_ghs_i)
    C_ghs_i, q_ghs_i = ghs_iv_fit(sl.K, C, sl.S0, sl.r, sl.T, sl.q, sl.F, sl.K, s, 2.0 * h_ghs_i)
    calls = {
        "Ours": fit["C"],
        "Aït-Sahalia–Duarte": C_asd,
        "Aït-Sahalia–Lo": C_asl,
        "GHS call, CV": C_ghs_cv,
        "GHS call, 2×": C_ghs_c,
        "GHS IV, CV": C_ghs_iv_cv,
        "GHS IV, 2×": C_ghs_i,
        "PCA": C_pca,
    }
    dens = {
        "Ours": fit["q"],
        "Aït-Sahalia–Duarte": q_asd,
        "Aït-Sahalia–Lo": q_asl,
        "GHS call, CV": q_ghs_cv,
        "GHS call, 2×": q_ghs_c,
        "GHS IV, CV": q_ghs_iv_cv,
        "GHS IV, 2×": q_ghs_i,
        "PCA": q_pca,
    }
    keys = {
        "Ours": "ours",
        "Aït-Sahalia–Duarte": "asd",
        "Aït-Sahalia–Lo": "asl",
        "GHS call, CV": "ghs_call_cv",
        "GHS call, 2×": "ghs_call_2",
        "GHS IV, CV": "ghs_iv_cv",
        "GHS IV, 2×": "ghs_iv_2",
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
        q_ghs_cv=q_ghs_cv, q_ghs_iv_cv=q_ghs_iv_cv,
        C_pca=C_pca, C_asl=C_asl, C_ghs_c=C_ghs_c, C_ghs_i=C_ghs_i,
        C_ghs_cv=C_ghs_cv, C_ghs_iv_cv=C_ghs_iv_cv,
        h_asd=h_asd, h_pca=h_pca,
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
            f"  {title:8} F={sl.F:.1f} h={bundle['h']:.4f} c={bundle['fit']['c']:.6e} "
            f"shape={bundle['fit']['shape']:.4f} "
            f"ASD {bundle['h_asd']:.4g} ASL {bundle['h_asl']:.4g} "
            f"GHSc {bundle['h_ghs_c']:.4g} GHSi {bundle['h_ghs_i']:.4g}"
        )
        for name in _METHODS:
            peaks = bundle["peaks"][name]
            tv = bundle["tv"][name]
            m = metrics[name]
            print(
                f"  {name:22} OTM {m[0]:.3f} puts {m[1]:.3f} calls {m[2]:.3f} "
                f"hold {m[3]:.3f} mass {m[4]:.1f} peaks {peaks} tv {tv:.1f}"
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
        q_asl = bundle["q_asl"]
        C_pca, C_ghs_c = bundle["C_pca"], bundle["C_ghs_c"]
        C_asl = bundle["C_asl"]
        print(
            f"  fig {title}: ASD h={bundle['h_asd']:.1f}  PCA h={bundle['h_pca']:.4f}  "
            f"ASL h={bundle['h_asl']:.3f}  GHS IV h={bundle['h_ghs_i']:.1f}"
        )
        q_ghs_i = bundle["q_ghs_i"]
        ax = axes[row, 0]
        core = (s >= 0.65 * sl.F) & (s <= 1.30 * sl.F)
        base = np.maximum(np.concatenate([fit["q"][core], q_asl[core], q_pca[core]]), 0)
        spike = np.maximum(q_ghs_i[core], 0)
        ymax = 1.15 * max(float(np.nanmax(base)), min(float(np.nanmax(spike)), 1.35 * float(np.nanmax(base))))
        h_ours, = ax.plot(s, np.maximum(fit["q"], 0), color="#1f77b4", lw=1.5, label="Ours", zorder=5)
        h_pca, = ax.plot(s, np.maximum(q_pca, 0), color="#2ca02c", lw=1.15, ls="-.", label="PCA", zorder=4)
        h_ghs_c, = ax.plot(s, np.maximum(q_ghs_c, 0), color="#ff7f0e", lw=1.15, label="GHS, call, 2×", zorder=4)
        h_ghs_i, = ax.plot(s, np.maximum(q_ghs_i, 0), color="#17becf", lw=1.15, ls=(0, (4, 1.5)), label="GHS, IV, 2×", zorder=4)
        h_asd, = ax.plot(s, np.maximum(q_asd, 0), color="#8c564b", lw=1.15, ls="--", label="Aït-Sahalia–Duarte", zorder=3)
        h_asl_d, = ax.plot(s, np.maximum(q_asl, 0), color="#d62728", lw=1.05, ls=":", label="Aït-Sahalia–Lo", zorder=4)
        ax.axvline(sl.F, color="0.45", ls="--", lw=0.8)
        ax.set_xlim(0.55 * sl.F, 1.40 * sl.F)
        ax.set_ylim(0, ymax)
        ax.set_title(title)
        ax.set_xlabel(r"Strike $K$")
        ax.legend(
            [h_ours, h_pca, h_ghs_c, h_ghs_i, h_asd, h_asl_d],
            ["Ours", "PCA", "GHS, call, 2×", "GHS, IV, 2×", "Aït-Sahalia–Duarte", "Aït-Sahalia–Lo"],
            frameon=False, fontsize=6.5, loc="upper right",
        )
        if row == 0:
            ax.set_ylabel(r"$f_{\mathbb{Q}}(K)$")

        P_c, C_c = _parity(sl, fit["C"])
        P_pca, C_pca = _parity(sl, C_pca)
        P_asl, C_asl = _parity(sl, C_asl)
        P_ghs_c, C_ghs_c = _parity(sl, C_ghs_c)
        P_ghs_i, C_ghs_i = _parity(sl, bundle["C_ghs_i"])
        iv = _otm_iv(sl, P_c, C_c)
        iv_pca = _otm_iv(sl, P_pca, C_pca)
        iv_asl = _otm_iv(sl, P_asl, C_asl)
        iv_ghs_c = _otm_iv(sl, P_ghs_c, C_ghs_c)
        iv_ghs_i = _otm_iv(sl, P_ghs_i, C_ghs_i)
        iv_m = _otm_iv(sl, sl.P, sl.C)
        lo, hi = 0.55 * sl.F, 1.40 * sl.F
        show = np.isfinite(iv_m) & (sl.K >= lo) & (sl.K <= hi)
        ax = axes[row, 1]
        ax.set_xlim(lo, hi)
        y1, y2 = _iv_ylim(
            [iv_m[show], iv[show], iv_pca[show], iv_asl[show], iv_ghs_c[show], iv_ghs_i[show]]
        )
        ax.set_ylim(y1, y2)
        h_mkt, = ax.plot(sl.K[show], iv_m[show], "k.", ms=2.6, alpha=0.40, label="Market", zorder=2)
        h_asl, = ax.plot(sl.K[show], iv_asl[show], color="#d62728", lw=1.0, ls=":", label="Aït-Sahalia–Lo", zorder=4)
        h_ghs_c, = ax.plot(sl.K[show], iv_ghs_c[show], color="#ff7f0e", lw=1.15, label="GHS, call, 2×", zorder=5)
        h_ghs_i, = ax.plot(sl.K[show], iv_ghs_i[show], color="#17becf", lw=1.15, ls=(0, (4, 1.5)), label="GHS, IV, 2×", zorder=5)
        h_pca, = ax.plot(sl.K[show], iv_pca[show], color="#2ca02c", lw=1.05, ls="-.", label="PCA", zorder=5)
        h_ours, = ax.plot(sl.K[show], iv[show], color="#1f77b4", lw=1.4, label="Ours", zorder=6)
        ax.axvline(sl.F, color="0.45", ls="--", lw=0.8)
        ax.set_xlabel(r"Strike $K$")
        ax.legend(
            [h_mkt, h_ours, h_ghs_c, h_ghs_i, h_pca, h_asl],
            ["Market", "Ours", "GHS, call, 2×", "GHS, IV, 2×", "PCA", "Aït-Sahalia–Lo"],
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
        base = np.maximum(np.concatenate([
            fit["q"][core], bundle["q_asl"][core], bundle["q_pca"][core],
        ]), 0)
        spike = np.maximum(bundle["q_ghs_i"][core], 0)
        ymax = 1.15 * max(
            float(np.nanmax(base)),
            min(float(np.nanmax(spike)), 1.35 * float(np.nanmax(base))),
        )
        ax = axes[row, 0]
        ax.plot(s, np.maximum(bundle["q_ghs_cv"], 0), color="#ff7f0e", lw=1.0, ls=":", label="GHS call, CV")
        ax.plot(s, np.maximum(bundle["q_ghs_iv_cv"], 0), color="#17becf", lw=1.0, ls="--", label="GHS IV, CV")
        ax.axvline(sl.F, color="0.45", ls="--", lw=0.8)
        ax.set_xlim(0.55 * sl.F, 1.40 * sl.F)
        ax.set_ylim(0, ymax)
        ax.set_title(title)
        ax.set_xlabel(r"Strike $K$")
        ax.legend(frameon=False, fontsize=6.5, loc="upper right")
        if row == 0:
            ax.set_ylabel(r"$f_{\mathbb{Q}}(K)$")

        P_cv, C_cv = _parity(sl, bundle["C_ghs_cv"])
        P_iv, C_iv = _parity(sl, bundle["C_ghs_iv_cv"])
        iv_cv = _otm_iv(sl, P_cv, C_cv)
        iv_iv = _otm_iv(sl, P_iv, C_iv)
        iv_m = _otm_iv(sl, sl.P, sl.C)
        lo, hi = 0.55 * sl.F, 1.40 * sl.F
        show = np.isfinite(iv_m) & (sl.K >= lo) & (sl.K <= hi)
        ax = axes[row, 1]
        y1, y2 = _iv_ylim([iv_m[show], iv_cv[show], iv_iv[show]])
        ax.set_xlim(lo, hi)
        ax.set_ylim(y1, y2)
        ax.plot(sl.K[show], iv_m[show], "k.", ms=2.6, alpha=0.40, label="Market", zorder=2)
        ax.plot(sl.K[show], iv_cv[show], color="#ff7f0e", lw=1.0, ls=":", label="GHS call, CV", zorder=4)
        ax.plot(sl.K[show], iv_iv[show], color="#17becf", lw=1.0, ls="--", label="GHS IV, CV", zorder=5)
        ax.axvline(sl.F, color="0.45", ls="--", lw=0.8)
        ax.set_xlabel(r"Strike $K$")
        ax.legend(frameon=False, fontsize=6.5, loc="upper right")
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
    """Mean ISE on Heston quotes with N(0, sd^2) noise in implied volatility.

    One generator, dense chain then the 32-strike chain, so the two columns
    share the noise stream in the paper.
    """
    print(f"\nNoisy Heston  reps={n_reps}  iv sd={sd}  seed={seed}")
    rng = np.random.default_rng(seed)
    p = BCC97
    K_eval = np.linspace(60.0, 150.0, 401)
    q_true = heston_spot_density(K_eval, p)
    calls = lambda K: carr_madan_puts(K, p) + p.S0 * np.exp(-p.q * p.T) - K * p.disc
    for chain, K in (
        ("dense", np.linspace(30.0, 220.0, 256)),
        ("sparse", np.linspace(70.0, 140.0, 32)),
    ):
        C_true = np.maximum(calls(K), 0.0)
        acc = {}
        for _rep in range(n_reps):
            C_obs = _iv_noise(K, C_true, p.S0, p.r, p.T, p.q, rng, sd=sd)
            C_proj = _projected_calls(K, C_obs, p.r, p.T, p.forward)
            fit = estimate_rnd(K, C_proj, p.S0, p.r, p.T, p.q, K_eval=K_eval)
            acc.setdefault("Ours", []).append(_ise(K_eval, fit["q"], q_true))
            for name, _h, q in _competitor_rows(
                K, C_proj, p.S0, p.r, p.T, p.q, p.forward, K_eval
            ):
                acc.setdefault(name, []).append(_ise(K_eval, q, q_true))
        print(f"  -- {chain}")
        for name, vals in acc.items():
            print(f"    {name:22} {np.mean(vals):.4e}")


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

    Aït-Sahalia–Lo uses the Appendix A moneyness bandwidth. Aït-Sahalia–Duarte
    uses the Fan–Gijbels plug-in for one local linear. The Grith–Härdle–Schienle
    rows in this diagnostic still search an even/odd pricing grid; the listed
    table uses leave-one-out and twice that width.
    """
    from src.competitors import (
        asl_fit,
        ghs_call_fit,
        ghs_iv_fit,
        kernel_price_cv,
        second_derivative_h,
    )

    fits = {
        "GHS call": ghs_call_fit,
        "GHS IV": ghs_iv_fit,
    }
    kinds = {"GHS call": "ghs_call", "GHS IV": "ghs_iv"}

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
        h_asl = asl_bandwidth(K, F)
        C_asl, q_asl = asl_fit(K, C, S0, r, T, q, F, K, K_eval, h_asl)
        otm_asl = _otm_rmse(K, C_asl, C, S0, r, T, q, F)
        neg_l, pk_l = _kernel_stats(K_eval, q_asl)
        ise_l = _ise(K_eval, q_asl, q_true) if q_true is not None else float("nan")
        print(
            f"    {'AS-Lo':8} h={h_asl:.4g} OTM {otm_asl:.4g} ISE {ise_l:.4e} "
            f"peaks {pk_l} neg {neg_l:.3e}"
        )
        h_asd = asd_bandwidth(K, C, r, T, F)
        C_asd, q_asd, _ = asd_fit(K, C, S0, r, T, q, F, K, K_eval, h_asd)
        otm_asd = _otm_rmse(K, C_asd, C, S0, r, T, q, F)
        neg_a, pk_a = _kernel_stats(K_eval, q_asd)
        ise_a = _ise(K_eval, q_asd, q_true) if q_true is not None else float("nan")
        print(
            f"    {'ASD':8} h={h_asd:.4g} OTM {otm_asd:.4g} "
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
            h_asl = asl_bandwidth(K, p.forward)
            _, q_asl = asl_fit(K, C_proj, p.S0, p.r, p.T, p.q, p.forward, K[:1], K_eval, h_asl)
            acc["AS-Lo"].append(_ise(K_eval, q_asl, q_true))
            acc_cv["AS-Lo"].append(_ise(K_eval, q_asl, q_true))
            h_asd = asd_bandwidth(K, C_proj, p.r, p.T, p.forward)
            _, q_asd, _ = asd_fit(K, C_proj, p.S0, p.r, p.T, p.q, p.forward, K[:1], K_eval, h_asd)
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
        h_asd = asd_bandwidth(sl.K, C, sl.r, sl.T, sl.F)
        C_asd, q_asd, _ = asd_fit(sl.K, C, sl.S0, sl.r, sl.T, sl.q, sl.F, sl.K, s, h_asd)
        o, pu, ca = _otm(sl, C_asd)
        mass, peaks = _mass_peaks(sl.F, q_asd, s)
        neg = _kernel_stats(s, q_asd)[0]
        ho = _holdout_method(sl, "asd")
        print(
            f"    {'ASD':16} h={h_asd:.4g} OTM {o:.3f} puts {pu:.3f} calls {ca:.3f} "
            f"hold {ho:.3f} mass {mass:.2f} peaks {peaks} neg {neg:.3e}"
        )
        h_asl = asl_bandwidth(sl.K, sl.F)
        C_asl, q_asl = asl_fit(sl.K, C, sl.S0, sl.r, sl.T, sl.q, sl.F, sl.K, s, h_asl)
        o, pu, ca = _otm(sl, C_asl)
        mass, peaks = _mass_peaks(sl.F, q_asl, s)
        neg = _kernel_stats(s, q_asl)[0]
        ho = _holdout_method(sl, "asl")
        print(
            f"    {'AS-Lo':16} h={h_asl:.4g} OTM {o:.3f} puts {pu:.3f} calls {ca:.3f} "
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



def _secant_spline(K, C, S0, r, T, q, h, mult):
    """Least-squares natural cubic of the midpoint secants of P.

    The secant between two strikes is the average slope on that interval, so
    the observation sits at the midpoint. Knots are equally spaced by
    ``mult * h``.
    """
    from scipy.interpolate import CubicSpline

    K = np.asarray(K, dtype=float)
    C = np.asarray(C, dtype=float)
    stock = float(S0) * np.exp(-float(q) * float(T))
    P = np.clip(1.0 - C / max(stock, 1e-12), 0.0, 1.0)
    mids = 0.5 * (K[:-1] + K[1:])
    sec = np.diff(P) / np.maximum(np.diff(K), 1e-12)
    step = max(float(h) * float(mult), float(np.median(np.diff(K))))
    lo, hi = float(mids[0]), float(mids[-1])
    knots = np.arange(lo, hi + 0.5 * step, step)
    if knots.size < 4:
        knots = np.linspace(lo, hi, 4)
    eye = np.eye(knots.size)
    B = np.column_stack([
        CubicSpline(knots, eye[j], bc_type="natural")(mids) for j in range(knots.size)
    ])
    coef = np.linalg.lstsq(B, sec, rcond=None)[0]
    return stock, P, knots, CubicSpline(knots, coef, bc_type="natural")



def _slope_cells(K, C, r, T, F, knots, spl):
    """Pieces of the secant spline, with constant-slope tails matched at the ends."""
    cells = []
    k0 = float(knots[0])
    if k0 > 1e-8:
        cells.append((0.0, k0, float(spl(k0)), 0.0, 0.0, 0.0))
    for L, R in zip(knots[:-1], knots[1:]):
        cells.append((
            float(L), float(R),
            float(spl(L, 0)), float(spl(L, 1)),
            0.5 * float(spl(L, 2)), float(spl(L, 3)) / 6.0,
        ))
    disc = float(np.exp(-float(r) * float(T)))
    spec = _right_wing(K, C, F, disc)
    k1 = float(knots[-1])
    if spec is not None and float(spec[3]) > k1 + 1e-6:
        cells.append((k1, float(spec[3]), float(spl(k1)), 0.0, 0.0, 0.0))
    return cells


def _check_slope_closed_form():
    """Quadratic cells match ``_smooth``. A cubic cell matches a trapezoidal integral."""
    from scipy.interpolate import CubicSpline

    K = np.linspace(70.0, 140.0, 32)
    p = BCC97
    C = np.maximum(
        carr_madan_puts(K, p) + p.S0 * np.exp(-p.q * p.T) - K * p.disc, 0.0,
    )
    stock = p.S0 * np.exp(-p.q * p.T)
    Ks, Ps = _knots(K, C, stock, p.forward, p.disc, tails=True)
    spl = CubicSpline(Ks, Ps, bc_type="natural")
    cells = [
        (
            float(L), float(R),
            float(spl(L, 1)), float(spl(L, 2)), 0.5 * float(spl(L, 3)), 0.0,
        )
        for L, R in zip(Ks[:-1], Ks[1:])
    ]
    x = np.linspace(60.0, 150.0, 81)
    h = 2.5
    d_new, G_new = _convolve_slope(x, cells, h)
    d_old, G_old, _ = _smooth(x, Ks, spl, h)
    err_d = float(np.max(np.abs(d_new - d_old)))
    err_G = float(np.max(np.abs(G_new - G_old)))
    A, B, Cc, D = 0.01, 1.2e-4, -1.5e-7, 3e-10
    y = np.linspace(0.0, 160.0, 20001)
    gy = A + B * y + Cc * y ** 2 + D * y ** 3
    xs = np.array([40.0, 80.0, 120.0])
    d_hat, G_hat = _convolve_slope(xs, [(0.0, 160.0, A, B, Cc, D)], h)
    eps = 1e-2
    def _conv_at(z):
        kap = np.exp(-0.5 * ((z - y) / h) ** 2) / (h * np.sqrt(2.0 * np.pi))
        return float(np.trapezoid(gy * kap, y))
    d_num = np.array([(_conv_at(z + eps) - _conv_at(z - eps)) / (2.0 * eps) for z in xs])
    level = np.array([
        float(np.trapezoid(gy * _Phi((z - y) / h), y)) for z in xs
    ])
    err_dnum = float(np.max(np.abs(d_hat - d_num)))
    err_level = float(np.max(np.abs(G_hat - level)))
    print(
        f"  closed form  vs spline formula  d={err_d:.2e} G={err_G:.2e}  "
        f"vs trapezoid  d={err_dnum:.2e} level={err_level:.2e}"
    )
    if max(err_d, err_G, err_dnum, err_level) > 1e-5:
        raise RuntimeError("slope convolution does not match the closed form")
    return err_d, err_G, err_dnum, err_level


def _direct_secant_fit(K, C, S0, r, T, q, mult, K_eval=None, K_price=None):
    """Thrice the midpoint-secant spline directly. No second interpolant.

    The spline is the slope in the closed-form convolution. The level's
    constant of integration is the mean residual against P at the quoted
    strikes; the density does not depend on it.
    """
    K = np.asarray(K, dtype=float)
    C = np.asarray(C, dtype=float)
    h = estimate_rnd(K, C, S0, r, T, q, h="deriv")["h"]
    stock, P, knots, spl = _secant_spline(K, C, S0, r, T, q, h, mult)
    F = float(S0) * np.exp((float(r) - float(q)) * float(T))
    cells = _slope_cells(K, C, r, T, F, knots, spl)
    weights = ((1.0, 8.0 / 3.0), (np.sqrt(2.0), -2.0), (2.0, 1.0 / 3.0))
    targets = []
    if K_eval is not None:
        targets.append(np.asarray(K_eval, dtype=float))
    targets.append(K)
    if K_price is not None:
        targets.append(np.asarray(K_price, dtype=float))
    grid = np.unique(np.concatenate(targets))
    d_tot = np.zeros(grid.size, dtype=float)
    G_tot = np.zeros(grid.size, dtype=float)
    for fac, w in weights:
        d_one, G_one = _convolve_slope(grid, cells, h * fac)
        d_tot += w * d_one
        G_tot += w * G_one
    anchor = float(np.mean(P - np.interp(K, grid, G_tot)))
    G_tot = G_tot + anchor
    out = {"h": float(h), "knots": int(knots.size), "q": None, "C": None}
    if K_eval is not None:
        out["q"] = -F * np.interp(np.asarray(K_eval, dtype=float), grid, d_tot)
    if K_price is not None:
        G = np.clip(np.interp(np.asarray(K_price, dtype=float), grid, G_tot), 0.0, 1.0)
        out["C"] = np.maximum(stock * (1.0 - G), 0.0)
    return out


def _direct_secant_holdout(sl, mult):
    errs = []
    for train, test in _folds(len(sl.K)):
        Kt = sl.K[train]
        Ct = _projected_calls(sl.K[train], sl.C[train], sl.r, sl.T, sl.F)
        fit = _direct_secant_fit(
            Kt, Ct, sl.S0, sl.r, sl.T, sl.q, mult, K_price=sl.K[test],
        )
        errs.append(_otm_rmse_slice(sl, fit["C"], test))
    return float(np.mean(errs))


def direct_secant_pilot(n_reps=30, seed=20260923, sd=0.01, mults=(2.0, 4.0)):
    """Appendix: convolve the midpoint-secant spline itself, at the locked h.

    Same noise stream as the other pilots: one Generator, dense then sparse.
    """
    print(f"\nDirect secant  reps={n_reps}  iv sd={sd}  seed={seed}")
    _check_slope_closed_form()
    K_eval = np.linspace(60.0, 150.0, 401)
    designs = (
        ("Heston", BCC97,
         lambda K, p: carr_madan_puts(K, p) + p.S0 * np.exp(-p.q * p.T) - K * p.disc,
         lambda ev, p: heston_spot_density(ev, p)),
        ("VG", CM99,
         lambda K, p: vg_calls(K, p),
         lambda ev, p: vg_spot_density(ev, p)),
    )
    chains = (
        ("dense", np.linspace(30.0, 220.0, 256)),
        ("sparse", np.linspace(70.0, 140.0, 32)),
    )
    for label, p, calls, dens in designs:
        q_true = dens(K_eval, p)
        print(f"  -- exact {label}")
        for chain, K in chains:
            C = np.maximum(calls(K, p), 0.0)
            for mult in mults:
                fit = _direct_secant_fit(
                    K, C, p.S0, p.r, p.T, p.q, mult, K_eval=K_eval,
                )
                print(
                    f"    {chain:7} {mult:g}h  ISE={_ise(K_eval, fit['q'], q_true):.4e}  "
                    f"h={fit['h']:.4g}  knots={fit['knots']}"
                )
    rng = np.random.default_rng(seed)
    p = BCC97
    q_true = heston_spot_density(K_eval, p)
    calls = lambda K: carr_madan_puts(K, p) + p.S0 * np.exp(-p.q * p.T) - K * p.disc
    for chain, K in chains:
        C_true = np.maximum(calls(K), 0.0)
        acc = {mult: [] for mult in mults}
        for _rep in range(n_reps):
            C = _projected_calls(
                K, _iv_noise(K, C_true, p.S0, p.r, p.T, p.q, rng, sd=sd),
                p.r, p.T, p.forward,
            )
            for mult in mults:
                fit = _direct_secant_fit(
                    K, C, p.S0, p.r, p.T, p.q, mult, K_eval=K_eval,
                )
                acc[mult].append(_ise(K_eval, fit["q"], q_true))
        print(f"  -- noisy Heston {chain}")
        for mult in mults:
            print(f"    {mult:g}h  ISE={np.mean(acc[mult]):.4e}")
    print("  -- listed")
    for csv, title in (
        ("spx_20261218.csv", "SPX Dec"),
        ("spx_20270319.csv", "SPX Mar"),
        ("ndx_20261218.csv", "NDX Dec"),
        ("rut_20261218.csv", "RUT Dec"),
    ):
        sl = load_slice(RES / csv)
        C = _projected_calls(sl.K, sl.C, sl.r, sl.T, sl.F)
        s = np.linspace(max(50.0, 0.2 * sl.F), 2.4 * sl.F, 1601)
        for mult in mults:
            fit = _direct_secant_fit(
                sl.K, C, sl.S0, sl.r, sl.T, sl.q, mult, K_eval=s, K_price=sl.K,
            )
            o, pu, ca = _otm(sl, fit["C"])
            mass, peaks = _mass_peaks(sl.F, fit["q"], s)
            tv = _variation(sl.F, fit["q"], s)
            ho = _direct_secant_holdout(sl, mult)
            print(
                f"    {title:8} {mult:g}h  OTM {o:.3f} puts {pu:.3f} calls {ca:.3f} "
                f"hold {ho:.3f} mass {mass:.1f} peaks {peaks} tv {tv:.1f} "
                f"h={fit['h']:.2f} knots={fit['knots']}"
            )



def midpoint_average_calls(K, C, S0, r, T, q):
    """Calls at the strike midpoints, each the average of the adjacent quotes.

    The natural spline then interpolates these midpoints. Adjacent noise in
    the complementary cdf is averaged before the spline is built.
    """
    K = np.asarray(K, dtype=float)
    C = np.asarray(C, dtype=float)
    stock = float(S0) * np.exp(-float(q) * float(T))
    P = np.clip(1.0 - C / max(stock, 1e-12), 0.0, 1.0)
    mids = 0.5 * (K[:-1] + K[1:])
    Pav = 0.5 * (P[:-1] + P[1:])
    return mids, np.maximum(stock * (1.0 - Pav), 0.0)


def _midpoint_average_fit(K, C, S0, r, T, q, K_eval=None, K_price=None):
    """Spline through midpoint averages, then tails and thricing at the original h."""
    h = estimate_rnd(K, C, S0, r, T, q, h="deriv")["h"]
    Km, Cm = midpoint_average_calls(K, C, S0, r, T, q)
    fit = estimate_rnd(
        Km, Cm, S0, r, T, q, K_eval=K_eval, K_price=K_price, h=h,
        tails=True, higher=True,
    )
    fit["knots"] = int(Km.size)
    return fit


def _midpoint_average_holdout(sl):
    errs = []
    for train, test in _folds(len(sl.K)):
        Kt = sl.K[train]
        Ct = _projected_calls(sl.K[train], sl.C[train], sl.r, sl.T, sl.F)
        fit = _midpoint_average_fit(
            Kt, Ct, sl.S0, sl.r, sl.T, sl.q, K_price=sl.K[test],
        )
        errs.append(_otm_rmse_slice(sl, fit["C"], test))
    return float(np.mean(errs))


def midpoint_average_pilot(n_reps=30, seed=20260923, sd=0.01):
    """Natural spline through averaged midpoints, then thrice at the locked h.

    Same noise stream as ``midpoint_pilot``: one Generator, dense then sparse.
    """
    print(f"\nMidpoint average  reps={n_reps}  iv sd={sd}  seed={seed}")
    K_eval = np.linspace(60.0, 150.0, 401)
    designs = (
        ("Heston", BCC97,
         lambda K, p: carr_madan_puts(K, p) + p.S0 * np.exp(-p.q * p.T) - K * p.disc,
         lambda ev, p: heston_spot_density(ev, p)),
        ("VG", CM99,
         lambda K, p: vg_calls(K, p),
         lambda ev, p: vg_spot_density(ev, p)),
    )
    chains = (
        ("dense", np.linspace(30.0, 220.0, 256)),
        ("sparse", np.linspace(70.0, 140.0, 32)),
    )
    for label, p, calls, dens in designs:
        q_true = dens(K_eval, p)
        print(f"  -- exact {label}")
        for chain, K in chains:
            C = np.maximum(calls(K, p), 0.0)
            fit = _midpoint_average_fit(
                K, C, p.S0, p.r, p.T, p.q, K_eval=K_eval,
            )
            print(
                f"    {chain:7} ISE={_ise(K_eval, fit['q'], q_true):.4e}  "
                f"h={fit['h']:.4g}  knots={fit['knots']}"
            )
    rng = np.random.default_rng(seed)
    p = BCC97
    q_true = heston_spot_density(K_eval, p)
    calls = lambda K: carr_madan_puts(K, p) + p.S0 * np.exp(-p.q * p.T) - K * p.disc
    for chain, K in chains:
        C_true = np.maximum(calls(K), 0.0)
        acc = []
        for _rep in range(n_reps):
            C = _projected_calls(
                K, _iv_noise(K, C_true, p.S0, p.r, p.T, p.q, rng, sd=sd),
                p.r, p.T, p.forward,
            )
            fit = _midpoint_average_fit(
                K, C, p.S0, p.r, p.T, p.q, K_eval=K_eval,
            )
            acc.append(_ise(K_eval, fit["q"], q_true))
        print(f"  -- noisy Heston {chain}  ISE={np.mean(acc):.4e}")
    print("  -- listed")
    for csv, title in (
        ("spx_20261218.csv", "SPX Dec"),
        ("spx_20270319.csv", "SPX Mar"),
        ("ndx_20261218.csv", "NDX Dec"),
        ("rut_20261218.csv", "RUT Dec"),
    ):
        sl = load_slice(RES / csv)
        C = _projected_calls(sl.K, sl.C, sl.r, sl.T, sl.F)
        s = np.linspace(max(50.0, 0.2 * sl.F), 2.4 * sl.F, 1601)
        fit = _midpoint_average_fit(
            sl.K, C, sl.S0, sl.r, sl.T, sl.q, K_eval=s, K_price=sl.K,
        )
        o, pu, ca = _otm(sl, fit["C"])
        mass, peaks = _mass_peaks(sl.F, fit["q"], s)
        tv = _variation(sl.F, fit["q"], s)
        ho = _midpoint_average_holdout(sl)
        print(
            f"    {title:8} OTM {o:.3f} puts {pu:.3f} calls {ca:.3f} "
            f"hold {ho:.3f} mass {mass:.1f} peaks {peaks} tv {tv:.1f} "
            f"h={fit['h']:.2f} knots={fit['knots']}"
        )



# Dimensionless penalty c in λ = c F^3 ∫(P'')^2. Zero is the interpolant.
_SMOOTH_C = np.array([0.0, *np.geomspace(1e-7, 3e-4, 16)])


def _smoothing_base(K, C, S0, r, T, q, h_rule="deriv"):
    K = np.asarray(K, float)
    C = np.asarray(C, float)
    h = estimate_rnd(K, C, S0, r, T, q, h=h_rule)["h"]
    disc = float(np.exp(-float(r) * float(T)))
    stock = float(S0) * np.exp(-float(q) * float(T))
    F = float(S0) * np.exp((float(r) - float(q)) * float(T))
    Ks, Ps = _knots(K, C, stock, F, disc, tails=True)
    return K, C, h, stock, F, Ks, Ps


def _thrice_from_spline(spl, Ks, h, F, x):
    """Thriced convolution of a cubic spline of P, as in estimate_rnd."""
    from rnd import _convolve

    x = np.asarray(x, float)
    wts = ((1.0, 8.0 / 3.0), (np.sqrt(2.0), -2.0), (2.0, 1.0 / 3.0))
    return _convolve(spl, Ks, h, F, x, wts, level_at=x)


def _score_smoothing_c(base, c):
    from scipy.interpolate import make_smoothing_spline

    K, C, h, stock, F, Ks, Ps = base
    lam = 0.0 if c <= 0.0 else float(c) * F ** 3
    spl = make_smoothing_spline(Ks, Ps, lam=lam)
    s = np.linspace(max(50.0, 0.2 * F), 2.4 * F, 700)
    q, _ = _thrice_from_spline(spl, Ks, h, F, s)
    # Plain Gaussian, for the gap that the corner watches.
    from rnd import _smooth
    q1 = -F * _smooth(s, Ks, spl, h)[0]
    m = (s >= 0.55 * F) & (s <= 1.40 * F)
    corr = float(np.sqrt(np.mean((q[m] - q1[m]) ** 2)))
    _, G = _thrice_from_spline(spl, Ks, h, F, K)
    Cf = np.maximum(stock * (1.0 - np.clip(G, 0.0, 1.0)), 0.0)
    price = float(np.sqrt(np.mean((Cf - C) ** 2)))
    tv = _variation(F, q, s)
    mass, peaks = _mass_peaks(F, q, s)
    return dict(
        c=float(c), tv=tv, peaks=peaks, mass=mass, price=price, corr=corr,
        spl=spl, Ks=Ks, h=h, stock=stock, F=F,
    )


def _choose_smoothing_c(rows, curv_min=0.5):
    """Smallest unimodal penalty, then the L-curve corner when the bend is sharp.

    Unimodal means one peak and total variation at most 1 on [0.60F, 1.30F].
    The corner is of log call residual against log gap between the thriced
    density and the plain Gaussian. A flat bend keeps the unimodal penalty.
    """
    uni = [r for r in rows if r["peaks"] <= 1 and r["tv"] <= 1.0]
    c0 = uni[0]["c"] if uni else rows[-1]["c"]
    sub = [r for r in rows if r["c"] + 1e-15 >= c0 and r["price"] > 1e-8 and r["corr"] > 0.0]
    if len(sub) < 5:
        return c0
    x = np.log([r["price"] for r in sub])
    y = np.log([r["corr"] for r in sub])
    dx, dy = np.gradient(x), np.gradient(y)
    ddx, ddy = np.gradient(dx), np.gradient(dy)
    kap = np.abs(dx * ddy - dy * ddx) / np.maximum((dx * dx + dy * dy) ** 1.5, 1e-18)
    kap[0] = kap[-1] = -1.0
    i = int(np.argmax(kap))
    if float(kap[i]) >= curv_min:
        return sub[i]["c"]
    return c0


def _smoothing_fit_at(base, c, K_eval=None, K_price=None):
    row = _score_smoothing_c(base, c)
    out = {"c": row["c"], "h": row["h"], "q": None, "C": None, "mass": row["mass"],
           "peaks": row["peaks"], "tv": row["tv"]}
    if K_eval is not None:
        q, _ = _thrice_from_spline(row["spl"], row["Ks"], row["h"], row["F"], K_eval)
        out["q"] = q
    if K_price is not None:
        _, G = _thrice_from_spline(row["spl"], row["Ks"], row["h"], row["F"], K_price)
        out["C"] = np.maximum(row["stock"] * (1.0 - np.clip(G, 0.0, 1.0)), 0.0)
    return out


def smoothing_spline_fit(
    K, C, S0, r, T, q, K_eval=None, K_price=None, h_rule="deriv", method="fast",
    penalty="closed", tv_cap=None,
):
    """The paper's estimator. ``estimate_rnd`` with its default closed-form penalty."""
    return estimate_rnd(
        K, C, S0, r, T, q, K_eval=K_eval, K_price=K_price, h=h_rule, method=method,
        penalty=penalty, tv_cap=TV_CAP if tv_cap is None else tv_cap,
    )


def _smoothing_holdout(sl, h_rule="deriv", penalty="closed", tv_cap=None):
    errs = []
    for train, test in _folds(len(sl.K)):
        Ct = _projected_calls(sl.K[train], sl.C[train], sl.r, sl.T, sl.F)
        fit = smoothing_spline_fit(
            sl.K[train], Ct, sl.S0, sl.r, sl.T, sl.q, K_price=sl.K[test],
            h_rule=h_rule, penalty=penalty,
            tv_cap=TV_CAP if tv_cap is None else tv_cap,
        )
        errs.append(_otm_rmse_slice(sl, fit["C"], test))
    return float(np.mean(errs))


def smoothing_spline_pilot(n_reps=30, seed=20260923, sd=0.01):
    """Appendix: smoothing spline of P, penalty chosen per chain, then thrice."""
    print(f"\nSmoothing spline  reps={n_reps}  iv sd={sd}  seed={seed}")
    K_eval = np.linspace(60.0, 150.0, 401)
    designs = (
        ("Heston", BCC97,
         lambda K, p: carr_madan_puts(K, p) + p.S0 * np.exp(-p.q * p.T) - K * p.disc,
         lambda ev, p: heston_spot_density(ev, p)),
        ("VG", CM99,
         lambda K, p: vg_calls(K, p),
         lambda ev, p: vg_spot_density(ev, p)),
    )
    chains = (
        ("dense", np.linspace(30.0, 220.0, 256)),
        ("sparse", np.linspace(70.0, 140.0, 32)),
    )
    for label, p, calls, dens in designs:
        q_true = dens(K_eval, p)
        print(f"  -- exact {label}")
        for chain, K in chains:
            C = np.maximum(calls(K, p), 0.0)
            fit = smoothing_spline_fit(K, C, p.S0, p.r, p.T, p.q, K_eval=K_eval)
            print(
                f"    {chain:7} ISE={_ise(K_eval, fit['q'], q_true):.4e}  "
                f"c={fit['c']:.2e}"
            )
    rng = np.random.default_rng(seed)
    p = BCC97
    q_true = heston_spot_density(K_eval, p)
    calls = lambda K: carr_madan_puts(K, p) + p.S0 * np.exp(-p.q * p.T) - K * p.disc
    for chain, K in chains:
        C_true = np.maximum(calls(K), 0.0)
        acc = []
        for _rep in range(n_reps):
            C = _projected_calls(
                K, _iv_noise(K, C_true, p.S0, p.r, p.T, p.q, rng, sd=sd),
                p.r, p.T, p.forward,
            )
            fit = smoothing_spline_fit(K, C, p.S0, p.r, p.T, p.q, K_eval=K_eval)
            acc.append(_ise(K_eval, fit["q"], q_true))
        print(f"  -- noisy Heston {chain}  ISE={np.mean(acc):.4e}")
    print("  -- listed")
    for csv, title in (
        ("spx_20261218.csv", "SPX Dec"),
        ("spx_20270319.csv", "SPX Mar"),
        ("ndx_20261218.csv", "NDX Dec"),
        ("rut_20261218.csv", "RUT Dec"),
    ):
        sl = load_slice(RES / csv)
        C = _projected_calls(sl.K, sl.C, sl.r, sl.T, sl.F)
        s = np.linspace(max(50.0, 0.2 * sl.F), 2.4 * sl.F, 1601)
        fit = smoothing_spline_fit(
            sl.K, C, sl.S0, sl.r, sl.T, sl.q, K_eval=s, K_price=sl.K,
        )
        o, pu, ca = _otm(sl, fit["C"])
        mass, peaks = _mass_peaks(sl.F, fit["q"], s)
        tv = _variation(sl.F, fit["q"], s)
        ho = _smoothing_holdout(sl)
        print(
            f"    {title:8} c={fit['c']:.2e} OTM {o:.3f} puts {pu:.3f} calls {ca:.3f} "
            f"hold {ho:.3f} mass {mass:.1f} peaks {peaks} tv {tv:.2f}"
        )


def _half_up(x, digits):
    scale = 10.0 ** int(digits)
    return float(np.floor(float(x) * scale + 0.5) / scale)


def _timing_designs():
    """The eight designs of the computational table, inputs built once."""
    def heston_calls(K, p=BCC97):
        return np.maximum(
            carr_madan_puts(K, p) + p.S0 * np.exp(-p.q * p.T) - K * p.disc, 0.0,
        )

    def vg_calls_pos(K, p=CM99):
        return np.maximum(vg_calls(K, p), 0.0)

    rows = []
    for name, p, calls, K in (
        ("Heston, dense", BCC97, heston_calls, np.linspace(30.0, 220.0, 256)),
        ("Heston, sparse", BCC97, heston_calls, np.linspace(70.0, 140.0, 32)),
        ("Variance gamma, dense", CM99, vg_calls_pos, np.linspace(30.0, 220.0, 256)),
        ("Variance gamma, sparse", CM99, vg_calls_pos, np.linspace(70.0, 140.0, 32)),
    ):
        rows.append((name, np.asarray(K, float), calls(K), p.S0, p.r, p.T, p.q))
    for csv, title in (
        ("spx_20261218.csv", "SPX 18 Dec 2026"),
        ("spx_20270319.csv", "SPX 19 Mar 2027"),
        ("ndx_20261218.csv", "NDX 18 Dec 2026"),
        ("rut_20261218.csv", "RUT 18 Dec 2026"),
    ):
        sl = load_slice(RES / csv)
        C = _projected_calls(sl.K, sl.C, sl.r, sl.T, sl.F)
        rows.append((title, sl.K, C, sl.S0, sl.r, sl.T, sl.q))
    return rows


def convolution_times(repeats=5, warmup=1):
    """One pricing at penalty zero. The penalty rule is not applied.

    Each timing is ``estimate_rnd(..., c=0, K_price=K)``: the natural cubic
    through the knots, then the density and the calls at the quoted strikes.
    One warmup call is discarded. The recorded time is the minimum of
    ``repeats``. The parenthetical is that time in seconds, half-up to
    three decimals, which is the figure in the paper. Ratios use the raw
    times, then half-up to one decimal. ``fast`` is the FFT+ME column.
    """
    import time

    print(
        f"\nConvolution times at c=0, penalty rule off. "
        f"warmup {warmup}, min of {repeats}."
    )
    methods = ("naive", "fft", "fast")
    for name, K, C, S0, r, T, q in _timing_designs():
        raw = {}
        for method in methods:
            def once(method=method, K=K, C=C, S0=S0, r=r, T=T, q=q):
                return estimate_rnd(
                    K, C, S0, r, T, q, c=0.0, K_price=K, method=method,
                )

            for _ in range(warmup):
                once()
            best = None
            for _ in range(repeats):
                t0 = time.perf_counter()
                once()
                dt = time.perf_counter() - t0
                best = dt if best is None else min(best, dt)
            raw[method] = best
        print(
            f"  {name:24} "
            f"naive {raw['naive']:.6f} ({_half_up(raw['naive'], 3):.3f}) "
            f"fft {raw['fft']:.6f} ({_half_up(raw['fft'], 3):.3f}) "
            f"fast {raw['fast']:.6f} ({_half_up(raw['fast'], 3):.3f}) "
            f"fft/naive {_half_up(raw['naive'] / raw['fft'], 1):.1f} "
            f"fast/naive {_half_up(raw['naive'] / raw['fast'], 1):.1f}"
        )


def extra_chains():
    """Appendix chains: the closed-form rule and PCA on expiries outside the four slices.

    Not called from ``main``. The quotes are not shipped. Cboe's North American
    Data Policies, effective 1 September 2026, do not permit a recipient of
    delayed Cboe options quotes to redistribute them externally except to a
    named affiliate, and historical quotes require a data agreement and
    approval before they go to anyone else. This reads the local dumps.

    Same quote screen as the paper. Maturity runs from 14 days to two years,
    each chain has at least 40 strikes, and each root keeps at most eight
    expiries spread across the calendar. Weekly roots are left out. The four
    published slices are left out. SPX, NDX, and RUT are the 23 September 2026
    dumps; DJX is 19 September 2026.
    """
    from datetime import date

    from src.spx import asof_from_raw, build_otm_slice, fetch_cboe

    paper = {
        ("SPX", date(2026, 12, 18)),
        ("SPX", date(2027, 3, 19)),
        ("NDX", date(2026, 12, 18)),
        ("RUT", date(2026, 12, 18)),
    }
    print(f"\nAppendix chains  penalty=closed  tv_cap={TV_CAP:.6f}")
    for symbol, name in (
        ("SPX", "cboe_spx.json"),
        ("NDX", "cboe_ndx.json"),
        ("RUT", "cboe_rut.json"),
        ("DJX", "cboe_djx.json"),
    ):
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
        built = []
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
            built.append(sl)
        if len(built) > 8:
            pick = [built[int(round(i))] for i in np.linspace(0, len(built) - 1, 8)]
            seen, uniq = set(), []
            for sl in pick:
                if sl.expiry in seen:
                    continue
                seen.add(sl.expiry)
                uniq.append(sl)
            built = uniq
        print(f"  {symbol} asof {asof.isoformat()}  n_chains {len(built)}")
        for sl in built:
            C = _projected_calls(sl.K, sl.C, sl.r, sl.T, sl.F)
            s = np.linspace(max(50.0, 0.2 * sl.F), 2.4 * sl.F, 1601)
            fit = smoothing_spline_fit(
                sl.K, C, sl.S0, sl.r, sl.T, sl.q, K_eval=s, K_price=sl.K,
            )
            h_pca = pca_cv_bandwidth(sl.K, C, sl.S0, sl.r, sl.T, sl.q, sl.F)
            C_pca, q_pca, _, _, _ = pca_fit(
                sl.K, C, sl.r, sl.T, sl.F, h_pca, sl.K, s,
            )
            rows = (
                ("Ours", fit["C"], fit["q"], _smoothing_holdout(sl)),
                ("PCA", C_pca, q_pca, _holdout_method(sl, "pca")),
            )
            for name, chat, qhat, ho in rows:
                o, pu, ca = _otm(sl, chat)
                mass, peaks = _mass_peaks(sl.F, qhat, s)
                tv = _variation(sl.F, qhat, s)
                note = ""
                if name == "Ours":
                    note = f" h={fit['h']:.4f} c={fit['c']:.6e} shape={fit['shape']:.4f}"
                if peaks >= 2:
                    yy = np.maximum(np.nan_to_num(qhat), 0.0)
                    core = (s >= _REPORT_LO * sl.F) & (s <= _REPORT_HI * sl.F)
                    yy = yy[core]
                    loc = (yy[1:-1] > yy[:-2]) & (yy[1:-1] > yy[2:]) & (yy[1:-1] > 0.2 * yy.max())
                    heights = np.sort(yy[1:-1][loc])
                    note += f" second/mode {heights[-2] / yy.max():.4f}"
                print(
                    f"  {sl.root} {sl.expiry.isoformat()} {name:4} n={len(sl.K):4d} "
                    f"OTM {o:.3f} puts {pu:.3f} calls {ca:.3f} "
                    f"hold {ho:.3f} mass {mass:.1f} tv {tv:.4f} peaks {peaks}{note}",
                    flush=True,
                )


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
    noisy_heston()
    listed()
    convolution_times()
    print("DONE")


if __name__ == "__main__":
    main()
