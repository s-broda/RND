"""Replication for the paper.

Run from the repository root::

    python3 python/replicate.py

Prints the Heston and variance-gamma ISE tables and the listed pricing table,
and writes figures/ccdf_heston_rnd.pdf, figures/ccdf_vg_rnd.pdf, and
figures/ccdf_listed.pdf.
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
            ("Tails+Twice", True, True),
        )
        for name, tails, twice in specs:
            best, best_h = np.inf, hs[len(hs) // 2]
            for h in hs:
                q = estimate_rnd(
                    K, C, p.S0, p.r, p.T, p.q, K_eval=K_eval,
                    h=float(h), tails=tails, twice=twice,
                )["q"]
                err = _ise(K_eval, q, q_true)
                if err < best:
                    best, best_h = err, float(h)
            fit_m = estimate_rnd(
                K, C, p.S0, p.r, p.T, p.q, K_eval=K_eval,
                h="mesh", tails=tails, twice=twice,
            )
            fit_d = estimate_rnd(
                K, C, p.S0, p.r, p.T, p.q, K_eval=K_eval,
                tails=tails, twice=twice,
            )
            q_m, q_d = fit_m["q"], fit_d["q"]
            rec[chain][name] = dict(
                h_star=best_h, ise_star=best,
                h_mesh=fit_m["h"], ise_mesh=_ise(K_eval, q_m, q_true),
                h_den=fit_d["h"], ise_den=_ise(K_eval, q_d, q_true),
            )
            row = rec[chain][name]
            print(
                f"    {name:16} oracle {row['h_star']:.4g} {row['ise_star']:.4e}  "
                f"mesh {row['h_mesh']:.4g} {row['ise_mesh']:.4e}  "
                f"rule {row['h_den']:.4g} {row['ise_den']:.4e}"
            )
            plot.setdefault(chain, {})[{"Quoted spline": "quoted", "Tails": "tails", "Tails+Twice": "twice"}[name]] = q_m
            plot[chain]["K"] = K
    return rec, plot, K_eval, q_true


def _figure_exact(path, K_eval, q_true, plot, title):
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6), sharey=True)
    for ax, chain in zip(axes, ("dense", "sparse")):
        ax.plot(K_eval, q_true, color="black", lw=1.8, label=title, zorder=2)
        ax.plot(K_eval, np.maximum(plot[chain]["quoted"], 0), color="#7f7f7f", lw=1.15, ls=":", label="Quoted spline", zorder=3)
        ax.plot(K_eval, np.maximum(plot[chain]["tails"], 0), color="#d62728", lw=1.15, ls="--", label="Tails", zorder=4)
        ax.plot(K_eval, np.maximum(plot[chain]["twice"], 0), color="#1f77b4", lw=1.35, label="Tails+Twice", zorder=5)
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
                Kt, Ct, sl.S0, sl.r, sl.T, sl.q, K_price=Ke, tails=True, twice=True,
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
            h = estimate_rnd(Kt, Ct, sl.S0, sl.r, sl.T, sl.q, tails=True, twice=True)["h"]
            _, _, C_te = priestley_chao_cubic(
                Kt, Ct, sl.S0, sl.r, sl.T, Ke, q=sl.q, h=h, return_call=True
            )
        else:
            raise ValueError(method)
        errs.append(_otm_rmse_slice(sl, C_te, test))
    return float(np.mean(errs))


def _row_metrics(sl, C_hat, q, s, ho):
    o, pu, ca = _otm(sl, C_hat)
    mass, _ = _mass_peaks(sl.F, q, s)
    return np.array([o, pu, ca, ho, mass], dtype=float)


_METHODS = ("Ours", "Yatchew–Härdle", "Aït-Sahalia–Duarte", "PCA", "Priestley–Chao")


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
        sl.K, C, sl.S0, sl.r, sl.T, sl.q, K_eval=s, K_price=sl.K, tails=True, twice=True,
    )
    h = fit["h"]
    lam = yatchew_cv_lambda(sl.K, C, sl.S0, sl.r, sl.T, sl.q, sl.F)
    C_yh, m_yh = yatchew_fit(sl.K, C, sl.S0, sl.r, sl.T, sl.q, sl.F, sl.K, lam)
    q_yh = np.interp(s, sl.K, _second_diff_q(sl.K, m_yh, sl.r * sl.T), left=0.0, right=0.0)
    h_asd, h_asd_d = asd_cv_bandwidth(sl.K, C, sl.S0, sl.r, sl.T, sl.q, sl.F)
    C_asd, q_asd, _ = asd_fit(
        sl.K, C, sl.S0, sl.r, sl.T, sl.q, sl.F, sl.K, s, h_asd, h_asd_d
    )
    h_pca = pca_cv_bandwidth(sl.K, C, sl.S0, sl.r, sl.T, sl.q, sl.F)
    C_pca, q_pca, _, _, _ = pca_fit(sl.K, C, sl.r, sl.T, sl.F, h_pca, sl.K, s)
    q_pc, _, _ = priestley_chao_cubic(sl.K, C, sl.S0, sl.r, sl.T, s, q=sl.q, h=h, return_call=True)
    _, _, C_pc = priestley_chao_cubic(
        sl.K, C, sl.S0, sl.r, sl.T, sl.K, q=sl.q, h=h, return_call=True
    )
    calls = {
        "Ours": fit["C"],
        "Yatchew–Härdle": C_yh,
        "Aït-Sahalia–Duarte": C_asd,
        "PCA": C_pca,
        "Priestley–Chao": C_pc,
    }
    dens = {
        "Ours": fit["q"],
        "Yatchew–Härdle": q_yh,
        "Aït-Sahalia–Duarte": q_asd,
        "PCA": q_pca,
        "Priestley–Chao": q_pc,
    }
    keys = {"Ours": "ours", "Yatchew–Härdle": "yh", "Aït-Sahalia–Duarte": "asd", "PCA": "pca", "Priestley–Chao": "pc"}
    metrics = {
        name: _row_metrics(sl, calls[name], dens[name], s, _holdout_method(sl, keys[name]))
        for name in _METHODS
    }
    bundle = dict(
        C=C, fit=fit, h=h, s=s, q_pc=q_pc, q_asd=q_asd, q_pca=q_pca,
        C_yh=C_yh, C_pca=C_pca, C_pc=C_pc,
        h_asd_d=h_asd_d, h_pca=h_pca, lam=lam,
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
        print(f"  {title:8} F={sl.F:.1f} h={bundle['h']:.2f}")
        for name in _METHODS:
            _print_metrics(name, metrics[name])
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
        q_asd, q_pca, q_pc = bundle["q_asd"], bundle["q_pca"], bundle["q_pc"]
        C_yh, C_pca, C_pc = bundle["C_yh"], bundle["C_pca"], bundle["C_pc"]
        print(
            f"  fig {title}: ASD h={bundle['h_asd_d']:.1f}  PCA h={bundle['h_pca']:.4f}  "
            f"YH λ={bundle['lam']:.4g}  PC h={h:.1f}"
        )
        ax = axes[row, 0]
        ax.plot(s, np.maximum(fit["q"], 0), color="#1f77b4", lw=1.4, label="Ours")
        ax.plot(s, np.maximum(q_asd, 0), color="#8c564b", lw=1.15, ls="--", label="Aït-Sahalia–Duarte")
        ax.plot(s, np.maximum(q_pca, 0), color="#2ca02c", lw=1.15, ls="-.", label="PCA")
        ax.plot(s, np.maximum(q_pc, 0), color="#9467bd", lw=1.15, ls=":", label="Priestley–Chao")
        ax.axvline(sl.F, color="0.45", ls="--", lw=0.8)
        ax.set_xlim(0.55 * sl.F, 1.40 * sl.F)
        core = (s >= 0.65 * sl.F) & (s <= 1.30 * sl.F)
        ymax = np.nanmax(np.maximum(fit["q"][core], 0))
        ax.set_ylim(0, 1.15 * ymax)
        ax.set_title(title)
        ax.set_xlabel(r"Strike $K$")
        ax.legend(frameon=False, fontsize=7.5, loc="upper right")
        if row == 0:
            ax.set_ylabel(r"$f_{\mathbb{Q}}(K)$")

        P_c, C_c = _parity(sl, fit["C"])
        P_yh, C_yh = _parity(sl, C_yh)
        P_pca, C_pca = _parity(sl, C_pca)
        P_pc, C_pc = _parity(sl, C_pc)
        iv = _otm_iv(sl, P_c, C_c)
        iv_yh = _otm_iv(sl, P_yh, C_yh)
        iv_pca = _otm_iv(sl, P_pca, C_pca)
        iv_pc = _otm_iv(sl, P_pc, C_pc)
        iv_m = _otm_iv(sl, sl.P, sl.C)
        lo, hi = 0.55 * sl.F, 1.40 * sl.F
        show = np.isfinite(iv_m) & (sl.K >= lo) & (sl.K <= hi)
        ax = axes[row, 1]
        ax.set_xlim(lo, hi)
        y1, y2 = _iv_ylim(
            [iv_m[show], iv[show], iv_yh[show], iv_pca[show], iv_pc[show]]
        )
        ax.set_ylim(y1, y2)
        ax.plot(sl.K[show], iv_m[show], "k.", ms=2.6, alpha=0.40, label="Market", zorder=2)
        ax.plot(sl.K[show], iv_pc[show], color="#9467bd", lw=1.15, ls=":", label="Priestley–Chao", zorder=3)
        ax.plot(sl.K[show], iv_pca[show], color="#2ca02c", lw=1.15, ls="-.", label="PCA", zorder=4)
        ax.plot(sl.K[show], iv_yh[show], color="#ff7f0e", lw=1.15, ls=":", label="Yatchew–Härdle", zorder=5)
        ax.plot(sl.K[show], iv[show], color="#1f77b4", lw=1.35, label="Ours", zorder=6)
        ax.axvline(sl.F, color="0.45", ls="--", lw=0.8)
        ax.set_xlabel(r"Strike $K$")
        ax.legend(frameon=False, fontsize=7.5, loc="upper right")
        if row == 0:
            ax.set_ylabel("OTM implied vol")
    fig.tight_layout()
    fig.savefig(FIG / "ccdf_listed.pdf", facecolor="white")
    plt.close(fig)
    print(" wrote ccdf_listed.pdf")


def main():
    K_eval = np.linspace(60.0, 150.0, 401)
    p = BCC97
    rec_h, plot_h, _, qh = _exact(
        "Heston", p, lambda K: carr_madan_puts(K, p) + p.S0 * np.exp(-p.q * p.T) - K * p.disc, 
        heston_spot_density(K_eval, p), K_eval,
    )
    # the calls() above double-counts because carr_madan_puts returns puts; fix in _exact by passing C
    _figure_exact(FIG / "ccdf_heston_rnd.pdf", K_eval, qh, plot_h, "Heston")
    v = CM99
    rec_v, plot_v, _, qv = _exact("VG", v, lambda K: vg_calls(K, v), vg_spot_density(K_eval, v), K_eval)
    _figure_exact(FIG / "ccdf_vg_rnd.pdf", K_eval, qv, plot_v, "Variance gamma")
    listed()
    print("DONE")


if __name__ == "__main__":
    main()
