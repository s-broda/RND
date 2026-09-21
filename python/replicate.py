#!/usr/bin/env python3
"""Replication script for the closed-form call-on-K RND paper.

The shareable estimator is ``rnd.estimate_rnd``. This script is the
single entry point: it prints Heston and variance-gamma ISE and listed
SPX/NDX/RUT pricing tables, and writes figures/ccdf_heston_rnd.pdf, figures/ccdf_vg_rnd.pdf,
and figures/ccdf_listed.pdf.

Run from the repository root::

    python3 python/replicate.py
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PY = Path(__file__).resolve().parent
ROOT = PY.parent
import sys

sys.path.insert(0, str(PY))

from rnd import estimate_rnd
from src import black_scholes as bs
from src.competitors import (
    ait_sahalia_duarte,
    convex_decreasing_ls,
    pca_lognormal,
    priestley_chao_bl,
    priestley_chao_cubic,
    yatchew_hardle,
)
from src.heston import BCC97, carr_madan_puts, heston_spot_density
from src.vg import CM99, vg_calls, vg_spot_density
from src.spx import atm_iv, build_otm_slice, fetch_cboe, load_slice, save_slice

FIG = ROOT / "figures"
RES = PY / "results"
FIG.mkdir(parents=True, exist_ok=True)


def _rmse(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    ok = np.isfinite(a) & np.isfinite(b)
    return float(np.sqrt(np.mean((a[ok] - b[ok]) ** 2))) if ok.any() else float("nan")


def _otm(sl, P, C):
    return np.where(sl.K <= sl.F, P, C)


def _otm_iv(sl, P, C):
    otm = _otm(sl, P, C)
    is_c = sl.K > sl.F
    iv = np.empty_like(sl.K)
    iv[is_c] = bs.implied_vol(C[is_c], sl.S0, sl.K[is_c], sl.r, sl.T, sl.q, True)
    iv[~is_c] = bs.implied_vol(P[~is_c], sl.S0, sl.K[~is_c], sl.r, sl.T, sl.q, False)
    return otm, iv


def _mass_mean(s, q):
    q = np.maximum(np.asarray(q, dtype=float), 0.0)
    s = np.asarray(s, dtype=float)
    w = 0.5 * (q[1:] + q[:-1]) * np.diff(s)
    sm = 0.5 * (s[1:] + s[:-1])
    mass = float(w.sum())
    mean = float((w * sm).sum() / mass) if mass > 1e-16 else float("nan")
    return mass, mean


def _prices_from_q(s, q, K, disc):
    q = np.maximum(np.asarray(q, dtype=float), 0.0)
    s = np.asarray(s, dtype=float)
    K = np.asarray(K, dtype=float)
    w = 0.5 * (q[1:] + q[:-1]) * np.diff(s)
    sm = 0.5 * (s[1:] + s[:-1])
    I0 = np.concatenate([[0.0], np.cumsum(w)])
    I1 = np.concatenate([[0.0], np.cumsum(w * sm)])
    mass = float(I0[-1])
    i0 = np.interp(K, s, I0)
    i1 = np.interp(K, s, I1)
    P = disc * (K * i0 - i1)
    C = disc * ((I1[-1] - i1) - K * (mass - i0))
    return np.maximum(P, 0.0), np.maximum(C, 0.0), mass, (
        float(I1[-1] / mass) if mass > 1e-16 else float("nan")
    )


def _ise(K, qh, q):
    return float(np.trapezoid((qh - q) ** 2, K))


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


def heston_ise():
    p = BCC97
    F = p.forward
    K_eval = np.linspace(60.0, 150.0, 401)
    q_true = heston_spot_density(K_eval, p)
    print(f"Heston  S0={p.S0}  F={F:.4f}  T={p.T}")
    rec = {}
    plot_q = {}
    for chain, K_obs in (
        ("dense", np.linspace(30.0, 220.0, 256)),
        ("sparse", np.linspace(70.0, 140.0, 64)),
    ):
        P = carr_madan_puts(K_obs, p)
        C = np.maximum(P + p.S0 * np.exp(-p.q * p.T) - K_obs * p.disc, 0.0)
        specs = (
            ("Baseline", dict(tails=False, midpoints=False, twice=False)),
            ("Tail completion", dict(tails=True, midpoints=False, twice=False)),
            ("Tail+Midpoint", dict(tails=True, midpoints=True, twice=False)),
            ("Tail+Mid+Twice", dict(tails=True, midpoints=True, twice=True)),
        )
        delta = float(np.median(np.diff(K_obs)))
        hs = np.geomspace(max(0.20 * delta, 1e-3), max(30.0 * delta, 40.0), 40)
        print(f"\n-- {chain}  m={len(K_obs)}")
        rec[chain] = {}
        for name, kw in specs:
            def ise_at(h, _kw=kw):
                out = estimate_rnd(
                    K_obs, C, p.S0, p.r, p.T, p.q, K_eval=K_eval, h=h, **_kw
                )
                return _ise(K_eval, out["q"], q_true)

            best, best_h = np.inf, hs[len(hs) // 2]
            for h in hs:
                err = ise_at(h)
                if err < best:
                    best, best_h = err, float(h)
            mesh = estimate_rnd(
                K_obs, C, p.S0, p.r, p.T, p.q, K_eval=K_eval, h="mesh", **kw
            )
            den = estimate_rnd(
                K_obs, C, p.S0, p.r, p.T, p.q, K_eval=K_eval, h="density", **kw
            )
            ise_m = _ise(K_eval, mesh["q"], q_true)
            ise_d = _ise(K_eval, den["q"], q_true)
            rec[chain][name] = {
                "h_star": best_h,
                "ise_star": best,
                "h_mesh": mesh["h"],
                "ise_mesh": ise_m,
                "h_den": den["h"],
                "ise_den": ise_d,
            }
            print(
                f"  [{name:18s}]  oracle h={best_h:.4g} ISE={best:.4e}  "
                f"mesh h={mesh['h']:.4g} ISE={ise_m:.4e}  "
                f"den h={den['h']:.4g} ISE={ise_d:.4e}"
            )
            if name == "Tail+Midpoint":
                plot_q.setdefault(chain, {})["mid"] = mesh["q"]
                plot_q[chain]["K"] = K_obs
            if name == "Tail+Mid+Twice":
                plot_q.setdefault(chain, {})["twice"] = mesh["q"]
        _pc_ise_row(K_obs, C, p, K_eval, q_true)
    _density_figure(
        FIG / "ccdf_heston_rnd.pdf", K_eval, q_true, plot_q, "Heston RND"
    )
    return rec


def _density_figure(path, K_eval, q_true, plot_q, true_label):
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6), sharey=True)
    for ax, chain in zip(axes, ("dense", "sparse")):
        ax.plot(K_eval, q_true, color="black", lw=2.0, label=true_label)
        ax.plot(K_eval, plot_q[chain]["mid"], color="#1f77b4", lw=1.35, label="Tail+Midpoint")
        ax.plot(
            K_eval,
            plot_q[chain]["twice"],
            color="#d62728",
            lw=1.35,
            ls="--",
            label="Tail+Mid+Twice",
        )
        if chain == "sparse":
            ax.plot(
                plot_q[chain]["K"],
                np.interp(plot_q[chain]["K"], K_eval, q_true),
                "k.",
                ms=4,
                alpha=0.55,
            )
        ax.set_xlabel(r"Strike $K$")
        ax.set_xlim(60.0, 150.0)
        ax.set_ylim(bottom=0.0)
        ax.set_title(r"Dense, $m=256$" if chain == "dense" else r"Sparse, $m=64$")
        ax.legend(frameon=False, loc="upper right")
    axes[0].set_ylabel(r"$f_{\mathbb{Q}}(K)$")
    fig.tight_layout()
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    print(f"  wrote {path.name}")


def _exact_ise(label, p, C_obs_fn, q_true, K_eval):
    """Oracle / mesh / density-scale ISE for the four nested estimators."""
    rec = {}
    print(f"{label}  S0={p.S0}  F={p.forward:.4f}  T={p.T}")
    for chain, K_obs in (
        ("dense", np.linspace(30.0, 220.0, 256)),
        ("sparse", np.linspace(70.0, 140.0, 64)),
    ):
        C = np.maximum(C_obs_fn(K_obs), 0.0)
        specs = (
            ("Baseline", dict(tails=False, midpoints=False, twice=False)),
            ("Tail completion", dict(tails=True, midpoints=False, twice=False)),
            ("Tail+Midpoint", dict(tails=True, midpoints=True, twice=False)),
            ("Tail+Mid+Twice", dict(tails=True, midpoints=True, twice=True)),
        )
        delta = float(np.median(np.diff(K_obs)))
        hs = np.geomspace(max(0.20 * delta, 1e-3), max(30.0 * delta, 40.0), 40)
        print(f"\n-- {chain}  m={len(K_obs)}")
        rec[chain] = {}
        for name, kw in specs:
            def ise_at(h, _kw=kw):
                out = estimate_rnd(
                    K_obs, C, p.S0, p.r, p.T, p.q, K_eval=K_eval, h=h, **_kw
                )
                return _ise(K_eval, out["q"], q_true)

            best, best_h = np.inf, hs[len(hs) // 2]
            for h in hs:
                err = ise_at(h)
                if err < best:
                    best, best_h = err, float(h)
            mesh = estimate_rnd(
                K_obs, C, p.S0, p.r, p.T, p.q, K_eval=K_eval, h="mesh", **kw
            )
            den = estimate_rnd(
                K_obs, C, p.S0, p.r, p.T, p.q, K_eval=K_eval, h="density", **kw
            )
            ise_m = _ise(K_eval, mesh["q"], q_true)
            ise_d = _ise(K_eval, den["q"], q_true)
            rec[chain][name] = {
                "h_star": best_h,
                "ise_star": best,
                "h_mesh": mesh["h"],
                "ise_mesh": ise_m,
                "h_den": den["h"],
                "ise_den": ise_d,
            }
            print(
                f"  [{name:18s}]  oracle h={best_h:.4g} ISE={best:.4e}  "
                f"mesh h={mesh['h']:.4g} ISE={ise_m:.4e}  "
                f"den h={den['h']:.4g} ISE={ise_d:.4e}"
            )
        _pc_ise_row(K_obs, C, p, K_eval, q_true)
    return rec


def _pc_ise_row(K_obs, C, p, K_eval, q_true):
    """Priestley–Chao of C, with and without a cubic interpolant. n^{-1/9} h."""
    from rnd import _mesh_h

    delta = float(np.median(np.diff(K_obs)))
    hs = np.geomspace(max(0.20 * delta, 1e-3), max(30.0 * delta, 40.0), 40)
    h_mesh = _mesh_h(K_obs)
    for label, fn in (
        ("PC cubic", priestley_chao_cubic),
        ("PC no cubic", priestley_chao_bl),
    ):
        best, best_h = np.inf, hs[len(hs) // 2]
        for h in hs:
            qh, _ = fn(K_obs, C, p.S0, p.r, p.T, K_eval, q=p.q, h=h)
            err = _ise(K_eval, qh, q_true)
            if err < best:
                best, best_h = err, float(h)
        q_m, _ = fn(K_obs, C, p.S0, p.r, p.T, K_eval, q=p.q, h=h_mesh)
        q_d, h_d = fn(K_obs, C, p.S0, p.r, p.T, K_eval, q=p.q, h=None)
        print(
            f"  [{label:18s}]  oracle h={best_h:.4g} ISE={best:.4e}  "
            f"mesh h={h_mesh:.4g} ISE={_ise(K_eval, q_m, q_true):.4e}  "
            f"C'' h={h_d:.4g} ISE={_ise(K_eval, q_d, q_true):.4e}"
        )


def vg_ise():
    p = CM99
    K_eval = np.linspace(60.0, 150.0, 401)
    q_true = vg_spot_density(K_eval, p)
    rec = _exact_ise("VG", p, lambda K: vg_calls(K, p), q_true, K_eval)
    plot_q = {}
    for chain, K_obs in (
        ("dense", np.linspace(30.0, 220.0, 256)),
        ("sparse", np.linspace(70.0, 140.0, 64)),
    ):
        C = np.maximum(vg_calls(K_obs, p), 0.0)
        mid = estimate_rnd(
            K_obs, C, p.S0, p.r, p.T, p.q, K_eval=K_eval, h="mesh",
            tails=True, midpoints=True, twice=False,
        )
        tw = estimate_rnd(
            K_obs, C, p.S0, p.r, p.T, p.q, K_eval=K_eval, h="mesh",
            tails=True, midpoints=True, twice=True,
        )
        plot_q[chain] = {"mid": mid["q"], "twice": tw["q"], "K": K_obs}
    _density_figure(FIG / "ccdf_vg_rnd.pdf", K_eval, q_true, plot_q, "VG RND")
    return rec


def _holdout_ours(sl, twice=True):
    idx = np.arange(len(sl.K))
    rmses = []
    for train, test in ((idx % 2 == 0, idx % 2 == 1), (idx % 2 == 1, idx % 2 == 0)):
        out = estimate_rnd(
            sl.K[train], sl.C[train], sl.S0, sl.r, sl.T, sl.q, twice=twice
        )
        P_te, C_te = out["interpolant"](sl.K[test])
        otm_hat = np.where(sl.K[test] <= sl.F, P_te, C_te)
        otm_true = np.where(sl.K[test] <= sl.F, sl.P[test], sl.C[test])
        rmses.append(_rmse(otm_hat, otm_true))
    return float(np.mean(rmses))


def _holdout_yh(sl):
    idx = np.arange(len(sl.K))
    rmses = []
    for train, test in ((idx % 2 == 0, idx % 2 == 1), (idx % 2 == 1, idx % 2 == 0)):
        intrinsic = sl.disc * np.maximum(sl.F - sl.K[train], 0.0)
        m = convex_decreasing_ls(
            sl.K[train], sl.C[train], sl.disc, intrinsic=intrinsic
        )
        C_te = np.interp(sl.K[test], sl.K[train], m)
        P_te = C_te - sl.S0 * np.exp(-sl.q * sl.T) + sl.K[test] * sl.disc
        otm_hat = np.where(sl.K[test] <= sl.F, P_te, C_te)
        otm_true = np.where(sl.K[test] <= sl.F, sl.P[test], sl.C[test])
        rmses.append(_rmse(otm_hat, otm_true))
    return float(np.mean(rmses))


def _holdout_pca(sl, s_grid):
    idx = np.arange(len(sl.K))
    rmses = []
    for train, test in ((idx % 2 == 0, idx % 2 == 1), (idx % 2 == 1, idx % 2 == 0)):
        q, _ = pca_lognormal(
            sl.K[train], sl.C[train], sl.S0, sl.r, sl.T, s_grid, q=sl.q
        )
        P, C, _, _ = _prices_from_q(s_grid, q, sl.K[test], sl.disc)
        otm_hat = np.where(sl.K[test] <= sl.F, P, C)
        otm_true = np.where(sl.K[test] <= sl.F, sl.P[test], sl.C[test])
        rmses.append(_rmse(otm_hat, otm_true))
    return float(np.mean(rmses))


def _holdout_asd(sl):
    idx = np.arange(len(sl.K))
    rmses = []
    for train, test in ((idx % 2 == 0, idx % 2 == 1), (idx % 2 == 1, idx % 2 == 0)):
        C_te, _, _, _ = ait_sahalia_duarte(
            sl.K[train], sl.C[train], sl.S0, sl.r, sl.T, sl.K[test], q=sl.q
        )
        P_te = C_te - sl.S0 * np.exp(-sl.q * sl.T) + sl.K[test] * sl.disc
        otm_hat = np.where(sl.K[test] <= sl.F, P_te, C_te)
        otm_true = np.where(sl.K[test] <= sl.F, sl.P[test], sl.C[test])
        rmses.append(_rmse(otm_hat, otm_true))
    return float(np.mean(rmses))


def _pc_listed_h(K, C, S0, r, T, q):
    """Second-derivative scale with ATM IV, matching the listed PC row."""
    n = max(len(K), 8)
    F = S0 * np.exp((r - q) * T)
    iv = bs.implied_vol(C, S0, K, r, T, q)
    atm = np.nanmedian(iv[np.abs(K - F) <= 0.03 * F])
    if not np.isfinite(atm):
        atm = np.nanmedian(iv)
    if not np.isfinite(atm):
        atm = 0.2
    return float(1.06 * F * atm * np.sqrt(T) * n ** (-1.0 / 9.0))


def _holdout_pc(sl):
    idx = np.arange(len(sl.K))
    rmses = []
    for train, test in ((idx % 2 == 0, idx % 2 == 1), (idx % 2 == 1, idx % 2 == 0)):
        h = _pc_listed_h(sl.K[train], sl.C[train], sl.S0, sl.r, sl.T, sl.q)
        _, _, C_te = priestley_chao_cubic(
            sl.K[train],
            sl.C[train],
            sl.S0,
            sl.r,
            sl.T,
            sl.K[test],
            q=sl.q,
            h=h,
            return_call=True,
        )
        P_te = C_te - sl.S0 * np.exp(-sl.q * sl.T) + sl.K[test] * sl.disc
        otm_hat = np.where(sl.K[test] <= sl.F, P_te, C_te)
        otm_true = np.where(sl.K[test] <= sl.F, sl.P[test], sl.C[test])
        rmses.append(_rmse(otm_hat, otm_true))
    return float(np.mean(rmses))


def _pack(sl, P, C, q, s_grid, ho, iv_mkt, **extra):
    otm, iv = _otm_iv(sl, P, C)
    otm_mkt = _otm(sl, sl.P, sl.C)
    left, right = sl.K <= sl.F, sl.K > sl.F
    mass, mean = _mass_mean(s_grid, q)
    rec = {
        "otm": _rmse(otm, otm_mkt),
        "puts": _rmse(otm[left], otm_mkt[left]),
        "calls": _rmse(otm[right], otm_mkt[right]),
        "iv": _rmse(iv, iv_mkt),
        "mass": mass,
        "mean": mean,
        "ho": ho,
        "q": q,
        "iv_series": iv,
        "P": P,
        "C": C,
    }
    rec.update(extra)
    return rec


def _print_row(name, rec, extra=""):
    print(
        f"  [{name:22s}]  OTM={rec['otm']:.2f} (puts {rec['puts']:.2f}, "
        f"calls {rec['calls']:.2f})  mass={rec['mass']:.3f}  "
        f"hold-out={rec['ho']:.2f}{extra}"
    )


def score_listed(sl, title):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)
    s_grid = np.linspace(max(50.0, 0.20 * sl.F), 2.40 * sl.F, 2401)
    otm_mkt = _otm(sl, sl.P, sl.C)
    _, iv_mkt = _otm_iv(sl, sl.P, sl.C)
    left, right = sl.K <= sl.F, sl.K > sl.F
    print(
        f"  n={len(sl.K)}  F={sl.F:.0f}  S={sl.S0:.0f}  T={sl.T:.3f}  "
        f"σ={atm_iv(sl):.3f}  K=[{sl.K[0]:.0f},{sl.K[-1]:.0f}]"
    )

    ours = estimate_rnd(sl.K, sl.C, sl.S0, sl.r, sl.T, sl.q, K_eval=s_grid, twice=True)
    rec_o = _pack(
        sl, ours["P"], ours["C"], ours["q"], s_grid, _holdout_ours(sl, twice=True), iv_mkt, h=ours["h"]
    )
    _print_row("Ours", rec_o, extra=f"  h={ours['h']:.1f}")

    h_pc = _pc_listed_h(sl.K, sl.C, sl.S0, sl.r, sl.T, sl.q)
    _, _, C_pc = priestley_chao_cubic(
        sl.K, sl.C, sl.S0, sl.r, sl.T, sl.K, q=sl.q, h=h_pc, return_call=True
    )
    q_pc_g, _, _ = priestley_chao_cubic(
        sl.K, sl.C, sl.S0, sl.r, sl.T, s_grid, q=sl.q, h=h_pc, return_call=True
    )
    P_pc = C_pc - np.exp(-sl.q * sl.T) * sl.S0 + sl.K * sl.disc
    rec_pc = _pack(sl, P_pc, C_pc, q_pc_g, s_grid, _holdout_pc(sl), iv_mkt, h=h_pc)
    _print_row("Priestley–Chao", rec_pc, extra=f"  h={h_pc:.1f}")

    disc = sl.disc
    dfq = np.exp(-sl.q * sl.T)
    intrinsic = disc * np.maximum(sl.F - sl.K, 0.0)
    C_yh = convex_decreasing_ls(sl.K, sl.C, disc, intrinsic=intrinsic)
    P_yh = C_yh - dfq * sl.S0 + sl.K * disc
    _, q_yh, _, _ = yatchew_hardle(
        sl.K, sl.C, sl.S0, sl.r, sl.T, s_grid, q=sl.q, lam=0.0
    )
    rec_y = _pack(sl, P_yh, C_yh, q_yh, s_grid, _holdout_yh(sl), iv_mkt)
    _print_row("Yatchew–Härdle λ=0", rec_y)

    C_asd, q_asd, _, _ = ait_sahalia_duarte(
        sl.K, sl.C, sl.S0, sl.r, sl.T, s_grid, q=sl.q
    )
    C_asd_q, _, _, _ = ait_sahalia_duarte(
        sl.K, sl.C, sl.S0, sl.r, sl.T, sl.K, q=sl.q
    )
    P_asd_q = C_asd_q - np.exp(-sl.q * sl.T) * sl.S0 + sl.K * sl.disc
    rec_asd = _pack(sl, P_asd_q, C_asd_q, q_asd, s_grid, _holdout_asd(sl), iv_mkt)
    _print_row("Aït-Sahalia–Duarte", rec_asd)

    q_pca, _ = pca_lognormal(sl.K, sl.C, sl.S0, sl.r, sl.T, s_grid, q=sl.q)
    P_p, C_p, _, _ = _prices_from_q(s_grid, q_pca, sl.K, sl.disc)
    rec_p = _pack(sl, P_p, C_p, q_pca, s_grid, _holdout_pca(sl, s_grid), iv_mkt)
    _print_row("PCA", rec_p)
    return {
        "sl": sl,
        "s_grid": s_grid,
        "iv_mkt": iv_mkt,
        "ours": rec_o,
        "pc": rec_pc,
        "yh": rec_y,
        "asd": rec_asd,
        "pca": rec_p,
        "title": title,
    }


def plot_listed(rows):
    """rows: list of score_listed dicts to show (December SPX, NDX, RUT)."""
    _style()
    fig, axes = plt.subplots(len(rows), 2, figsize=(9.6, 3.15 * len(rows)))
    if len(rows) == 1:
        axes = np.array([axes])
    for axrow, rec in zip(axes, rows):
        axq, axiv = axrow
        sl, s = rec["sl"], rec["s_grid"]
        lo, hi = 0.55 * sl.F, 1.40 * sl.F
        q_o = np.maximum(rec["ours"]["q"], 0.0)
        q_pc = np.maximum(rec["pc"]["q"], 0.0)
        q_a = np.maximum(rec["asd"]["q"], 0.0)
        q_p = np.maximum(rec["pca"]["q"], 0.0)
        axq.plot(s, q_o, color="#1f77b4", lw=1.4, label="Ours")
        axq.plot(s, q_pc, color="#9467bd", lw=1.15, ls=":", label="Priestley–Chao")
        axq.plot(s, q_a, color="#8c564b", lw=1.15, ls="--", label="Aït-Sahalia–Duarte")
        axq.plot(s, q_p, color="#2ca02c", lw=1.4, ls="-.", label="PCA")
        core = (s >= 0.65 * sl.F) & (s <= 1.30 * sl.F)
        ymax = 1.12 * max(
            float(q_o[core].max()),
            float(q_pc[core].max()),
            float(q_a[core].max()),
            float(q_p[core].max()),
            1e-12,
        )
        axq.axvline(sl.F, color="0.5", ls="--", lw=0.8)
        axq.set_xlim(lo, hi)
        axq.set_ylim(0.0, ymax)
        axq.set_ylabel(r"$f_{\mathbb{Q}}(K)$")
        axq.set_title(rec["title"])
        axq.legend(frameon=False, loc="upper right")
        axq.set_xlabel(r"Strike $K$")

        iv_m = rec["iv_mkt"]
        show = np.isfinite(iv_m) & (sl.K >= lo) & (sl.K <= hi)
        axiv.set_xlim(lo, hi)
        ivs = iv_m[show]
        y1 = float(np.nanpercentile(ivs, 5)) - 0.02 if ivs.size else 0.08
        y2 = float(np.nanpercentile(ivs, 95)) + 0.04 if ivs.size else 0.45
        axiv.set_ylim(max(0.05, y1), min(0.55, y2))
        axiv.plot(sl.K[show], iv_m[show], "k.", ms=3, alpha=0.40, label="Market")
        axiv.plot(sl.K[show], rec["ours"]["iv_series"][show], color="#1f77b4", lw=1.2, label="Ours")
        axiv.plot(
            sl.K[show], rec["yh"]["iv_series"][show], color="#ff7f0e", lw=1.2, ls=":", label="Yatchew–Härdle"
        )
        axiv.plot(
            sl.K[show], rec["pca"]["iv_series"][show], color="#2ca02c", lw=1.2, ls="-.", label="PCA"
        )
        axiv.axvline(sl.F, color="0.5", ls="--", lw=0.8)
        axiv.set_ylabel("OTM implied vol")
        axiv.set_xlabel(r"Strike $K$")
        axiv.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(FIG / "ccdf_listed.pdf", facecolor="white")
    plt.close(fig)
    print("  wrote ccdf_listed.pdf")


def listed_slice(csv_name, json_name, symbol, expiry, asof, root="SPX"):
    """Load the working OTM slice from CSV; fall back to a CBOE JSON snapshot."""
    csv_path = RES / csv_name
    if csv_path.exists():
        return load_slice(csv_path)
    raw = fetch_cboe(RES / json_name, symbol=symbol)
    sl = build_otm_slice(raw, expiry=expiry, r=0.04, root=root, asof=asof)
    save_slice(sl, csv_path)
    return sl


def main():
    heston_ise()
    vg_ise()
    spx_dec = score_listed(
        listed_slice(
            "spx_20261218.csv", "cboe_spx.json", "SPX",
            date(2026, 12, 18), date(2026, 9, 6),
        ),
        "SPX 18 Dec 2026",
    )
    spx_mar = score_listed(
        listed_slice(
            "spx_20270319.csv", "cboe_spx.json", "SPX",
            date(2027, 3, 19), date(2026, 9, 6),
        ),
        "SPX 19 Mar 2027",
    )
    ndx_dec = score_listed(
        listed_slice(
            "ndx_20261218.csv", "cboe_ndx.json", "NDX",
            date(2026, 12, 18), date(2026, 9, 8), root="NDX",
        ),
        "NDX 18 Dec 2026",
    )
    rut_dec = score_listed(
        listed_slice(
            "rut_20261218.csv", "cboe_rut.json", "RUT",
            date(2026, 12, 18), date(2026, 9, 19), root="RUT",
        ),
        "RUT 18 Dec 2026",
    )
    plot_listed([spx_dec, ndx_dec, rut_dec])
    return spx_dec, spx_mar, ndx_dec, rut_dec


if __name__ == "__main__":
    main()
