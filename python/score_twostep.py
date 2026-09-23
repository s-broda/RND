#!/usr/bin/env python3
"""Score the two-step estimator on the v2 noise design and the four listed slices.

Run from the repository root::

    python3 python/score_twostep.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PY = Path(__file__).resolve().parent
sys.path.insert(0, str(PY))

from replicate_v2 import _add_iv_noise, _irregular_strikes, _ise, _style
from rnd import estimate_rnd
from src.competitors_v2 import diagnostics, otm_rmse
from src.heston import BCC97, carr_madan_calls, heston_spot_density
from src.spx import load_slice
from src.twostep import estimate_twostep
from src.vg import CM99, vg_calls, vg_spot_density

ROOT = PY.parent
FIG = ROOT / "figures_v2"
RES = PY / "results"
N_REPS = 30
SEED = 20260923


def _noise(label, call_fn, dens_fn, params):
    rng = np.random.default_rng(SEED)
    F = params.forward
    K = _irregular_strikes(F)
    Ctrue = np.maximum(call_fn(K), 0.0)
    Kise = np.linspace(60.0, 140.0, 241)
    qtrue = dens_fn(Kise)
    ises, rms = [], []
    show = None
    show_gap = -1.0
    for i in range(N_REPS):
        Cobs, _ = _add_iv_noise(K, Ctrue, params.S0, params.r, params.T, params.q, rng)
        est = estimate_twostep(
            K, Cobs, params.S0, params.r, params.T, params.q, F=F, K_eval=Kise, h="density", twice=True
        )
        raw = estimate_rnd(
            K, Cobs, params.S0, params.r, params.T, params.q, K_eval=Kise, h="density", twice=True
        )
        ise = _ise(Kise, est["q"], qtrue)
        ise_raw = _ise(Kise, raw["q"], qtrue)
        rm = otm_rmse(K, est["interpolant"](K)[1], Ctrue, params.S0, params.r, params.T, params.q, F)
        ises.append(ise)
        rms.append(rm)
        # The figure shows the draw on which the projection removes the most error.
        if label == "Heston" and ise_raw - ise > show_gap:
            show_gap = ise_raw - ise
            show = (Kise, qtrue, est["q"], raw["q"], ise, ise_raw, i)
    a = np.asarray(ises)
    b = np.asarray(rms)
    print(
        f"{label} n={N_REPS} ISE {a.mean():.6e} sd {a.std(ddof=1):.6e} "
        f"RMSE {b.mean():.4f} sd {b.std(ddof=1):.4f}"
    )
    return a, b, show


def _listed():
    print("\nlisted two-step")
    for name in ("spx_20261218.csv", "spx_20270319.csv", "ndx_20261218.csv", "rut_20261218.csv"):
        sl = load_slice(RES / name)
        grid = np.linspace(max(1.0, 0.15 * sl.F), 2.60 * sl.F, 1601)
        est = estimate_twostep(
            sl.K, sl.C, sl.S0, sl.r, sl.T, sl.q, F=sl.F, K_eval=grid, h="density", twice=True
        )
        rm = otm_rmse(sl.K, est["interpolant"](sl.K)[1], sl.C, sl.S0, sl.r, sl.T, sl.q, sl.F)
        d = diagnostics(grid, est["q"], sl.F)
        idx = np.arange(len(sl.K))
        errs = []
        for train, test in ((idx % 2 == 0, idx % 2 == 1), (idx % 2 == 1, idx % 2 == 0)):
            hold = estimate_twostep(
                sl.K[train], sl.C[train], sl.S0, sl.r, sl.T, sl.q, F=sl.F, h="density", twice=True
            )
            _, c_te = hold["interpolant"](sl.K[test])
            errs.append(otm_rmse(sl.K[test], c_te, sl.C[test], sl.S0, sl.r, sl.T, sl.q, sl.F))
        print(
            f"  {name:22} h={est['h']:.1f} rmse={rm:.4f} hold={float(np.mean(errs)):.4f} "
            f"pos={d['positive']:.3f} neg={d['negative']:.3f}"
        )


def _figure(pack):
    if pack is None:
        return
    Kise, qtrue, q_two, q_raw, ise, ise_raw, draw = pack
    _style()
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.plot(Kise, qtrue, color="black", lw=2.0, label="True")
    ax.plot(Kise, np.maximum(q_two, 0.0), color="#1f77b4", lw=1.5, label="Two-step")
    ax.plot(Kise, np.maximum(q_raw, 0.0), color="#ff7f0e", lw=1.2, ls="--", label="Kernel on raw quotes")
    ax.set_xlim(60.0, 140.0)
    ax.set_ylim(bottom=0.0)
    ax.set_xlabel(r"Strike $K$")
    ax.set_ylabel(r"$f_{\mathbb{Q}}(K)$")
    ax.set_title("Heston draw where the raw kernel misses by the most")
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    path = FIG / "noise_twostep.pdf"
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    print(f"wrote {path}")


def main():
    h_ise, _, pack = _noise(
        "Heston",
        lambda K: carr_madan_calls(K, BCC97),
        lambda K: heston_spot_density(K, BCC97),
        BCC97,
    )
    _noise(
        "VG",
        lambda K: vg_calls(K, CM99),
        lambda K: vg_spot_density(K, CM99),
        CM99,
    )
    _figure(pack)
    _listed()
    return h_ise


if __name__ == "__main__":
    main()
