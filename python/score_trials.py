"""Score the three trial switches against the paper rule.

Not part of the replication. From the repository root::

    python3 python/score_trials.py factors
    python3 python/score_trials.py exact
    python3 python/score_trials.py listed
    python3 python/score_trials.py noise
    python3 python/score_trials.py appendix

``factors`` prints the raw butterfly scale and how many gaps exceed ``h``.
The other stages print the paper row beside each switch. Noise uses the
paper's seed and draws, and scores only this estimator.
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import numpy as np

PY = Path(__file__).resolve().parent
ROOT = PY.parent
sys.path.insert(0, str(PY))

from replicate import (  # noqa: E402
    RES,
    _folds,
    _ise,
    _iv_noise,
    _mass_peaks,
    _otm,
    _otm_rmse_slice,
    _projected_calls,
    _variation,
)
from rnd import _butterfly_factor, estimate_rnd  # noqa: E402
from src.heston import BCC97, carr_madan_puts, heston_spot_density  # noqa: E402
from src.spx import asof_from_raw, build_otm_slice, fetch_cboe, load_slice  # noqa: E402
from src.vg import CM99, vg_calls, vg_spot_density  # noqa: E402

VARIANTS = (
    ("paper", {}),
    ("hole", {"fill": "hole"}),
    ("fill", {"fill": "interior"}),
    ("pieces", {"pieces": True}),
    ("shape", {"shape": True, "shape_lo": 0.90, "shape_hi": 1.10}),
    ("shape95", {"shape": True, "shape_lo": 0.95, "shape_hi": 1.05}),
)


def _variants():
    """Optional comma-separated names in argv[2] restrict a stage."""
    if len(sys.argv) > 2:
        want = {name.strip() for name in sys.argv[2].split(",") if name.strip()}
        picked = tuple(v for v in VARIANTS if v[0] in want)
        if not picked:
            raise SystemExit(f"no variants in {sys.argv[2]}")
        return picked
    return VARIANTS

LISTED = (
    ("spx_20261218.csv", "SPX Dec"),
    ("spx_20270319.csv", "SPX Mar"),
    ("ndx_20261218.csv", "NDX Dec"),
    ("rut_20261218.csv", "RUT Dec"),
)

EXACT_GRIDS = (
    ("dense", np.linspace(30.0, 220.0, 256)),
    ("sparse", np.linspace(70.0, 140.0, 32)),
)


def _heston_calls(K, p=BCC97):
    return np.maximum(
        carr_madan_puts(K, p) + p.S0 * np.exp(-p.q * p.T) - K * p.disc, 0.0,
    )


def _designs():
    return (
        ("Heston", BCC97, _heston_calls, lambda ev, p: heston_spot_density(ev, p)),
        ("VG", CM99, lambda K, p: np.maximum(vg_calls(K, p), 0.0),
         lambda ev, p: vg_spot_density(ev, p)),
    )


def _say(msg):
    print(msg, flush=True)


def factors():
    """Raw scale, and how many quoted gaps exceed the paper bandwidth."""
    _say("\nFactors")
    K_eval = np.linspace(60.0, 150.0, 5)
    for label, p, calls, _dens in _designs():
        for chain, K in EXACT_GRIDS:
            C = calls(K, p)
            fit = estimate_rnd(K, C, p.S0, p.r, p.T, p.q, K_eval=K_eval[:1])
            raw = _butterfly_factor(K, C, p.S0, p.r, p.T, p.q, p.forward)
            gaps = np.diff(K)
            _say(
                f"  {label:8} {chain:7} h={fit['h']:.4g} "
                f"factor={raw:.3f} gaps>h {(gaps > fit['h']).sum()}/{gaps.size} "
                f"maxgap={gaps.max():.3g}"
            )
    for csv, title in LISTED:
        sl = load_slice(RES / csv)
        C = _projected_calls(sl.K, sl.C, sl.r, sl.T, sl.F)
        fit = estimate_rnd(sl.K, C, sl.S0, sl.r, sl.T, sl.q, K_price=sl.K[:1])
        raw = _butterfly_factor(sl.K, C, sl.S0, sl.r, sl.T, sl.q, sl.F)
        gaps = np.diff(np.sort(sl.K))
        wide = gaps > fit["h"]
        _say(
            f"  {title:8} n={len(sl.K):4d} h={fit['h']:.2f} c={fit['c']:.3e} "
            f"factor={raw:.3f} gaps>h {int(wide.sum())}/{gaps.size} "
            f"maxgap={float(gaps.max()):.1f} fill={fit['n_fill']}"
        )
        for train, _test in _folds(len(sl.K)):
            Kt = sl.K[train]
            Ct = _projected_calls(Kt, sl.C[train], sl.r, sl.T, sl.F)
            half = estimate_rnd(Kt, Ct, sl.S0, sl.r, sl.T, sl.q, K_price=Kt[:1])
            g = np.diff(np.sort(Kt))
            _say(
                f"    half n={len(Kt):4d} h={half['h']:.2f} "
                f"gaps>h {int((g > half['h']).sum())}/{g.size} maxgap={float(g.max()):.1f}"
            )


def exact():
    _say("\nExact ISE")
    K_eval = np.linspace(60.0, 150.0, 401)
    for label, p, calls, dens in _designs():
        q_true = dens(K_eval, p)
        _say(f"  -- {label}")
        for chain, K in EXACT_GRIDS:
            C = calls(K, p)
            bits = []
            for name, kw in _variants():
                fit = estimate_rnd(K, C, p.S0, p.r, p.T, p.q, K_eval=K_eval, **kw)
                err = _ise(K_eval, fit["q"], q_true)
                bits.append(
                    f"{name} {err:.4e} h={fit['h']:.4g} c={fit['c']:.2e} "
                    f"fill={fit['n_fill']} pcs={fit['n_pieces']} s={fit['shape']:.3f}"
                )
            _say(f"    {chain}")
            for bit in bits:
                _say(f"      {bit}")


def _listed_one(sl, kw):
    C = _projected_calls(sl.K, sl.C, sl.r, sl.T, sl.F)
    s = np.linspace(max(50.0, 0.2 * sl.F), 2.4 * sl.F, 1601)
    fit = estimate_rnd(
        sl.K, C, sl.S0, sl.r, sl.T, sl.q, K_eval=s, K_price=sl.K, **kw,
    )
    o, pu, ca = _otm(sl, fit["C"])
    mass, peaks = _mass_peaks(sl.F, fit["q"], s)
    tv = _variation(sl.F, fit["q"], s)
    errs = []
    for train, test in _folds(len(sl.K)):
        Ct = _projected_calls(sl.K[train], sl.C[train], sl.r, sl.T, sl.F)
        half = estimate_rnd(
            sl.K[train], Ct, sl.S0, sl.r, sl.T, sl.q, K_price=sl.K[test], **kw,
        )
        errs.append(_otm_rmse_slice(sl, half["C"], test))
    ho = float(np.mean(errs))
    return fit, o, pu, ca, ho, mass, peaks, tv


def listed():
    _say("\nListed")
    totals = {name: [] for name, _kw in _variants()}
    for csv, title in LISTED:
        sl = load_slice(RES / csv)
        _say(f"  -- {title}")
        for name, kw in _variants():
            fit, o, pu, ca, ho, mass, peaks, tv = _listed_one(sl, kw)
            totals[name].append(ho)
            _say(
                f"    {name:12} h={fit['h']:.2f} c={fit['c']:.3e} "
                f"OTM {o:.3f} puts {pu:.3f} calls {ca:.3f} hold {ho:.3f} "
                f"mass {mass:.1f} peaks {peaks} tv {tv:.2f} "
                f"fill={fit['n_fill']} pcs={fit['n_pieces']} s={fit['shape']:.3f}"
            )
    _say("  -- average hold-out")
    for name, _kw in _variants():
        _say(f"    {name:12} {float(np.mean(totals[name])):.4f}")


def noise(n_reps=30, seed=20260923, sd=0.01):
    """Same draws as the paper: dense reps first, then the 32-strike reps."""
    _say(f"\nNoisy Heston  reps={n_reps}  iv sd={sd}  seed={seed}")
    K_eval = np.linspace(60.0, 150.0, 401)
    p = BCC97
    q_true = heston_spot_density(K_eval, p)
    for chain, K in EXACT_GRIDS:
        C_true = _heston_calls(K, p)
        acc = {name: [] for name, _kw in _variants()}
        meta = {name: [] for name, _kw in _variants()}
        # The paper uses one generator for both chains, dense first.
        stream = np.random.default_rng(seed)
        if chain == "sparse":
            dense_k = EXACT_GRIDS[0][1]
            for _rep in range(n_reps):
                _iv_noise(
                    dense_k, _heston_calls(dense_k, p),
                    p.S0, p.r, p.T, p.q, stream, sd=sd,
                )
        for rep in range(n_reps):
            C_obs = _iv_noise(K, C_true, p.S0, p.r, p.T, p.q, stream, sd=sd)
            C_proj = _projected_calls(K, C_obs, p.r, p.T, p.forward)
            for name, kw in _variants():
                fit = estimate_rnd(K, C_proj, p.S0, p.r, p.T, p.q, K_eval=K_eval, **kw)
                acc[name].append(_ise(K_eval, fit["q"], q_true))
                meta[name].append((fit["c"], fit["n_pieces"], fit["shape"], fit["n_fill"]))
            if (rep + 1) % 10 == 0:
                _say(f"    {chain} {rep + 1}/{n_reps}")
        _say(f"  -- {chain}")
        for name, _kw in _variants():
            vals = np.asarray(acc[name], float)
            pcs = np.mean([m[1] for m in meta[name]])
            shp = np.mean([m[2] for m in meta[name]])
            _say(
                f"    {name:12} {vals.mean():.4e}  "
                f"median c={np.median([m[0] for m in meta[name]]):.3e} "
                f"pcs={pcs:.2f} s={shp:.3f}"
            )


def _appendix_slices():
    paper = {
        ("SPX", date(2026, 12, 18)),
        ("SPX", date(2027, 3, 19)),
        ("NDX", date(2026, 12, 18)),
        ("RUT", date(2026, 12, 18)),
    }
    out = []
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
        out.extend(built)
    return out


def appendix():
    _say("\nAppendix hold-out")
    slices = _appendix_slices()
    _say(f"  chains {len(slices)}")
    totals = {name: [] for name, _kw in _variants()}
    for sl in slices:
        _say(f"  -- {sl.root} {sl.expiry.isoformat()} n={len(sl.K)}")
        for name, kw in _variants():
            fit, o, _pu, _ca, ho, _mass, peaks, tv = _listed_one(sl, kw)
            totals[name].append((ho, o, peaks, tv))
            _say(
                f"    {name:12} h={fit['h']:.2f} c={fit['c']:.3e} "
                f"OTM {o:.3f} hold {ho:.3f} peaks {peaks} tv {tv:.2f} "
                f"fill={fit['n_fill']} pcs={fit['n_pieces']} s={fit['shape']:.3f}"
            )
    _say("  -- average hold-out / OTM")
    for name, _kw in _variants():
        rows = totals[name]
        _say(
            f"    {name:12} hold {np.mean([r[0] for r in rows]):.4f} "
            f"OTM {np.mean([r[1] for r in rows]):.4f}"
        )


def main():
    stage = sys.argv[1] if len(sys.argv) > 1 else "factors"
    dispatch = {
        "factors": factors,
        "exact": exact,
        "listed": listed,
        "noise": noise,
        "appendix": appendix,
    }
    if stage == "all":
        for name in ("factors", "exact", "listed", "noise", "appendix"):
            dispatch[name]()
        return
    if stage not in dispatch:
        raise SystemExit(f"unknown stage {stage}")
    dispatch[stage]()


if __name__ == "__main__":
    main()
