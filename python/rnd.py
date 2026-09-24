"""Closed-form risk-neutral density from a single-maturity call strip.

The public entry point is ``estimate_rnd``. Copy this file: it is self-contained
apart from NumPy and SciPy. The estimator rescales calls to a complementary
cdf, completes the unquoted tails, places masses at cell centers, convolves with a
Gaussian kernel, and (by default) applies Schucany–Sommers twicing.
"""

from __future__ import annotations

import numpy as np
from scipy.special import erf
from scipy.stats import norm


def estimate_rnd(
    K,
    C,
    S0,
    r,
    T,
    q=0.0,
    K_eval=None,
    h=None,
    twice=True,
    tails=True,
    midpoints=True,
    n_left=None,
    n_right=None,
):
    """Headline estimator of the risk-neutral density of ``S_T``.

    Parameters
    ----------
    K, C : array_like
        Quoted strikes and European call prices, one expiry.
    S0, r, T : float
        Spot, continuously compounded rate, time to expiry in years.
    q : float
        Continuous dividend yield.
    K_eval : array_like, optional
        Strikes at which to return the density. Defaults to ``K``.
    h : float, optional
        Gaussian bandwidth. Default is the density-scale rule
        ``1.06 F σ_ATM √T n^{-1/5}`` with ``n`` the number of masses after the
        right wing. Pass a number, or ``"mesh"`` for ``h = 1.2 × median ΔK``.
    twice : bool
        If True (default), return ``2 f_h − f_{h√2}`` and the same combination
        of the interpolant.
    tails : bool
        Linear left wing through the origin and linear call decay to zero.
    midpoints : bool
        Place each jump at the center of its cell rather than the right end.
    n_left : int, optional
        Number of filler knots on the left wing. Default is
        ``round(K_1 / median ΔK)``, so the completed mesh continues the
        quoted spacing through the splice at ``K_1``.
    n_right : int, optional
        Number of filler knots on the right wing. Default is
        ``round(8 (K_end - K_m) / (K_m - K_{m-1}))``, eight knots per last
        quoted gap along the linear call tail.

    Returns
    -------
    dict
        ``q`` density on ``K_eval``; ``P``, ``C`` put and call interpolants on
        the quoted ``K``; ``h`` the bandwidth used; ``K_mass``, ``dp`` the
        completed midpoint support; ``n_ext`` mass count after the right wing;
        ``n_left`` the left-wing count used; ``n_right`` the right-wing count
        used; ``interpolant(K_pts) -> (P, C)``.
    """
    K = np.asarray(K, dtype=float)
    C = np.asarray(C, dtype=float)
    order = np.argsort(K)
    K, C = K[order], C[order]
    disc = float(np.exp(-r * T))
    dfq = float(np.exp(-q * T))
    stock = float(S0) * dfq
    F = float(S0) * np.exp((r - q) * T)
    if K_eval is None:
        K_eval = K
    else:
        K_eval = np.asarray(K_eval, dtype=float)

    if n_left is None:
        n_left = _n_left_from_mesh(K)
    else:
        n_left = max(int(n_left), 2)
    if n_right is None:
        n_right = _n_right_from_gap(K, C, F, disc)
    else:
        n_right = max(int(n_right), 0)

    K_work, C_work = K, C
    n_ext = len(K)
    if tails:
        K_work, C_work = _complete_c_tail(K, C, F, disc, n_tail=n_right)
        n_ext = len(K_work)
    Ko, dG = _normalized_jumps(K_work, C_work, stock)
    if tails:
        Ko, dG = _complete_left(Ko, dG, n_left=n_left)
    if midpoints:
        Ko, dG = _midpoint_support(Ko, dG)

    if h is None or h == "density":
        h_use = _density_h(K, C, S0, r, T, q, F, n_ext)
    elif h == "mesh":
        h_use = _mesh_h(K)
    else:
        h_use = float(h)

    qhat = _qhat(K_eval, Ko, dG, h_use, F)

    def interpolant(K_pts):
        G = _Fhat(K_pts, Ko, dG, h_use)
        if twice:
            G = 2.0 * G - _Fhat(K_pts, Ko, dG, h_use * np.sqrt(2.0))
        G = np.clip(G, 0.0, 1.0)
        C_hat = stock * np.clip(1.0 - G, 0.0, 1.0)
        P_hat = np.maximum(C_hat - stock + np.asarray(K_pts, dtype=float) * disc, 0.0)
        return P_hat, C_hat

    if twice:
        qhat = 2.0 * qhat - _qhat(K_eval, Ko, dG, h_use * np.sqrt(2.0), F)
    P_hat, C_hat = interpolant(K)
    return {
        "q": qhat,
        "P": P_hat,
        "C": C_hat,
        "h": float(h_use),
        "K_mass": Ko,
        "dp": dG,
        "n_ext": int(n_ext),
        "n_left": int(n_left),
        "n_right": int(n_right),
        "F": F,
        "interpolant": interpolant,
    }


def _normalized_jumps(K, C, stock):
    c = np.clip(C / max(float(stock), 1e-16), 0.0, 1.0)
    c = np.minimum.accumulate(c)
    G = np.maximum.accumulate(np.clip(1.0 - c, 0.0, 1.0))
    dG = np.maximum(np.diff(G, prepend=0.0), 0.0)
    return K, dG


def _right_wing(K, C, F, disc, k_max=None):
    """Linear intercept of leftover C_m, or None if the wing is empty."""
    K = np.asarray(K, dtype=float)
    C = np.asarray(C, dtype=float)
    if len(K) == 0:
        return None
    K_m = float(K[-1])
    C_m = float(max(C[-1], 0.0))
    if C_m <= 1e-8:
        return None
    slp = None
    # A flat last step (two quotes on the premium floor) is not a decay.
    # Counting the cap-to-8F quadrature as observations then shrinks h.
    for i in range(len(K) - 1, 0, -1):
        dk = float(K[i] - K[i - 1])
        if dk <= 0.0:
            continue
        step = float(C[i] - C[i - 1]) / dk
        if step < -1e-8:
            slp = step
            break
    if slp is None:
        slp = -float(disc) * C_m / max(K_m, 1.0)
    slp = float(np.clip(slp, -float(disc), -1e-8))
    K_end = K_m + C_m / (-slp)
    cap = 8.0 * float(F) if k_max is None else float(k_max)
    K_end = min(max(K_end, K_m * 1.01), cap)
    if K_end <= K_m * 1.001:
        return None
    return K_m, C_m, slp, K_end


def _n_right_from_gap(K, C, F, disc, k_max=None):
    """Eight knots per last quoted gap along the linear call tail."""
    spec = _right_wing(K, C, F, disc, k_max=k_max)
    if spec is None:
        return 0
    K_m, _, _, K_end = spec
    last = float(K[-1] - K[-2]) if len(K) >= 2 else 0.0
    if last <= 0.0:
        return 2
    return max(1, int(round(8.0 * (K_end - K_m) / last)))


def _complete_c_tail(K, C, F, disc, n_tail=None, k_max=None):
    """Linear no-arbitrage decay of leftover C_m to 0."""
    K = np.asarray(K, dtype=float)
    C = np.asarray(C, dtype=float)
    spec = _right_wing(K, C, F, disc, k_max=k_max)
    if spec is None:
        return K, C
    K_m, C_m, slp, K_end = spec
    if n_tail is None:
        n_tail = _n_right_from_gap(K, C, F, disc, k_max=k_max)
    n_tail = int(n_tail)
    if n_tail < 1:
        return K, C
    K_t = np.linspace(K_m, K_end, n_tail + 1)[1:]
    C_t = np.maximum(C_m + slp * (K_t - K_m), 0.0)
    return np.concatenate([K, K_t]), np.concatenate([C, C_t])


def _n_left_from_mesh(K):
    """Continue the quoted median spacing through (0, K_1]."""
    K = np.asarray(K, dtype=float)
    if K.size < 2 or K[0] <= 0.0:
        return 2
    delta = float(np.median(np.diff(K)))
    if not np.isfinite(delta) or delta <= 0.0:
        return 2
    return max(2, int(round(float(K[0]) / delta)))


def _complete_left(K, dG, n_left):
    K = np.asarray(K, dtype=float)
    dG = np.asarray(dG, dtype=float)
    g1 = float(dG[0])
    if g1 <= 1e-16 or K[0] <= 1e-8:
        return K, dG
    K_l = np.linspace(K[0] / n_left, K[0], n_left)
    G_l = g1 * (K_l / K[0])
    dG_l = np.maximum(np.diff(G_l, prepend=0.0), 0.0)
    dG = dG.copy()
    dG[0] = 0.0
    return np.concatenate([K_l, K[1:]]), np.concatenate([dG_l, dG[1:]])


def _midpoint_support(K, dp):
    """Place each jump at the center of (K_{i-1}, K_i] with K_0 := 0."""
    K = np.asarray(K, dtype=float)
    dp = np.asarray(dp, dtype=float)
    left = np.empty_like(K)
    left[0] = 0.0
    left[1:] = K[:-1]
    return 0.5 * (left + K), dp


def _mesh_h(K):
    K = np.sort(np.asarray(K, dtype=float))
    delta = float(np.median(np.diff(K)))
    span = float(K[-1] - K[0])
    return float(min(1.2 * delta, 0.30 * span))


def _gauss_pdf(u):
    kap = np.exp(-0.5 * u * u) / np.sqrt(2.0 * np.pi)
    return kap, -u * kap


def _gauss_cdf(u):
    return 0.5 * (1.0 + erf(np.asarray(u, dtype=float) / np.sqrt(2.0)))


def _Fhat(K_eval, K_obs, dp, h):
    K_eval = np.atleast_1d(np.asarray(K_eval, dtype=float))
    K_obs = np.asarray(K_obs, dtype=float)
    dp = np.asarray(dp, dtype=float)
    hv = max(float(h), 1e-16)
    u = (K_eval[:, None] - K_obs[None, :]) / hv
    return (dp[None, :] * _gauss_cdf(u)).sum(axis=1)


def _qhat(K_eval, K_obs, dG, h, F):
    hv = max(float(h), 1e-16)
    u = (K_eval[:, None] - K_obs[None, :]) / hv
    _kap, dkap = _gauss_pdf(u)
    gp = (dG[None, :] * dkap / (hv * hv)).sum(axis=1)
    return -float(F) * gp


def _density_h(K, C, S0, r, T, q, F, n_ext):
    n = max(int(n_ext), 8)
    iv = _implied_vol(C, S0, K, r, T, q)
    atm = np.nanmedian(iv[np.abs(K - F) <= 0.03 * F])
    if not np.isfinite(atm):
        med = np.nanmedian(iv)
        atm = float(med) if np.isfinite(med) else 0.16
    return float(1.06 * F * atm * np.sqrt(T) * n ** (-0.2))


def _d1_d2(S, K, r, T, sig, q=0.0):
    S = np.asarray(S, dtype=float)
    K = np.asarray(K, dtype=float)
    sig = np.maximum(np.asarray(sig, dtype=float), 1e-12)
    sqrtT = np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sig**2) * T) / (sig * sqrtT)
    return d1, d1 - sig * sqrtT


def _call_price(S, K, r, T, sig, q=0.0):
    d1, d2 = _d1_d2(S, K, r, T, sig, q)
    return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def _vega(S, K, r, T, sig, q=0.0):
    d1, _ = _d1_d2(S, K, r, T, sig, q)
    return S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T)


def _implied_vol(price, S, K, r, T, q=0.0, is_call=True, tol=1e-8, maxiter=40):
    """Implied vol via Newton on the OTM premium, bisection fallback."""
    K = np.atleast_1d(np.asarray(K, dtype=float))
    price = np.broadcast_to(np.asarray(price, dtype=float), K.shape).copy()
    disc = np.exp(-r * T)
    dfq = np.exp(-q * T)
    F = S * np.exp((r - q) * T)
    if is_call:
        otm = np.where(K < F, price - dfq * S + disc * K, price)
    else:
        otm = np.where(K >= F, price + dfq * S - disc * K, price)
    intrinsic = np.maximum(dfq * S - disc * K, 0.0)
    call = np.where(K < F, otm + dfq * S - disc * K, otm)
    lo_bound = intrinsic + 1e-12
    hi_bound = dfq * S - 1e-12
    ok = np.isfinite(call) & (call > lo_bound) & (call < hi_bound)
    otm_prem = np.where(K < F, call - intrinsic, call)
    sig = np.sqrt(2.0 * np.pi / max(T, 1e-12)) * np.maximum(otm_prem, 1e-12) / max(S * dfq, 1e-12)
    sig = np.clip(sig, 0.05, 1.5)
    sig = np.where(ok, sig, 0.2)
    lo = np.full(K.shape, 1e-5)
    hi = np.full(K.shape, 4.0)
    for _ in range(maxiter):
        pr = _call_price(S, K, r, T, sig, q)
        vg = _vega(S, K, r, T, sig, q)
        diff = pr - call
        hi = np.where(diff > 0, np.minimum(hi, sig), hi)
        lo = np.where(diff < 0, np.maximum(lo, sig), lo)
        step = np.zeros_like(sig)
        good = ok & (vg > 1e-10)
        step[good] = diff[good] / vg[good]
        sig_n = sig - step
        outside = (sig_n <= lo) | (sig_n >= hi) | ~np.isfinite(sig_n)
        sig_n = np.where(outside, 0.5 * (lo + hi), sig_n)
        sig = np.where(ok, np.clip(sig_n, 1e-5, 4.0), sig)
        if np.max(np.abs(diff[ok])) < tol if np.any(ok) else True:
            break
    sig = np.where(ok, sig, np.nan)
    if sig.size == 1:
        return float(sig.reshape(-1)[0])
    return sig
