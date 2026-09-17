"""Black–Scholes prices, greeks, implied volatility, and lognormal RND."""

from __future__ import annotations

import numpy as np
from scipy.stats import norm


def _d1_d2(S, K, r, T, sig, q=0.0):
    S = np.asarray(S, dtype=float)
    K = np.asarray(K, dtype=float)
    sig = np.asarray(sig, dtype=float)
    sqrtT = np.sqrt(T)
    sig = np.maximum(sig, 1e-12)
    d1 = (np.log(S / K) + (r - q + 0.5 * sig**2) * T) / (sig * sqrtT)
    d2 = d1 - sig * sqrtT
    return d1, d2


def call_price(S, K, r, T, sig, q=0.0):
    d1, d2 = _d1_d2(S, K, r, T, sig, q)
    return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def put_price(S, K, r, T, sig, q=0.0):
    d1, d2 = _d1_d2(S, K, r, T, sig, q)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)


def vega(S, K, r, T, sig, q=0.0):
    d1, _ = _d1_d2(S, K, r, T, sig, q)
    return S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T)


def implied_vol(price, S, K, r, T, q=0.0, is_call=True, tol=1e-8, maxiter=40):
    """Implied vol via Newton on the OTM premium, bisection fallback."""
    K = np.atleast_1d(np.asarray(K, dtype=float))
    price = np.broadcast_to(np.asarray(price, dtype=float), K.shape).copy()
    disc = np.exp(-r * T)
    dfq = np.exp(-q * T)
    F = S * np.exp((r - q) * T)
    # Put-call parity maps any quote to the OTM premium (same IV).
    if is_call:
        otm = np.where(K < F, price - dfq * S + disc * K, price)
    else:
        otm = np.where(K >= F, price + dfq * S - disc * K, price)
        # `otm` is now a call price; invert as a call.
    intrinsic = np.maximum(dfq * S - disc * K, 0.0)
    call = np.where(K < F, otm + dfq * S - disc * K, otm)
    lo_bound = intrinsic + 1e-12
    hi_bound = dfq * S - 1e-12
    ok = np.isfinite(call) & (call > lo_bound) & (call < hi_bound)

    # Brenner–Subrahmanyam on the OTM premium.
    otm_prem = np.where(K < F, call - intrinsic, call)
    sig = np.sqrt(2.0 * np.pi / max(T, 1e-12)) * np.maximum(otm_prem, 1e-12) / max(S * dfq, 1e-12)
    sig = np.clip(sig, 0.05, 1.5)
    sig = np.where(ok, sig, 0.2)

    lo = np.full(K.shape, 1e-5)
    hi = np.full(K.shape, 4.0)
    for _ in range(maxiter):
        pr = call_price(S, K, r, T, sig, q)
        vg = vega(S, K, r, T, sig, q)
        diff = pr - call
        # Maintain a bracket.
        too_high = diff > 0
        too_low = diff < 0
        hi = np.where(too_high, np.minimum(hi, sig), hi)
        lo = np.where(too_low, np.maximum(lo, sig), lo)
        step = np.zeros_like(sig)
        good = ok & (vg > 1e-10)
        step[good] = diff[good] / vg[good]
        sig_n = sig - step
        # If Newton jumps outside the bracket, take the midpoint.
        outside = (sig_n <= lo) | (sig_n >= hi) | ~np.isfinite(sig_n)
        sig_n = np.where(outside, 0.5 * (lo + hi), sig_n)
        sig = np.where(ok, np.clip(sig_n, 1e-5, 4.0), sig)
        if np.max(np.abs(diff[ok])) < tol if np.any(ok) else True:
            break

    sig = np.where(ok, sig, np.nan)
    if sig.size == 1:
        return float(sig.reshape(-1)[0])
    return sig


def lognormal_density(K, S, r, T, sig, q=0.0):
    """Risk-neutral density of S_T in the Black–Scholes model."""
    K = np.asarray(K, dtype=float)
    F = S * np.exp((r - q) * T)
    s = sig * np.sqrt(T)
    m = np.log(F) - 0.5 * s**2
    return np.exp(-0.5 * ((np.log(K) - m) / s) ** 2) / (K * s * np.sqrt(2.0 * np.pi))


def forward(S, r, T, q=0.0):
    return S * np.exp((r - q) * T)
