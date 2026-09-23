"""Two-step estimator: convex projection, flat-volatility wings, then the kernel.

The kernel in ``rnd.estimate_rnd`` is unchanged. This module builds the call
curve it is applied to when the quotes can violate convexity. On an exact
convex strip the projection is the identity.
"""

from __future__ import annotations

import numpy as np

from rnd import estimate_rnd
from src.black_scholes import call_price, implied_vol
from src.competitors_v2 import _shape_fit


def project_calls(K, C, S0, r, T, q, F):
    """Euclidean projection onto decreasing convex calls with slopes in [-e^{-rT}, 0]."""
    K = np.asarray(K, dtype=float)
    C = np.asarray(C, dtype=float)
    disc = float(np.exp(-r * T))
    intrinsic = disc * np.maximum(F - K, 0.0)
    return _shape_fit(K, C, disc, lam=0.0, intrinsic=intrinsic)


def flat_vol_wings(K, C, S0, r, T, q, F, n_side=30, k_min=0.15, k_max=3.0):
    """Extend past the quoted strikes at constant wing implied volatility.

    Total variance is held at its value on the first and last finite implied
    volatility. Strikes are added on a log grid down to ``k_min * F`` and up
    to ``k_max * F``.
    """
    K = np.asarray(K, dtype=float)
    C = np.asarray(C, dtype=float)
    iv = implied_vol(C, S0, K, r, T, q)
    iv = np.where(np.isfinite(iv), iv, np.nan)
    if not np.isfinite(iv).any():
        return K, C
    good = np.where(np.isfinite(iv))[0]
    iv = np.interp(np.arange(len(iv)), good, iv[good])
    iv = np.maximum(iv, 1e-3)
    log_k = np.log(np.maximum(K, 1e-8) / F)
    left = np.linspace(np.log(k_min), log_k[0], n_side, endpoint=False)
    right = np.linspace(log_k[-1], np.log(k_max), n_side, endpoint=False)[1:]
    K_left = F * np.exp(left)
    K_right = F * np.exp(right)
    C_left = call_price(S0, K_left, r, T, iv[0], q)
    C_right = call_price(S0, K_right, r, T, iv[-1], q)
    return (
        np.concatenate([K_left, K, K_right]),
        np.concatenate([np.asarray(C_left, dtype=float), C, np.asarray(C_right, dtype=float)]),
    )


def estimate_twostep(K, C, S0, r, T, q=0.0, F=None, K_eval=None, h="density", twice=True):
    """Project, complete the wings at constant implied volatility, then the kernel."""
    K = np.asarray(K, dtype=float)
    C = np.asarray(C, dtype=float)
    if F is None:
        F = float(S0) * float(np.exp((r - q) * T))
    projected = project_calls(K, C, S0, r, T, q, F)
    K_ext, C_ext = flat_vol_wings(K, projected, S0, r, T, q, F)
    out = estimate_rnd(
        K_ext, C_ext, S0, r, T, q, K_eval=K_eval, h=h, twice=twice
    )
    out["C_projected"] = projected
    return out
