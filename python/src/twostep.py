"""Two-step estimator: convex projection, flat-volatility wings, then the kernel.

The kernel in ``rnd.estimate_rnd`` is unchanged. This module builds the call
curve it is applied to when the quotes can violate convexity. On an exact
convex strip the projection is the identity.
"""

from __future__ import annotations

import numpy as np

from scipy.special import sici

from rnd import _gauss_cdf, _qhat, estimate_rnd
from src.black_scholes import call_price, implied_vol
from src.competitors_v2 import _shape_fit

# Equal weight on the twiced Gaussian and a sinc kernel three times as wide.
# Chosen on the first eight replications of the Heston and variance-gamma
# noise designs; see the manuscript.
MIX_ALPHA = 0.5
SINC_FACTOR = 3.0


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


def _sinc_q_G(K, mu, dp, h, F):
    """Unit-integral sinc. The cdf is the sine integral."""
    K = np.atleast_1d(np.asarray(K, dtype=float))
    mu = np.asarray(mu, dtype=float)
    dp = np.asarray(dp, dtype=float)
    h = max(float(h), 1e-8)
    x = K[:, None] - mu[None, :]
    ax = (np.pi / h) * x
    small = np.abs(x) < 1e-6 * h
    numer = (np.pi / h) * x * np.cos(ax) - np.sin(ax)
    deriv = np.zeros_like(x)
    np.divide(numer, np.pi * x * x, out=deriv, where=~small)
    q = -float(F) * (dp[None, :] * deriv).sum(axis=1)
    Si, _ = sici(ax)
    G = (dp[None, :] * (0.5 + Si / np.pi)).sum(axis=1)
    return q, G


def _gauss_twice_q_G(K, mu, dp, h, F):
    q = 2.0 * _qhat(K, mu, dp, h, F) - _qhat(K, mu, dp, h * np.sqrt(2.0), F)
    u = (np.atleast_1d(np.asarray(K, dtype=float))[:, None] - np.asarray(mu)[None, :]) / max(float(h), 1e-8)
    u2 = u / np.sqrt(2.0)
    G = (np.asarray(dp)[None, :] * (2.0 * _gauss_cdf(u) - _gauss_cdf(u2))).sum(axis=1)
    return q, G


def estimate_combined(K, C, S0, r, T, q=0.0, F=None, K_eval=None, alpha=MIX_ALPHA, sinc_factor=SINC_FACTOR):
    """Two-step masses, then a Gaussian-sinc mixture at the density-scale bandwidth.

    The Gaussian piece is the twiced kernel at bandwidth ``h``. The sinc piece
    uses bandwidth ``sinc_factor * h``. ``alpha`` is the Gaussian weight.
    """
    base = estimate_twostep(K, C, S0, r, T, q=q, F=F, K_eval=K, h="density", twice=True)
    mu, dp, h, F_used = base["K_mass"], base["dp"], base["h"], base["F"]
    if K_eval is None:
        K_eval = np.asarray(K, dtype=float)
    q_g, _ = _gauss_twice_q_G(K_eval, mu, dp, h, F_used)
    q_s, _ = _sinc_q_G(K_eval, mu, dp, h * sinc_factor, F_used)
    q_hat = alpha * q_g + (1.0 - alpha) * q_s

    def interpolant(K_pts):
        _, G_g = _gauss_twice_q_G(K_pts, mu, dp, h, F_used)
        _, G_s = _sinc_q_G(K_pts, mu, dp, h * sinc_factor, F_used)
        G = np.clip(alpha * G_g + (1.0 - alpha) * G_s, 0.0, 1.0)
        stock = float(S0) * float(np.exp(-q * T))
        disc = float(np.exp(-r * T))
        C_hat = stock * np.clip(1.0 - G, 0.0, 1.0)
        P_hat = np.maximum(C_hat - stock + np.asarray(K_pts, dtype=float) * disc, 0.0)
        return P_hat, C_hat

    base["q"] = q_hat
    base["interpolant"] = interpolant
    _, C_hat = interpolant(K)
    base["C"] = C_hat
    base["alpha"] = float(alpha)
    base["sinc_factor"] = float(sinc_factor)
    return base
