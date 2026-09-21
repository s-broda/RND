"""Competitors used in the paper: Priestley–Chao, YH λ=0, ASD, Bondarenko PCA."""

from __future__ import annotations

import numpy as np
from scipy.optimize import nnls
from scipy.stats import norm

from . import black_scholes as bs


def _sorted(K, C):
    K = np.asarray(K, dtype=float)
    C = np.asarray(C, dtype=float)
    o = np.argsort(K)
    K, C = K[o], C[o]
    _, u = np.unique(K, return_index=True)
    return K[u], C[u]


def priestley_chao_bl(K_obs, C_obs, S0, r, T, K_eval, q=0.0, h=None):
    """Priestley–Chao kernel of call *levels*, then C''. q̂ = e^{rT} Ĉ''."""
    K_obs, C_obs = _sorted(K_obs, C_obs)
    K_eval = np.asarray(K_eval, dtype=float)
    F = S0 * np.exp((r - q) * T)
    if h is None:
        n = max(len(K_obs), 8)
        iv = bs.implied_vol(C_obs, S0, K_obs, r, T, q)
        atm = np.nanmedian(iv)
        if not np.isfinite(atm):
            atm = 0.2
        h = float(1.06 * F * atm * np.sqrt(T) * n ** (-1.0 / 9.0))
    h = max(float(h), 1e-6)
    dK = np.empty_like(K_obs)
    dK[0] = K_obs[1] - K_obs[0] if len(K_obs) > 1 else 1.0
    dK[1:] = np.diff(K_obs)
    u = (K_eval[:, None] - K_obs[None, :]) / h
    kap = np.exp(-0.5 * u * u) / (h * np.sqrt(2.0 * np.pi))
    kap2 = kap * (u * u - 1.0) / (h * h)
    qhat = np.exp(r * T) * (C_obs * dK)[None, :] * kap2
    qhat = np.nan_to_num(qhat.sum(axis=1), nan=0.0, posinf=0.0, neginf=0.0)
    return qhat, h


def local_cubic_bl(K_obs, C_obs, S0, r, T, K_eval, q=0.0, h=None):
    """Local cubic of C(K); q̂ = e^{rT} · 2 β₂."""
    K_obs, C_obs = _sorted(K_obs, C_obs)
    K_eval = np.asarray(K_eval, dtype=float)
    if h is None:
        n = max(len(K_obs), 8)
        iv = bs.implied_vol(C_obs, S0, K_obs, r, T, q)
        atm = np.nanmedian(iv)
        if not np.isfinite(atm):
            atm = 0.2
        F = S0 * np.exp((r - q) * T)
        h = float(0.9 * F * atm * np.sqrt(T) * n ** (-1.0 / 9.0))
    h = max(float(h), 1e-6)
    out = np.zeros_like(K_eval)
    scale = np.exp(r * T)
    for j, x in enumerate(K_eval):
        dx = K_obs - x
        w = np.exp(-0.5 * (dx / h) ** 2)
        if w.sum() < 1e-12:
            continue
        X = np.column_stack([np.ones_like(dx), dx, dx**2, dx**3])
        sw = np.sqrt(w)
        beta, *_ = np.linalg.lstsq(X * sw[:, None], C_obs * sw, rcond=None)
        out[j] = scale * 2.0 * beta[2]
    return out, h


def pca_lognormal(K_obs, C_obs, S0, r, T, K_eval, q=0.0, h=None, n_centers=41):
    """Bondarenko PCA: q = μ * lognormal kernel, μ ≥ 0 fitted by NNLS."""
    K_obs, C_obs = _sorted(K_obs, C_obs)
    F = S0 * np.exp((r - q) * T)
    disc = np.exp(-r * T)
    iv = bs.implied_vol(C_obs, S0, K_obs, r, T, q)
    atm = np.nanmedian(iv)
    if not np.isfinite(atm):
        atm = 0.2
    s_atm = float(np.clip(atm * np.sqrt(T), 0.02, 1.5))
    if h is None:
        h = 0.65 * s_atm
    h = float(np.clip(h, 0.20 * s_atm, 1.25 * s_atm))
    z = np.linspace(np.log(F) - 4.0 * s_atm, np.log(F) + 4.0 * s_atm, n_centers)
    Kcol = K_obs[:, None]
    zrow = z[None, :]
    d2 = (zrow - np.log(Kcol)) / h
    d1 = d2 + h
    Fm = np.exp(z + 0.5 * h**2)
    W = disc * (Fm[None, :] * norm.cdf(d1) - Kcol * norm.cdf(d2))
    W = np.nan_to_num(W, nan=0.0, posinf=0.0, neginf=0.0)
    col = np.maximum(np.linalg.norm(W, axis=0), 1e-12)
    scale = max(float(np.median(np.abs(C_obs))), 1e-6)
    lam = 8.0 * scale
    W_aug = np.vstack([W / col, lam / col, lam * Fm / (F * col)])
    c_aug = np.concatenate([C_obs, [lam, lam]])
    W_aug = np.nan_to_num(W_aug, nan=0.0, posinf=0.0, neginf=0.0)
    try:
        with np.errstate(all="ignore"):
            a_n, _ = nnls(W_aug, c_aug, maxiter=800)
    except Exception:
        a_n = np.ones(n_centers)
    a = np.maximum(a_n / col, 0.0)
    if a.sum() <= 1e-16:
        a = np.ones_like(a)
    a = a / a.sum()
    K_eval = np.asarray(K_eval, dtype=float)
    x = np.log(np.maximum(K_eval, 1e-12))[:, None]
    kern = np.exp(-np.clip(0.5 * ((x - z[None, :]) / h) ** 2, 0.0, 60.0)) / (
        np.maximum(K_eval[:, None], 1e-12) * h * np.sqrt(2.0 * np.pi)
    )
    qhat = np.nan_to_num(kern @ a, nan=0.0, posinf=0.0, neginf=0.0)
    mass = np.trapezoid(qhat, K_eval)
    if mass > 1e-16:
        qhat = qhat / mass
    return qhat, h


def _pava_increasing(y, w=None):
    """Weighted pool-adjacent-violators, increasing."""
    y = np.asarray(y, dtype=float)
    n = y.size
    if n <= 1:
        return y.copy()
    w = np.ones(n) if w is None else np.asarray(w, dtype=float)
    sw, swy, ln = [], [], []
    for i in range(n):
        sw.append(float(w[i]))
        swy.append(float(w[i] * y[i]))
        ln.append(1)
        while len(sw) >= 2 and swy[-2] / sw[-2] > swy[-1] / sw[-1] + 1e-15:
            sw[-2] += sw[-1]
            swy[-2] += swy[-1]
            ln[-2] += ln[-1]
            sw.pop()
            swy.pop()
            ln.pop()
    out = np.empty(n)
    idx = 0
    for s, sy, L in zip(sw, swy, ln):
        out[idx : idx + L] = sy / s
        idx += L
    return out


def convex_decreasing_ls(K, C, disc, intrinsic=None, weights=None):
    """L2 projection onto decreasing convex calls with slopes in [-disc, 0].

    Shared first step of Aït-Sahalia–Duarte (2003) and the interpolating
    case of Yatchew–Härdle (2006). If the quotes already satisfy the
    constraints, the projection is the identity.
    """
    from scipy.optimize import Bounds, LinearConstraint, minimize

    K = np.asarray(K, dtype=float)
    C = np.asarray(C, dtype=float)
    n = len(K)
    if n < 3:
        return C.copy()
    dK = np.diff(K)
    disc = float(disc)
    w = np.ones(n) if weights is None else np.asarray(weights, dtype=float)
    w = np.maximum(w, 1e-12)

    def m_from_x(x):
        m0, v = x[0], np.maximum(x[1:], 0.0)
        s = -disc + np.cumsum(v)
        return np.concatenate([[m0], m0 + np.cumsum(s * dK)])

    def obj(x):
        r = m_from_x(x) - C
        return 0.5 * float(np.dot(w * r, r))

    def grad(x):
        r = m_from_x(x) - C
        wr = w * r
        g = np.zeros_like(x)
        g[0] = float(wr.sum())
        prefix = np.cumsum(wr[::-1])[::-1]
        acc = 0.0
        g_v = np.empty(n - 1)
        for i in range(n - 2, -1, -1):
            acc += dK[i] * prefix[i + 1]
            g_v[i] = acc
        g[1:] = g_v
        return g

    s0 = np.clip(np.diff(C) / np.maximum(dK, 1e-16), -disc, 0.0)
    s0 = _pava_increasing(s0)
    s0 = np.clip(s0, -disc, 0.0)
    v0 = np.diff(s0, prepend=s0[0] + disc)
    v0 = np.maximum(v0, 0.0)
    if v0.sum() > disc:
        v0 *= disc / max(v0.sum(), 1e-16)
    x0 = np.concatenate([[float(C[0])], v0])
    bounds = Bounds(
        lb=np.concatenate([[-np.inf], np.zeros(n - 1)]),
        ub=np.concatenate([[np.inf], np.full(n - 1, disc)]),
    )
    cons = LinearConstraint(np.concatenate([[0.0], np.ones(n - 1)]), -np.inf, disc)
    res = minimize(
        obj,
        x0,
        jac=grad,
        method="SLSQP",
        bounds=bounds,
        constraints=cons,
        options={"maxiter": 800, "ftol": 1e-14, "disp": False},
    )
    m = m_from_x(res.x if res.success else x0)
    if intrinsic is not None:
        m = np.maximum(m, np.asarray(intrinsic, dtype=float))
        s = np.clip(np.diff(m) / np.maximum(dK, 1e-16), -disc, 0.0)
        s = _pava_increasing(s)
        s = np.clip(s, -disc, 0.0)
        m = np.concatenate([[m[0]], m[0] + np.cumsum(s * dK)])
        m = np.maximum(m, np.asarray(intrinsic, dtype=float))
    return np.maximum(m, 0.0)


def local_linear_fit(K_obs, Y, K_eval, h):
    """Local linear of Y(K); returns level and slope at K_eval."""
    K_obs, Y = _sorted(K_obs, Y)
    K_eval = np.asarray(K_eval, dtype=float)
    h = max(float(h), 1e-6)
    level = np.zeros_like(K_eval)
    slope = np.zeros_like(K_eval)
    for j, x in enumerate(K_eval):
        dx = K_obs - x
        w = np.exp(-0.5 * (dx / h) ** 2)
        if w.sum() < 1e-12:
            continue
        X = np.column_stack([np.ones_like(dx), dx])
        sw = np.sqrt(w)
        beta, *_ = np.linalg.lstsq(X * sw[:, None], Y * sw, rcond=None)
        level[j] = beta[0]
        slope[j] = beta[1]
    return level, slope


def _second_diff_q(K, C, r):
    """Breeden–Litzenberger from a call curve on a (possibly uneven) grid."""
    K = np.asarray(K, dtype=float)
    C = np.asarray(C, dtype=float)
    q = np.zeros_like(K)
    if len(K) < 3:
        return q
    dK = np.diff(K)
    s = np.diff(C) / np.maximum(dK, 1e-16)
    span = np.maximum(0.5 * (dK[:-1] + dK[1:]), 1e-16)
    q[1:-1] = np.exp(r) * np.diff(s) / span
    return q


def ait_sahalia_duarte(K_obs, C_obs, S0, r, T, K_eval, q=0.0, h=None):
    """Aït-Sahalia–Duarte (2003): constrained LS of C, then local linear.

    Returns (C_hat, q_hat, h, m) on K_eval. C_hat is the local linear of the
    constrained values; q_hat is e^{rT} times the local-cubic second
    derivative of those same constrained values.
    """
    K_obs, C_obs = _sorted(K_obs, C_obs)
    F = S0 * np.exp((r - q) * T)
    disc = np.exp(-r * T)
    intrinsic = disc * np.maximum(F - K_obs, 0.0)
    m = convex_decreasing_ls(K_obs, C_obs, disc, intrinsic=intrinsic)
    n = max(len(K_obs), 8)
    iv = bs.implied_vol(m, S0, K_obs, r, T, q)
    atm = np.nanmedian(iv)
    if not np.isfinite(atm):
        atm = 0.2
    if h is None:
        h = float(0.9 * F * atm * np.sqrt(T) * n ** (-1.0 / 9.0))
    h = max(float(h), 1e-6)
    h_price = float(1.06 * F * atm * np.sqrt(T) * n ** (-0.2))
    h_price = max(h_price, 1e-6)
    C_hat, _ = local_linear_fit(K_obs, m, K_eval, h_price)
    q_hat, _ = local_cubic_bl(K_obs, m, S0, r, T, K_eval, q=q, h=h)
    return C_hat, q_hat, h, m


def yatchew_hardle(K_obs, C_obs, S0, r, T, K_eval, q=0.0, lam=0.0):
    """Yatchew–Härdle (2006): shape-constrained LS of C.

    λ=0 is the interpolating projection used in the paper.
    """
    from scipy.optimize import Bounds, LinearConstraint, minimize

    K_obs, C_obs = _sorted(K_obs, C_obs)
    F = S0 * np.exp((r - q) * T)
    disc = np.exp(-r * T)
    intrinsic = disc * np.maximum(F - K_obs, 0.0)
    n = len(K_obs)
    dK = np.diff(K_obs)
    span = np.maximum(0.5 * (dK[:-1] + dK[1:]), 1e-16)
    lam = float(lam)

    def m_from_x(x):
        m0, v = x[0], np.maximum(x[1:], 0.0)
        s = -disc + np.cumsum(v)
        return np.concatenate([[m0], m0 + np.cumsum(s * dK)])

    def obj(x):
        m = m_from_x(x)
        r0 = m - C_obs
        sse = 0.5 * float(np.dot(r0, r0))
        if lam > 0.0 and n >= 3:
            s = np.diff(m) / np.maximum(dK, 1e-16)
            d2 = np.diff(s) / span
            sse += 0.5 * lam * float(np.dot(d2, d2))
        return sse

    s0 = np.clip(np.diff(C_obs) / np.maximum(dK, 1e-16), -disc, 0.0)
    s0 = _pava_increasing(s0)
    s0 = np.clip(s0, -disc, 0.0)
    v0 = np.diff(s0, prepend=s0[0] + disc)
    v0 = np.maximum(v0, 0.0)
    if v0.sum() > disc:
        v0 *= disc / max(v0.sum(), 1e-16)
    x0 = np.concatenate([[float(C_obs[0])], v0])
    bounds = Bounds(
        lb=np.concatenate([[-np.inf], np.zeros(n - 1)]),
        ub=np.concatenate([[np.inf], np.full(n - 1, disc)]),
    )
    cons = LinearConstraint(np.concatenate([[0.0], np.ones(n - 1)]), -np.inf, disc)
    res = minimize(
        obj,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=cons,
        options={"maxiter": 400, "ftol": 1e-12, "disp": False},
    )
    m = m_from_x(res.x if res.success else x0)
    m = np.maximum(m, intrinsic)
    m = np.maximum(m, 0.0)
    C_hat = np.interp(K_eval, K_obs, m, left=m[0], right=m[-1])
    q_obs = _second_diff_q(K_obs, m, r * T)
    q_hat = np.interp(K_eval, K_obs, q_obs, left=0.0, right=0.0)
    return C_hat, q_hat, lam, m
