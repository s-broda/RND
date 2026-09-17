"""Kernel and shape-constrained RND estimators on a single-maturity slice.

Local cubic of C (Härdle–Grith / Dalderop), Priestley–Chao convolution of
prices, Nadaraya–Watson in implied vol then BL, Bondarenko PCA,
Aït-Sahalia–Duarte constrained LS plus local linear, Birke–Pilz
rearrangements, Yatchew–Härdle constrained LS, and raw SVI.
"""

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


def _pilot_h(K, C, S0, r, T, q=0.0, c=0.9):
    """Strike bandwidth for a second derivative: Silverman with n^{-1/9}."""
    K = np.asarray(K, dtype=float)
    n = max(len(K), 8)
    iv = bs.implied_vol(C, S0, K, r, T, q)
    atm = np.nanmedian(iv)
    if not np.isfinite(atm):
        atm = 0.2
    F = S0 * np.exp((r - q) * T)
    return float(c * F * atm * np.sqrt(T) * n ** (-1.0 / 9.0))


def complete_call(K_obs, C_obs, K_grid, F, disc):
    """Clamped cubic through quotes, linear no-arbitrage wings."""
    from scipy.interpolate import CubicSpline, PchipInterpolator

    K_obs, C_obs = _sorted(K_obs, C_obs)
    dC = np.gradient(C_obs, K_obs)
    sl_l = float(np.clip(dC[0], -disc, 0.0))
    sl_r = float(np.clip(dC[-1], -disc, 0.0))
    if len(K_obs) >= 4:
        spl = CubicSpline(K_obs, C_obs, bc_type=((1, sl_l), (1, sl_r)))
        C = np.asarray(spl(K_grid), dtype=float)
    else:
        C = np.asarray(PchipInterpolator(K_obs, C_obs, extrapolate=True)(K_grid), dtype=float)
    left, right = K_grid < K_obs[0], K_grid > K_obs[-1]
    C[left] = C_obs[0] + sl_l * (K_grid[left] - K_obs[0])
    C[right] = C_obs[-1] + sl_r * (K_grid[right] - K_obs[-1])
    intrinsic = disc * np.maximum(F - K_grid, 0.0)
    C = np.maximum(C, intrinsic)
    C = np.maximum.accumulate(C[::-1])[::-1]
    return np.maximum(C, 0.0)


def local_cubic_bl(K_obs, C_obs, S0, r, T, K_eval, q=0.0, h=None):
    """Local cubic of C(K); q̂ = e^{rT} · 2 β₂ (Fan–Gijbels / Härdle–Grith)."""
    K_obs, C_obs = _sorted(K_obs, C_obs)
    if h is None:
        h = _pilot_h(K_obs, C_obs, S0, r, T, q, c=0.9)
    h = max(float(h), 1e-6)
    K_eval = np.asarray(K_eval, dtype=float)
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


def priestley_chao_bl(K_obs, C_obs, S0, r, T, K_eval, q=0.0, h=None):
    """Priestley–Chao: convolve completed C with a Gaussian κ_h'', scale by e^{rT}."""
    K_obs, C_obs = _sorted(K_obs, C_obs)
    F = S0 * np.exp((r - q) * T)
    disc = np.exp(-r * T)
    if h is None:
        h = _pilot_h(K_obs, C_obs, S0, r, T, q, c=0.70)
    h = max(float(h), 1e-6)
    K_grid = np.linspace(max(1e-6, K_obs.min() - 6 * h), K_obs.max() + 6 * h, 1600)
    C_grid = complete_call(K_obs, C_obs, K_grid, F, disc)
    dK = K_grid[1] - K_grid[0]
    half = int(min(K_grid.size // 2 - 1, np.ceil(6.0 * h / dK)))
    z = np.arange(-half, half + 1) * dK
    kap = np.exp(-0.5 * (z / h) ** 2) / (h * np.sqrt(2.0 * np.pi))
    kap2 = kap * (z**2 / h**4 - 1.0 / h**2)
    second = np.convolve(C_grid, kap2, mode="same") * dK
    q_grid = np.exp(r * T) * second
    q_eval = np.interp(K_eval, K_grid, q_grid, left=0.0, right=0.0)
    return q_eval, h


def iv_spline_bl(K_obs, C_obs, S0, r, T, K_eval, q=0.0):
    """Cubic GCV smoothing spline of implied vol, then Black prices and BL.

    Practitioner pipeline of Bliss and Panigirtzoglou (2002): smooth the smile,
    convert to calls, differentiate twice. Applied in strike, not delta.
    Outside the quoted range, implied volatility is held at the nearest
    observed value.
    """
    from scipy.interpolate import make_smoothing_spline

    K_obs, C_obs = _sorted(K_obs, C_obs)
    iv = bs.implied_vol(C_obs, S0, K_obs, r, T, q)
    ok = np.isfinite(iv) & (iv > 1e-4)
    K_eval = np.asarray(K_eval, dtype=float)
    if ok.sum() < 4:
        z = np.zeros_like(K_eval)
        return z, z, None
    K_obs, iv = K_obs[ok], iv[ok]
    spl = make_smoothing_spline(K_obs, iv)

    def iv_at(K):
        K = np.asarray(K, dtype=float)
        v = np.asarray(spl(K), dtype=float)
        v = np.where(K < K_obs[0], iv[0], v)
        v = np.where(K > K_obs[-1], iv[-1], v)
        return np.clip(v, 1e-4, 5.0)

    iv_e = iv_at(K_eval)
    C_hat = bs.call_price(S0, K_eval, r, T, iv_e, q)
    if len(K_eval) >= 3:
        dK = np.diff(K_eval)
        if np.allclose(dK, dK[0], rtol=0.05, atol=1e-6):
            second = np.zeros_like(K_eval)
            second[1:-1] = (C_hat[2:] - 2.0 * C_hat[1:-1] + C_hat[:-2]) / dK[0] ** 2
            q_hat = np.exp(r * T) * second
        else:
            q_hat = _second_diff_q(K_eval, C_hat, r * T)
    else:
        q_hat = np.zeros_like(K_eval)
    return C_hat, q_hat, iv_at


def kernel_iv_bl(K_obs, C_obs, S0, r, T, K_eval, q=0.0, h=None):
    """Nadaraya–Watson on implied vol, convert to C, then BL."""
    K_obs, C_obs = _sorted(K_obs, C_obs)
    iv = bs.implied_vol(C_obs, S0, K_obs, r, T, q)
    ok = np.isfinite(iv)
    if ok.sum() < 4:
        return np.zeros_like(np.asarray(K_eval, dtype=float)), 1.0
    K_obs, iv, C_obs = K_obs[ok], iv[ok], C_obs[ok]
    F = S0 * np.exp((r - q) * T)
    if h is None:
        h = 1.55 * _pilot_h(K_obs, C_obs, S0, r, T, q, c=0.9)
    h = max(float(h), 1e-6)
    K_grid = np.linspace(max(1e-6, K_obs.min() - 4 * h), K_obs.max() + 4 * h, 1600)
    u2 = 0.5 * ((K_grid[:, None] - K_obs[None, :]) / h) ** 2
    W = np.exp(-np.clip(u2, 0.0, 60.0))
    iv_grid = (W @ iv) / np.clip(W.sum(axis=1), 1e-30, None)
    iv_grid = np.where(K_grid < K_obs[0], iv[0], iv_grid)
    iv_grid = np.where(K_grid > K_obs[-1], iv[-1], iv_grid)
    iv_grid = np.clip(iv_grid, 1e-4, 5.0)
    C_grid = bs.call_price(S0, K_grid, r, T, iv_grid, q)
    dK = K_grid[1] - K_grid[0]
    second = (C_grid[2:] - 2.0 * C_grid[1:-1] + C_grid[:-2]) / dK**2
    q_grid = np.zeros_like(K_grid)
    q_grid[1:-1] = np.exp(r * T) * second
    q_eval = np.interp(K_eval, K_grid, q_grid, left=0.0, right=0.0)
    return q_eval, h


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
        # prefix[k] = sum_{i>=k} wr[i]
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
        if intrinsic is not None:
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
    # second derivative at interior knots: slope increment over mid-span
    span = np.maximum(0.5 * (dK[:-1] + dK[1:]), 1e-16)
    q[1:-1] = np.exp(r) * np.diff(s) / span
    return q


def ait_sahalia_duarte(K_obs, C_obs, S0, r, T, K_eval, q=0.0, h=None):
    """Aït-Sahalia–Duarte (2003): constrained LS of C, then local linear.

    Returns (C_hat, q_hat, h) on K_eval. C_hat is the local linear of the
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
    # Level bandwidth for the call; second-derivative bandwidth for the density.
    if h is None:
        h = float(0.9 * F * atm * np.sqrt(T) * n ** (-1.0 / 9.0))
    h = max(float(h), 1e-6)
    h_price = float(1.06 * F * atm * np.sqrt(T) * n ** (-0.2))
    h_price = max(h_price, 1e-6)
    C_hat, _ = local_linear_fit(K_obs, m, K_eval, h_price)
    q_hat, _ = local_cubic_bl(K_obs, m, S0, r, T, K_eval, q=q, h=h)
    return C_hat, q_hat, h, m


def _rearrange_increasing(x, y, n_grid=None):
    """Increasing rearrangement of y as a function of x, Lebesgue on [x0,x1]."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n_grid = int(n_grid or max(8 * len(x), 800))
    xu = np.linspace(x[0], x[-1], n_grid)
    yu = np.interp(xu, x, y)
    ys = np.sort(yu)
    return np.interp(x, xu, ys)


def _rearrange_decreasing(x, y, n_grid=None):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n_grid = int(n_grid or max(8 * len(x), 800))
    xu = np.linspace(x[0], x[-1], n_grid)
    yu = np.interp(xu, x, y)
    ys = np.sort(yu)[::-1]
    return np.interp(x, xu, ys)


def birke_pilz(K_obs, C_obs, S0, r, T, K_eval, q=0.0, h=None):
    """Birke–Pilz (2009): kernel of C, convex then monotone rearrangements.

    Unconstrained local linear of C, increasing rearrangement of C' clipped
    to [-e^{-rT}, 0], integrate (L2-optimal origin = mean of C), then
    decreasing rearrangement of C if needed. Prices are the constrained
    call; q is e^{rT} C''.
    """
    K_obs, C_obs = _sorted(K_obs, C_obs)
    F = S0 * np.exp((r - q) * T)
    disc = np.exp(-r * T)
    if h is None:
        # level bandwidth: they estimate C, not C''
        n = max(len(K_obs), 8)
        iv = bs.implied_vol(C_obs, S0, K_obs, r, T, q)
        atm = np.nanmedian(iv)
        if not np.isfinite(atm):
            atm = 0.2
        h = float(1.06 * F * atm * np.sqrt(T) * n ** (-0.2))
    h = max(float(h), 1e-6)
    # fine uniform grid for rearrangement (Lebesgue on the quoted span)
    K_grid = np.linspace(float(K_obs[0]), float(K_obs[-1]), max(8 * len(K_obs), 800))
    C_u, Cp_u = local_linear_fit(K_obs, C_obs, K_grid, h)
    Cp_u = np.clip(Cp_u, -disc, 0.0)
    Cp_inc = np.sort(Cp_u)  # increasing rearrangement on a uniform grid
    Cp_inc = np.clip(Cp_inc, -disc, 0.0)
    # integrate; L2-optimal origin is the average over a of ρ(x,a)
    dK = K_grid[1] - K_grid[0]
    C_from_left = C_u[0] + np.concatenate([[0.0], np.cumsum(Cp_inc[:-1] * dK)])
    # shift so mean(C) is preserved (discrete L2-optimal intercept)
    C_from_left = C_from_left + (C_u.mean() - C_from_left.mean())
    if np.any(np.diff(C_from_left) > 1e-12):
        C_from_left = np.sort(C_from_left)[::-1]
    C_from_left = np.maximum(C_from_left, disc * np.maximum(F - K_grid, 0.0))
    C_from_left = np.maximum.accumulate(C_from_left[::-1])[::-1]
    C_from_left = np.maximum(C_from_left, 0.0)
    C_hat = np.interp(K_eval, K_grid, C_from_left, left=C_from_left[0], right=C_from_left[-1])
    q_grid = _second_diff_q(K_grid, C_from_left, r * T)
    q_hat = np.interp(K_eval, K_grid, q_grid, left=0.0, right=0.0)
    return C_hat, q_hat, h


def yatchew_hardle(K_obs, C_obs, S0, r, T, K_eval, q=0.0, lam=None):
    """Yatchew–Härdle (2006): shape-constrained LS of C, optional roughness.

    λ=0 is the interpolating projection (quotes already in the cone are
    unchanged). A positive λ penalises second differences of C, i.e. the
    SPD, which is their smoothness control. Default λ is GCV on a small
    log grid, under the same cone.
    """
    from scipy.optimize import Bounds, LinearConstraint, minimize

    K_obs, C_obs = _sorted(K_obs, C_obs)
    F = S0 * np.exp((r - q) * T)
    disc = np.exp(-r * T)
    intrinsic = disc * np.maximum(F - K_obs, 0.0)
    n = len(K_obs)
    dK = np.diff(K_obs)
    # second-difference operator on levels, scaled as a roughness of C''
    span = np.maximum(0.5 * (dK[:-1] + dK[1:]), 1e-16)

    def m_from_x(x):
        m0, v = x[0], np.maximum(x[1:], 0.0)
        s = -disc + np.cumsum(v)
        return np.concatenate([[m0], m0 + np.cumsum(s * dK)])

    def fit(lam_):
        def obj(x):
            m = m_from_x(x)
            r0 = m - C_obs
            sse = 0.5 * float(np.dot(r0, r0))
            if lam_ > 0.0 and n >= 3:
                s = np.diff(m) / np.maximum(dK, 1e-16)
                d2 = np.diff(s) / span
                sse += 0.5 * float(lam_) * float(np.dot(d2, d2))
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
        return np.maximum(m, 0.0)

    if lam is None:
        # GCV among λ=0 and a few positive roughnesses; λ=0 is interpolation
        cands = [0.0]
        scale = float(np.median(dK) ** 4)
        cands.extend([scale * 10.0 ** p for p in (-4, -2, 0, 2)])
        best, best_m, best_lam = np.inf, None, 0.0
        for lam_ in cands:
            m = fit(lam_)
            rss = float(np.mean((m - C_obs) ** 2))
            # trace proxy: smaller λ interpolates more
            df = n if lam_ <= 0.0 else n / (1.0 + lam_ / max(scale, 1e-16))
            df = min(max(df, 2.0), n - 1.0)
            gcv = rss / max(1.0 - df / n, 1e-3) ** 2
            if gcv < best:
                best, best_m, best_lam = gcv, m, lam_
        m, lam = best_m, best_lam
    else:
        m = fit(float(lam))
    C_hat = np.interp(K_eval, K_obs, m, left=m[0], right=m[-1])
    q_obs = _second_diff_q(K_obs, m, r * T)
    q_hat = np.interp(K_eval, K_obs, q_obs, left=0.0, right=0.0)
    return C_hat, q_hat, float(lam), m


def svi_iv_bl(K_obs, C_obs, S0, r, T, K_eval, q=0.0):
    """Raw SVI smile fitted to listed IV, then Black prices and BL density."""
    from scipy.optimize import least_squares

    K_obs, C_obs = _sorted(K_obs, C_obs)
    F = S0 * np.exp((r - q) * T)
    iv = bs.implied_vol(C_obs, S0, K_obs, r, T, q)
    ok = np.isfinite(iv) & (iv > 1e-4)
    if ok.sum() < 5:
        return np.zeros_like(np.asarray(K_eval, dtype=float)), np.zeros_like(
            np.asarray(K_eval, dtype=float)
        ), iv
    K_obs, iv = K_obs[ok], iv[ok]
    k = np.log(np.maximum(K_obs, 1e-12) / F)
    w = np.maximum(iv, 1e-4) ** 2 * T
    atm = float(np.median(iv))
    # (a, b, rho, m, sigma)
    x0 = np.array([max(atm**2 * T * 0.5, 1e-6), 0.1, 0.0, 0.0, 0.1])

    def w_svi(x, k_):
        a, b, rho, m, sig = x
        b = np.abs(b)
        rho = np.clip(rho, -0.999, 0.999)
        sig = max(float(sig), 1e-4)
        z = k_ - m
        return a + b * (rho * z + np.sqrt(z * z + sig * sig))

    def resid(x):
        ww = w_svi(x, k)
        return ww - w

    bounds = ([0.0, 0.0, -0.999, -2.0, 1e-4], [2.0, 5.0, 0.999, 2.0, 2.0])
    try:
        res = least_squares(resid, x0, bounds=bounds, max_nfev=400)
        xhat = res.x
    except Exception:
        xhat = x0
    K_eval = np.asarray(K_eval, dtype=float)
    ke = np.log(np.maximum(K_eval, 1e-12) / F)
    we = np.maximum(w_svi(xhat, ke), 1e-8)
    iv_e = np.sqrt(we / max(T, 1e-12))
    iv_e = np.clip(iv_e, 1e-4, 5.0)
    C_hat = bs.call_price(S0, K_eval, r, T, iv_e, q)
    dK = np.median(np.diff(np.sort(K_eval))) if len(K_eval) > 1 else 1.0
    # BL on the evaluation grid if it is regular enough
    if len(K_eval) >= 3 and np.allclose(np.diff(K_eval), dK, rtol=0.05, atol=1e-6):
        second = np.zeros_like(K_eval)
        second[1:-1] = (C_hat[2:] - 2.0 * C_hat[1:-1] + C_hat[:-2]) / dK**2
        q_hat = np.exp(r * T) * second
    else:
        q_hat = _second_diff_q(K_eval, C_hat, r * T)
    return C_hat, q_hat, xhat
