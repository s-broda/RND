"""Competitors for the listed comparison.

The historical λ=0 / fixed-bandwidth routines remain in this file. The
listed table uses the routines at the bottom: a solved call-coordinate
projection, cross-validated tuning, Bondarenko weights that are not
rescaled, and the cubic Priestley--Chao smoother.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import LinearConstraint, minimize, nnls
from scipy.stats import norm

from . import black_scholes as bs


def _sorted(K, C):
    K = np.asarray(K, dtype=float)
    C = np.asarray(C, dtype=float)
    o = np.argsort(K)
    K, C = K[o], C[o]
    _, u = np.unique(K, return_index=True)
    return K[u], C[u]


def _c_second_h(K_obs, C_obs, S0, r, T, q=0.0):
    """Silverman bandwidth for a second derivative of C: n^{-1/9}."""
    n = max(len(K_obs), 8)
    iv = bs.implied_vol(C_obs, S0, K_obs, r, T, q)
    atm = np.nanmedian(iv)
    if not np.isfinite(atm):
        atm = 0.2
    F = S0 * np.exp((r - q) * T)
    return float(1.06 * F * atm * np.sqrt(T) * n ** (-1.0 / 9.0))


def _cubic_call(K_obs, C_obs, K_grid, disc):
    """Clamped cubic through quotes, linear wings with slope in [-disc, 0]."""
    from scipy.interpolate import CubicSpline, PchipInterpolator

    K_obs, C_obs = _sorted(K_obs, C_obs)
    dC = np.gradient(C_obs, K_obs)
    sl_l = float(np.clip(dC[0], -disc, 0.0))
    sl_r = float(np.clip(dC[-1], -disc, 0.0))
    if len(K_obs) >= 4:
        spl = CubicSpline(K_obs, C_obs, bc_type=((1, sl_l), (1, sl_r)))
        C = np.asarray(spl(K_grid), dtype=float)
    else:
        C = np.asarray(
            PchipInterpolator(K_obs, C_obs, extrapolate=True)(K_grid), dtype=float
        )
    left, right = K_grid < K_obs[0], K_grid > K_obs[-1]
    C[left] = C_obs[0] + sl_l * (K_grid[left] - K_obs[0])
    C[right] = C_obs[-1] + sl_r * (K_grid[right] - K_obs[-1])
    return np.maximum(C, 0.0)


def priestley_chao_bl(K_obs, C_obs, S0, r, T, K_eval, q=0.0, h=None):
    """Priestley–Chao kernel of quoted call *levels*, then C''. No interpolant."""
    K_obs, C_obs = _sorted(K_obs, C_obs)
    K_eval = np.asarray(K_eval, dtype=float)
    if h is None:
        h = _c_second_h(K_obs, C_obs, S0, r, T, q)
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


def priestley_chao_cubic(K_obs, C_obs, S0, r, T, K_eval, q=0.0, h=None, return_call=False):
    """Cubic spline of C, then Gaussian convolution of C and of C''."""
    K_obs, C_obs = _sorted(K_obs, C_obs)
    K_eval = np.asarray(K_eval, dtype=float)
    disc = np.exp(-r * T)
    if h is None:
        h = _c_second_h(K_obs, C_obs, S0, r, T, q)
    h = max(float(h), 1e-6)
    k_lo = max(1e-6, min(float(K_obs[0]), float(K_eval.min())) - 6.0 * h)
    k_hi = max(float(K_obs[-1]), float(K_eval.max())) + 6.0 * h
    # Resolve the kernel: a fixed 1600-point mesh is coarser than a narrow h
    # and the second-derivative kernel then fails to integrate.
    span = max(k_hi - k_lo, h)
    dK_target = min(span / 1599.0, h / 12.0)
    n_grid = int(np.clip(np.ceil(span / dK_target) + 1, 400, 20000))
    K_grid = np.linspace(k_lo, k_hi, n_grid)
    C_grid = _cubic_call(K_obs, C_obs, K_grid, disc)
    dK = float(K_grid[1] - K_grid[0])
    half = int(min(K_grid.size // 2 - 1, np.ceil(6.0 * h / dK)))
    z = np.arange(-half, half + 1) * dK
    kap = np.exp(-0.5 * (z / h) ** 2) / (h * np.sqrt(2.0 * np.pi))
    kap2 = kap * (z**2 / h**4 - 1.0 / h**2)
    C_smooth = np.convolve(C_grid, kap, mode="same") * dK
    second = np.convolve(C_grid, kap2, mode="same") * dK
    C_eval = np.interp(K_eval, K_grid, np.maximum(C_smooth, 0.0), left=0.0, right=0.0)
    q_eval = np.exp(r * T) * np.interp(K_eval, K_grid, second, left=0.0, right=0.0)
    if return_call:
        return q_eval, h, C_eval
    return q_eval, h


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


# --- Listed comparison: the methods as the source papers specify them ---


def _otm_rmse(K, C_hat, C_ref, S0, r, T, q, F):
    """Floored out-of-the-money premium. Puts at K <= F, calls at K >= F."""
    K = np.asarray(K, dtype=float)
    disc = float(np.exp(-r * T))
    stock = float(S0) * float(np.exp(-q * T))
    C_hat = np.maximum(np.asarray(C_hat, dtype=float), 0.0)
    C_ref = np.maximum(np.asarray(C_ref, dtype=float), 0.0)
    P_hat = np.maximum(C_hat - stock + K * disc, 0.0)
    P_ref = np.maximum(C_ref - stock + K * disc, 0.0)
    a = np.where(K <= F, P_hat, C_hat)
    b = np.where(K <= F, P_ref, C_ref)
    ok = np.isfinite(a) & np.isfinite(b)
    if not np.any(ok):
        return float("nan")
    d = a[ok] - b[ok]
    return float(np.sqrt(np.mean(d * d)))


def _splits(n):
    idx = np.arange(n)
    return ((idx % 2 == 0, idx % 2 == 1), (idx % 2 == 1, idx % 2 == 0))


def _atm_vol(C, S0, K, r, T, q, F):
    iv = bs.implied_vol(C, S0, K, r, T, q)
    atm = np.nanmedian(iv[np.abs(K - F) <= 0.03 * F])
    if not np.isfinite(atm):
        atm = np.nanmedian(iv)
    if not np.isfinite(atm) or atm <= 0.0:
        atm = 0.2
    return float(atm)


def shape_fit_calls(K, C, disc, lam=0.0, intrinsic=None):
    """Euclidean projection onto decreasing convex calls, in call coordinates.

    Slopes lie in [-e^{-rT}, 0]. A positive ``lam`` penalizes squared second
    differences of the slope, scaled by the fourth power of the median gap
    so that ``lam`` is dimensionless. The unknown is the call, not a
    slope increment: the pricing Hessian is the identity.
    """
    K = np.asarray(K, dtype=float)
    C = np.asarray(C, dtype=float)
    n = len(K)
    if n < 3:
        return C.copy()
    dK = np.maximum(np.diff(K), 1e-16)
    disc = float(disc)
    lam = float(lam)
    span = np.maximum(0.5 * (dK[:-1] + dK[1:]), 1e-16)
    B = np.zeros((max(n - 2, 0), n))
    for i in range(n - 2):
        B[i, i] = -1.0 / (dK[i] * span[i])
        B[i, i + 1] = (1.0 / dK[i] + 1.0 / dK[i + 1]) / span[i]
        B[i, i + 2] = -1.0 / (dK[i + 1] * span[i])
    rows, rhs = [], []
    for i in range(n - 1):
        dec = np.zeros(n)
        dec[i], dec[i + 1] = -1.0, 1.0
        rows.append(dec)
        rhs.append(0.0)
        floor = np.zeros(n)
        floor[i], floor[i + 1] = 1.0, -1.0
        rows.append(floor)
        rhs.append(disc * dK[i])
    for i in range(n - 2):
        rows.append(B[i])
        rhs.append(0.0)
    if intrinsic is not None:
        intrinsic = np.asarray(intrinsic, dtype=float)
        for i in range(n):
            row = np.zeros(n)
            row[i] = -1.0
            rows.append(row)
            rhs.append(-float(intrinsic[i]))
    A = np.vstack(rows)
    ub = np.asarray(rhs, dtype=float)
    cons = LinearConstraint(A, -np.inf * np.ones(len(ub)), ub)
    pen_scale = float(np.median(span) ** 4)

    def objective(m):
        r = m - C
        sse = 0.5 * float(r @ r)
        if lam > 0.0 and n >= 3:
            d2 = np.nan_to_num(B @ m, nan=0.0, posinf=0.0, neginf=0.0)
            sse += 0.5 * lam * pen_scale * float(d2 @ d2)
        return sse

    def gradient(m):
        g = m - C
        if lam > 0.0 and n >= 3:
            Bm = np.nan_to_num(B @ m, nan=0.0, posinf=0.0, neginf=0.0)
            g = g + (lam * pen_scale) * (B.T @ Bm)
            g = np.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0)
        return g

    res = minimize(
        objective,
        C.copy(),
        jac=gradient,
        method="SLSQP",
        constraints=cons,
        options={"maxiter": 200, "ftol": 1e-12, "disp": False},
    )
    m = np.asarray(res.x if res.success else C, dtype=float)
    s = np.diff(m) / dK
    if (not res.success) or np.any(s < -disc - 1e-6) or np.any(s > 1e-6) or np.any(np.diff(s) < -1e-6):
        s = np.clip(s, -disc, 0.0)
        s = _pava_increasing(s)
        s = np.clip(s, -disc, 0.0)
        m = np.concatenate([[float(m[0])], m[0] + np.cumsum(s * dK)])
    if intrinsic is not None:
        m = np.maximum(m, np.asarray(intrinsic, dtype=float))
    return np.maximum(m, 0.0)


def _lambda_grid():
    return np.concatenate([[0.0], np.logspace(-2, 4, 7)])


def yatchew_cv_lambda(K, C, S0, r, T, q, F):
    """Even/odd penalty for Yatchew--Härdle. Returns λ."""
    K, C = _sorted(K, C)
    disc = float(np.exp(-r * T))
    best_lam, best = 0.0, np.inf
    for lam in _lambda_grid():
        errs = []
        for train, test in _splits(len(K)):
            if int(train.sum()) < 4 or int(test.sum()) < 2:
                continue
            m = shape_fit_calls(K[train], C[train], disc, lam=float(lam))
            C_te = np.interp(K[test], K[train], m)
            errs.append(_otm_rmse(K[test], C_te, C[test], S0, r, T, q, F))
        if errs and float(np.mean(errs)) < best:
            best = float(np.mean(errs))
            best_lam = float(lam)
    return best_lam


def yatchew_fit(K, C, S0, r, T, q, F, K_eval, lam):
    K, C = _sorted(K, C)
    disc = float(np.exp(-r * T))
    intrinsic = disc * np.maximum(F - K, 0.0)
    m = shape_fit_calls(K, C, disc, lam=float(lam), intrinsic=intrinsic)
    C_hat = np.interp(np.asarray(K_eval, dtype=float), K, m, left=m[0], right=m[-1])
    return C_hat, m


def _local_linear(K, Y, K_eval, h):
    K = np.asarray(K, dtype=float)
    Y = np.asarray(Y, dtype=float)
    K_eval = np.asarray(K_eval, dtype=float)
    h = max(float(h), 1e-6)
    level = np.empty_like(K_eval)
    half = 5.0 * h
    for j, x in enumerate(K_eval):
        sel = np.abs(K - x) <= half
        if int(sel.sum()) < 2:
            sel = np.ones(K.size, dtype=bool)
        dx = K[sel] - x
        yy = Y[sel]
        w = np.exp(-0.5 * (dx / h) ** 2)
        sw = np.sqrt(np.maximum(w, 0.0))
        X = np.column_stack([np.ones_like(dx), dx])
        beta, *_ = np.linalg.lstsq(X * sw[:, None], yy * sw, rcond=None)
        level[j] = beta[0]
    return level


def _local_cubic_second(K, Y, K_eval, h):
    """Twice the quadratic coefficient of a local cubic. Caller scales by e^{rT}."""
    K = np.asarray(K, dtype=float)
    Y = np.asarray(Y, dtype=float)
    K_eval = np.asarray(K_eval, dtype=float)
    h = max(float(h), 1e-6)
    out = np.zeros_like(K_eval)
    half = 5.0 * h
    for j, x in enumerate(K_eval):
        sel = np.abs(K - x) <= half
        if int(sel.sum()) < 4:
            sel = np.ones(K.size, dtype=bool)
        dx = K[sel] - x
        yy = Y[sel]
        w = np.exp(-0.5 * (dx / h) ** 2)
        if w.sum() < 1e-14:
            continue
        sw = np.sqrt(w)
        X = np.column_stack([np.ones_like(dx), dx, dx**2, dx**3])
        beta, *_ = np.linalg.lstsq(X * sw[:, None], yy * sw, rcond=None)
        out[j] = 2.0 * beta[2]
    return out


def asd_cv_bandwidth(K, C, S0, r, T, q, F):
    """Even/odd local-linear bandwidth on the solved projection.

    Returns ``(h_price, h_density)``. The density bandwidth is the fixed
    second-derivative rate ``0.9 F σ √T n^{-1/9}``.
    """
    K, C = _sorted(K, C)
    disc = float(np.exp(-r * T))
    n = max(len(K), 8)
    sig = _atm_vol(C, S0, K, r, T, q, F)
    s = F * sig * np.sqrt(T)
    h_dens = float(0.9 * s * n ** (-1.0 / 9.0))
    grid = s * n ** (-0.2) * np.array([0.25, 0.5, 1.0, 2.0, 4.0])
    best_h, best = float(grid[2]), np.inf
    folds = []
    for train, test in _splits(len(K)):
        if int(train.sum()) < 4 or int(test.sum()) < 2:
            continue
        m_tr = shape_fit_calls(K[train], C[train], disc, lam=0.0)
        folds.append((train, test, m_tr))
    for h in grid:
        errs = []
        for train, test, m_tr in folds:
            C_te = _local_linear(K[train], m_tr, K[test], float(h))
            errs.append(_otm_rmse(K[test], C_te, C[test], S0, r, T, q, F))
        if errs and float(np.mean(errs)) < best:
            best = float(np.mean(errs))
            best_h = float(h)
    return best_h, h_dens


def asd_fit(K, C, S0, r, T, q, F, K_price, K_dens, h_price, h_dens):
    K, C = _sorted(K, C)
    disc = float(np.exp(-r * T))
    intrinsic = disc * np.maximum(F - K, 0.0)
    m = shape_fit_calls(K, C, disc, lam=0.0, intrinsic=intrinsic)
    C_hat = _local_linear(K, m, K_price, h_price)
    q_hat = np.exp(r * T) * _local_cubic_second(K, m, K_dens, h_dens)
    return C_hat, q_hat, m


def _pca_centers(K, n_centers=41):
    K = np.asarray(K, dtype=float)
    lo = np.log(max(float(K[0]), 1e-8))
    hi = np.log(max(float(K[-1]), float(K[0]) * 1.01))
    if hi <= lo:
        hi = lo + 0.05
    return np.linspace(lo, hi, int(n_centers))


def _pca_h_grid(K):
    z = _pca_centers(K, 41)
    dz = float(z[1] - z[0]) if z.size > 1 else 0.05
    return np.unique(np.clip(dz * np.array([1.0, 2.0, 4.0, 8.0, 16.0]), 0.015, 1.25))


def _pca_matrices(K, z, h, F, disc):
    h = max(float(h), 1e-4)
    K = np.asarray(K, dtype=float)
    z = np.asarray(z, dtype=float)
    d2 = np.clip((z[None, :] - np.log(np.maximum(K, 1e-12))[:, None]) / h, -30.0, 30.0)
    d1 = d2 + h
    Fm = np.exp(z + 0.5 * h * h)
    W = disc * (Fm[None, :] * norm.cdf(d1) - K[:, None] * norm.cdf(d2))
    return np.nan_to_num(W, nan=0.0, posinf=0.0, neginf=0.0), Fm


def _pca_density(K_eval, z, h, a):
    h = max(float(h), 1e-4)
    K_eval = np.asarray(K_eval, dtype=float)
    x = np.log(np.maximum(K_eval, 1e-12))[:, None]
    u = np.clip((x - z[None, :]) / h, -20.0, 20.0)
    kern = np.exp(-0.5 * u * u) / (
        np.maximum(K_eval[:, None], 1e-12) * h * np.sqrt(2.0 * np.pi)
    )
    a = np.clip(np.nan_to_num(np.asarray(a, dtype=float), nan=0.0), 0.0, 1.0)
    return np.nan_to_num(kern @ a, nan=0.0, posinf=0.0, neginf=0.0)


def _pca_weights(W, C, Fm, F):
    """NNLS with sum-to-one and forward penalties. Weights are not rescaled."""
    n = W.shape[1]
    col = np.maximum(np.linalg.norm(W, axis=0), 1e-12)
    scale = max(float(np.median(np.abs(C))), 1e-6)
    lam = 8.0 * scale
    W_aug = np.vstack([W / col, lam / col, lam * Fm / (max(float(F), 1e-12) * col)])
    c_aug = np.concatenate([np.asarray(C, dtype=float), [lam, lam]])
    W_aug = np.nan_to_num(W_aug, nan=0.0, posinf=0.0, neginf=0.0)
    try:
        with np.errstate(all="ignore"):
            a_n, _ = nnls(W_aug, c_aug, maxiter=500)
    except Exception:
        a_n = np.zeros(n)
    a = np.maximum(a_n / col, 0.0)
    return np.clip(np.nan_to_num(a, nan=0.0), 0.0, 1.0)


def pca_cv_bandwidth(K, C, S0, r, T, q, F):
    K, C = _sorted(K, C)
    disc = float(np.exp(-r * T))
    grid = _pca_h_grid(K)
    best_h, best = float(grid[0]), np.inf
    for h in grid:
        errs = []
        for train, test in _splits(len(K)):
            if int(train.sum()) < 6 or int(test.sum()) < 2:
                continue
            z = _pca_centers(K[train], 41)
            W, Fm = _pca_matrices(K[train], z, float(h), F, disc)
            a = _pca_weights(W, C[train], Fm, F)
            W_te, _ = _pca_matrices(K[test], z, float(h), F, disc)
            pred = np.nan_to_num(W_te @ a, nan=0.0, posinf=0.0, neginf=0.0)
            errs.append(_otm_rmse(K[test], pred, C[test], S0, r, T, q, F))
        if errs and float(np.mean(errs)) < best:
            best = float(np.mean(errs))
            best_h = float(h)
    return best_h


def pca_fit(K, C, r, T, F, h, K_price, K_dens, n_centers=41):
    K, C = _sorted(K, C)
    disc = float(np.exp(-r * T))
    z = _pca_centers(K, n_centers)
    W, Fm = _pca_matrices(K, z, h, F, disc)
    a = _pca_weights(W, C, Fm, F)
    W_price, _ = _pca_matrices(np.asarray(K_price, dtype=float), z, h, F, disc)
    C_hat = np.nan_to_num(W_price @ a, nan=0.0, posinf=0.0, neginf=0.0)
    q_hat = _pca_density(K_dens, z, h, a)
    return C_hat, q_hat, a, z, h


def _spacing(K):
    K = np.asarray(K, dtype=float)
    dK = np.empty_like(K)
    if K.size < 2:
        dK[:] = 1.0
        return dK
    dK[0] = K[1] - K[0]
    dK[1:] = np.diff(K)
    return np.maximum(dK, 0.0)


def priestley_chao_kernel(K, C, K_eval, h, r=0.0):
    """Priestley--Chao (1972) kernel of the call, and its Breeden--Litzenberger density.

    ``sum ΔK_i C_i κ_h(K-K_i)``, with ``ΔK_1 = K_2-K_1``. The density is
    ``e^{rT}`` times the second strike derivative of that kernel.
    """
    K, C = _sorted(K, C)
    K_eval = np.asarray(K_eval, dtype=float)
    h = max(float(h), 1e-6)
    dK = _spacing(K)
    u = (K_eval[:, None] - K[None, :]) / h
    kap = np.exp(-0.5 * u * u) / (h * np.sqrt(2.0 * np.pi))
    kap2 = kap * (u * u - 1.0) / (h * h)
    weight = (dK * C)[None, :]
    call = np.nan_to_num((weight * kap).sum(axis=1), nan=0.0, posinf=0.0, neginf=0.0)
    dens = np.exp(r) * np.nan_to_num((weight * kap2).sum(axis=1), nan=0.0, posinf=0.0, neginf=0.0)
    return call, dens


def _pc_h_grid(K, C, S0, r, T, q, F):
    K = np.asarray(K, dtype=float)
    n = max(len(K), 8)
    delta = float(np.median(np.diff(K))) if K.size > 1 else 1.0
    sig = _atm_vol(C, S0, K, r, T, q, F)
    s = F * sig * np.sqrt(T)
    raw = np.concatenate(
        [
            delta * np.array([1.0, 2.0, 4.0, 8.0, 16.0]),
            s * n ** (-0.2) * np.array([0.25, 0.5, 1.0, 2.0]),
        ]
    )
    return np.unique(np.clip(raw, max(delta, 1e-3), max(2.5 * s, delta * 4)))


def priestley_chao_cv(K, C, S0, r, T, q, F):
    """Even/odd bandwidth for the Priestley--Chao call kernel."""
    K, C = _sorted(K, C)
    grid = _pc_h_grid(K, C, S0, r, T, q, F)
    best_h, best = float(grid[len(grid) // 2]), np.inf
    for h in grid:
        errs = []
        for train, test in _splits(len(K)):
            if int(train.sum()) < 6 or int(test.sum()) < 2:
                continue
            C_te, _ = priestley_chao_kernel(K[train], C[train], K[test], float(h), r=r * T)
            errs.append(_otm_rmse(K[test], C_te, C[test], S0, r, T, q, F))
        if errs and float(np.mean(errs)) < best:
            best = float(np.mean(errs))
            best_h = float(h)
    return best_h
