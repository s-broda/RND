"""Retuned benchmarks for the v2 comparison.

The v1 competitors in ``competitors.py`` are unchanged. This module

- places Bondarenko's lognormal mixture on the quoted log-strike range and
  does not rescale the fitted weights,
- chooses the Yatchew--Härdle roughness penalty, the Priestley--Chao
  bandwidth, and the Aït-Sahalia--Duarte pricing bandwidth by even/odd
  cross-validation,
- fits a raw SVI slice penalized for butterfly arbitrage.

Cross-validation scores out-of-the-money puts and calls and never sees the
risk-neutral density.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import LinearConstraint, least_squares, minimize, nnls
from scipy.stats import norm

from . import black_scholes as bs
from .competitors import _pava_increasing, _sorted


def rmse(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    ok = np.isfinite(a) & np.isfinite(b)
    if not np.any(ok):
        return float("nan")
    d = a[ok] - b[ok]
    return float(np.sqrt(np.mean(d * d)))


def otm_prices(K, C, S0, r, T, q, F):
    """Out-of-the-money premium: put on K <= F, call on K >= F."""
    K = np.asarray(K, dtype=float)
    C = np.asarray(C, dtype=float)
    disc = float(np.exp(-r * T))
    stock = float(S0) * float(np.exp(-q * T))
    # Premiums are floored at zero. A negative parity put is not the quote
    # the estimator returns, and it is not a traded out-of-the-money price.
    C = np.maximum(C, 0.0)
    P = np.maximum(C - stock + K * disc, 0.0)
    return np.where(K <= F, P, C)


def otm_rmse(K, C_hat, C_ref, S0, r, T, q, F):
    return rmse(
        otm_prices(K, C_hat, S0, r, T, q, F),
        otm_prices(K, C_ref, S0, r, T, q, F),
    )


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


def integrate_density(K, q):
    """Trapezoid integrals of a density on a sorted grid.

    Returns signed mass, positive mass, negative mass, and the signed
    first moment ∫ K q(K) dK. Positive and negative masses split the
    integrand before integration, so they add to the signed mass.
    """
    K = np.asarray(K, dtype=float)
    q = np.asarray(q, dtype=float)
    order = np.argsort(K)
    K, q = K[order], q[order]
    dK = np.diff(K)

    def _trap(vals):
        return float(np.sum(0.5 * (vals[1:] + vals[:-1]) * dK))

    def _moment(vals):
        mid_q = 0.5 * (vals[1:] + vals[:-1])
        mid_k = 0.5 * (K[1:] + K[:-1])
        return float(np.sum(mid_q * mid_k * dK))

    signed = _trap(q)
    pos = _trap(np.maximum(q, 0.0))
    neg = _trap(np.minimum(q, 0.0))
    moment = _moment(q)
    return {
        "signed": signed,
        "positive": pos,
        "negative": neg,
        "moment": moment,
    }


def diagnostics(K, q, F):
    out = integrate_density(K, q)
    out["forward_error"] = float(out["moment"] - F)
    out["F"] = float(F)
    return out


def _shape_fit(K, C, disc, lam=0.0, intrinsic=None, weights=None):
    """Decreasing convex projection with slope bounds and roughness λ.

    The unknown is the call itself. The Hessian of the pricing term is the
    identity, so a slope-increment parameterization is not used: that map is
    ill-conditioned and the solver stops on an infeasible quadratic
    subproblem. ``weights`` is accepted for signature compatibility and is
    not used; the empirical criterion is unweighted pricing error.
    """
    del weights
    K = np.asarray(K, dtype=float)
    C = np.asarray(C, dtype=float)
    n = len(K)
    if n < 3:
        return C.copy()
    dK = np.maximum(np.diff(K), 1e-16)
    disc = float(disc)
    lam = float(lam)
    span = np.maximum(0.5 * (dK[:-1] + dK[1:]), 1e-16)
    # Second-difference matrix of the slopes, shape (n-2, n).
    B = np.zeros((n - 2, n))
    for i in range(n - 2):
        B[i, i] = -1.0 / (dK[i] * span[i])
        B[i, i + 1] = (1.0 / dK[i] + 1.0 / dK[i + 1]) / span[i]
        B[i, i + 2] = -1.0 / (dK[i + 1] * span[i])
    rows = []
    rhs = []
    for i in range(n - 1):
        dec = np.zeros(n)
        dec[i] = -1.0
        dec[i + 1] = 1.0
        rows.append(dec)
        rhs.append(0.0)
        floor = np.zeros(n)
        floor[i] = 1.0
        floor[i + 1] = -1.0
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
        if lam > 0.0:
            with np.errstate(all="ignore"):
                d2 = B @ m
            d2 = np.nan_to_num(d2, nan=0.0, posinf=0.0, neginf=0.0)
            sse += 0.5 * lam * pen_scale * float(d2 @ d2)
        return sse

    def gradient(m):
        g = m - C
        if lam > 0.0:
            with np.errstate(all="ignore"):
                Bm = B @ m
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
    # One convex repair if the solver returns an infeasible point.
    s = np.diff(m) / dK
    if (not res.success) or np.any(s < -disc - 1e-6) or np.any(s > 1e-6) or np.any(np.diff(s) < -1e-6):
        s = np.clip(s, -disc, 0.0)
        s = _pava_increasing(s)
        s = np.clip(s, -disc, 0.0)
        m = np.concatenate([[m[0]], m[0] + np.cumsum(s * dK)])
    if intrinsic is not None:
        m = np.maximum(m, np.asarray(intrinsic, dtype=float))
    return np.maximum(m, 0.0)


def _lambda_grid(K, C, disc):
    del K, C, disc
    # Penalty is multiplied by median(gap)^4 inside the fit, so these values
    # are dimensionless relative to a unit second difference of the slope.
    return np.concatenate([[0.0], np.logspace(-2, 4, 7)])


def yatchew_cv_lambda(K, C, S0, r, T, q, F):
    """Even/odd λ on the given quotes. Returns the penalty."""
    K, C = _sorted(K, C)
    disc = float(np.exp(-r * T))
    grid = _lambda_grid(K, C, disc)
    best_lam, best = 0.0, np.inf
    for lam in grid:
        errs = []
        for train, test in _splits(len(K)):
            if train.sum() < 4 or test.sum() < 2:
                continue
            m = _shape_fit(K[train], C[train], disc, lam=float(lam))
            C_te = np.interp(K[test], K[train], m)
            errs.append(otm_rmse(K[test], C_te, C[test], S0, r, T, q, F))
        if errs and np.mean(errs) < best:
            best = float(np.mean(errs))
            best_lam = float(lam)
    return best_lam


def yatchew_fit(K, C, S0, r, T, q, F, K_eval, lam):
    K, C = _sorted(K, C)
    disc = float(np.exp(-r * T))
    intrinsic = disc * np.maximum(F - K, 0.0)
    m = _shape_fit(K, C, disc, lam=float(lam), intrinsic=intrinsic)
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
        sw = np.sqrt(w)
        X = np.column_stack([np.ones_like(dx), dx])
        beta, *_ = np.linalg.lstsq(X * sw[:, None], yy * sw, rcond=None)
        level[j] = beta[0]
    return level


def _local_cubic_second(K, Y, K_eval, h):
    """e^{rT} is applied by the caller. Returns 2 β₂ of a local cubic."""
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
    """Cross-validate the local-linear pricing bandwidth on a fixed projection."""
    K, C = _sorted(K, C)
    disc = float(np.exp(-r * T))
    intrinsic = disc * np.maximum(F - K, 0.0)
    m = _shape_fit(K, C, disc, lam=0.0, intrinsic=intrinsic)
    n = max(len(K), 8)
    sig = _atm_vol(m, S0, K, r, T, q, F)
    s = F * sig * np.sqrt(T)
    h_dens = float(0.9 * s * n ** (-1.0 / 9.0))
    grid = s * n ** (-0.2) * np.array([0.25, 0.5, 1.0, 2.0, 4.0])
    best_h, best = float(grid[2]), np.inf
    fold_proj = []
    for train, test in _splits(len(K)):
        if train.sum() < 4 or test.sum() < 2:
            continue
        m_tr = _shape_fit(K[train], C[train], disc, lam=0.0)
        fold_proj.append((train, test, m_tr))
    for h in grid:
        errs = []
        for train, test, m_tr in fold_proj:
            C_te = _local_linear(K[train], m_tr, K[test], float(h))
            errs.append(otm_rmse(K[test], C_te, C[test], S0, r, T, q, F))
        if errs and np.mean(errs) < best:
            best = float(np.mean(errs))
            best_h = float(h)
    return best_h, h_dens, m


def asd_fit(K, C, S0, r, T, q, F, K_price, K_dens, h_price, h_dens):
    K, C = _sorted(K, C)
    disc = float(np.exp(-r * T))
    intrinsic = disc * np.maximum(F - K, 0.0)
    m = _shape_fit(K, C, disc, lam=0.0, intrinsic=intrinsic)
    C_hat = _local_linear(K, m, K_price, h_price)
    q_hat = np.exp(r * T) * _local_cubic_second(K, m, K_dens, h_dens)
    return C_hat, q_hat, m


def _pca_matrices(K, z, h, F, disc):
    """Call design and lognormal density kernel. h is log-space width."""
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
    a = np.nan_to_num(np.asarray(a, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    with np.errstate(all="ignore"):
        q_hat = kern @ a
    return np.nan_to_num(q_hat, nan=0.0, posinf=0.0, neginf=0.0)


def _pca_weights(W, C, Fm, F):
    """Nonnegative least squares. Penalties encourage sum a = 1 and mean F.

    The fitted weights are returned as solved. They are not rescaled.
    """
    n = W.shape[1]
    col = np.maximum(np.linalg.norm(W, axis=0), 1e-12)
    scale = max(float(np.median(np.abs(C))), 1e-6)
    lam = 8.0 * scale
    W_aug = np.vstack([W / col, lam / col, lam * Fm / (max(F, 1e-12) * col)])
    c_aug = np.concatenate([np.asarray(C, dtype=float), [lam, lam]])
    W_aug = np.nan_to_num(W_aug, nan=0.0, posinf=0.0, neginf=0.0)
    try:
        with np.errstate(all="ignore"):
            a_n, _ = nnls(W_aug, c_aug, maxiter=500)
    except Exception:
        a_n = np.zeros(n)
    a = np.maximum(a_n / col, 0.0)
    a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
    # A probability weight cannot exceed one once the simplex penalty has
    # done its job. The cap only binds on a singular design.
    return np.clip(a, 0.0, 1.0)


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
    # At least one center spacing: a narrower kernel leaves holes between nodes.
    return np.unique(np.clip(dz * np.array([1.0, 2.0, 4.0, 8.0, 16.0]), 0.015, 1.25))


def pca_cv_bandwidth(K, C, S0, r, T, q, F):
    K, C = _sorted(K, C)
    disc = float(np.exp(-r * T))
    grid = _pca_h_grid(K)
    best_h, best = float(grid[0]), np.inf
    for h in grid:
        errs = []
        for train, test in _splits(len(K)):
            if train.sum() < 6 or test.sum() < 2:
                continue
            z = _pca_centers(K[train], 41)
            W, Fm = _pca_matrices(K[train], z, float(h), F, disc)
            a = _pca_weights(W, C[train], Fm, F)
            W_te, _ = _pca_matrices(K[test], z, float(h), F, disc)
            with np.errstate(all="ignore"):
                pred = np.nan_to_num(W_te @ a, nan=0.0, posinf=0.0, neginf=0.0)
            errs.append(otm_rmse(K[test], pred, C[test], S0, r, T, q, F))
        if errs and np.mean(errs) < best:
            best = float(np.mean(errs))
            best_h = float(h)
    return best_h


def pca_fit(K, C, r, T, F, h, K_price, K_dens, n_centers=41):
    K, C = _sorted(K, C)
    disc = float(np.exp(-r * T))
    z = _pca_centers(K, n_centers)
    W, Fm = _pca_matrices(K, z, h, F, disc)
    a = _pca_weights(W, C, Fm, F)
    W_price, _ = _pca_matrices(K_price, z, h, F, disc)
    q_hat = _pca_density(K_dens, z, h, a)
    with np.errstate(all="ignore"):
        C_hat = np.nan_to_num(W_price @ a, nan=0.0, posinf=0.0, neginf=0.0)
    return C_hat, q_hat, a, z


def _pc_h_grid(K, C, S0, r, T, q, F):
    K = np.asarray(K, dtype=float)
    n = max(len(K), 8)
    delta = float(np.median(np.diff(K)))
    sig = _atm_vol(C, S0, K, r, T, q, F)
    s = F * sig * np.sqrt(T)
    raw = np.concatenate(
        [
            delta * np.array([2.0, 4.0, 8.0, 16.0]),
            s * n ** (-0.2) * np.array([0.35, 0.7, 1.4]),
            s * n ** (-1.0 / 9.0) * np.array([0.5, 1.0]),
        ]
    )
    return np.unique(np.clip(raw, max(delta, 1e-3), max(2.5 * s, delta * 4)))


def priestley_cv_bandwidth(K, C, S0, r, T, q, F, price_fn):
    """``price_fn(K_obs, C_obs, K_eval, h) -> call prices``."""
    K, C = _sorted(K, C)
    grid = _pc_h_grid(K, C, S0, r, T, q, F)
    best_h, best = float(grid[len(grid) // 2]), np.inf
    for h in grid:
        errs = []
        for train, test in _splits(len(K)):
            if train.sum() < 6 or test.sum() < 2:
                continue
            C_te = price_fn(K[train], C[train], K[test], float(h))
            errs.append(otm_rmse(K[test], C_te, C[test], S0, r, T, q, F))
        if errs and np.mean(errs) < best:
            best = float(np.mean(errs))
            best_h = float(h)
    return best_h


def _svi_total_var(k, a, b, rho, m, sig):
    km = k - m
    return a + b * (rho * km + np.sqrt(km * km + sig * sig))


def _svi_g(k, a, b, rho, m, sig):
    """Gatheral--Jacquier butterfly function of a raw SVI slice."""
    km = k - m
    rad = np.sqrt(km * km + sig * sig)
    w = a + b * (rho * km + rad)
    wp = b * (rho + km / rad)
    wpp = b * sig * sig / rad**3
    w = np.maximum(w, 1e-12)
    return (1.0 - k * wp / (2.0 * w)) ** 2 - 0.25 * wp**2 * (1.0 / w + 0.25) + 0.5 * wpp


def _svi_unpack(th):
    a, log_b, rho, m, log_sig = th
    return float(a), float(np.exp(log_b)), float(rho), float(m), float(np.exp(log_sig))


def svi_fit(K, C, S0, r, T, q, F, K_price, K_dens):
    """Raw SVI calibrated to call prices, penalized when g(k) < 0.

    Returns call prices, a Breeden--Litzenberger density on ``K_dens``,
    the parameter vector, and the minimum of g on a fixed log-moneyness grid.
    """
    K, C = _sorted(K, C)
    k = np.log(np.maximum(K, 1e-12) / F)
    scale = max(float(np.median(np.abs(C))), 1e-4)
    k_pen = np.linspace(-1.6, 1.6, 96)

    def residuals(th):
        a, b, rho, m, sig = _svi_unpack(th)
        w = _svi_total_var(k, a, b, rho, m, sig)
        bad = w <= 1e-8
        vol = np.sqrt(np.maximum(w, 1e-8) / max(T, 1e-8))
        model = bs.call_price(S0, K, r, T, vol, q)
        price_r = (model - C) / scale
        price_r = np.where(bad, price_r + 10.0, price_r)
        g = _svi_g(k_pen, a, b, rho, m, sig)
        pen = 8.0 * np.minimum(g, 0.0)
        return np.concatenate([price_r, pen])

    sig0 = _atm_vol(C, S0, K, r, T, q, F)
    w_atm = max(sig0**2 * T, 1e-4)
    starts = []
    for rho0, m0 in ((-0.6, -0.05), (-0.3, 0.0), (-0.75, 0.1), (0.0, 0.0)):
        starts.append(
            np.array([0.5 * w_atm, np.log(0.15), rho0, m0, np.log(0.25)])
        )
    lb = np.array([-0.5, np.log(1e-4), -0.995, -1.5, np.log(1e-3)])
    ub = np.array([2.0, np.log(3.0), 0.995, 1.5, np.log(2.0)])
    best, best_cost = starts[0], np.inf
    for th0 in starts:
        res = least_squares(
            residuals,
            th0,
            bounds=(lb, ub),
            method="trf",
            ftol=1e-10,
            xtol=1e-10,
            max_nfev=80,
        )
        cost = float(np.dot(res.fun, res.fun))
        if cost < best_cost:
            best, best_cost = res.x, cost
    a, b, rho, m, sig = _svi_unpack(best)
    g_min = float(np.min(_svi_g(k_pen, a, b, rho, m, sig)))

    def _calls(KK):
        KK = np.asarray(KK, dtype=float)
        kk = np.log(np.maximum(KK, 1e-12) / F)
        w = np.maximum(_svi_total_var(kk, a, b, rho, m, sig), 1e-8)
        vol = np.sqrt(w / max(T, 1e-8))
        return bs.call_price(S0, KK, r, T, vol, q)

    C_price = _calls(K_price)
    # Uniform grid for a stable second derivative, then interpolate.
    lo = max(float(np.min(K_dens)), 1e-3)
    hi = float(np.max(K_dens))
    grid = np.linspace(lo, hi, 2401)
    Cg = _calls(grid)
    d1 = np.gradient(Cg, grid)
    d2 = np.gradient(d1, grid)
    q_grid = np.exp(r * T) * d2
    q_hat = np.interp(np.asarray(K_dens, dtype=float), grid, q_grid)
    params = {"a": a, "b": b, "rho": rho, "m": m, "sigma": sig, "g_min": g_min}
    return C_price, q_hat, params
