"""Closed-form risk-neutral density from a single-maturity call strip.

The public entry point is ``estimate_rnd``. Copy this file: it is self-contained
apart from NumPy and SciPy. The estimator rescales calls to a complementary
cdf, fits that function with a smoothing spline, completes the unquoted tails,
and convolves the spline with a Gaussian. Thricing and the penalty rule are
on by default. The default bandwidth is ``0.38 F σ_ATM √T n^{-1/9}``.

Pass ``c=0`` for the cubic through the knots, ``kernel="twice"`` or
``kernel="plain"`` for the other convolutions, and ``tails=False`` to drop
the endpoint knots. ``h`` accepts a number or ``"deriv"``.
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.special import erf
from scipy.stats import norm

DERIV_C = 0.38

# Dimensionless penalty in λ = c F^3 ∫(f'')^2. Zero passes through the knots.
SMOOTH_C = np.array([0.0, *np.geomspace(1e-7, 3e-4, 16)])


def _kernel_weights(kernel):
    """Convolution weights ``(bandwidth multiple, weight)``."""
    if kernel in (True, "thrice", "thricing"):
        return ((1.0, 8.0 / 3.0), (np.sqrt(2.0), -2.0), (2.0, 1.0 / 3.0))
    if kernel in ("twice", "twicing"):
        return ((1.0, 2.0), (np.sqrt(2.0), -1.0))
    if kernel in (False, "plain", "gaussian"):
        return ((1.0, 1.0),)
    raise ValueError("kernel must be 'thrice', 'twice', or 'plain'")


def estimate_rnd(
    K,
    C,
    S0,
    r,
    T,
    q=0.0,
    K_eval=None,
    K_price=None,
    h=None,
    tails=True,
    kernel="thrice",
    c="rule",
    higher=None,
    deriv_c=None,
):
    """Risk-neutral density of ``S_T`` from one expiry's calls.

    The default is the estimator in the paper: tails, thricing, bandwidth
    ``deriv_c F σ_ATM √T n^{-1/9}`` with ``deriv_c=0.38``, and the penalty
    chosen from ``SMOOTH_C``.

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
    K_price : array_like, optional
        Strikes at which to return calls. Omitted if None.
    h : float or "deriv", optional
        Bandwidth. ``"deriv"`` (the default) is the rule above. A number
        is used as given.
    tails : bool
        Add the knots ``(0, 0)`` and, when the call has not died,
        ``(K_end, 1)``.
    kernel : {"thrice", "twice", "plain"}
        Thricing is ``(8/3) f_h − 2 f_{h√2} + (1/3) f_{2h}``. Twicing is
        ``2 f_h − f_{h√2}``. ``"plain"`` is one Gaussian.
    c : "rule" or float
        ``"rule"`` picks the penalty from ``SMOOTH_C``. A number, including
        ``0``, is used as given. Zero is the natural cubic through the knots.
    higher : bool or str, optional
        Alias of ``kernel`` kept for older calls. ``True`` is thricing,
        ``False`` is one Gaussian, ``"twice"`` is twicing.
    deriv_c : float, optional
        Replaces ``0.38`` in the default bandwidth.

    Returns
    -------
    dict
        ``q`` on ``K_eval``, ``C`` on ``K_price`` or None, ``h`` the
        bandwidth, and ``c`` the penalty.
    """
    if higher is not None:
        kernel = higher
    weights = _kernel_weights(kernel)
    K = np.asarray(K, dtype=float)
    C = np.asarray(C, dtype=float)
    order = np.argsort(K)
    K, C = K[order], C[order]
    disc = float(np.exp(-float(r) * float(T)))
    stock = float(S0) * np.exp(-float(q) * float(T))
    F = float(S0) * np.exp((float(r) - float(q)) * float(T))
    if K_eval is None:
        K_eval = K
    else:
        K_eval = np.asarray(K_eval, dtype=float)
    if h is None or h == "deriv":
        h_use = _bandwidth(K, C, S0, r, T, q, F, deriv_c=deriv_c)
    else:
        h_use = float(h)
    Ks, Ps = _knots(K, C, stock, F, disc, tails=tails)
    if c == "rule":
        c_use = _choose_penalty(K, C, Ks, Ps, h_use, stock, F, weights)
        spl = _rule_spline(Ks, Ps, c_use, F)
    else:
        c_use = float(c)
        spl = CubicSpline(Ks, Ps, bc_type="natural") if c_use <= 0.0 else _rule_spline(Ks, Ps, c_use, F)
    qhat, G = _convolve(spl, Ks, h_use, F, K_eval, weights, level_at=None if K_price is None else np.asarray(K_price, dtype=float))
    C_hat = None
    if G is not None:
        C_hat = stock * np.clip(1.0 - np.clip(G, 0.0, 1.0), 0.0, 1.0)
        C_hat = np.maximum(C_hat, 0.0)
    return {"q": qhat, "C": C_hat, "h": float(h_use), "c": float(c_use)}


def _knots(K, C, stock, F, disc, tails=True):
    K = np.asarray(K, dtype=float)
    C = np.asarray(C, dtype=float)
    P = 1.0 - C / max(float(stock), 1e-16)
    if tails:
        Ks = np.concatenate([[0.0], K])
        Ps = np.concatenate([[0.0], P])
        spec = _right_wing(K, C, F, disc)
        if spec is not None and spec[3] > Ks[-1] + 1e-6:
            Ks = np.concatenate([Ks, [float(spec[3])]])
            Ps = np.concatenate([Ps, [1.0]])
    else:
        Ks, Ps = K.copy(), P.copy()
    keep = np.concatenate([[True], np.diff(Ks) > 1e-8])
    return Ks[keep], Ps[keep]


def _smooth(x, Ks, spl, h):
    """Gaussian convolution of a cubic spline.

    Returns the smoothed second derivative, the smoothed level, and the
    smoothed first derivative. The call density is ``-F`` times the first
    of these.
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    h = max(float(h), 1e-6)
    gprime = np.zeros_like(x)
    G = np.zeros_like(x)
    gfirst = np.zeros_like(x)
    for L, R in zip(Ks[:-1], Ks[1:]):
        alpha = float(spl(L, 1))
        beta = float(spl(L, 2))
        gamma = 0.5 * float(spl(L, 3))
        uL = (x - L) / h
        uR = (x - R) / h
        A0 = alpha + beta * h * uL + gamma * (h * uL) ** 2
        A1 = -beta * h - 2.0 * gamma * h * h * uL
        A2 = gamma * h * h
        I0 = _Phi(uL) - _Phi(uR)
        I1 = -_phi(uL) + _phi(uR)
        I2 = (-uL * _phi(uL) + _Phi(uL)) - (-uR * _phi(uR) + _Phi(uR))
        dA0 = beta + 2.0 * gamma * h * uL
        dA1 = -2.0 * gamma * h
        dI0 = (_phi(uL) - _phi(uR)) / h
        dI1 = (uL * _phi(uL) - uR * _phi(uR)) / h
        dI2 = (uL * uL * _phi(uL) - uR * uR * _phi(uR)) / h
        gprime += dA0 * I0 + A0 * dI0 + dA1 * I1 + A1 * dI1 + A2 * dI2
        gfirst += A0 * I0 + A1 * I1 + A2 * I2

        def J0(u):
            return u * _Phi(u) + _phi(u)

        def J1(u):
            return 0.5 * u * u * _Phi(u) + 0.5 * u * _phi(u) - 0.5 * _Phi(u)

        def J2(u):
            return u ** 3 * _Phi(u) / 3.0 + u * u * _phi(u) / 3.0 + 2.0 * _phi(u) / 3.0

        G += h * (
            A0 * (J0(uL) - J0(uR))
            + A1 * (J1(uL) - J1(uR))
            + A2 * (J2(uL) - J2(uR))
        )
    return gprime, G, gfirst


def _Phi(u):
    return 0.5 * (1.0 + erf(np.asarray(u, dtype=float) / np.sqrt(2.0)))


def _phi(u):
    u = np.asarray(u, dtype=float)
    return np.exp(-0.5 * u * u) / np.sqrt(2.0 * np.pi)


def _atm_sigma(K, C, S0, r, T, q, F):
    iv = _implied_vol(C, S0, K, r, T, q)
    iv = np.asarray(iv, dtype=float)
    atm = np.nanmedian(iv[np.abs(K - F) <= 0.03 * F])
    if not np.isfinite(atm):
        med = np.nanmedian(iv)
        atm = float(med) if np.isfinite(med) else 0.16
    return float(atm)


def _bandwidth(K, C, S0, r, T, q, F, deriv_c=None):
    """``deriv_c F σ_ATM √T n^{-1/9}``. ``deriv_c`` defaults to ``0.38``."""
    n = max(len(K), 8)
    sig = _atm_sigma(K, C, S0, r, T, q, F)
    const = DERIV_C if deriv_c is None else float(deriv_c)
    return float(const * F * sig * np.sqrt(T) * n ** (-1.0 / 9.0))


def _convolve(spl, Ks, h, F, x, weights, level_at=None):
    """Density and, if requested, the smoothed complementary cdf."""
    x = np.asarray(x, dtype=float)
    q = np.zeros(x.shape, dtype=float)
    G = None if level_at is None else np.zeros(np.shape(level_at), dtype=float)
    for fac, w in weights:
        g2, _, _ = _smooth(x, Ks, spl, h * fac)
        q += w * (-float(F) * g2)
        if level_at is not None:
            G += w * _smooth(level_at, Ks, spl, h * fac)[1]
    return q, G


def _rule_spline(Ks, Ps, c, F):
    from scipy.interpolate import make_smoothing_spline

    lam = 0.0 if c <= 0.0 else float(c) * float(F) ** 3
    return make_smoothing_spline(Ks, Ps, lam=lam)


def _shape_scores(q, q_plain, F, s):
    m = (s >= 0.55 * F) & (s <= 1.40 * F)
    corr = float(np.sqrt(np.mean((q[m] - q_plain[m]) ** 2)))
    p = np.maximum(np.nan_to_num(np.asarray(q, float)), 0.0)
    core = p[m]
    if core.size < 3 or float(core.max()) <= 0.0:
        tv = float("nan")
    else:
        tv = float(np.sum(np.abs(np.diff(core))) / (2.0 * float(core.max())))
    peaks = 0
    if core.size > 4 and np.nanmax(core) > 0:
        peaks = int(np.sum(
            (core[1:-1] > core[:-2]) & (core[1:-1] > core[2:]) & (core[1:-1] > 0.2 * core.max())
        ))
    return tv, peaks, corr


def _choose_penalty(K, C, Ks, Ps, h, stock, F, weights, curv_min=0.5):
    """Smallest unimodal penalty, then the corner when the bend is sharp."""
    s = np.linspace(max(50.0, 0.2 * F), 2.4 * F, 700)
    rows = []
    for c in SMOOTH_C:
        spl = _rule_spline(Ks, Ps, c, F)
        q, G = _convolve(spl, Ks, h, F, s, weights, level_at=K)
        q_plain = -F * _smooth(s, Ks, spl, h)[0]
        tv, peaks, corr = _shape_scores(q, q_plain, F, s)
        Cf = np.maximum(stock * (1.0 - np.clip(G, 0.0, 1.0)), 0.0)
        price = float(np.sqrt(np.mean((Cf - C) ** 2)))
        rows.append(dict(c=float(c), tv=tv, peaks=peaks, price=price, corr=corr))
    uni = [r for r in rows if r["peaks"] <= 1 and r["tv"] <= 1.02]
    c0 = uni[0]["c"] if uni else rows[-1]["c"]
    sub = [r for r in rows if r["c"] + 1e-15 >= c0 and r["price"] > 1e-8 and r["corr"] > 0.0]
    if len(sub) < 5:
        return c0
    x = np.log([r["price"] for r in sub])
    y = np.log([r["corr"] for r in sub])
    dx, dy = np.gradient(x), np.gradient(y)
    ddx, ddy = np.gradient(dx), np.gradient(dy)
    kap = np.abs(dx * ddy - dy * ddx) / np.maximum((dx * dx + dy * dy) ** 1.5, 1e-18)
    kap[0] = kap[-1] = -1.0
    i = int(np.argmax(kap))
    if float(kap[i]) >= curv_min:
        return sub[i]["c"]
    return c0


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


def _mesh_h(K):
    K = np.sort(np.asarray(K, dtype=float))
    delta = float(np.median(np.diff(K)))
    span = float(K[-1] - K[0])
    return float(min(1.2 * delta, 0.30 * span))


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
