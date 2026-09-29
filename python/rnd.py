"""Risk-neutral density from a single-maturity call strip.

The public entry point is ``estimate_rnd``. Copy this file: it is self-contained
apart from NumPy and SciPy. The estimator rescales calls to a complementary
cdf, fits that function with a smoothing spline, completes the unquoted tails,
and convolves the spline with a Gaussian by a fast Fourier transform.
Thricing and the penalty rule are on by default. The penalty is
``((t - 1.02)_+ / (v_* - 1)) h^4 / (δ F^3)``: ``t`` is the clipped total
variation of the thriced interpolant on ``[0.55F, 1.40F]``, ``v_* - 1`` is
the excess variation of that kernel, and ``δ`` is the median knot spacing.
The default bandwidth is ``0.38 F σ_ATM √T n^{-1/9}``.

Pass ``c=0`` for the cubic through the knots, ``kernel="twice"`` or
``kernel="plain"`` for the other convolutions, and ``tails=False`` to drop
the endpoint knots. ``h`` accepts a number or ``"deriv"``.
"""

from __future__ import annotations

import numpy as np
from scipy.fft import irfft, next_fast_len, rfft, rfftfreq
from scipy.interpolate import CubicSpline
from scipy.special import erf
from scipy.stats import norm

DERIV_C = 0.38

# A unimodal curve scores 1. The penalty stays off until the clipped total
# variation exceeds this fixed allowance.
TV_ALLOW = 1.02


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
    ``((t - 1.02)_+ / (v_* - 1)) h^4 / (δ F^3)``.

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
        ``"rule"`` is ``((t - 1.02)_+ / (v_* - 1)) h^4 / (δ F^3)``. A number,
        including ``0``, is used as given. Zero is the natural cubic through
        the knots.
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
        c_use = _choose_penalty(Ks, Ps, h_use, F, weights)
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


def _conv_dx(Ks, h):
    """Step for the sampled convolution. Finer than the knots and than h."""
    gaps = np.diff(np.asarray(Ks, dtype=float))
    gaps = gaps[gaps > 1e-12]
    gap = float(np.median(gaps)) if gaps.size else float(h)
    return max(min(gap, float(h)) / 64.0, 1e-4)


def _transfer(omega, h):
    """Fourier multiplier of a Gaussian of width ``h``."""
    return np.exp(-0.5 * (omega * max(float(h), 1e-6)) ** 2)


def _fft_grid(Ks, spl, h, factors, cover):
    """Real FFT of the spline, held flat outside the knots.

    Equation (10) convolves the slope. That slope is the derivative of this
    extension, so one transform of the sampled spline feeds every bandwidth:
    multiplication by ``iω`` is the slope, and the Gaussian multiplier is the
    convolution. ``cover`` is every strike the result is read at. The grid
    keeps ten bandwidths of the widest kernel beyond those strikes, which is
    where a circular convolution would otherwise wrap.
    """
    h = max(float(h), 1e-6)
    facs = np.asarray(list(factors), dtype=float)
    dx = _conv_dx(Ks, h * float(facs.min()))
    pad = 10.0 * h * float(facs.max())
    cover = np.asarray(cover, dtype=float).ravel()
    cover = cover[np.isfinite(cover)]
    lo = float(Ks[0])
    hi = float(Ks[-1])
    if cover.size:
        lo = min(lo, float(cover.min()))
        hi = max(hi, float(cover.max()))
    lo -= pad
    hi += pad
    n = int(next_fast_len(int(np.ceil((hi - lo) / dx)) + 1))
    grid = lo + dx * np.arange(n)
    values = np.empty(n)
    left = float(spl(float(Ks[0])))
    right = float(spl(float(Ks[-1])))
    values[:] = right
    values[grid < Ks[0]] = left
    inside = (grid >= Ks[0]) & (grid <= Ks[-1])
    if np.any(inside):
        values[inside] = np.asarray(spl(grid[inside]), dtype=float)
    omega = 2.0 * np.pi * rfftfreq(n, d=dx)
    return grid, omega, rfft(values), left


def _smooth(x, Ks, spl, h):
    """Gaussian convolution of a cubic spline, by one real FFT.

    Returns the smoothed second derivative, the smoothed level, and the
    smoothed first derivative. The call density is ``-F`` times the first
    of these. The level is equation (10): the convolved extension minus the
    value at the left knot, which is the slope convolved with the Gaussian cdf.
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    grid, omega, spec, left = _fft_grid(Ks, spl, h, (1.0,), x)
    n = grid.size
    damped = spec * _transfer(omega, h)
    iw = 1j * omega
    g2 = irfft(damped * iw * iw, n=n)
    g1 = irfft(damped * iw, n=n)
    level = irfft(damped, n=n) - left
    return np.interp(x, grid, g2), np.interp(x, grid, level), np.interp(x, grid, g1)


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


def _convolve(spl, Ks, h, F, x, weights, level_at=None, with_plain=False):
    """Density and, if requested, the smoothed complementary cdf.

    One FFT of the spline. Thricing is the multiplier
    ``(8/3) e^{-h^2 ω^2/2} - 2 e^{-h^2 ω^2} + (1/3) e^{-2 h^2 ω^2}``
    applied to that transform. ``with_plain`` also returns the untwiced
    density, from the same transform, for the penalty rule.
    """
    x = np.asarray(x, dtype=float)
    cover = x.ravel()
    if level_at is not None:
        cover = np.concatenate([cover, np.asarray(level_at, dtype=float).ravel()])
    facs = [float(f) for f, _ in weights]
    if with_plain:
        facs.append(1.0)
    grid, omega, spec, left = _fft_grid(Ks, spl, h, facs, cover)
    n = grid.size
    iw2 = (1j * omega) ** 2
    transfer = np.zeros(omega.shape, dtype=float)
    for fac, w in weights:
        transfer += float(w) * _transfer(omega, h * float(fac))
    g2 = irfft(spec * transfer * iw2, n=n)
    q = -float(F) * np.interp(x, grid, g2)
    G = None
    if level_at is not None:
        # The weights sum to one, so the left endpoint is subtracted once.
        level = irfft(spec * transfer, n=n) - left * float(sum(w for _, w in weights))
        G = np.interp(np.asarray(level_at, dtype=float), grid, level)
    if not with_plain:
        return q, G
    g2_plain = irfft(spec * _transfer(omega, h) * iw2, n=n)
    return q, G, -float(F) * np.interp(x, grid, g2_plain)


def _rule_spline(Ks, Ps, c, F):
    from scipy.interpolate import make_smoothing_spline

    lam = 0.0 if c <= 0.0 else float(c) * float(F) ** 3
    return make_smoothing_spline(Ks, Ps, lam=lam)


def _kernel_excess(weights):
    """Excess signed variation of the equivalent kernel, ``∫|L| - 1``.

    ``L`` is the kernel on the bandwidth scale, so the integral does not
    depend on ``h``. A nonnegative kernel has excess zero; one extra mode
    is then one unit of the penalty.
    """
    z = np.linspace(-40.0, 40.0, 80001)
    L = np.zeros_like(z)
    for fac, w in weights:
        fac = float(fac)
        L += float(w) * norm.pdf(z / fac) / fac
    excess = float(np.trapezoid(np.abs(L), z) - 1.0)
    if excess < 1e-8:
        return 1.0
    return excess


def _clipped_variation(q, F, s):
    """Total variation of the positive part on ``[0.55F, 1.40F]``, over twice the max."""
    m = (s >= 0.55 * F) & (s <= 1.40 * F)
    core = np.maximum(np.nan_to_num(np.asarray(q, float)), 0.0)[m]
    if core.size < 3 or float(core.max()) <= 0.0:
        return float("nan")
    return float(np.sum(np.abs(np.diff(core))) / (2.0 * float(core.max())))


def _choose_penalty(Ks, Ps, h, F, weights):
    """``((t - 1.02)_+ / (v_* - 1)) h^4 / (δ F^3)`` from one interpolant.

    ``t`` is the clipped total variation of the thriced interpolant. ``δ``
    is the median knot spacing, the gap in the smoothing-spline damper
    ``1/(1 + λ δ ω^4)``. The frequency ``ω = 1/h`` turns an excess count
    ``ρ`` into ``λ δ = ρ h^4``.
    """
    s = np.linspace(max(50.0, 0.2 * float(F)), 2.4 * float(F), 700)
    q, _G = _convolve(_rule_spline(Ks, Ps, 0.0, F), Ks, h, F, s, weights)
    t = _clipped_variation(q, F, s)
    gaps = np.diff(np.asarray(Ks, float))
    gaps = gaps[gaps > 1e-12]
    delta = float(np.median(gaps)) if gaps.size else float(h)
    excess = _kernel_excess(weights)
    if not np.isfinite(t) or delta <= 0.0 or float(F) == 0.0:
        return 0.0
    rho = max(t - TV_ALLOW, 0.0) / excess
    return float(rho * (float(h) ** 4) / (delta * float(F) ** 3))


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
