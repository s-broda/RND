"""Risk-neutral density from a single-maturity call strip.

The public entry point is ``estimate_rnd``. Copy this file together with
``density_closed.c``. The estimator depends only on NumPy and SciPy, and on a
C compiler when one is present. It rescales calls to a complementary cdf,
fits that function with a smoothing spline, completes the unquoted tails, and
convolves the spline with a Gaussian. That integral is elementary on each
cell of the cubic. The expansion in ``density_closed.c`` evaluates it
by one FFT of the shared box moments, with thricing as one multiplier;
if that file cannot be compiled, the same convolution is one real FFT
of the sampled spline.
Thricing and the penalty rule are on by default. The default bandwidth is
``0.38 F σ_ATM √T n^{-1/9}``.

Pass ``c=0`` for the cubic through the knots, ``kernel="twice"`` or
``kernel="plain"`` for the other convolutions, and ``tails=False`` to drop
the endpoint knots. ``h`` accepts a number or ``"deriv"``.
"""

from __future__ import annotations

import ctypes
import hashlib
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from scipy.fft import irfft, next_fast_len, rfft, rfftfreq
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


def _convolve_fft(spl, Ks, h, F, x, weights, level_at=None, with_plain=False):
    """Density and level from one real FFT of the flat-extended spline.

    Thricing is the multiplier
    ``(8/3) e^{-h^2 ω^2/2} - 2 e^{-h^2 ω^2} + (1/3) e^{-2 h^2 ω^2}``
    applied to that transform. ``with_plain`` also returns the untwiced
    density, from the same transform, for the penalty rule. This is the
    fallback when ``density_closed.c`` cannot be compiled.
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


_CLOSED = None
_CLOSED_TRIED = False


def _spline_jets(spl, t):
    """Value, slope, and second derivative of ``spl`` at the knots ``t``."""
    t = np.ascontiguousarray(np.asarray(t, dtype=np.float64).ravel())
    lib = _closed_library()
    m = int(t.size)
    p0 = np.empty(m, dtype=np.float64)
    p1 = np.empty(m, dtype=np.float64)
    sec = np.empty(m, dtype=np.float64)
    c = getattr(spl, "c", None)
    breaks = getattr(spl, "x", None)
    if (
        lib is not None
        and c is not None
        and breaks is not None
        and np.ndim(c) == 2
        and c.shape[0] == 4
        and np.shape(breaks) == t.shape
        and np.array_equal(np.asarray(breaks), t)
    ):
        coef = np.ascontiguousarray(c, dtype=np.float64)
        lib.ppoly_jets(
            _c_ptr(coef), _c_ptr(t), ctypes.c_int(m),
            _c_ptr(p0), _c_ptr(p1), _c_ptr(sec),
        )
        return t, p0, p1, sec
    knots = getattr(spl, "t", None)
    degree = getattr(spl, "k", None)
    if lib is not None and knots is not None and c is not None and degree == 3 and np.ndim(c) == 1:
        tk = np.ascontiguousarray(knots, dtype=np.float64)
        coef = np.ascontiguousarray(c, dtype=np.float64)
        rc = lib.bspline_jets(
            _c_ptr(tk), ctypes.c_int(tk.size),
            _c_ptr(coef), ctypes.c_int(coef.size), ctypes.c_int(3),
            _c_ptr(t), ctypes.c_int(m),
            _c_ptr(p0), _c_ptr(p1), _c_ptr(sec),
        )
        if rc == 0:
            return t, p0, p1, sec
    return (
        t,
        np.ascontiguousarray(spl(t), dtype=np.float64),
        np.ascontiguousarray(spl(t, nu=1), dtype=np.float64),
        np.ascontiguousarray(spl(t, nu=2), dtype=np.float64),
    )


def _closed_library():
    """Compile ``density_closed.c`` once. None when no compiler is available."""
    global _CLOSED, _CLOSED_TRIED
    if _CLOSED_TRIED:
        return _CLOSED
    _CLOSED_TRIED = True
    src = Path(__file__).resolve().parent / "density_closed.c"
    compiler = shutil.which("clang") or shutil.which("gcc")
    if compiler is None or not src.is_file():
        return None
    digest = hashlib.sha256(src.read_bytes()).hexdigest()[:16]
    so = Path(tempfile.gettempdir()) / f"rnd_density_{digest}.so"
    if not so.is_file():
        cmd = [
            compiler, "-O3", "-march=native", "-ffast-math", "-fno-math-errno",
            "-shared", "-fPIC", "-o", str(so), str(src), "-lm",
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True)
        except (OSError, subprocess.CalledProcessError):
            cmd = [a for a in cmd if a != "-march=native"]
            try:
                subprocess.run(cmd, check=True, capture_output=True)
            except (OSError, subprocess.CalledProcessError):
                return None
    try:
        lib = ctypes.CDLL(str(so))
    except OSError:
        return None
    P = ctypes.POINTER(ctypes.c_double)
    lib.ppoly_jets.restype = None
    lib.ppoly_jets.argtypes = [P, P, ctypes.c_int, P, P, P]
    lib.bspline_jets.restype = ctypes.c_int
    lib.bspline_jets.argtypes = [
        P, ctypes.c_int, P, ctypes.c_int, ctypes.c_int,
        P, ctypes.c_int, P, P, P,
    ]
    lib.rnd_conv.restype = ctypes.c_int
    lib.rnd_conv.argtypes = [
        ctypes.POINTER(ctypes.c_double), ctypes.c_int,
        ctypes.POINTER(ctypes.c_double), ctypes.c_int,
        ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
        ctypes.c_int, ctypes.c_double,
        ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
    ]
    _CLOSED = lib
    return lib


def _as_f64(a):
    arr = np.asarray(a, dtype=np.float64)
    if not arr.flags.c_contiguous:
        arr = np.ascontiguousarray(arr)
    return arr


def _c_ptr(a):
    if a is None or a.size == 0:
        return None
    return a.ctypes.data_as(ctypes.POINTER(ctypes.c_double))


def _convolve_series(spl, Ks, h, F, x, weights, level_at=None, with_plain=False):
    """Density and level from one FFT of the order-eight box moments."""
    lib = _closed_library()
    if lib is None:
        raise RuntimeError("closed-form convolution is not compiled")
    x_arr = np.asarray(x, dtype=np.float64)
    shape = x_arr.shape
    xf = np.ascontiguousarray(x_arr.ravel())
    lshape = None
    lf = None
    if level_at is not None:
        la = np.asarray(level_at, dtype=np.float64)
        lshape = la.shape
        if la.shape == x_arr.shape and np.array_equal(la, x_arr):
            lf = xf
        else:
            lf = np.ascontiguousarray(la.ravel())
    t, p0, p1, sec = _spline_jets(spl, Ks)
    fac = np.ascontiguousarray([float(f) for f, _ in weights], dtype=np.float64)
    wt = np.ascontiguousarray([float(w) for _, w in weights], dtype=np.float64)
    g2 = np.empty(xf.size, dtype=np.float64)
    level = np.empty(0 if lf is None else lf.size, dtype=np.float64)
    plain = np.empty(xf.size, dtype=np.float64) if with_plain else None
    if xf.size == 0 and (lf is None or lf.size == 0):
        q = np.zeros(shape, dtype=float)
        G = None if lf is None else np.zeros(lshape, dtype=float)
        if with_plain:
            return q, G, np.zeros(shape, dtype=float)
        return q, G
    rc = lib.rnd_conv(
        _c_ptr(xf), ctypes.c_int(xf.size),
        _c_ptr(lf), ctypes.c_int(0 if lf is None else lf.size),
        _c_ptr(t), _c_ptr(p0), _c_ptr(p1), _c_ptr(sec), ctypes.c_int(t.size),
        ctypes.c_double(float(h)),
        _c_ptr(fac), _c_ptr(wt), ctypes.c_int(fac.size),
        _c_ptr(g2), _c_ptr(level), _c_ptr(plain),
    )
    if rc != 0:
        raise RuntimeError("closed-form convolution could not allocate")
    q = (-float(F) * g2).reshape(shape)
    G = None if lf is None else level.reshape(lshape)
    if not with_plain:
        return q, G
    return q, G, (-float(F) * plain).reshape(shape)


def _convolve(spl, Ks, h, F, x, weights, level_at=None, with_plain=False):
    """Density and, if requested, the smoothed complementary cdf.

    One FFT of the shared box moments, with thricing as one multiplier.
    The sampled-spline FFT is the fallback when that file cannot be
    compiled. ``with_plain`` also returns the untwiced density for the
    penalty rule.
    """
    if _closed_library() is not None:
        return _convolve_series(spl, Ks, h, F, x, weights, level_at=level_at, with_plain=with_plain)
    return _convolve_fft(spl, Ks, h, F, x, weights, level_at=level_at, with_plain=with_plain)


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
        q, G, q_plain = _convolve(spl, Ks, h, F, s, weights, level_at=K, with_plain=True)
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
