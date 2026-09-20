"""Variance-gamma CF, Carr–Madan prices, and the Bessel density of S_T.

Process parameters (sigma, nu, theta) are the Carr–Madan (1999) numerical
example. Spot, rate, and maturity match the Heston illustration.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import trapezoid
from scipy.special import gamma, kve


@dataclass
class VGParams:
    S0: float = 100.0
    r: float = 0.05
    q: float = 0.0
    T: float = 0.25
    sigma: float = 0.12
    nu: float = 0.2
    theta: float = -0.14

    @property
    def disc(self) -> float:
        return float(np.exp(-self.r * self.T))

    @property
    def forward(self) -> float:
        return float(self.S0 * np.exp((self.r - self.q) * self.T))

    @property
    def omega(self) -> float:
        """Martingale correction: E[exp(X_T)] = exp(-ω T)."""
        inner = 1.0 - self.theta * self.nu - 0.5 * self.sigma**2 * self.nu
        if inner <= 0.0:
            raise ValueError("VG martingale correction is not defined")
        return float(np.log(inner) / self.nu)


# Carr–Madan (1999, §5): σ=0.12, ν=0.2, θ=-0.14.
CM99 = VGParams()


def vg_cf(u, p: VGParams):
    """φ(u) = E[exp(i u log S_T)] under Q."""
    u = np.asarray(u, dtype=np.complex128)
    sig2 = p.sigma * p.sigma
    z = 1.0 - 1j * p.theta * p.nu * u + 0.5 * sig2 * p.nu * u * u
    loc = np.log(p.S0) + (p.r - p.q + p.omega) * p.T
    return np.exp(1j * u * loc - (p.T / p.nu) * np.log(z))


def _v_grid(n: int = 4096, v_max: float = 250.0) -> np.ndarray:
    return np.linspace(1e-6, v_max, n)


def vg_log_density_bessel(x, p: VGParams):
    """Density of X_T, the VG increment in log S_T = loc + X_T.

    Madan–Carr–Chang (1998, eq. for h(z) with the location stripped).
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    sig2 = p.sigma * p.sigma
    t_nu = p.T / p.nu
    nu_k = t_nu - 0.5
    c = np.sqrt(p.theta * p.theta + 2.0 * sig2 / p.nu)
    ax = np.abs(x)
    small = ax < 1e-12
    den = p.nu**t_nu * np.sqrt(2.0 * np.pi) * p.sigma * gamma(t_nu)
    out = np.empty_like(x, dtype=float)
    if np.any(~small):
        xs = ax[~small]
        arg = xs * c / sig2
        # kv(ν,z) = e^{-z} kve(ν,z); fold the exp(θx/σ²) into the same exponent.
        expo = p.theta * x[~small] / sig2 - arg
        out[~small] = (
            2.0 * np.exp(expo) / den * (xs / c) ** nu_k * kve(nu_k, arg)
        )
    if np.any(small):
        # kv(ν,z) ~ ½ Γ(ν) (z/2)^{-ν} as z→0, so (|x|/c)^{ν} K_ν(c|x|/σ²)
        # → ½ Γ(ν) (2σ²)^ν / c^{2ν}.
        lim = 0.5 * gamma(nu_k) * (2.0 * sig2) ** nu_k / (c ** (2.0 * nu_k))
        out[small] = 2.0 * np.exp(p.theta * x[small] / sig2) / den * lim
    return np.maximum(out, 0.0)


def vg_spot_density(K, p: VGParams):
    """f_Q(K) from the Bessel density of the VG increment, Jacobian 1/K."""
    K = np.asarray(K, dtype=float)
    loc = np.log(p.S0) + (p.r - p.q + p.omega) * p.T
    x = np.log(np.maximum(K, 1e-16)) - loc
    return vg_log_density_bessel(x, p) / np.maximum(K, 1e-16)


def vg_log_density_fourier(x, p: VGParams, n_v: int = 4096, v_max: float = 250.0):
    x = np.atleast_1d(np.asarray(x, dtype=float))
    v = _v_grid(n_v, v_max)
    loc = np.log(p.S0) + (p.r - p.q + p.omega) * p.T
    # φ of log S, so density of log S at loc+x uses φ of log S.
    phi = vg_cf(v, p)
    y = loc + x
    integ = np.real(np.exp(-1j * np.outer(y, v)) * phi)
    return np.maximum(trapezoid(integ, v, axis=1) / np.pi, 0.0)


def carr_madan_calls(K, p: VGParams, alpha: float = 1.5, n_v: int = 4096, v_max: float = 250.0):
    """European calls via the Carr–Madan (1999) damped CF integrand."""
    K = np.atleast_1d(np.asarray(K, dtype=float))
    k = np.log(K)
    v = _v_grid(n_v, v_max)
    denom = alpha**2 + alpha - v**2 + 1j * v * (2.0 * alpha + 1.0)
    psi = np.exp(-p.r * p.T) * vg_cf(v - 1j * (alpha + 1.0), p) / denom
    integ = np.real(np.exp(-1j * np.outer(k, v)) * psi)
    C = np.exp(-alpha * k) / np.pi * trapezoid(integ, v, axis=1)
    return np.maximum(C, 0.0)


def vg_calls(K, p: VGParams, n_grid: int = 8001, k_max: float | None = None):
    """Discounted (S_T-K)^+ against the Bessel density (trapezoid)."""
    K = np.atleast_1d(np.asarray(K, dtype=float))
    hi = 8.0 * p.forward if k_max is None else float(k_max)
    s = np.linspace(1.0, hi, n_grid)
    q = vg_spot_density(s, p)
    ds = np.diff(s)
    I0 = np.concatenate([[0.0], np.cumsum(0.5 * (q[1:] + q[:-1]) * ds)])
    I1 = np.concatenate([[0.0], np.cumsum(0.5 * (q[1:] * s[1:] + q[:-1] * s[:-1]) * ds)])
    i0 = np.interp(K, s, I0)
    i1 = np.interp(K, s, I1)
    C = p.disc * ((I1[-1] - i1) - K * (I0[-1] - i0))
    return np.maximum(C, 0.0)


def vg_puts(K, p: VGParams, **kwargs):
    C = vg_calls(K, p, **kwargs)
    K = np.atleast_1d(np.asarray(K, dtype=float))
    return np.maximum(C - p.S0 * np.exp(-p.q * p.T) + K * p.disc, 0.0)


def carr_madan_puts(K, p: VGParams, **kwargs):
    C = carr_madan_calls(K, p, **kwargs)
    K = np.atleast_1d(np.asarray(K, dtype=float))
    return np.maximum(C - p.S0 * np.exp(-p.q * p.T) + K * p.disc, 0.0)
