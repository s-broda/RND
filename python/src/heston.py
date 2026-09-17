"""Heston CF (Albrecher little-trap), Carr–Madan prices, and the true RND."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import trapezoid


@dataclass
class HestonParams:
    S0: float = 100.0
    r: float = 0.05
    q: float = 0.0
    T: float = 0.25
    kappa: float = 1.15
    theta: float = 0.04
    sigma: float = 0.39
    rho: float = -0.64
    v0: float = 0.04

    @property
    def disc(self) -> float:
        return float(np.exp(-self.r * self.T))

    @property
    def forward(self) -> float:
        return float(self.S0 * np.exp((self.r - self.q) * self.T))


BCC97 = HestonParams()


def heston_cf(u, p: HestonParams):
    """φ(u) = E[exp(i u log S_T)] under Q. Little-trap branch of Albrecher et al."""
    u = np.asarray(u, dtype=np.complex128)
    i = 1j
    sigma = p.sigma
    xi = p.kappa - p.rho * sigma * i * u
    d = np.sqrt(xi**2 + sigma**2 * (i * u + u**2))
    d = np.where(np.real(d) >= 0.0, d, -d)
    g = (xi - d) / (xi + d)
    exp_dt = np.exp(-d * p.T)
    g_exp = g * exp_dt
    log_term = np.log((1.0 - g_exp) / (1.0 - g))
    C = (p.kappa * p.theta / sigma**2) * ((xi - d) * p.T - 2.0 * log_term)
    D = ((xi - d) / sigma**2) * (1.0 - exp_dt) / (1.0 - g_exp)
    return np.exp(i * u * (np.log(p.S0) + (p.r - p.q) * p.T) + C + D * p.v0)


def _v_grid(n: int = 4096, v_max: float = 250.0) -> np.ndarray:
    return np.linspace(1e-6, v_max, n)


def heston_log_density(x, p: HestonParams, n_v: int = 4096, v_max: float = 250.0):
    x = np.atleast_1d(np.asarray(x, dtype=float))
    v = _v_grid(n_v, v_max)
    phi = heston_cf(v, p)
    integ = np.real(np.exp(-1j * np.outer(x, v)) * phi)
    return np.maximum(trapezoid(integ, v, axis=1) / np.pi, 0.0)


def heston_spot_density(K, p: HestonParams, n_v: int = 4096, v_max: float = 250.0):
    K = np.asarray(K, dtype=float)
    return heston_log_density(np.log(K), p, n_v=n_v, v_max=v_max) / K


def carr_madan_calls(K, p: HestonParams, alpha: float = 1.5, n_v: int = 4096, v_max: float = 250.0):
    """European calls via the Carr–Madan (1999) damped CF integrand."""
    K = np.atleast_1d(np.asarray(K, dtype=float))
    k = np.log(K)
    v = _v_grid(n_v, v_max)
    denom = alpha**2 + alpha - v**2 + 1j * v * (2.0 * alpha + 1.0)
    psi = np.exp(-p.r * p.T) * heston_cf(v - 1j * (alpha + 1.0), p) / denom
    integ = np.real(np.exp(-1j * np.outer(k, v)) * psi)
    C = np.exp(-alpha * k) / np.pi * trapezoid(integ, v, axis=1)
    return np.maximum(C, 0.0)


def carr_madan_puts(K, p: HestonParams, **kwargs):
    C = carr_madan_calls(K, p, **kwargs)
    K = np.atleast_1d(np.asarray(K, dtype=float))
    return np.maximum(C - p.S0 * np.exp(-p.q * p.T) + K * p.disc, 0.0)


def gil_pelaez_calls(K, p: HestonParams, n_v: int = 4096, v_max: float = 250.0):
    """P1/P2 inversion; used only as a check on Carr–Madan."""
    K = np.atleast_1d(np.asarray(K, dtype=float))
    k = np.log(K)
    v = _v_grid(n_v, v_max)
    phi = heston_cf(v, p)
    phi_star = heston_cf(v - 1j, p)
    phi_mi = p.forward
    expo = np.exp(-1j * np.outer(k, v))
    P2 = 0.5 + trapezoid(np.real(expo * phi / (1j * v)), v, axis=1) / np.pi
    P1 = 0.5 + trapezoid(np.real(expo * (phi_star / phi_mi) / (1j * v)), v, axis=1) / np.pi
    return p.S0 * np.exp(-p.q * p.T) * P1 - K * p.disc * P2
