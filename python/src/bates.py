"""Bates (1996) stochastic-volatility jump-diffusion: Heston plus Merton jumps.

The parameters are the SVJ estimates of Bakshi, Cao and Chen (1997,
Table III, all options): kappa = 2.03, theta = 0.04, sigma = 0.38,
rho = -0.52, lambda = 0.59, mu_J = -0.05, sigma_J = 0.07, v0 = 0.04.
Spot, rate and maturity are those of the Heston design.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import trapezoid

from .heston import HestonParams, heston_cf


@dataclass
class BatesParams(HestonParams):
    kappa: float = 2.03
    theta: float = 0.04
    sigma: float = 0.38
    rho: float = -0.52
    v0: float = 0.04
    lam: float = 0.59
    mu_j: float = -0.05
    sig_j: float = 0.07


BCC97_SVJ = BatesParams()


def bates_cf(u, p: BatesParams):
    """phi(u) = E[exp(i u log S_T)]. Jumps are compensated so that E[S_T] = F."""
    u = np.asarray(u, dtype=np.complex128)
    base = heston_cf(u, p)
    kbar = np.exp(p.mu_j + 0.5 * p.sig_j**2) - 1.0
    jump = p.lam * p.T * (np.exp(1j * u * p.mu_j - 0.5 * p.sig_j**2 * u**2) - 1.0)
    comp = -1j * u * p.lam * p.T * kbar
    return base * np.exp(jump + comp)


def _v_grid(n: int = 4096, v_max: float = 250.0) -> np.ndarray:
    return np.linspace(1e-6, v_max, n)


def bates_log_density(x, p: BatesParams, n_v: int = 4096, v_max: float = 250.0):
    x = np.atleast_1d(np.asarray(x, dtype=float))
    v = _v_grid(n_v, v_max)
    phi = bates_cf(v, p)
    integ = np.real(np.exp(-1j * np.outer(x, v)) * phi)
    return np.maximum(trapezoid(integ, v, axis=1) / np.pi, 0.0)


def bates_spot_density(K, p: BatesParams, n_v: int = 4096, v_max: float = 250.0):
    K = np.asarray(K, dtype=float)
    return bates_log_density(np.log(K), p, n_v=n_v, v_max=v_max) / K


def bates_calls(K, p: BatesParams, alpha: float = 1.5, n_v: int = 4096, v_max: float = 250.0):
    """European calls via the Carr-Madan damped integrand."""
    K = np.atleast_1d(np.asarray(K, dtype=float))
    k = np.log(K)
    v = _v_grid(n_v, v_max)
    denom = alpha**2 + alpha - v**2 + 1j * v * (2.0 * alpha + 1.0)
    psi = np.exp(-p.r * p.T) * bates_cf(v - 1j * (alpha + 1.0), p) / denom
    integ = np.real(np.exp(-1j * np.outer(k, v)) * psi)
    C = np.exp(-alpha * k) / np.pi * trapezoid(integ, v, axis=1)
    return np.maximum(C, 0.0)
