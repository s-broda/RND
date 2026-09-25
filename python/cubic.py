"""Cubic-spline empirical measure, Gaussian inversion in closed form.

P = 1 - C / (S0 e^{-qT}) is interpolated by a natural cubic spline through
the knots, including a point at 0 and the linear right-wing intercept.
The risk-neutral density is -F times the derivative of that spline's
derivative convolved with a Gaussian. Twicing uses h and h*sqrt(2).
The listed bandwidth is 0.55 times the density-scale rule.
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.special import erf

from rnd import _density_h, _mesh_h, _n_right_from_gap, _right_wing, estimate_rnd

FACTOR = 0.55


def _Phi(u):
    return 0.5 * (1.0 + erf(np.asarray(u, dtype=float) / np.sqrt(2.0)))


def _phi(u):
    u = np.asarray(u, dtype=float)
    return np.exp(-0.5 * u * u) / np.sqrt(2.0 * np.pi)


def knots(K, C, stock, F, disc, tails=True):
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


def _smooth(x, Ks, spl, h, F):
    x = np.atleast_1d(np.asarray(x, dtype=float))
    h = max(float(h), 1e-6)
    gprime = np.zeros_like(x)
    G = np.zeros_like(x)
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
    return -float(F) * gprime, G


def bandwidth(K, C, S0, r, T, q, F, rule="cubic"):
    """``mesh`` is 1.2 times the median gap. ``cubic`` is 0.55 times density-scale."""
    if rule == "mesh":
        return _mesh_h(K)
    disc = float(np.exp(-r * T))
    n_right = _n_right_from_gap(K, C, F, disc)
    n_ext = len(K) + int(n_right)
    h = _density_h(K, C, S0, r, T, q, F, n_ext)
    if rule == "density":
        return h
    return FACTOR * h


def estimate_cubic(
    K, C, S0, r, T, q, F, x_dens, x_price=None, h=None, tails=True, twice=True
):
    stock = float(S0) * np.exp(-float(q) * float(T))
    disc = float(np.exp(-float(r) * float(T)))
    if h is None or h == "cubic":
        h_use = bandwidth(K, C, S0, r, T, q, F, "cubic")
    elif h == "mesh":
        h_use = bandwidth(K, C, S0, r, T, q, F, "mesh")
    elif h == "density":
        h_use = bandwidth(K, C, S0, r, T, q, F, "density")
    else:
        h_use = float(h)
    Ks, Ps = knots(K, C, stock, F, disc, tails=tails)
    spl = CubicSpline(Ks, Ps, bc_type="natural")
    q1, _ = _smooth(x_dens, Ks, spl, h_use, F)
    if x_price is None:
        G1 = None
    else:
        _, G1 = _smooth(x_price, Ks, spl, h_use, F)
    if twice:
        q2, _ = _smooth(x_dens, Ks, spl, h_use * np.sqrt(2.0), F)
        qhat = 2.0 * q1 - q2
        if G1 is not None:
            _, G2 = _smooth(x_price, Ks, spl, h_use * np.sqrt(2.0), F)
            G = 2.0 * G1 - G2
        else:
            G = None
    else:
        qhat, G = q1, G1
    C_hat = None
    if G is not None:
        G = np.clip(G, 0.0, 1.0)
        C_hat = stock * np.clip(1.0 - G, 0.0, 1.0)
    return {"q": qhat, "C": C_hat, "h": float(h_use)}
