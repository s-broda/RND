"""Live CBOE index option slice: OTM puts/calls via parity, wing filters."""

from __future__ import annotations

import json
import math
import re
import urllib.request
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np

from . import black_scholes as bs

CBOE_URL = "https://cdn.cboe.com/api/global/delayed_quotes/options/_SPX.json"
OCC = re.compile(r"^([A-Z]+)(\d{2})(\d{2})(\d{2})([CP])(\d{8})$")


def cboe_url(symbol: str = "SPX") -> str:
    return f"https://cdn.cboe.com/api/global/delayed_quotes/options/_{symbol}.json"


@dataclass
class SpxSlice:
    S0: float
    r: float
    q: float
    T: float
    F: float
    expiry: date
    asof: str
    K: np.ndarray
    P: np.ndarray
    C: np.ndarray
    source: np.ndarray  # "P" or "C"
    root: str = "SPX"

    @property
    def disc(self) -> float:
        return float(np.exp(-self.r * self.T))


def fetch_cboe(path: Path | None = None, symbol: str = "SPX") -> dict:
    if path is not None and path.exists():
        return json.loads(path.read_text())
    req = urllib.request.Request(cboe_url(symbol), headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        raw = json.loads(resp.read().decode())
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(raw))
    return raw


def asof_from_raw(raw: dict, default: date | None = None) -> date:
    ts = str(raw.get("timestamp", ""))
    try:
        return date.fromisoformat(ts[:10])
    except ValueError:
        return default if default is not None else date.today()


def _parse_rows(raw: dict):
    rows = []
    for o in raw["data"]["options"]:
        m = OCC.match(o["option"])
        if not m:
            continue
        root, yy, mm, dd, cp, kdig = m.groups()
        exp = date(2000 + int(yy), int(mm), int(dd))
        bid = float(o["bid"] or 0.0)
        ask = float(o["ask"] or 0.0)
        oi = float(o["open_interest"] or 0.0)
        if bid <= 0.0 or ask < bid:
            continue
        rows.append(
            dict(
                root=root,
                exp=exp,
                cp=cp,
                K=int(kdig) / 1000.0,
                bid=bid,
                ask=ask,
                mid=0.5 * (bid + ask),
                oi=oi,
                spread=ask - bid,
            )
        )
    return rows, float(raw["data"]["current_price"]), str(raw.get("timestamp", ""))


def _keep_quote(row, min_mid=0.25, max_rel_spread=0.50, max_abs_spread=15.0):
    """Drop tick-noise and crossed/gaping wings; keep crash puts with real bids."""
    if row["mid"] < min_mid:
        return False
    if row["spread"] > max(max_abs_spread, max_rel_spread * row["mid"]):
        return False
    return True


def build_otm_slice(
    raw: dict,
    expiry: date,
    r: float = 0.04,
    root: str = "SPX",
    asof: date | None = None,
) -> SpxSlice:
    """OTM puts (K≤F) and OTM calls (K≥F) converted to puts by parity."""
    rows, S0, ts = _parse_rows(raw)
    if asof is None:
        asof = date.today()
    T = max((expiry - asof).days, 1) / 365.25
    cand = [x for x in rows if x["root"] == root and x["exp"] == expiry and _keep_quote(x)]
    puts = {x["K"]: x for x in cand if x["cp"] == "P"}
    calls = {x["K"]: x for x in cand if x["cp"] == "C"}
    both = sorted(set(puts) & set(calls))
    if not both:
        raise RuntimeError(f"no put-call pairs for {expiry}")
    # Forward from near-ATM parity: F = K + e^{rT}(C-P)
    atm = min(both, key=lambda K: abs(K - S0))
    # Keep the SPX 50-point band; scale to ~0.65% of spot on other roots.
    band_width = 50.0 if root in ("SPX", "SPXW") else max(50.0, 0.0065 * float(S0))
    band = [K for K in both if abs(K - atm) <= band_width]
    Fs = []
    for K in band:
        Fs.append(K + math.exp(r * T) * (calls[K]["mid"] - puts[K]["mid"]))
    F = float(np.median(Fs))
    q = r - math.log(F / S0) / T
    disc = math.exp(-r * T)
    dfq = math.exp(-q * T)

    Ks, Ps, Cs, src = [], [], [], []
    for K in sorted(set(puts) | set(calls)):
        if K <= F and K in puts:
            P = puts[K]["mid"]
            C = P + dfq * S0 - K * disc
            side = "P"
        elif K >= F and K in calls:
            C = calls[K]["mid"]
            P = C - dfq * S0 + K * disc
            side = "C"
        else:
            continue
        if P <= 0.0 or C <= 0.0:
            continue
        Ks.append(K)
        Ps.append(P)
        Cs.append(C)
        src.append(side)

    Ks, Ps, Cs = np.asarray(Ks), np.asarray(Ps), np.asarray(Cs)
    # Drop leftover deep-OTM puts that are still ticks after parity (left wing only).
    keep = np.ones(len(Ks), dtype=bool)
    left = Ks < 0.70 * F
    keep[left] &= Ps[left] >= 0.50
    Ks, Ps, Cs, src = Ks[keep], Ps[keep], Cs[keep], np.asarray(src)[keep]
    return SpxSlice(
        S0=S0, r=r, q=q, T=T, F=F, expiry=expiry, asof=ts, K=Ks, P=Ps, C=Cs, source=src, root=root
    )


def save_slice(sl: SpxSlice, path: Path) -> None:
    """Write the working OTM slice as CSV with a metadata header."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        f.write(
            f"# root={sl.root} expiry={sl.expiry.isoformat()} asof={sl.asof} "
            f"S0={sl.S0:.8g} r={sl.r:.8g} q={sl.q:.12g} T={sl.T:.12g} F={sl.F:.12g}\n"
        )
        f.write("K,P,C,source\n")
        for K, P, C, src in zip(sl.K, sl.P, sl.C, sl.source):
            f.write(f"{K:.8g},{P:.8g},{C:.8g},{src}\n")


def load_slice(path: Path) -> SpxSlice:
    """Load a slice written by ``save_slice``."""
    path = Path(path)
    lines = path.read_text().splitlines()
    meta = {}
    start = 0
    if lines and lines[0].startswith("#"):
        for part in lines[0][1:].split():
            if "=" in part:
                k, v = part.split("=", 1)
                meta[k] = v
        start = 1
    Ks, Ps, Cs, src = [], [], [], []
    for line in lines[start + 1 :]:
        if not line.strip():
            continue
        k, p, c, s = line.split(",")
        Ks.append(float(k))
        Ps.append(float(p))
        Cs.append(float(c))
        src.append(s.strip())
    exp = date.fromisoformat(meta["expiry"])
    return SpxSlice(
        S0=float(meta["S0"]),
        r=float(meta["r"]),
        q=float(meta["q"]),
        T=float(meta["T"]),
        F=float(meta["F"]),
        expiry=exp,
        asof=meta.get("asof", ""),
        K=np.asarray(Ks),
        P=np.asarray(Ps),
        C=np.asarray(Cs),
        source=np.asarray(src),
        root=meta.get("root", "SPX"),
    )


def atm_iv(sl: SpxSlice) -> float:
    iv = bs.implied_vol(sl.C, sl.S0, sl.K, sl.r, sl.T, sl.q)
    atm = np.nanmedian(iv[np.abs(sl.K - sl.F) <= 0.03 * sl.F])
    if not np.isfinite(atm):
        atm = float(np.nanmedian(iv))
    return float(atm) if np.isfinite(atm) else 0.16
