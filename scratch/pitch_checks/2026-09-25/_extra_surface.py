"""Extra surface for the 2026-09-25 map: series outside the 218-name tape.

Same conventions as the tape: pct_rank of trailing returns vs own 252d history,
z10 = 10d return / (21d daily vol * sqrt(10)), distances from 252d high/low.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from pitch_lab import close_panel, load_prices  # noqa: E402

T = ["HG=F", "GC=F", "SI=F", "PL=F", "PA=F", "CL=F", "NG=F", "ZC=F", "ZS=F", "ZW=F",
     "KC=F", "CC=F", "SB=F", "CT=F", "LE=F", "HE=F", "TIP", "IEF", "TLT", "^FVX", "^IRX",
     "^TNX", "JPY=X", "EURUSD=X", "GBPUSD=X", "AUDUSD=X", "USDMXN=X", "USDCNY=X",
     "AUDJPY=X", "CAD=X", "USDBRL=X", "^VVIX", "^VIX", "^MOVE", "^N225", "^HSI",
     "^BVSP", "^GDAXI", "^FTSE", "^KS11", "EWZ", "EWJ", "KWEB", "MU", "IYT", "XHB",
     "ITB", "COPX", "^RUT", "IWM", "TMF", "ES=F", "NQ=F", "UNG"]


def pr(s: pd.Series, h: int) -> float:
    r = s.pct_change(h)
    w = r.iloc[-252:]
    return float((w <= w.iloc[-1]).mean() * 100)


def main() -> None:
    px = load_prices(T)
    cp = close_panel(T + ["GLD"])
    rows = []
    for t in T:
        if t not in cp:
            continue
        s = cp[t].dropna()
        if len(s) < 260:
            continue
        d = s.pct_change()
        vol21 = d.iloc[-21:].std()
        z10 = (s.iloc[-1] / s.iloc[-11] - 1) / (vol21 * np.sqrt(10)) if vol21 > 0 else np.nan
        hi, lo = s.iloc[-252:].max(), s.iloc[-252:].min()
        rows.append(dict(t=t, last=str(s.index[-1].date()), close=s.iloc[-1],
                         r1=100 * d.iloc[-1], r5=100 * (s.iloc[-1] / s.iloc[-6] - 1),
                         r21=100 * (s.iloc[-1] / s.iloc[-22] - 1),
                         r63=100 * (s.iloc[-1] / s.iloc[-64] - 1),
                         rk5=pr(s, 5), rk21=pr(s, 21), rk63=pr(s, 63), z10=z10,
                         offhi=100 * (s.iloc[-1] / hi - 1), offlo=100 * (s.iloc[-1] / lo - 1),
                         lvl_pct=float((s.iloc[-252:] <= s.iloc[-1]).mean() * 100)))
    df = pd.DataFrame(rows).set_index("t")
    pd.set_option("display.width", 250)
    print(df.round(2).to_string())
    # ratios
    for a, b in [("HG=F", "GC=F"), ("TIP", "IEF"), ("SI=F", "GC=F"), ("CL=F", "GC=F")]:
        if a in cp and b in cp:
            r = (cp[a] / cp[b]).dropna()
            print(f"{a}/{b}: last {r.iloc[-1]:.5f} rk21 {pr(r, 21):.1f} rk63 {pr(r, 63):.1f} "
                  f"lvl_pct252 {(r.iloc[-252:] <= r.iloc[-1]).mean() * 100:.1f} "
                  f"21d chg {100 * (r.iloc[-1] / r.iloc[-22] - 1):+.2f}%")
    if "^TNX" in cp and "^IRX" in cp:
        c = (cp["^TNX"] - cp["^IRX"]).dropna()
        print(f"10y-3m: {c.iloc[-1]:.3f} 21d chg {c.iloc[-1] - c.iloc[-22]:+.3f} lvl_pct252 "
              f"{(c.iloc[-252:] <= c.iloc[-1]).mean() * 100:.1f}")
    if "^TNX" in cp and "^FVX" in cp:
        c = (cp["^TNX"] - cp["^FVX"]).dropna()
        print(f"10y-5y: {c.iloc[-1]:.3f} 21d chg {c.iloc[-1] - c.iloc[-22]:+.3f} lvl_pct252 "
              f"{(c.iloc[-252:] <= c.iloc[-1]).mean() * 100:.1f}")
    v = cp["^VIX"].dropna()
    m = cp["^MOVE"].dropna()
    print(f"VIX lvl pct252 {(v.iloc[-252:] <= v.iloc[-1]).mean() * 100:.1f}  "
          f"MOVE lvl pct252 {(m.iloc[-252:] <= m.iloc[-1]).mean() * 100:.1f}")
    md = m.pct_change().dropna()
    print(f"MOVE 1d +{100 * md.iloc[-1]:.2f}% = pctile {(md <= md.iloc[-1]).mean() * 100:.2f} of all daily moves")
    g = cp["GLD"].dropna() if "GLD" in cp else None
    u = px.get("UNG")
    if u is not None and len(u):
        vr = u["Volume"] / u["Volume"].rolling(63).mean().shift(1)
        print("UNG last 8 sessions ret / vol ratio:")
        for d0 in u.index[-8:]:
            print(f"  {pd.Timestamp(d0).date()} {100 * u['Close'].pct_change()[d0]:+.2f}% {vr[d0]:.2f}x")


if __name__ == "__main__":
    main()
