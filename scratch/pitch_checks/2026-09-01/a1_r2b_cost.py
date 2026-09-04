"""A1 blocker (b): settle the cost bar EXPLICITLY.

The parked arm: "TURNS ON at ... (b) a two-leg round trip under 4.4 bps".
4.4 bps is exactly 22.2 bps / 5, i.e. the h=8 episode mean at the 5x bar.

The denominator matters and is easy to get wrong.  `vehicle_ret` returns
w_IEF * r_IEF + w_TLT * r_TLT with w_IEF = 1.0 and w_TLT = -1/beta.  A return
quoted that way is per $1 of IEF NOTIONAL, so the cost must be quoted in the
same denominator: 1.00 x (IEF round-trip cost) + |w_TLT| x (TLT round-trip
cost), each leg's cost as a fraction of that leg's OWN notional.

Four crossing conventions, cheapest to most punitive.  Entry and exit are both
MOC in the cell's own geometry (close-to-close returns), which is the reason
the auction convention is not a cheat -- an MOC order fills AT the official
closing price, which IS the number the backtest differences.  The spread
conventions are charged anyway as the pessimistic bound.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

import numpy as np
import pandas as pd

ASOF = pd.Timestamp("2026-08-31")
px = close_panel(["^TNX", "TLT", "IEF"]).dropna(how="any")
px = px[px.index <= ASOF]
d = px[["TLT", "IEF"]].pct_change().dropna()
BETA = float(np.polyfit(d["IEF"].values, d["TLT"].values, 1)[0])
W_TLT = 1.0 / BETA

P_TLT = float(px["TLT"].iloc[-1])
P_IEF = float(px["IEF"].iloc[-1])
TICK = 0.01                      # both names quote in pennies
COMM_FIXED = 0.005               # IBKR fixed tier, $/share, min $1/order
COMM_TIERED = 0.0035             # IBKR tiered, $/share, before exch fees
AUCTION_FEE = 0.0005             # closing-auction fee, $/share, conservative

print("=" * 100)
print("A1 COST -- long 1.00 IEF / short %.4f TLT, MOC in and MOC out" % W_TLT)
print("=" * 100)
print("  live closes 2026-08-31: TLT %.2f   IEF %.2f   (beta TLT~IEF %.4f)"
      % (P_TLT, P_IEF, BETA))
sp_tlt = TICK / P_TLT * 10000
sp_ief = TICK / P_IEF * 10000
print("  one-penny spread: TLT %.3f bps   IEF %.3f bps" % (sp_tlt, sp_ief))
for lbl, c in (("fixed $0.0050/sh", COMM_FIXED), ("tiered $0.0035/sh", COMM_TIERED),
               ("auction fee $0.0005/sh", AUCTION_FEE)):
    print("  %-24s -> TLT %.3f bps/side   IEF %.3f bps/side"
          % (lbl, c / P_TLT * 10000, c / P_IEF * 10000))


def rt(price, comm, spread_share):
    """Round-trip cost in bps of that leg's notional.
    spread_share: 0 = auction print (no crossing), 0.5 = half spread each
    side, 1.0 = full spread each side."""
    per_side = (comm / price * 10000) + spread_share * (TICK / price * 10000)
    return 2.0 * per_side


CONV = [
    ("A  MOC auction print, tiered comm + auction fee",
     COMM_TIERED + AUCTION_FEE, 0.0),
    ("B  MOC auction print, FIXED comm + auction fee",
     COMM_FIXED + AUCTION_FEE, 0.0),
    ("C  half-spread crossed each side, FIXED comm + auction fee",
     COMM_FIXED + AUCTION_FEE, 0.5),
    ("D  FULL spread crossed each side, FIXED comm + auction fee",
     COMM_FIXED + AUCTION_FEE, 1.0),
]
EDGE_H8 = 22.2     # bps, reproduced exactly today
EDGE_H8_PIT = 20.6  # bps, parked point-in-time hedge-ratio variant

print("\n  %-56s %8s %8s %8s %7s" % ("convention", "IEF bps", "TLTx%.3f" % W_TLT,
                                     "TOTAL", "x@22.2"))
rows = []
for lbl, comm, ss in CONV:
    c_ief = rt(P_IEF, comm, ss)
    c_tlt = rt(P_TLT, comm, ss)
    tot = 1.0 * c_ief + W_TLT * c_tlt
    rows.append((lbl, tot))
    print("  %-56s %8.3f %8.3f %8.3f %7.2f"
          % (lbl, c_ief, W_TLT * c_tlt, tot, EDGE_H8 / tot))

print("\n  the stated arm is 'two-leg round trip under 4.4 bps' (= 22.2 / 5):")
for lbl, tot in rows:
    print("    %-56s %.3f bps -> %s" % (lbl, tot, "CLEARS" if tot < 4.4 else "FAILS"))

print("\n  the book's own battery default (3 bps/leg x 2 legs = 6.0 bps, weight-blind):"
      "  %.2fx  -> %s" % (EDGE_H8 / 6.0, "CLEARS" if EDGE_H8 / 6.0 >= 5 else "FAILS"))
print("  under the PIT hedge ratio (20.6 bps): conv C %.2fx   conv D %.2fx"
      % (EDGE_H8_PIT / rows[2][1], EDGE_H8_PIT / rows[3][1]))

# --------------------------------------------------------------- financing leg
print("\n  FINANCING on the short TLT leg, 8 sessions held")
HOLD_TD = 8
for borrow in (0.0025, 0.0050, 0.0100):
    bps = W_TLT * borrow * (HOLD_TD / 252.0) * 10000
    print("    borrow %.2f%%/yr -> %.3f bps of the IEF-notional denominator" % (100 * borrow, bps))
print("    (short rebate on the proceeds is not credited in a retail margin"
      " account below the threshold; treated as a pure cost)")

print("\n  ADV sanity -- can an MOC of this size print without impact?")
for t in ("TLT", "IEF"):
    v = load_prices([t])[t]
    v = v[v.index <= ASOF]
    dollar_adv = float((v["Close"] * v["Volume"]).iloc[-21:].mean())
    print("    %s 21d dollar ADV $%.0fm ; closing auction is typically 5-10%% of that"
          % (t, dollar_adv / 1e6))

print("\n  VERDICT ON BLOCKER (b): the cheapest honest convention that matches the")
print("  cell's own close-to-close geometry (A/B: MOC auction, no crossing) lands")
print("  at %.2f-%.2f bps and CLEARS 4.4.  Half-spread (C) at %.2f bps CLEARS."
      % (rows[0][1], rows[1][1], rows[2][1]))
print("  Only the full-spread bound (D, %.2f bps) fails, and a market-order full")
print("  cross is not what an MOC does (D = %.2f bps)." % rows[3][1])
