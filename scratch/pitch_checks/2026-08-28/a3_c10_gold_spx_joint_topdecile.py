"""C10 round 1 -- GLD and SPY both in the top decile of their trailing-252
21-day return distribution. 88 days of 5,205 against 75 expected under
independence, so the PRIOR is that the join is decoration.

Attribution is the whole point: the joint cell has to beat BOTH single-condition
cells, or the interaction does no work (registry 2026-08-10, "adding confirming
legs to a momentum state does not create a state").

Expressions run: long GLD, long SPY, long the 50/50 basket, and both pair
directions.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from pitch_lab import *  # noqa: E402,F403

ASOF = pd.Timestamp("2026-08-27")
raw = load_prices(["GLD", "SPY", "GDX", "TLT"])
gld = raw["GLD"]["Close"].dropna()
spy = raw["SPY"]["Close"].dropna()

px = pd.DataFrame({"GLD": gld, "SPY": spy.reindex(gld.index),
                   "GDX": raw["GDX"]["Close"].reindex(gld.index),
                   "TLT": raw["TLT"]["Close"].reindex(gld.index)}).dropna(subset=["GLD", "SPY"])

g21 = pct_rank(px["GLD"], 21)
s21 = pct_rank(px["SPY"], 21)
g_hi = px["GLD"] / px["GLD"].rolling(252).max() - 1.0

G = (g21 >= 90).fillna(False)
S = (s21 >= 90).fillna(False)
J = (G & S).fillna(False)

print("sample %s .. %s" % (px.index[0].date(), px.index[-1].date()))
print("LIVE  GLD r21 %.1f (%+.2f%%)  SPY r21 %.1f (%+.2f%%)  GLD dist52wH %+.2f%%"
      % (g21.loc[ASOF], 100 * (px["GLD"].loc[ASOF] / px["GLD"].shift(21).loc[ASOF] - 1),
         s21.loc[ASOF], 100 * (px["SPY"].loc[ASOF] / px["SPY"].shift(21).loc[ASOF] - 1),
         100 * g_hi.loc[ASOF]))
print("joint %d | GLD alone %d | SPY alone %d | independence %.0f"
      % (int(J.sum()), int(G.sum()), int(S.sum()),
         G.sum() * S.sum() / max(1, (g21.notna() & s21.notna()).sum())))

# ---- ATTRIBUTION: does the join beat either parent? --------------------------
print("\n[attribution] episode-level (min_gap=10) forward return by cell and leg")
for h in (5, 10):
    rows = []
    for lbl, m in [("JOINT (both top decile)", J),
                   ("GLD-only cell (SPY not)", (G & ~S).fillna(False)),
                   ("SPY-only cell (GLD not)", (S & ~G).fillna(False)),
                   ("GLD top decile, uncond.", G),
                   ("SPY top decile, uncond.", S)]:
        d = declusters(px.index[m.values], 10, px.index)
        for legname, legs in [("GLD", [("GLD", 1.0)]), ("SPY", [("SPY", 1.0)]),
                              ("50/50", [("GLD", 0.5), ("SPY", 0.5)]),
                              ("GLD-SPY", [("GLD", 1.0), ("SPY", -1.0)])]:
            r = vehicle_ret(px, legs, h, 1).reindex(d).dropna()
            rows.append({**summarize(r.values, f"{lbl} | {legname}")})
    show(rows, f"h={h}")
    base = {}
    for legname, legs in [("GLD", [("GLD", 1.0)]), ("SPY", [("SPY", 1.0)]),
                          ("50/50", [("GLD", 0.5), ("SPY", 0.5)]),
                          ("GLD-SPY", [("GLD", 1.0), ("SPY", -1.0)])]:
        base[legname] = 100 * vehicle_ret(px, legs, h, 1).dropna().mean()
    print("  ALL-DAY baselines h=%d: %s" % (h, {k: round(v, 3) for k, v in base.items()}))

# ---- the GLD-still-in-drawdown sub-state (today's actual shape) --------------
sub = (J & (g_hi <= -0.10)).fillna(False)
print("\n[today's exact shape] joint AND GLD still >=10%% below its own 52wH: %d days"
      % int(sub.sum()))
for h in (5, 10):
    d = declusters(px.index[sub.values], 10, px.index)
    show([summarize(vehicle_ret(px, [("GLD", 1.0)], h, 1).reindex(d).dropna().values,
                    f"h={h} GLD"),
          summarize(vehicle_ret(px, [("SPY", 1.0)], h, 1).reindex(d).dropna().values,
                    f"h={h} SPY")], f"sub-state episodes (N={len(d)})")

# ---- round 1 batteries -------------------------------------------------------
variants = {
    "both >= 80th pctile": ((g21 >= 80) & (s21 >= 80)).fillna(False),
    "both >= 95th pctile": ((g21 >= 95) & (s21 >= 95)).fillna(False),
    "GATE-OFF GLD alone >=90": G,
    "GATE-OFF SPY alone >=90": S,
    "63d ranks instead of 21d": ((pct_rank(px["GLD"], 63) >= 90)
                                 & (pct_rank(px["SPY"], 63) >= 90)).fillna(False),
    "10d ranks instead of 21d": ((pct_rank(px["GLD"], 10) >= 90)
                                 & (pct_rank(px["SPY"], 10) >= 90)).fillna(False),
}
for h in (5, 10):
    for legname, legs, cost in [("long GLD", [("GLD", 1.0)], 3.0),
                                ("long 50/50 GLD+SPY", [("GLD", 0.5), ("SPY", 0.5)], 3.0),
                                ("pair long GLD short SPY", [("GLD", 1.0), ("SPY", -1.0)], 3.0)]:
        battery(px, J, legs, h, f"C10 joint top-decile -> {legname}, h={h}",
                cost_bps=cost, variants=variants if legname == "long GLD" else None,
                min_gap=10)
