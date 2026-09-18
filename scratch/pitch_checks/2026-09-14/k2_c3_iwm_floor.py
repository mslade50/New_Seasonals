"""C3: IWM r21 AND r63 at a floor while SPY within 2% of its 252 high.
Long IWM / short SPY (equal-dollar + beta-matched) and IWM outright, h=5..10."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa
sys.path.insert(0, str(Path(__file__).resolve().parent))
from k2_common import cell, fmt, conc, near_high  # noqa

px = close_panel(["IWM", "SPY"])
r21, r63 = pct_rank(px["IWM"], 21), pct_rank(px["IWM"], 63)
nh = near_high(px["SPY"])
rel = px["IWM"] / px["SPY"]
rr21, rr63 = pct_rank(rel, 21), pct_rank(rel, 63)
# trailing 252d beta of IWM on SPY (lagged 1)
dr = px.pct_change()
beta = (dr["IWM"].rolling(252).cov(dr["SPY"]) / dr["SPY"].rolling(252).var()).shift(1)
print("live: IWM r21", round(r21.iloc[-1], 1), "r63", round(r63.iloc[-1], 1), "SPY nh",
      bool(nh.iloc[-1]), "rel r21", round(rr21.iloc[-1], 1), "rel r63", round(rr63.iloc[-1], 1),
      "beta", round(beta.iloc[-1], 2))
bmean = float(beta.dropna().iloc[-756:].mean())
print("3y mean beta", round(bmean, 2))

M = (r21 <= 10) & (r63 <= 10) & nh
LP = [("IWM", 1.0), ("SPY", -1.0)]
LB = [("IWM", 1.0 / bmean), ("SPY", -1.0)]
LX = [("IWM", 1.0)]
battery(px, M, LP, 5, "C3 long IWM / short SPY, IWM r21<=10 & r63<=10 & SPY nh", 3,
        variants={"r21<=7 & r63<=7 & nh": (r21 <= 7) & (r63 <= 7) & nh,
                  "r21<=15 & r63<=15 & nh": (r21 <= 15) & (r63 <= 15) & nh,
                  "r21<=10 & r63<=10 NO nh": (r21 <= 10) & (r63 <= 10),
                  "rel r21<=10 & rel r63<=10 & nh": (rr21 <= 10) & (rr63 <= 10) & nh,
                  "nh within 3%": (r21 <= 10) & (r63 <= 10) & near_high(px["SPY"], 0.03)},
        event_kinds=("fomc_decision",))

print("\n=== gate attribution + vehicles (episodes) ===")
for h in (5, 10):
    for L, nm in ((LP, "pair eq$"), (LB, "pair beta"), (LX, "IWM outright")):
        print(f"-- h={h} {nm}")
        print(fmt(cell(px, M, L, h), "joint r21<=10 & r63<=10 & nh"))
        print(fmt(cell(px, (r21 <= 10) & (r63 <= 10), L, h), "no nh gate"))
        print(fmt(cell(px, (r21 <= 10) & (r63 <= 10) & ~nh, L, h), "floors & SPY NOT near high"))
        print(fmt(cell(px, (r21 <= 10) & nh, L, h), "r21<=10 & nh (drop r63)"))
        print(fmt(cell(px, (r63 <= 10) & nh, L, h), "r63<=10 & nh (drop r21)"))
        print(fmt(cell(px, (r21 <= 10) & (r63 > 10) & nh, L, h), "r21 floor, r63>10, nh"))
        print(fmt(cell(px, nh, L, h), "SPY nh alone (control)"))
        print(fmt(cell(px, (rr21 <= 10) & (rr63 <= 10) & nh, L, h), "RELATIVE floors & nh"))
c = cell(px, M, LP, 5)
print("\nconcentration pair h5:", conc(c))
print("episodes:", ", ".join(str(d.date()) for d in c["epi"]))
print("ep rets %:", [round(100 * x, 2) for x in c["ep"]])
mid = pd.Series(px.index.year % 4 == 2, index=px.index)
for h in (5, 10):
    print(f"-- era/regime h={h} pair")
    print(fmt(cell(px, M & (px.index < "2018-01-01"), LP, h), "pre-2018"))
    print(fmt(cell(px, M & (px.index >= "2018-01-01"), LP, h), "2018+"))
    print(fmt(cell(px, M & mid, LP, h), "midterm"))
    print(fmt(cell(px, M & ~mid, LP, h), "non-midterm"))
