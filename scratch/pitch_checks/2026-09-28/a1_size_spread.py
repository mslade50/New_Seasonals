import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa

px = close_panel(["QQQ", "IWM", "SPY"]).dropna()
idx = px.index
q21 = px["QQQ"] / px["QQQ"].shift(21) - 1
i21 = px["IWM"] / px["IWM"].shift(21) - 1
spread = q21 - i21
sp_rank = spread.rolling(252).rank(pct=True) * 100
sp_max = spread.rolling(252).max()
iwm_r63 = pct_rank(px["IWM"], 63)
qqq_r21 = pct_rank(px["QQQ"], 21)
qqq_hi = px["QQQ"] / px["QQQ"].rolling(252).max() - 1

d = idx[-1]
print("last bar", d.date())
print(f"today spread21 {100*spread[d]:.2f}pp  rank252 {sp_rank[d]:.1f}  at252max {spread[d] >= sp_max[d] - 1e-12}")
print(f"IWM r63 {iwm_r63[d]:.1f}  QQQ r21 {qqq_r21[d]:.1f}  QQQ off hi {100*qqq_hi[d]:.2f}%")
print("last 8 spread/rank:")
print(pd.DataFrame({"spread_pp": 100 * spread, "rank": sp_rank}).tail(8).round(2))

# 63d beta of IWM on QQQ (daily returns) for beta-matched form
r = px.pct_change()
beta = r["IWM"].rolling(63).cov(r["QQQ"]) / r["QQQ"].rolling(63).var()
print(f"63d beta IWM on QQQ today {beta[d]:.2f}; 252d {(r['IWM'].rolling(252).cov(r['QQQ'])/r['QQQ'].rolling(252).var())[d]:.2f}")

legs = [("IWM", 1.0), ("QQQ", -1.0)]
trig = sp_rank >= 99.5  # at / next to trailing-252 max
for h in (5, 10):
    battery(px, trig, legs, h, f"C1 long IWM / short QQQ, spread21 rank>=99.5 (h={h})",
            cost_bps=3,
            variants={"rank>=97": sp_rank >= 97, "rank>=99": sp_rank >= 99,
                      "at 252 max": spread >= sp_max - 1e-12,
                      "rank>=99.5 & IWM r63<=5": (sp_rank >= 99.5) & (iwm_r63 <= 5),
                      "plain QQQ strong (r21>=95)": qqq_r21 >= 95,
                      "QQQ r21>=95 & NOT spread>=97": (qqq_r21 >= 95) & (sp_rank < 97),
                      "spread21>=10pp abs": spread >= 0.10},
            event_kinds=("nfp",))
