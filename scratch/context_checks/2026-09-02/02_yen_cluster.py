"""The yen printed a broad 2-ATR rally against the crosses tonight.

Engine fired P6:two_atr_day down on EURJPY, GBPJPY, CHFJPY, NZDJPY and
P5:rank5_extreme bottom on GBPJPY/CHFJPY/NZDJPY, all on 2026-09-02. Base cells
are per-cross and n=21-34. The question the engine cannot ask: how often do
FOUR OR MORE crosses do this on the same session, and what follows?

Second half crosses that state with tomorrow's calendar cell: E:weekday_month
JPY=X says Thursdays in September went 70-39 up for the dollar, sign p 0.0019.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

CROSSES = ["EURJPY=X", "GBPJPY=X", "CHFJPY=X", "NZDJPY=X", "AUDJPY=X", "JPY=X"]
px = load_prices(CROSSES)

for t in CROSSES:
    d = px[t]
    print(f"{t:<10} last {d.index[-1].date()} close {d['Close'].iloc[-1]:.3f} "
          f"1d {100*(d['Close'].iloc[-1]/d['Close'].iloc[-2]-1):+.2f}%")

# --- 2-ATR down day per cross (yen STRONGER = cross lower; JPY=X is USDJPY) ---
def atr_wilder(df, n=14):
    h, l, c = df["High"], df["Low"], df["Close"]
    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()

flags = {}
for t in CROSSES:
    d = px[t]
    a = atr_wilder(d).shift(1)
    move = d["Close"] - d["Close"].shift(1)
    flags[t] = (move <= -2.0 * a)

F = pd.DataFrame(flags).fillna(False)
count = F.sum(axis=1)
print(f"\ntonight's count of 2-ATR-down crosses (of {len(CROSSES)}): "
      f"{int(count.iloc[-1])} -> {[t for t in CROSSES if F[t].iloc[-1]]}")

panel = close_panel(CROSSES + ["SPY", "^GSPC", "^VIX", "TLT"])
idx = panel.index
cnt = count.reindex(idx).fillna(0)

for thr in (3, 4):
    trig = idx[cnt >= thr]
    trig = trig[trig < idx[-1]]
    epi = declusters(trig, 5, idx)
    print(f"\n=== {thr}+ crosses 2-ATR down same session: {len(trig)} days, "
          f"{len(epi)} episodes ===")
    if len(epi) < 3:
        print("  too few"); continue
    rows = []
    for sub in ["JPY=X", "EURJPY=X", "SPY"]:
        for h in (1, 5):
            r = fwd_ret(panel[sub], h)
            v = r.reindex(epi).dropna()
            s = summarize(v.values, f"{sub} h={h}")
            base = r.dropna()
            s["ctl_pct"] = round(100 * base.mean(), 3)
            s["edge_pct"] = round(s["mean_pct"] - 100 * base.mean(), 3)
            up = int((v > 0).sum())
            s["sign_p"] = round(sign_test(max(up, len(v) - up), len(v)), 4)
            rows.append(s)
    show(rows, f"forward from the cluster session, {thr}+ crosses")
    print("  episodes:", [str(d.date()) for d in epi][-12:])

# --- the calendar half: Thursdays in September, USDJPY ---
print("\n\n=== E:weekday_month JPY=X drill: Thursdays in September ===")
usd = panel["JPY=X"].dropna()
r1 = usd.shift(-1) / usd - 1.0
# anchor = the session BEFORE a September Thursday (engine convention, td_ahead 1)
nxt = pd.Series(usd.index[1:], index=usd.index[:-1])
is_sep_thu = nxt.dt.month.eq(9) & nxt.dt.weekday.eq(3)
anch = usd.index[:-1][is_sep_thu.values]
v = r1.reindex(anch).dropna()
up = int((v > 0).sum())
s = summarize(v.values, "all years")
s["sign_p"] = round(sign_test(up, len(v)), 4)
s["up"], s["down"] = up, len(v) - up
rows = [s]
for lab, m in [("pre-2018", anch.year < 2018), ("2018+", anch.year >= 2018),
               ("midterm", anch.year % 4 == 2)]:
    vv = r1.reindex(anch[m]).dropna()
    if len(vv) < 5:
        continue
    ss = summarize(vv.values, lab)
    u = int((vv > 0).sum())
    ss["sign_p"] = round(sign_test(u, len(vv)), 4)
    ss["up"], ss["down"] = u, len(vv) - u
    rows.append(ss)
# control: every other September session, and every other Thursday
ctl_sep = usd.index[:-1][nxt.dt.month.eq(9).values & ~is_sep_thu.values]
ctl_thu = usd.index[:-1][nxt.dt.weekday.eq(3).values & ~is_sep_thu.values]
for lab, a in [("ctl: other Sep sessions", ctl_sep), ("ctl: other Thursdays", ctl_thu)]:
    vv = r1.reindex(a).dropna()
    ss = summarize(vv.values, lab)
    u = int((vv > 0).sum())
    ss["up"], ss["down"] = u, len(vv) - u
    ss["sign_p"] = round(sign_test(u, len(vv)), 4)
    rows.append(ss)
show(rows, "USDJPY, session after the anchor")
print(cluster_note(anch, r1.reindex(anch).values, k=2))

# does the September-Thursday cell survive when the yen just rallied hard?
z = zscore(usd, 5)
weak = z.reindex(anch) < -0.75
vv = r1.reindex(anch[weak.fillna(False).values]).dropna()
if len(vv) >= 5:
    ss = summarize(vv.values, "Sep Thu, USDJPY 5d z < -0.75 going in")
    u = int((vv > 0).sum())
    ss["up"], ss["down"], ss["sign_p"] = u, len(vv) - u, round(sign_test(u, len(vv)), 4)
    show([ss], "conditioned on the live state (tonight z5 = %.2f)" % z.iloc[-1])
