"""K3 follow-up: align the DX cell to the EXACT order the pitch card places.

The card enters MOO on the execute session (2026-08-06, = signal date + 1) and
exits MOC after `time_td` sessions counted from that execute session. The K3
check measured signal-close -> close(signal + k). This script re-measures the
same cell on the executable basis so the number quoted on the card is the
number the trade actually books, and reports the d=2 sub-cell (NFP two
sessions after the signal close), which is today's configuration.

Trigger (K3's honest/loose form): DX 5d return rank <= 25 AND 63d return rank
>= 70 AND an NFP print falls within the next 3 sessions.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import _common as C

DX = C.load(["DX-Y.NYB"])["DX-Y.NYB"]
idx = DX.index

ev = pd.read_csv(C.ROOT / "data" / "macro_events.csv")
nfp = pd.to_datetime(
    ev.loc[ev["event"].astype(str).str.lower().str.contains("nfp|payroll"), "date"]
).dt.normalize().unique()
nfp = pd.DatetimeIndex(sorted(nfp))
print(f"NFP prints in the calendar: {len(nfp)}  ({nfp.min().date()} .. {nfp.max().date()})")

close = DX["Close"]
rk5 = C.pct_rank(C.ret(close, 5))
rk63 = C.pct_rank(C.ret(close, 63))
base = (rk5 <= 25) & (rk63 >= 70)

# distance in SESSIONS from the signal close to the next NFP print
pos = {d: i for i, d in enumerate(idx)}
nfp_pos = np.array(sorted({pos[d] for d in nfp if d in pos}))
dist = np.full(len(idx), 10 ** 6)
for i in range(len(idx)):
    nxt = nfp_pos[nfp_pos > i]
    if len(nxt):
        dist[i] = nxt[0] - i
dist = pd.Series(dist, index=idx)

sig = base & (dist <= 3)
print(f"\nsignal days (loose trigger, NFP within 3 sessions): {int(sig.sum())}")
print("today:", idx[-1].date(), "| rk5", round(rk5.iloc[-1], 1),
      "| rk63", round(rk63.iloc[-1], 1), "| NFP distance", int(dist.iloc[-1]),
      "| fires", bool(sig.iloc[-1]))

o, c = DX["Open"].to_numpy(), close.to_numpy()
n = len(idx)


def executable(hold: int, mask: pd.Series) -> pd.Series:
    """Enter MOO at signal+1, exit MOC `hold` sessions later counted from the
    entry session inclusive. hold=3 -> exit close(signal+3)."""
    out = np.full(n, np.nan)
    for i in np.flatnonzero(mask.to_numpy()):
        j, k = i + 1, i + hold
        if k < n:
            out[i] = (c[k] / o[j] - 1.0) * 100.0
    return pd.Series(out, index=idx)


rows = []
for hold in (2, 3, 4, 5):
    for name, m in (("all signals", sig),
                    ("d=2 (today's config)", sig & (dist == 2))):
        s = executable(hold, m).dropna()
        ep = C.declusterize(s.index, gap_td=10)
        rows.append({**C.describe(f"hold {hold}td MOO->MOC | {name}", s), "hold": hold})
        if name.startswith("all"):
            rows.append({**C.describe(f"hold {hold}td episodes(gap10)", s[ep]),
                         "hold": hold})
C.show(rows)

print("\n-- era split, the card's horizon (hold 3td, all signals) --")
s3 = executable(3, sig).dropna()
C.show(C.era_split(s3.index, s3.values))

print("\n-- unconditional control: same MOO->MOC hold, every session --")
ctrl = []
for hold in (2, 3, 4, 5):
    allm = pd.Series(True, index=idx)
    u = executable(hold, allm).dropna()
    ctrl.append({**C.describe(f"unconditional hold {hold}td", u), "hold": hold})
C.show(ctrl)

print("\n-- the d=2 sub-cell, signal by signal (hold 3td) --")
d2 = executable(3, sig & (dist == 2)).dropna()
for d, v in d2.items():
    print(f"   {d.date()}  {v:+.3f}%")
print(f"   mean {d2.mean():+.3f}%  median {d2.median():+.3f}%  "
      f"hit {(d2 > 0).mean() * 100:.1f}%  t {C.tstat(d2.values):.2f}  "
      f"worst {d2.min():+.3f}%  N={len(d2)}")

print("\n-- ATR context for sizing (Wilder-14 on DX spot) --")
atr = C.wilder_atr(DX).iloc[-1]
print(f"   DX close {close.iloc[-1]:.3f}  ATR14 {atr:.4f}  "
      f"({atr / close.iloc[-1] * 100:.2f}% of spot)")
print(f"   1.0 ATR on one DX contract ($1,000/pt) = ${atr * 1000:,.0f}")

print("\n-- worst drawdown inside the hold (hold 3td, all signals) --")
worst = []
for i in np.flatnonzero(sig.to_numpy()):
    j, k = i + 1, i + 3
    if k < n:
        lo = DX["Low"].to_numpy()[j:k + 1].min()
        worst.append((lo / o[j] - 1.0) * 100.0)
worst = np.array(worst)
print(f"   mean worst intra-hold excursion {worst.mean():+.2f}%  "
      f"median {np.median(worst):+.2f}%  5th pctile {np.percentile(worst, 5):+.2f}%  "
      f"min {worst.min():+.2f}%")
