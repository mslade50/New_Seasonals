"""C1b - the same TLT trade gated on the investment-grade 52w-low rung.

Watchlist entry 6 (added 2026-08-12) is explicit that the TRIGGER is FRESHNESS,
not the price state: the tight rung pays +0.354pp excess at an 82.4% hit over 17
EPISODE-FIRST days (sign p 0.0101), while later days inside the same episode pay
-0.079pp at a 50.0% hit over 52 days. It TURNS ON only when the rung fires on a
day that is the FIRST trigger day in >= 10 sessions.

Three jobs:
  1. establish today's exact depth inside the live episode,
  2. measure the depth-N cell at that exact depth (not the pooled "later days"),
  3. gate attribution - does the rung ADD anything to C1's August calendar cell?
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

BAR = pd.Timestamp("2026-08-14")
px = close_panel(["TLT", "IEF", "LQD"])
px = px.dropna()
IDX = px.index
tlt = px["TLT"]

low52 = {t: px[t].rolling(252).min() for t in ("TLT", "IEF", "LQD")}
off = {t: (px[t] / low52[t] - 1.0) * 100 for t in ("TLT", "IEF", "LQD")}
print("today's rung values (close 2026-08-14, signal day for a Monday MOC entry):")
for t in ("TLT", "IEF", "LQD"):
    print(f"  {t} {off[t].loc[BAR]:.2f}% off its 52w low")

TIGHT = (off["TLT"] <= 0.5) & (off["IEF"] <= 1.0) & (off["LQD"] <= 1.0)
trig = IDX[TIGHT.values]
print(f"\ntight rung trigger days N={len(trig)}, span {trig[0].date()} .. {trig[-1].date()}")

pos = pd.Series(range(len(IDX)), index=IDX)
# episode-first = first trigger day in >= 10 sessions
first, later, depth = [], [], {}
last_p, d = -10**9, 0
for dt in trig:
    p = pos[dt]
    if p - last_p >= 10:
        first.append(dt)
        d = 1
    else:
        later.append(dt)
        d += 1
    depth[dt] = d
    last_p = p
first, later = pd.DatetimeIndex(first), pd.DatetimeIndex(later)
print(f"episode-FIRST days N={len(first)}, LATER (depth>1) days N={len(later)}")

print("\n--- the live episode ---")
live = [dt for dt in trig if dt >= pd.Timestamp("2026-07-01")]
for dt in live:
    print(f"  {dt.date()}  depth={depth[dt]}  TLT {off['TLT'].loc[dt]:.2f} "
          f"IEF {off['IEF'].loc[dt]:.2f} LQD {off['LQD'].loc[dt]:.2f}"
          f"{'   <-- FIRST' if dt in first else ''}")
today_depth = depth.get(BAR)
print(f"\nTODAY (signal close {BAR.date()}): depth = {today_depth}. "
      f"prior trigger {trig[list(trig).index(BAR)-1].date()} "
      f"= {pos[BAR]-pos[trig[list(trig).index(BAR)-1]]} sessions ago "
      f"(freshness leg needs >= 10). EPISODE-FIRST: {BAR in first}")

# ------------------------------------------------ 2. the depth-N cell
print("\n" + "=" * 92)
print("2. the cell BY DEPTH - is today's depth paid, or is it the pooled 'later'?")
print("=" * 92)
for h in (1, 3, 5, 10):
    f = fwd_lag(tlt, h, 1)
    base = f.dropna()
    rows = [summarize(f.reindex(first).dropna().values, f"depth==1 (episode-first) h={h}"),
            summarize(f.reindex(later).dropna().values, f"depth>1 (later) h={h}"),
            summarize(base.values, "CTRL all days")]
    dsel = pd.DatetimeIndex([dt for dt in trig if depth[dt] == today_depth])
    rows.insert(2, summarize(f.reindex(dsel).dropna().values,
                             f"depth=={today_depth} EXACTLY (today) h={h}"))
    show(rows, f"h={h}")
    c = base.mean()
    for lbl, s in (("depth==1", first), (f"depth=={today_depth}", dsel),
                   ("depth>1", later)):
        v = f.reindex(s).dropna()
        if len(v):
            w = int((v > 0).sum())
            print(f"  {lbl}: excess {100*(v.mean()-c):+.3f}pp  {w}-{len(v)-w}  "
                  f"sign p {sign_test(w, len(v)):.4f}")

# depth ladder at h=1 (the watchlist's pitched horizon)
print("\n--- depth ladder, h=1 ---")
f1 = fwd_lag(tlt, 1, 1)
c1 = f1.dropna().mean()
rows = []
for dd in range(1, 12):
    s = pd.DatetimeIndex([dt for dt in trig if depth[dt] == dd])
    v = f1.reindex(s).dropna()
    if len(v) < 3:
        continue
    rows.append({"depth": dd, "n": len(v), "mean_pct": round(100 * v.mean(), 3),
                 "excess_pct": round(100 * (v.mean() - c1), 3),
                 "hit": round(100 * (v > 0).mean(), 1)})
show(rows, "TLT h=1 by depth inside the rung episode")

# ------------------------------------------------ 3. GATE ATTRIBUTION vs C1
print("\n" + "=" * 92)
print("3. GATE ATTRIBUTION - does the IG rung add anything to the August cell?")
print("=" * 92)
aug_mid = pd.Series((IDX.month == 8) & (IDX.day >= 6) & (IDX.day <= 19), index=IDX)
for h in (5, 10):
    f = fwd_lag(tlt, h, 1)
    a = IDX[aug_mid.values]
    a_on = pd.DatetimeIndex([d for d in a if TIGHT.loc[d]])
    a_off = pd.DatetimeIndex([d for d in a if not TIGHT.loc[d]])
    show([summarize(f.reindex(a).dropna().values, f"August 6-19, gate OFF-or-ON h={h}"),
          summarize(f.reindex(a_on).dropna().values, f"August 6-19 x rung ON h={h}"),
          summarize(f.reindex(a_off).dropna().values, f"August 6-19 x rung OFF h={h}"),
          summarize(f.reindex(pd.DatetimeIndex(trig)).dropna().values,
                    f"rung ON, all months h={h}"),
          summarize(f.dropna().values, "CTRL all days")], f"h={h}")

# ------------------------------------------------ 4. era + 2022 dependence
print("\n" + "=" * 92)
print("4. era / year concentration of the episode-first cell (the watchlist number)")
print("=" * 92)
f1 = fwd_lag(tlt, 1, 1)
v = f1.reindex(first).dropna()
print("episode-first dates:", ", ".join(str(d.date()) for d in v.index))
print(f"  h=1 mean {100*v.mean():+.3f}%  excess {100*(v.mean()-f1.dropna().mean()):+.3f}pp  "
      f"{int((v>0).sum())}-{int((v<=0).sum())}  sign p {sign_test(int((v>0).sum()), len(v)):.4f}")
print("  ", cluster_note(v.index, v.values))
byyr = pd.Series(100 * v.values).groupby(v.index.year.values).agg(["count", "mean"])
print(byyr.round(3).to_string())
