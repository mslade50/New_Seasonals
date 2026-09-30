"""Posts check (2026-09-18): five tape cells for tonight's queue.

Today is Friday 2026-09-18, the September quad-witching session.

A. IWM 63-session return at an extreme low WHILE the S&P sits near a high.
   Trigger: pct_rank(IWM, 63, 252) <= 5 AND SPY within 3% of its trailing
   252-session closing high (inclusive). Novelty declustering: the state must
   have been absent for the prior 21 sessions. Outcomes IWM/SPY/spread at
   h=5/10/21, two controls, era split, episode list.
B. CL=F stretched >= 30% above its 200-session SMA, first in 63 sessions.
   h=5/21/63, control = all valid-SMA sessions. Plus the 252d-return >= +80%
   episode inventory. April 2020's negative settle is handled explicitly.
C. ^TNX: the last 6 closes, every close >= 5.0 since 2007, run lengths.
D. GLD >= 15% below its trailing-252 closing high while its 252-session
   return is still >= +15%, first in 63 sessions. h=21/63.
E. The Monday after September quad witching (h=1 from the quad close) for
   IWM, SPY, ^VIX, 2000-2025. Control: the session after every other
   September Friday. Brief - it re-anchors a cell already posted.

Everything is close-to-close, anchored on a close that is already printed, so
entry is lag=0 from the anchor (MOC that session), matching how a post reads.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa: E402,F401,F403

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ASOF = pd.Timestamp("2026-09-18")
ERA = "2018-01-01"
TK = ["IWM", "SPY", "CL=F", "^TNX", "GLD", "^VIX"]

px = load_prices(TK)
nyse = px["SPY"]["Close"].dropna().index
nyse = pd.DatetimeIndex(nyse[nyse <= ASOF])
C: dict[str, pd.Series] = {t: px[t]["Close"].astype(float).reindex(nyse) for t in TK}
POS = pd.Series(range(len(nyse)), index=nyse)

print(f"NYSE calendar (SPY): {nyse[0].date()} .. {nyse[-1].date()}  n={len(nyse)}")
print(f"ASOF = {ASOF.date()}; freshest SPY bar = {nyse[-1].date()} "
      f"({'MATCHES' if nyse[-1] == ASOF else 'DOES NOT MATCH'})")
for t in TK:
    v = C[t].dropna()
    print(f"  {t}: first bar {v.index[0].date()}  last {v.index[-1].date()}  n={len(v)}"
          f"  last close {v.iloc[-1]:.3f}")


# ------------------------------------------------------------------ helpers
def rec(v: np.ndarray) -> tuple[int, int, int]:
    v = np.asarray(v, float)
    v = v[~np.isnan(v)]
    return int((v > 0).sum()), int((v < 0).sum()), len(v)


def line(label: str, s: pd.Series, dates: pd.DatetimeIndex,
         ctrl: pd.Series | None = None, ctrl_label: str = "ctrl") -> pd.Series:
    v = s.reindex(pd.DatetimeIndex(dates)).dropna()
    if len(v) == 0:
        print(f"  {label}: n=0")
        return v
    up, dn, n = rec(v.values)
    sm = summarize(v.values)
    extra = ""
    if ctrl is not None:
        cv = ctrl.dropna()
        cu, cd, cn = rec(cv.values)
        extra = (f"\n      {ctrl_label}: n={cn} {cu}-{cd} mean {100*cv.mean():+.3f}% "
                 f"med {100*float(np.median(cv.values)):+.3f}% hit {100*(cv>0).mean():.1f}%")
    print(f"  {label}: n={n} {up}-{dn} mean {sm['mean_pct']:+.3f}% med {sm['median_pct']:+.3f}% "
          f"hit {sm['hit']:.1f}% t {sm['t']:+.2f} signp_up {sign_test(up, n):.4f} "
          f"signp_dn {sign_test(dn, n):.4f} worst {sm['worst_pct']:+.2f}% "
          f"({v.idxmin().date()}) best {sm['best_pct']:+.2f}% ({v.idxmax().date()}){extra}")
    return v


def third_friday(year: int, month: int) -> pd.Timestamp:
    days = pd.date_range(f"{year}-{month:02d}-01", periods=31, freq="D")
    fri = [d for d in days if d.month == month and d.weekday() == 4]
    return fri[2]


def fwd(s: pd.Series, h: int) -> pd.Series:
    """Close of anchor -> close h sessions later, aligned to the anchor.

    Computed on the series' OWN valid sessions then reindexed: CL=F trades
    some NYSE holidays and misses others, so a raw shift over the union
    calendar would count a hole as a session.
    """
    v = s.dropna()
    return (v.shift(-h) / v - 1.0).reindex(s.index)


def pct_change_valid(s: pd.Series, n: int) -> pd.Series:
    """n-VALID-session return, reindexed (pitch_lab._valid_pct_change, which is
    private and so not re-exported by the star import)."""
    v = s.dropna()
    return (v / v.shift(n) - 1.0).reindex(s.index)


def novelty(trig: pd.DatetimeIndex, win: int) -> pd.DatetimeIndex:
    """Keep a trigger only if the state was ABSENT for the prior `win`
    sessions (stricter than pitch_lab.declusters, which measures the gap to
    the last KEPT event rather than to the last trigger)."""
    tset = set(pd.DatetimeIndex(trig))
    keep = []
    for d in sorted(tset):
        p = int(POS[d])
        prior = nyse[max(0, p - win):p]
        if not any(x in tset for x in prior):
            keep.append(d)
    return pd.DatetimeIndex(keep)


def era(label: str, v: pd.Series) -> None:
    a, b = v[v.index < ERA], v[v.index >= ERA]
    def part(x):
        if len(x) == 0:
            return "n=0"
        u, d, n = rec(x.values)
        return (f"n={n} {u}-{d} mean {100*x.mean():+.3f}% "
                f"med {100*float(np.median(x.values)):+.3f}%")
    print(f"   {label} era: pre-2018 [{part(a)}] | 2018+ [{part(b)}]")


def episodes(dates: pd.DatetimeIndex, series: dict[str, pd.Series]) -> None:
    cols = list(series)
    print("   episode      " + "".join(f"{c:>12}" for c in cols))
    for d in dates:
        cells = []
        for c in cols:
            x = series[c].get(d, np.nan)
            cells.append("        n/a" if (x is None or np.isnan(x)) else f"{100*x:+11.2f}")
        print(f"   {d.date()}  " + "".join(f"{c:>12}" for c in cells))


# =========================================================================
# A. IWM 63d return at an extreme low while SPY sits near its high
# =========================================================================
print("\n" + "=" * 78)
print("=== A. IWM 63d-return rank <= 5 while SPY is within 3% of its 252d high ===")
rank63 = pct_rank(C["IWM"], 63, 252)
spy_hi = rolling_on_valid(C["SPY"], lambda x: x.rolling(252).max())
spy_dist = C["SPY"] / spy_hi - 1.0

print(f"   TODAY ({nyse[-1].date()}): IWM 63d-return rank = {rank63.iloc[-1]:.2f} "
      f"(state file says 0.8) | SPY {100*spy_dist.iloc[-1]:+.2f}% from its 252d "
      f"closing high (state file says -2.08%)")
print(f"   today qualifies: IWM rank<=5 {rank63.iloc[-1] <= 5} | "
      f"SPY>=-3% {spy_dist.iloc[-1] >= -0.03}")

valid = rank63.notna() & spy_dist.notna()
raw = pd.DatetimeIndex(nyse[(rank63 <= 5) & (spy_dist >= -0.03) & valid])
near_hi = pd.DatetimeIndex(nyse[(spy_dist >= -0.03) & valid])
print(f"   raw (non-declustered) trigger sessions: {len(raw)} "
      f"({raw[0].date()} .. {raw[-1].date()})")
trig = novelty(raw, 21)
dc = declusters(raw, 21, nyse)
print(f"   after 21-session NOVELTY filter: {len(trig)} "
      f"(pitch_lab.declusters(21) would keep {len(dc)})")
print(f"   control-1 universe (SPY within 3% of high, any IWM state): {len(near_hi)} sessions")
print(f"   control-2 universe (all sessions with both series valid): {int(valid.sum())}")
print("   trigger dates: " + ", ".join(str(d.date()) for d in trig))

for h in (5, 10, 21):
    print(f"\n   --- horizon h={h} ---")
    fi, fs = fwd(C["IWM"], h), fwd(C["SPY"], h)
    sp = fi - fs
    vi = line(f"IWM h={h}", fi, trig, ctrl=fi.reindex(near_hi),
              ctrl_label="ctrl1 SPY-near-high")
    line(f"IWM h={h} vs all days", fi, trig, ctrl=fi.dropna(), ctrl_label="ctrl2 all days")
    line(f"SPY h={h}", fs, trig, ctrl=fs.reindex(near_hi),
         ctrl_label="ctrl1 SPY-near-high")
    line(f"IWM-SPY spread h={h}", sp, trig, ctrl=sp.reindex(near_hi),
         ctrl_label="ctrl1 SPY-near-high")
    if len(vi):
        era(f"IWM h={h}", vi)

print("\n   episode table (IWM / SPY / spread, in %):")
ser = {}
for h in (5, 10, 21):
    ser[f"IWM_{h}"] = fwd(C["IWM"], h)
    ser[f"SPY_{h}"] = fwd(C["SPY"], h)
    ser[f"SPR_{h}"] = fwd(C["IWM"], h) - fwd(C["SPY"], h)
episodes(trig, ser)
print("   (n/a on the last row = the forward window has not finished yet)")

# =========================================================================
# B. CL=F stretched above its 200-day
# =========================================================================
print("\n" + "=" * 78)
print("=== B. CL=F >= 30% above its 200-session SMA (first in 63 sessions) ===")
cl = C["CL=F"]
BAD_CL = pd.DatetimeIndex([pd.Timestamp("2020-04-20"), pd.Timestamp("2020-04-21")])
neg = cl[cl <= 0].dropna()
print(f"   DATA CAVEAT: CL=F is a continuous front-month series and settled "
      f"NEGATIVE on {len(neg)} session(s): "
      + ", ".join(f"{d.date()} {v:.2f}" for d, v in neg.items()))
print("   Percentage returns across those bars are meaningless, so any anchor "
      "whose forward window spans 2020-04-20/21 is EXCLUDED from that horizon,")
print("   and those two sessions are never anchors themselves.")

cl_valid = cl.dropna()
cl_pos = pd.Series(range(len(cl_valid)), index=cl_valid.index)
bad_pos = [int(cl_pos[d]) for d in BAD_CL if d in cl_pos.index]


def cl_ok(dates: pd.DatetimeIndex, h: int) -> pd.DatetimeIndex:
    """Anchors whose h-session forward window avoids the negative bars."""
    out = []
    for d in pd.DatetimeIndex(dates):
        p = cl_pos.get(d)
        if p is None:
            continue
        p = int(p)
        if any(p <= bp <= p + h for bp in bad_pos):
            continue
        out.append(d)
    return pd.DatetimeIndex(out)


sma200 = rolling_on_valid(cl, lambda x: x.rolling(200).mean())
stretch = cl / sma200 - 1.0
cl_252 = pct_change_valid(cl, 252)
print(f"\n   TODAY ({cl_valid.index[-1].date()}): CL=F close {cl_valid.iloc[-1]:.2f}, "
      f"200-SMA {sma200.dropna().iloc[-1]:.2f} -> {100*stretch.dropna().iloc[-1]:+.2f}% "
      f"above it; 252-session return {100*cl_252.dropna().iloc[-1]:+.2f}%")

raw_b = pd.DatetimeIndex(nyse[(stretch >= 0.30) & stretch.notna()])
raw_b = pd.DatetimeIndex([d for d in raw_b if d not in BAD_CL])
trig_b = novelty(raw_b, 63)
print(f"   raw sessions >= +30% over the 200-SMA: {len(raw_b)}; "
      f"after the 63-session novelty filter: {len(trig_b)}")
print("   episode dates: " + ", ".join(str(d.date()) for d in trig_b))

base_b = pd.DatetimeIndex(stretch.dropna().index)
for h in (5, 21, 63):
    f = fwd(cl, h)
    a = cl_ok(trig_b, h)
    c = cl_ok(base_b, h)
    drop_a, drop_c = len(trig_b) - len(a), len(base_b) - len(c)
    v = line(f"CL=F h={h}", f, a, ctrl=f.reindex(c), ctrl_label="ctrl all valid-SMA days")
    print(f"      (excluded for the negative-settle window: {drop_a} anchor(s), "
          f"{drop_c} control day(s))")
    if len(v):
        era(f"CL=F h={h}", v)

print("\n   episode table (CL=F forward returns, %):")
episodes(trig_b, {"h=5": fwd(cl, 5), "h=21": fwd(cl, 21), "h=63": fwd(cl, 63),
                  "stretch": stretch, "ret252": cl_252})
print("   (stretch / ret252 are the STATE on the anchor day, not outcomes)")

print("\n   --- for the record: CL=F 252-session return >= +80%, episodes since 2000 ---")
hot = pd.DatetimeIndex(cl_252.dropna()[cl_252.dropna() >= 0.80].index)
hot = pd.DatetimeIndex([d for d in hot if d not in BAD_CL])
print(f"   sessions with 252d return >= +80%: {len(hot)}")
ep_starts, last = [], None
for d in hot:
    p = int(cl_pos[d])
    if last is None or p - last >= 63:
        ep_starts.append(d)
    last = p
print(f"   grouped into {len(ep_starts)} episode(s) (a 63+ session gap starts a new one):")
f63 = fwd(cl, 63)
for d in ep_starts:
    ok = len(cl_ok(pd.DatetimeIndex([d]), 63)) == 1
    v = f63.get(d, np.nan)
    if not ok:
        txt = "EXCLUDED (forward window spans the negative settle)"
    elif v is None or np.isnan(v):
        txt = "n/a (window not finished)"
    else:
        txt = f"{100*v:+.2f}%"
    print(f"     {d.date()}  252d ret {100*cl_252.loc[d]:+7.1f}%  ->  63d forward {txt}")

# =========================================================================
# C. The 10-year yield
# =========================================================================
print("\n" + "=" * 78)
print("=== C. ^TNX (10-year yield, index points = percent) ===")
tnx = C["^TNX"].dropna()
print("   last 6 closes:")
for d, v in tnx.tail(6).items():
    print(f"     {d.date()}  {v:.3f}")

t07 = tnx[tnx.index >= "2007-01-01"]
hi = t07[t07 >= 5.0]
print(f"\n   closes >= 5.00 since 2007-01-01: {len(hi)}")
for d, v in hi.items():
    print(f"     {d.date()}  {v:.3f}")

pos07 = pd.Series(range(len(t07)), index=t07.index)
runs, cur = [], []
for d in hi.index:
    p = int(pos07[d])
    if cur and p == cur[-1][1] + 1:
        cur.append((d, p))
    else:
        if cur:
            runs.append(cur)
        cur = [(d, p)]
if cur:
    runs.append(cur)
print(f"\n   consecutive runs of closes >= 5.00: {len(runs)}")
for r in runs:
    print(f"     {r[0][0].date()} .. {r[-1][0].date()}  length {len(r)} session(s)")

last_pre = hi[hi.index < "2026-01-01"]
since = tnx[(tnx.index >= "2007-07-19") & (tnx >= 5.0)]
print(f"\n   PLAINLY: closes >= 5.00 since 2007-07-19: {len(since)}"
      + (" -> " + ", ".join(f"{d.date()} ({v:.3f})" for d, v in since.items())
         if len(since) else ""))
if len(last_pre):
    print(f"   last close >= 5.00 BEFORE 2026: {last_pre.index[-1].date()} "
          f"({last_pre.iloc[-1]:.3f})")
else:
    print("   there is no close >= 5.00 before 2026 in the 2007+ window")
for d in ("2026-09-16", "2026-09-17", "2026-09-18"):
    ts = pd.Timestamp(d)
    if ts in tnx.index:
        v = float(tnx.loc[ts])
        print(f"   {d}: close {v:.3f} -> {'YES' if v >= 5.0 else 'no'}, >= 5.00")
    else:
        print(f"   {d}: no bar")

# =========================================================================
# D. Gold drawdown while still up on the year
# =========================================================================
print("\n" + "=" * 78)
print("=== D. GLD >= 15% below its 252d closing high while 252d return >= +15% ===")
gld = C["GLD"]
gld_hi = rolling_on_valid(gld, lambda x: x.rolling(252).max())
gld_dd = gld / gld_hi - 1.0
gld_252 = pct_change_valid(gld, 252)
print(f"   TODAY ({gld.dropna().index[-1].date()}): GLD close "
      f"{gld.dropna().iloc[-1]:.2f}, {100*gld_dd.dropna().iloc[-1]:+.2f}% from its "
      f"252d closing high (state file says -19.1%), 252d return "
      f"{100*gld_252.dropna().iloc[-1]:+.2f}% (state file says +19.05%)")
print(f"   today qualifies: dd<=-15% {gld_dd.dropna().iloc[-1] <= -0.15} | "
      f"ret252>=+15% {gld_252.dropna().iloc[-1] >= 0.15}")

valid_d = gld_dd.notna() & gld_252.notna()
raw_d = pd.DatetimeIndex(nyse[(gld_dd <= -0.15) & (gld_252 >= 0.15) & valid_d])
trig_d = novelty(raw_d, 63)
print(f"   raw trigger sessions: {len(raw_d)}; after the 63-session novelty "
      f"filter: {len(trig_d)}")
if len(raw_d):
    print(f"   raw span: {raw_d[0].date()} .. {raw_d[-1].date()}")
print("   episode dates: " + (", ".join(str(d.date()) for d in trig_d) or "(none)"))

base_d = pd.DatetimeIndex(gld.dropna().index)
for h in (21, 63):
    f = fwd(gld, h)
    v = line(f"GLD h={h}", f, trig_d, ctrl=f.reindex(base_d), ctrl_label="ctrl all GLD days")
    if len(v):
        era(f"GLD h={h}", v)

print("\n   episode table (GLD, %):")
episodes(trig_d, {"h=21": fwd(gld, 21), "h=63": fwd(gld, 63),
                  "dd": gld_dd, "ret252": gld_252})
print("   (dd / ret252 are the STATE on the anchor day)")

# =========================================================================
# E. The session after September quad witching (brief re-anchor)
# =========================================================================
print("\n" + "=" * 78)
print("=== E. h=1 from the September quad-witching close (the Monday after) ===")
qw = load_events(["quad_witching"])
qw_all = pd.DatetimeIndex(pd.to_datetime(qw["date"].unique()))
sep_qw = pd.DatetimeIndex([d for d in qw_all if d.month == 9 and d in nyse and d <= ASOF])
other_sep_fri = pd.DatetimeIndex(
    [d for d in nyse if d.month == 9 and d.weekday() == 4 and d not in sep_qw])
nxt = {d: nyse[int(POS[d]) + 1] for d in sep_qw if int(POS[d]) + 1 < len(nyse)}
wk = pd.Series({d: nxt[d].weekday() for d in nxt})
print(f"   September quad anchors on the NYSE calendar: {len(sep_qw)} "
      f"({sep_qw[0].year}..{sep_qw[-1].year}); next session is a Monday in "
      f"{int((wk == 0).sum())} of {len(wk)} cases "
      f"(others: {sorted({int(w) for w in wk if w != 0})} weekday codes)")
for t in ("IWM", "SPY", "^VIX"):
    f1 = fwd(C[t], 1)
    line(f"{t} h=1 after Sept quad", f1, pd.DatetimeIndex(list(nxt)),
         ctrl=f1.reindex(other_sep_fri), ctrl_label="ctrl other Sept Fridays")
v = fwd(C["IWM"], 1).reindex(pd.DatetimeIndex(list(nxt))).dropna()
print("   IWM by year: " + ", ".join(f"{d.year}:{100*x:+.2f}%" for d, x in v.items()))

print("\n" + "=" * 78)
print(f"FROZEN closes on the last bar:")
for t in TK:
    s = C[t].dropna()
    print(f"   {t:5s} {s.index[-1].date()}  {s.iloc[-1]:.3f}")
