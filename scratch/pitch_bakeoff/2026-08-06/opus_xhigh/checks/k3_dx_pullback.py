"""K3 ADVERSARIAL CHECK — LONG the dollar on a washout-inside-an-uptrend,
with NFP as the claimed catalyst.

Candidate trigger (DX-Y.NYB spot index):
    5d return rank <= 10th pctile (252d window)  AND
    63d return rank >= 80th pctile
Forward 3/5/10 sessions. Vehicle claimed: DX futures (front month DXU6).

The brief is to KILL this. Tests run:
  A. Does the base pullback cell beat DX's unconditional drift at all?
  B. Era split at 2018 (both halves reported, no cherry-picking).
  C. Episode declustering (a washout persists for days -> raw N is fake).
  D. NFP interaction: does "NFP lands inside the forward window" add anything
     over the base cell? If not, the NFP framing is decoration and must die.
  E. The plain "day before NFP -> +5 sessions" DX cell on the FULL NFP sample
     (the baseline the conditional story must beat).
  F. Worst window / worst year.
  G. Cost math: DX futures tick/spread/carry vs the measured edge.

Entry bases measured BOTH ways: close-to-close from the signal close (what a
backtest usually books) and next-open MOO -> close k sessions later (what the
pitch would actually do on 2026-08-06).

Run: python k3_dx_pullback.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import _common as C
from macro_calendar import event_dates
from scipy import stats as sps

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 40)

TKR = "DX-Y.NYB"
HORIZONS = (3, 5, 10)

P = C.load([TKR])
dx = P[TKR]
close = dx["Close"]
idx = close.index


def hdr(t: str) -> None:
    print("\n" + "=" * 96 + f"\n{t}\n" + "=" * 96)


# ---------------------------------------------------------------- trigger ---
r5 = C.ret(close, 5)
r63 = C.ret(close, 63)
rk5 = C.pct_rank(r5)
rk63 = C.pct_rank(r63)
sma200 = close.rolling(200).mean()

cond = (rk5 <= 10) & (rk63 >= 80)

hdr("K3.0  TODAY'S READING (last usable bar = 2026-08-05)")
print(f"  DX close            {close.iloc[-1]:.3f}")
print(f"  5d return           {r5.iloc[-1]:+.2f}%   rank {rk5.iloc[-1]:.1f}")
print(f"  21d return          {C.ret(close, 21).iloc[-1]:+.2f}%   "
      f"rank {C.pct_rank(C.ret(close, 21)).iloc[-1]:.1f}")
print(f"  63d return          {r63.iloc[-1]:+.2f}%   rank {rk63.iloc[-1]:.1f}")
print(f"  z10                 {C.z10(close).iloc[-1]:+.2f}")
print(f"  vs 200d SMA         {(close.iloc[-1] / sma200.iloc[-1] - 1) * 100:+.2f}%")
print(f"  TRIGGER FIRES TODAY: {bool(cond.iloc[-1])}")
print(f"  raw signal days     {int(cond.sum())} of {int(cond.notna().sum())} bars "
      f"({close.index.min():%Y-%m} .. {close.index.max():%Y-%m})")


# --------------------------------------------------------- forward returns ---
def fwd_cc(k: int) -> pd.Series:
    return C.fwd(close, k)


def fwd_no(k: int) -> pd.Series:
    """MOO the session after the signal -> close k sessions after the signal."""
    entry = dx["Open"].shift(-1)
    exit_ = close.shift(-k)
    return (exit_ / entry - 1.0) * 100.0


hdr("K3.A  BASE PULLBACK CELL vs UNCONDITIONAL DRIFT  (close-to-close entry)")
rows = []
for k in HORIZONS:
    f = fwd_cc(k)
    sig = f[cond.fillna(False) & f.notna()]
    ep = C.declusterize(sig.index, gap_td=k)
    rows.append({**C.describe(f"h{k} all signal days", sig, baseline=f.dropna())})
    rows.append({**C.describe(f"h{k} EPISODES (gap {k}td)", sig[ep], baseline=f.dropna())})
    rows.append({**C.describe(f"h{k} unconditional", f.dropna())})
C.show(rows)

hdr("K3.A2 BASE PULLBACK CELL — next-open MOO entry (what the pitch would do)")
rows = []
for k in HORIZONS:
    f = fwd_no(k)
    sig = f[cond.fillna(False) & f.notna()]
    ep = C.declusterize(sig.index, gap_td=k)
    rows.append({**C.describe(f"h{k} all signal days", sig, baseline=f.dropna())})
    rows.append({**C.describe(f"h{k} EPISODES", sig[ep], baseline=f.dropna())})
    rows.append({**C.describe(f"h{k} unconditional", f.dropna())})
C.show(rows)


hdr("K3.B  ERA SPLIT AT 2018 (all signal days AND episodes, close-to-close)")
for k in HORIZONS:
    f = fwd_cc(k)
    sig = f[cond.fillna(False) & f.notna()]
    ep = C.declusterize(sig.index, gap_td=k)
    print(f"\n-- h{k} all signal days")
    C.show(C.era_split(sig.index, sig.values))
    print(f"-- h{k} episodes")
    C.show(C.era_split(sig[ep].index, sig[ep].values))
    print(f"-- h{k} unconditional (era baseline)")
    fu = f.dropna()
    C.show(C.era_split(fu.index, fu.values))


hdr("K3.B2 THIRD ERA CUT: pre-2015 / 2015-2021 / 2022+  (h5 episodes)")
f5 = fwd_cc(5)
sig5 = f5[cond.fillna(False) & f5.notna()]
ep5 = C.declusterize(sig5.index, gap_td=5)
s5 = sig5[ep5]
cuts = [("2000-2014", "2000-01-01", "2015-01-01"),
        ("2015-2021", "2015-01-01", "2022-01-01"),
        ("2022+", "2022-01-01", "2030-01-01")]
rows = []
for lab, a, b in cuts:
    m = (s5.index >= a) & (s5.index < b)
    rows.append(C.describe(lab, s5[m]))
    fb = f5.dropna()
    fm = (fb.index >= a) & (fb.index < b)
    rows.append(C.describe(f"  {lab} uncond", fb[fm]))
C.show(rows)


# ----------------------------------------------------------- NFP interaction -
hdr("K3.D  NFP INTERACTION — does 'NFP inside the forward window' add anything?")
nfp = event_dates("nfp")
nfp = nfp[(nfp >= idx.min()) & (nfp <= idx.max())]
pos = idx.searchsorted(nfp, side="left")
pos = np.unique(pos[pos < len(idx)])
is_nfp = np.zeros(len(idx), dtype=bool)
is_nfp[pos] = True
is_nfp = pd.Series(is_nfp, index=idx)
print(f"  NFP dates in calendar: {len(event_dates('nfp'))}, "
      f"mapped into the DX session index: {len(pos)}")

# td distance to the NEXT nfp (>=1 means it is ahead of us)
nxt_dist = np.full(len(idx), 10_000)
j = 0
for i in range(len(idx)):
    while j < len(pos) and pos[j] < i:
        j += 1
    if j < len(pos):
        nxt_dist[i] = pos[j] - i
nxt_dist = pd.Series(nxt_dist, index=idx, name="td_to_next_nfp")
print(f"  today's td to next NFP (from 2026-08-05 bar): {int(nxt_dist.iloc[-1])} "
      "(NFP 2026-08-07 is not in the price index yet, so this is informational)")

for k in HORIZONS:
    f = fwd_cc(k)
    base = cond.fillna(False) & f.notna()
    inwin = base & (nxt_dist >= 1) & (nxt_dist <= k)
    nowin = base & ~((nxt_dist >= 1) & (nxt_dist <= k))
    a, b = f[inwin], f[nowin]
    epa = C.declusterize(a.index, gap_td=k)
    epb = C.declusterize(b.index, gap_td=k)
    print(f"\n-- h{k}")
    C.show([C.describe(f"h{k} NFP IN window (all days)", a, baseline=f.dropna()),
            C.describe(f"h{k} NFP IN window (episodes)", a[epa], baseline=f.dropna()),
            C.describe(f"h{k} NFP NOT in window (all)", b, baseline=f.dropna()),
            C.describe(f"h{k} NFP NOT in window (eps)", b[epb], baseline=f.dropna())])
    if len(a) > 1 and len(b) > 1:
        from scipy import stats as sps
        tt = sps.ttest_ind(a.dropna(), b.dropna(), equal_var=False)
        print(f"   Welch t (IN vs NOT), all days: t={tt.statistic:+.2f} p={tt.pvalue:.3f}")

hdr("K3.D1b ERA SPLIT OF THE 'NFP IN WINDOW' CELL (the one that looks alive)")
for k in HORIZONS:
    f = fwd_cc(k)
    a = f[cond.fillna(False) & f.notna() & (nxt_dist >= 1) & (nxt_dist <= k)]
    print(f"-- h{k} NFP-in-window, all signal days (N={len(a)})")
    C.show(C.era_split(a.index, a.values))
    print(f"   dates: {[f'{d:%Y-%m-%d}:{v:+.2f}' for d, v in a.items()]}\n")

hdr("K3.D1c PLACEBO / PERMUTATION — is the NFP split special, or just a lucky 11?")
rng = np.random.default_rng(42)
for k in HORIZONS:
    f = fwd_cc(k)
    base = f[cond.fillna(False) & f.notna()]
    inmask = ((nxt_dist >= 1) & (nxt_dist <= k)).reindex(base.index).fillna(False)
    m = int(inmask.sum())
    obs = float(base[inmask].mean())
    draws = np.array([rng.choice(base.values, size=m, replace=False).mean()
                      for _ in range(20000)])
    p = float((draws >= obs).mean())
    print(f"  h{k}: observed in-window mean {obs:+.3f}%  (m={m} of {len(base)});"
          f"  P(random subset of size m >= obs) = {p:.4f}")
print("  ^ this is the honest within-cell test. It does NOT correct for the fact")
print("    that 3 horizons were tried, so multiply the smallest p by ~3.")

hdr("K3.D1d PLACEBO EVENTS — shift every NFP by +7 sessions and redo the split")
for shift in (5, 7, 11, -7):
    ppos = np.unique(np.clip(pos + shift, 0, len(idx) - 1))
    nd = np.full(len(idx), 10_000)
    j = 0
    for i in range(len(idx)):
        while j < len(ppos) and ppos[j] < i:
            j += 1
        if j < len(ppos):
            nd[i] = ppos[j] - i
    nd = pd.Series(nd, index=idx)
    f = fwd_cc(3)
    a = f[cond.fillna(False) & f.notna() & (nd >= 1) & (nd <= 3)]
    print(f"  fake-event shift {shift:+3d} td: h3 in-window N={len(a):3d} "
          f"avg {a.mean():+.3f}%  t {C.tstat(a.values):+.2f}")

hdr("K3.D2 EXACT NFP DISTANCE — today is d=2 (signal 08-05, entry 08-06, NFP 08-07)")
print("  The signal close is 2026-08-05. The next session (2026-08-06) is the")
print("  ENTRY. NFP prints 2026-08-07 = 2 sessions after the signal close.")
print("  So the historically matched cell is nxt_dist == 2, NOT == 1.\n")
for d in (1, 2, 3):
    for k in (3, 5):
        f = fwd_cc(k)
        m = cond.fillna(False) & (nxt_dist == d) & f.notna()
        sig = f[m]
        star = "  <-- TODAY" if d == 2 else ""
        print(f"  d={d} h{k}: N={len(sig):2d} avg {sig.mean():+.3f}% "
              f"t {C.tstat(sig.values):+.2f} hit "
              f"{(sig > 0).mean() * 100 if len(sig) else float('nan'):.0f}%{star}")
        if len(sig):
            print(f"        {[f'{x:%Y-%m-%d}:{v:+.2f}' for x, v in sig.items()]}")


# --------------------------------------------- E: plain day-before-NFP cell --
hdr("K3.E  BASELINE THE STORY MUST BEAT: plain 'day before NFP -> +k' on DX, full sample")
pre = pd.Series(False, index=idx)
pre.iloc[np.clip(pos - 1, 0, len(idx) - 1)] = True
pre.iloc[0] = False
rows = []
for k in HORIZONS:
    f = fwd_cc(k)
    sig = f[pre & f.notna()]
    rows.append(C.describe(f"h{k} day-before-NFP (N should be ~320)", sig,
                           baseline=f.dropna()))
C.show(rows)
print("\n  era split, h5 day-before-NFP:")
f5 = fwd_cc(5)
s = f5[pre & f5.notna()]
C.show(C.era_split(s.index, s.values))
print("\n  AUGUST-only day-before-NFP, h5:")
sa = s[s.index.month == 8]
C.show([C.describe("Aug day-before-NFP h5", sa, baseline=f5.dropna())])
print(f"   dates: {[f'{d:%Y-%m-%d}:{v:+.2f}' for d, v in sa.items()]}")
print("\n  MIDTERM August day-before-NFP, h5:")
sm = sa[[d.year % 4 == 2 for d in sa.index]]
C.show([C.describe("midterm-Aug day-before-NFP h5", sm)])
print(f"   dates: {[f'{d:%Y-%m-%d}:{v:+.2f}' for d, v in sm.items()]}")


# ---------------------------------------------------------- F worst / years --
hdr("K3.F  WORST WINDOW / PER-YEAR TABLE (h5, close-to-close, all signal days)")
f5 = fwd_cc(5)
sig5 = f5[cond.fillna(False) & f5.notna()]
ep5 = C.declusterize(sig5.index, gap_td=5)
byyear = pd.DataFrame({"r": sig5.values, "ep": ep5.astype(int)}, index=sig5.index)
g = byyear.groupby(byyear.index.year).agg(
    n=("r", "size"), eps=("ep", "sum"), avg=("r", "mean"),
    worst=("r", "min"), best=("r", "max"), hit=("r", lambda x: (x > 0).mean() * 100))
print(g.round(3).to_string())
print(f"\n  worst single h5 window: {sig5.min():+.2f}% on "
      f"{sig5.idxmin():%Y-%m-%d}")
print(f"  worst h10 window:      {fwd_cc(10)[cond.fillna(False)].min():+.2f}%")
print(f"  worst episode year (avg): {g['avg'].idxmin()} at {g['avg'].min():+.2f}%")


# ------------------------------------------------------------------- G cost --
hdr("K3.G  COST CHECK — DX futures (DXU6), $1,000/index point, tick 0.005 = $5")
px = float(close.iloc[-1])
notional = px * 1000.0
tick_bps = (0.005 / px) * 1e4
print(f"  DX spot            {px:.3f}   -> 1 contract notional ${notional:,.0f}")
print(f"  1 tick (0.005)     ${5:.2f} = {tick_bps:.2f} bps of notional")
print(f"  round-trip assumption: cross 1 tick in + 1 tick out = {2*tick_bps:.2f} bps")
print(f"  commission ~$2.50/side = $5 RT = {(5/notional)*1e4:.2f} bps")
print(f"  TOTAL round trip   ~{2*tick_bps + (5/notional)*1e4:.2f} bps of notional")
rt = 2 * tick_bps + (5 / notional) * 1e4
for k in HORIZONS:
    f = fwd_cc(k)
    sig = f[cond.fillna(False) & f.notna()]
    ep = C.declusterize(sig.index, gap_td=k)
    e = sig[ep].mean() * 100  # bps
    print(f"  h{k} episode edge  {e:+.1f} bps  -> edge/cost = {e/rt:+.1f}x "
          f"({'PASSES 2x' if e/rt >= 2 else 'FAILS 2x'})")
print("\n  BASIS NOTE: DX-Y.NYB is the SPOT index. DX futures price the forward,")
print("  so the futures leg differs from this measurement by the carry roll")
print("  (USD rate minus the basket-weighted foreign rate, EUR 57.6% dominant).")
print("  With USD rates above the basket, DX futures sit BELOW spot and converge")
print("  up: a LONG earns carry. Magnitude: a 100 bps annualized differential is")
print("  ~1.9 bps over 5 sessions / ~3.8 bps over 10 — same order as the round")
print("  trip, and it favors the long. It is NOT big enough to rescue a dead cell.")


hdr("K3.H  PARAMETER SENSITIVITY — is the trigger a knife edge? (h5 episodes)")
rows = []
for lo in (5, 10, 15, 20, 25):
    for hi in (70, 75, 80, 85, 90):
        c = (rk5 <= lo) & (rk63 >= hi)
        f = fwd_cc(5)
        s = f[c.fillna(False) & f.notna()]
        if len(s) < 5:
            continue
        e = C.declusterize(s.index, gap_td=5)
        rows.append({"rk5<=": lo, "rk63>=": hi, "n": len(s),
                     "avg": round(s.mean(), 3), "t": round(C.tstat(s.values), 2),
                     "eps": int(e.sum()), "ep_avg": round(s[e].mean(), 3),
                     "ep_t": round(C.tstat(s[e].values), 2)})
print(pd.DataFrame(rows).to_string(index=False))

hdr("K3.H2 SENSITIVITY OF THE NFP-IN-WINDOW h3 CELL (the only 'alive' cell)")
rows = []
for lo in (5, 10, 15, 20, 25):
    for hi in (70, 75, 80, 85, 90):
        c = (rk5 <= lo) & (rk63 >= hi)
        f = fwd_cc(3)
        s = f[c.fillna(False) & f.notna() & (nxt_dist >= 1) & (nxt_dist <= 3)]
        if len(s) < 3:
            continue
        rows.append({"rk5<=": lo, "rk63>=": hi, "n": len(s),
                     "avg": round(s.mean(), 3), "t": round(C.tstat(s.values), 2),
                     "hit": round((s > 0).mean() * 100, 1)})
print(pd.DataFrame(rows).to_string(index=False))


hdr("K3.I  RELAXED-PARAM VERSION OF THE NFP CELL, WITH ITS CONTROLS")
# The tight trigger (5/80) gives N=11. The sensitivity grid says the effect is
# not a knife edge, so re-measure at a LOOSER, still-defensible trigger and put
# the proper control next to it.
cond_loose = (rk5 <= 25) & (rk63 >= 70)
print(f"  loose trigger fires today: {bool(cond_loose.iloc[-1])}")
for k in (3, 5, 10):
    f = fwd_cc(k)
    base = cond_loose.fillna(False) & f.notna()
    a = f[base & (nxt_dist >= 1) & (nxt_dist <= k)]
    b = f[base & ~((nxt_dist >= 1) & (nxt_dist <= k))]
    epa = C.declusterize(a.index, gap_td=k)
    print(f"\n-- h{k}, loose trigger")
    C.show([C.describe(f"h{k} NFP IN window", a, baseline=f.dropna()),
            C.describe(f"h{k} NFP IN window (eps)", a[epa], baseline=f.dropna()),
            C.describe(f"h{k} NFP NOT in window", b, baseline=f.dropna())])
    tt = sps.ttest_ind(a.dropna(), b.dropna(), equal_var=False)
    print(f"   Welch IN vs NOT: t={tt.statistic:+.2f} p={tt.pvalue:.4f}")
    if k == 3:
        print("   era split:")
        C.show(C.era_split(a.index, a.values))
        yr = pd.Series(a.values, index=a.index).groupby(a.index.year).agg(
            ["size", "mean", "min"]).round(3)
        print(yr.to_string())

hdr("K3.J  CONFOUND #1 — NFP is the FIRST FRIDAY. Is this a turn-of-month cell?")
# "NFP at t+1..t+3" places the signal day in the last days of the prior month
# or the first days of the new one. Turn-of-month is a KNOWN and (per the
# negative registry) largely arbitraged calendar effect. Test the same cell
# against a pure turn-of-month window with no NFP reference at all.
dom = pd.Series(idx.day, index=idx)
mth = pd.Series(idx.month, index=idx)
is_last3 = pd.Series(False, index=idx)
is_first3 = pd.Series(False, index=idx)
grp = pd.Series(idx.to_period("M"), index=idx)
for _, g in grp.groupby(grp):
    is_last3.loc[g.index[-3:]] = True
    is_first3.loc[g.index[:3]] = True
tom = is_last3 | is_first3
for label, m in (("turn-of-month (last3|first3)", tom),
                 ("month-end last3", is_last3),
                 ("month-start first3", is_first3)):
    for k in (3,):
        f = fwd_cc(k)
        a = f[cond_loose.fillna(False) & f.notna() & m]
        b = f[cond_loose.fillna(False) & f.notna() & ~m]
        print(f"  {label:32s} h{k}: IN N={len(a):3d} avg {a.mean():+.3f}% "
              f"t {C.tstat(a.values):+.2f} | OUT N={len(b):3d} avg {b.mean():+.3f}% "
              f"t {C.tstat(b.values):+.2f}")
# And the decisive one: inside the turn-of-month window, does NFP-in-window
# still separate?
f3 = fwd_cc(3)
nfpwin = (nxt_dist >= 1) & (nxt_dist <= 3)
m = cond_loose.fillna(False) & f3.notna() & tom
print("\n  WITHIN turn-of-month days only (loose trigger, h3):")
print(f"    NFP in window:  N={int((m & nfpwin).sum())} "
      f"avg {f3[m & nfpwin].mean():+.3f}% t {C.tstat(f3[m & nfpwin].values):+.2f}")
print(f"    NFP not in win: N={int((m & ~nfpwin).sum())} "
      f"avg {f3[m & ~nfpwin].mean():+.3f}% t {C.tstat(f3[m & ~nfpwin].values):+.2f}")
print("\n  Overlap check: of the NFP-in-window signals, how many are ToM days?")
q = cond_loose.fillna(False) & f3.notna() & nfpwin
print(f"    {int((q & tom).sum())} of {int(q.sum())} "
      f"({(q & tom).sum() / max(1, q.sum()) * 100:.0f}%)")

hdr("K3.J2 IS THE TURN-OF-MONTH VERSION STILL ALIVE POST-2013?")
# The negative registry says famous calendar cells (turn-of-month included)
# were arbitraged away post-2013. If the pullback+ToM cell dies after 2013 the
# whole thing is a museum piece.
f3 = fwd_cc(3)
for lab, m in (("pullback + ToM", cond_loose.fillna(False) & tom),
               ("pullback + NFP-in-3", cond_loose.fillna(False) & nfpwin)):
    s = f3[m & f3.notna()]
    print(f"\n  {lab}: full N={len(s)} avg {s.mean():+.3f}% t {C.tstat(s.values):+.2f}")
    for a, b, e in (("2000-01-01", "2013-01-01", "pre-2013"),
                    ("2013-01-01", "2019-01-01", "2013-2018"),
                    ("2019-01-01", "2030-01-01", "2019+")):
        x = s[(s.index >= a) & (s.index < b)]
        ep = C.declusterize(x.index, gap_td=3) if len(x) else np.array([], bool)
        print(f"    {e:10s} N={len(x):3d} avg {x.mean():+.3f}% "
              f"t {C.tstat(x.values):+.2f}  hit {(x>0).mean()*100:.0f}%  "
              f"eps={int(ep.sum()) if len(x) else 0} "
              f"ep_t {C.tstat(x[ep].values) if len(x) else float('nan'):+.2f}")

hdr("K3.J3 WHERE IS TODAY IN THE MONTH? (does the ToM framing even apply?)")
last = idx[-1]
mon_sessions = idx[(idx.year == last.year) & (idx.month == last.month)]
print(f"  last bar {last:%Y-%m-%d} is session "
      f"{list(mon_sessions).index(last) + 1} of August 2026 so far")
print(f"  August 2026 sessions in the index: {[f'{d:%m-%d}' for d in mon_sessions]}")
print(f"  is_first3 on the last bar: {bool(is_first3.iloc[-1])}  "
      f"is_last3: {bool(is_last3.iloc[-1])}  ToM: {bool(tom.iloc[-1])}")
print("  NOTE: the ENTRY session is 2026-08-06 (4th session), the SIGNAL close")
print("  is 2026-08-05 (3rd session). The historical cell is indexed on the")
print("  SIGNAL close, so today qualifies as a first-3 turn-of-month day.")

hdr("K3.J4 DX DATA SANITY (spot index from yfinance — is the series clean?)")
print(f"  duplicate dates: {int(idx.duplicated().sum())}")
print(f"  zero/neg closes: {int((close <= 0).sum())}")
rr = close.pct_change().abs()
print(f"  |1d move| > 3%:  {int((rr > 0.03).sum())} bars, max {rr.max()*100:.2f}% "
      f"on {rr.idxmax():%Y-%m-%d}")
print(f"  repeated closes (stale prints): "
      f"{int((close.diff() == 0).sum())} bars")
print(f"  Volume column all-zero: {bool((dx['Volume'] == 0).all())} "
      "(expected for a spot index)")
gaps = pd.Series(idx).diff().dt.days
print(f"  max calendar gap between bars: {int(gaps.max())} days "
      f"(median {gaps.median():.0f})")

hdr("K3.K  CONFOUND #2 — does the pullback add anything, or is it pre-NFP drift?")
for k in (3, 5):
    f = fwd_cc(k)
    win = (nxt_dist >= 1) & (nxt_dist <= k) & f.notna()
    print(f"  h{k} ALL days with NFP in t+1..t+{k} (no pullback filter): "
          f"N={int(win.sum())} avg {f[win].mean():+.3f}% t {C.tstat(f[win].values):+.2f}")
    w2 = win & cond_loose.fillna(False)
    print(f"     ... AND loose pullback: N={int(w2.sum())} avg {f[w2].mean():+.3f}% "
          f"t {C.tstat(f[w2].values):+.2f}")
    w3 = win & (rk5 <= 25) & (rk63 < 70)
    print(f"     ... AND rk5<=25 but rk63<70 (pullback, NO uptrend): "
          f"N={int(w3.sum())} avg {f[w3].mean():+.3f}% t {C.tstat(f[w3].values):+.2f}")
    w4 = win & (rk5 > 25)
    print(f"     ... AND rk5>25 (no pullback): N={int(w4.sum())} "
          f"avg {f[w4].mean():+.3f}% t {C.tstat(f[w4].values):+.2f}")


hdr("K3.L  RETURN-PATH DECOMPOSITION — is the payoff the print, or the drift into it?")
# For each in-window signal, split the h3 return into
#   leg1: signal close -> the session BEFORE the print
#   leg2: print-eve close -> print-day close   (the event bet)
#   leg3: print close -> t+3
# If leg2 is the whole thing, this is a coin flip on a data release and must be
# graded as such. If leg1 carries it, it is a positioning/flow drift.
v = close.to_numpy()
n = len(v)
recs = []
for i in np.where((cond_loose.fillna(False) & (nxt_dist >= 1) & (nxt_dist <= 3)).to_numpy())[0]:
    d = int(nxt_dist.iloc[i])
    if i + 3 >= n:
        continue
    p_sig, p_eve, p_prn, p_end = v[i], v[i + d - 1], v[i + d], v[i + 3]
    recs.append({"date": idx[i], "d": d,
                 "leg1_pre": (p_eve / p_sig - 1) * 100,
                 "leg2_print": (p_prn / p_eve - 1) * 100,
                 "leg3_post": (p_end / p_prn - 1) * 100,
                 "total_h3": (p_end / p_sig - 1) * 100})
dec = pd.DataFrame(recs).set_index("date")
print(f"  N={len(dec)} in-window signals (loose trigger)")
print(dec[["leg1_pre", "leg2_print", "leg3_post", "total_h3"]].agg(
    ["mean", "median", "std", "min", "max"]).round(3).to_string())
for c in ("leg1_pre", "leg2_print", "leg3_post", "total_h3"):
    print(f"  {c:12s} mean {dec[c].mean():+.3f}%  t {C.tstat(dec[c].values):+.2f}  "
          f"hit {(dec[c] > 0).mean() * 100:.0f}%")
print("\n  by distance d:")
print(dec.groupby("d")[["leg1_pre", "leg2_print", "leg3_post", "total_h3"]]
      .agg(["size", "mean"]).round(3).to_string())

hdr("K3.N  THE NUMBER THAT ACTUALLY MATTERS: next-open MOO entry on the NFP cell")
# The signal is the 2026-08-05 close; the pitch enters 2026-08-06. Any edge
# that lives in the 08-05 -> 08-06 overnight is NOT capturable.
rows = []
for lab, c in (("TIGHT", cond), ("LOOSE", cond_loose)):
    for k in (3, 5):
        f = fwd_no(k)
        s = f[c.fillna(False) & f.notna() & (nxt_dist >= 1) & (nxt_dist <= k)]
        e = C.declusterize(s.index, gap_td=k)
        rows.append(C.describe(f"{lab} h{k} MOO all", s, baseline=f.dropna()))
        rows.append(C.describe(f"{lab} h{k} MOO episodes", s[e], baseline=f.dropna()))
C.show(rows)
print("\n  close-to-close vs MOO, same cells (how much edge is in the overnight):")
for lab, c in (("TIGHT", cond), ("LOOSE", cond_loose)):
    for k in (3, 5):
        m = c.fillna(False) & (nxt_dist >= 1) & (nxt_dist <= k)
        a = fwd_cc(k)[m].dropna()
        b = fwd_no(k)[m].dropna()
        print(f"   {lab} h{k}: CC {a.mean():+.3f}%  MOO {b.mean():+.3f}%  "
              f"overnight give-up {a.mean() - b.mean():+.3f}%")

hdr("K3.D3 d=2 CELL AT THE LOOSE TRIGGER (today's exact NFP distance)")
for k in (3, 5):
    for basis, ff in (("CC", fwd_cc(k)), ("MOO", fwd_no(k))):
        s = ff[cond_loose.fillna(False) & (nxt_dist == 2) & ff.notna()]
        print(f"  d=2 h{k} {basis}: N={len(s):2d} avg {s.mean():+.3f}% "
              f"t {C.tstat(s.values):+.2f} hit {(s > 0).mean() * 100:.0f}% "
              f"worst {s.min():+.2f}%")
s = fwd_cc(3)[cond_loose.fillna(False) & (nxt_dist == 2) & fwd_cc(3).notna()]
print(f"  d=2 h3 dates: {[f'{x:%Y-%m-%d}:{v:+.2f}' for x, v in s.items()]}")

hdr("K3.M  EPISODE BOOTSTRAP — P(mean <= 0) on the honest (declustered) sample")
rng2 = np.random.default_rng(11)
for lab, c in (("TIGHT trigger (pre-specified)", cond),
               ("LOOSE trigger (post-hoc)", cond_loose)):
    for k in (3, 5):
        f = fwd_cc(k)
        s = f[c.fillna(False) & f.notna() & (nxt_dist >= 1) & (nxt_dist <= k)]
        e = C.declusterize(s.index, gap_td=k)
        x = s[e].values
        if len(x) < 3:
            print(f"  {lab} h{k}: N_ep={len(x)} too small")
            continue
        bs = np.array([rng2.choice(x, size=len(x), replace=True).mean()
                       for _ in range(20000)])
        print(f"  {lab} h{k}: N_ep={len(x)} mean {x.mean():+.3f}% "
              f"t {C.tstat(x):+.2f}  P(bootstrap mean <= 0) = {(bs <= 0).mean():.4f}")


# ------------------------------------------------- multiplicity bookkeeping --
hdr("K3.X  MULTIPLICITY LEDGER")
print("""  Cells examined in this script:
    3 horizons x 2 entry bases                                  =  6
    x era splits (2018 cut, 2 halves; plus a 3-era cut on h5)   = +8
    NFP interaction: 3 horizons x 2 sides x (all/episodes)      = +12
    exact-cell (pullback & NFP=t+1): 3                          = +3
    day-before-NFP baseline: 3 horizons + era + Aug + midterm   = +7
  ~36 cells. A single |t| of 2.0 anywhere in here is worth nothing.
  Only a cell that is (a) the PRE-SPECIFIED one, (b) alive in BOTH eras and
  (c) alive on episodes should be allowed to carry a pitch.""")
