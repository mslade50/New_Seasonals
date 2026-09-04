"""Composer's own re-derivation of the two numbers the morning turns on.

Written because the survivor's whole case, and the whole argument against it,
came back inside subagent reports. A number quoted out of a report inherits
that report's controls (registry, 2026-08-10), so the composer re-derives the
live cell and the August cross from scratch before shipping anything.

Cell: long TLT, MOC on the session before a PPI release, exit MOC on the print
session. Live variant additionally requires a CPI print ON the entry session,
which is exactly tonight (CPI 2026-08-12, PPI 2026-08-13).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

px = close_panel(["TLT", "IEF", "LQD"])
idx = px.index
ret = fwd_lag(px["TLT"], 1, 1)          # enter MOC D+1, exit MOC D+2
ev = load_events(["ppi", "cpi"])
ppi = pd.DatetimeIndex(ev.loc[ev["event"] == "ppi", "date"])
cpi = set(pd.DatetimeIndex(ev.loc[ev["event"] == "cpi", "date"]).normalize())

# anchor = 2 sessions before the print, so entry lands on the print's eve
anch, eve_is_cpi = [], []
for d in ppi:
    loc = idx.searchsorted(pd.Timestamp(d))
    if loc >= len(idx):                  # never mint a fake anchor (registry)
        continue
    if idx[loc].normalize() != pd.Timestamp(d).normalize():
        continue                         # print landed on a non-session
    p = loc - 2
    if p < 0 or p + 2 >= len(idx):
        continue
    anch.append(idx[p])
    eve_is_cpi.append(idx[p + 1].normalize() in cpi)

anch = pd.DatetimeIndex(anch)
eve_is_cpi = np.array(eve_is_cpi)
ok = ret.reindex(anch).notna().values
anch, eve_is_cpi = anch[ok], eve_is_cpi[ok]
v = ret.reindex(anch).values
entry = pd.DatetimeIndex([idx[idx.searchsorted(a) + 1] for a in anch])

base = float((ret.dropna() > 0).mean())
print(f"TLT unconditional 1-day base rate = {100*base:.1f}%  "
      f"(N={ret.notna().sum()})\n")


def line(lbl, vals):
    if len(vals) == 0:
        print(f"{lbl:44s} N=0")
        return
    w = int((np.asarray(vals) > 0).sum())
    print(f"{lbl:44s} N={len(vals):4d}  mean={100*np.mean(vals):+7.3f}%  "
          f"hit={100*w/len(vals):5.1f}%  record {w}-{len(vals)-w}  "
          f"signp(base)={sign_test(w, len(vals), base):.4f}")


print("=== 1. the cell, day-level (each PPI print is its own episode) ===")
line("parent: every PPI print session", v)
line("live:   CPI printed on the eve", v[eve_is_cpi])
line("comp:   no CPI on the eve", v[~eve_is_cpi])

# tdom-matched control (registry: an all-days control on a rates event is
# invalid; TLT's own drift swings with trading-day-of-month)
tdom = pd.Series(index=idx, dtype=float)
for _, g in pd.Series(idx, index=idx).groupby([idx.year, idx.month]):
    tdom.loc[g.index] = np.arange(1, len(g) + 1)
ctrl_by_tdom = ret.groupby(tdom).mean()
entry_tdom = tdom.reindex(entry).values
matched = ctrl_by_tdom.reindex(entry_tdom).values
print()
print("=== 2. same, net of a trading-day-of-month matched control ===")
line("parent excess", v - matched)
line("live excess", (v - matched)[eve_is_cpi])
line("comp excess", (v - matched)[~eve_is_cpi])

print()
print("=== 3. the August cross (the decisive objection) ===")
mon = entry.month.values
for m_lbl, m in [("August", mon == 8), ("not August", mon != 8)]:
    line(f"parent, {m_lbl}", v[m])
    line(f"live,   {m_lbl}", v[eve_is_cpi & m])

aug_live = v[eve_is_cpi & (mon == 8)]
print(f"\n  live-cell August observations: "
      f"{[f'{100*x:+.2f}%' for x in aug_live]}")
print(f"  entry dates: {[str(d.date()) for d in entry[eve_is_cpi & (mon == 8)]]}")

# how surprising is the WORST month, given the month was found not predicted?
live_v = v[eve_is_cpi]
live_m = mon[eve_is_cpi]
obs = min((live_v[live_m == k].mean() for k in set(live_m)
           if (live_m == k).sum() >= 3), default=np.nan)
rng = np.random.default_rng(42)
worse = 0
for _ in range(20000):
    perm = rng.permutation(live_v)
    mn = min((perm[live_m == k].mean() for k in set(live_m)
              if (live_m == k).sum() >= 3))
    worse += mn <= obs
print(f"\n  worst month mean among months with N>=3 = {100*obs:+.3f}%")
print(f"  permutation P(some month this bad by chance) = {worse/20000:.3f}")
print("  (the month was FOUND, not pre-specified, so it owes this charge)")

print()
print("=== 4. is August weakness the cell's, or TLT's own? ===")
allm = pd.Series(ret.values, index=idx).dropna()
am = allm.index.month
print(f"  TLT unconditional 1d, August:     {100*allm[am == 8].mean():+.4f}%  "
      f"hit {100*(allm[am == 8] > 0).mean():.1f}%  N={(am == 8).sum()}")
print(f"  TLT unconditional 1d, other:      {100*allm[am != 8].mean():+.4f}%  "
      f"hit {100*(allm[am != 8] > 0).mean():.1f}%  N={(am != 8).sum()}")
print("  -> if these are close, August is the CELL's, not the instrument's")

print()
print("=== 5. what tonight actually risks ===")
print(f"  worst live-cell session: {100*live_v.min():+.2f}%  "
      f"on {entry[eve_is_cpi][int(np.argmin(live_v))].date()}")
print(f"  worst parent session:    {100*v.min():+.2f}%")
losers = live_v[live_v < 0]
print(f"  losers: {len(losers)} of {len(live_v)}, mean {100*losers.mean():+.3f}%")
nav, atr_pct, close = 750_000, 0.0070, 82.19
for bps in (20, 30):
    risk = nav * bps / 10_000
    sh = risk / (atr_pct * close)
    print(f"  at {bps} bps risk: {sh:,.0f} sh = ${sh*close:,.0f} notional "
          f"({100*sh*close/nav:.0f}% NAV); worst obs = "
          f"{sh*close*live_v.min():+,.0f} = {live_v.min()/atr_pct:+.2f}R")
