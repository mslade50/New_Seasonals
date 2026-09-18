"""B1 round 1 -- "The dollar that finally confirms".

PRE-SPECIFIED: LONG UUP (no short leg), entry lag=1 MOC, h=5.
TRIGGER: DX-Y.NYB 63d-return rank (trailing 252, PIT) <= 20
         AND ^TNX closes at a trailing-252d HIGH.

Second vehicle priced: DX-Y.NYB itself (untradeable index, mechanism proxy).

Order convention (rule 7): FILTER first, THEN decluster. Stated and fixed.
Calendar (rule 8): every series reindexed to SPY's NYSE calendar before any
differencing or ranking; ^TNX and DX-Y.NYB carry bars NYSE lacks.
Re-derived from scratch. Nothing reused from the 2026-09-09 registry entry.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

pd.set_option("display.width", 200)

TICKS = ["SPY", "UUP", "DX-Y.NYB", "^TNX", "EURUSD=X", "JPY=X", "TLT"]
raw = load_prices(TICKS)
cal = raw["SPY"].index                      # NYSE calendar, the tradeable one
px = pd.DataFrame({t: raw[t]["Close"].reindex(cal) for t in TICKS})
print(f"panel on SPY calendar: {len(px)} rows {px.index[0].date()}..{px.index[-1].date()}")
for t in TICKS:
    v = px[t].dropna()
    print(f"  {t:10s} valid={len(v):5d}  first={v.index[0].date()}  last={v.index[-1].date()}"
          f"  last_close={v.iloc[-1]:.4f}")

dx, tnx, uup = px["DX-Y.NYB"], px["^TNX"], px["UUP"]

# ---------------------------------------------------------------- state
dx_r63 = pct_rank(dx, 63)
dx_r21 = pct_rank(dx, 21)
uup_r63 = pct_rank(uup, 63)


def at_high(s: pd.Series, n: int) -> pd.Series:
    hi = rolling_on_valid(s, lambda x: x.rolling(n).max())
    return (s >= hi - 1e-9) & s.notna() & hi.notna()


tnx_hi252, tnx_hi126, tnx_hi63 = at_high(tnx, 252), at_high(tnx, 126), at_high(tnx, 63)

print("\n--- LIVE STATE CHECK (2026-09-09 close) ---")
last = px.index[-1]
print(f"  date              {last.date()}")
print(f"  ^TNX close        {tnx.iloc[-1]:.4f}   252d max {rolling_on_valid(tnx, lambda x: x.rolling(252).max()).iloc[-1]:.4f}"
      f"   at252high={bool(tnx_hi252.iloc[-1])}")
print(f"  DX r63            {dx_r63.iloc[-1]:.1f}   (brief says 15.1)")
print(f"  DX r21            {dx_r21.iloc[-1]:.1f}   (brief says 25.0)")
print(f"  UUP r63           {uup_r63.iloc[-1]:.1f}   (brief says 13.1)")
print(f"  trigger live?     {bool((dx_r63.iloc[-1] <= 20) and tnx_hi252.iloc[-1])}")

TRIG = (dx_r63 <= 20) & tnx_hi252
TRIG = TRIG.fillna(False)
print(f"\nstate days on the SPY calendar, DX history: {int(TRIG.sum())}")
print("  by year:", dict(TRIG[TRIG].groupby(TRIG[TRIG].index.year).size()))

# UUP-valid subset (UUP inception 2007-03-01 is a hard era cut)
uup_valid = uup.notna()
print(f"state days with UUP alive: {int((TRIG & uup_valid).sum())}  "
      f"| pre-UUP: {int((TRIG & ~uup_valid).sum())}")

H = 5

# ---------------------------------------------------------------- 1. battery
battery(px, TRIG, [("UUP", 1.0)], H, "B1 LONG UUP  h=5", cost_bps=3.5,
        variants={
            "DX r63<=10": (dx_r63 <= 10) & tnx_hi252,
            "DX r63<=15": (dx_r63 <= 15) & tnx_hi252,
            "DX r63<=20 (DEFENDED)": TRIG,
            "DX r63<=25": (dx_r63 <= 25) & tnx_hi252,
            "DX r63<=30": (dx_r63 <= 30) & tnx_hi252,
            "UUP-own r63<=20": (uup_r63 <= 20) & tnx_hi252,
            "TNX 126d high": (dx_r63 <= 20) & tnx_hi126,
            "TNX 63d high": (dx_r63 <= 20) & tnx_hi63,
        },
        event_kinds=("cpi",))

battery(px, TRIG, [("DX-Y.NYB", 1.0)], H, "B1 LONG DX-Y.NYB (index) h=5",
        cost_bps=0.0001, event_kinds=("cpi",))

# ---------------------------------------------------------------- helpers
def cell(mask, legs, h=H, min_gap=None, lbl=""):
    ret = vehicle_ret(px, legs, h, 1)
    valid = ret.notna()
    days = px.index[mask.reindex(px.index, fill_value=False).values & valid.values]
    if len(days) == 0:
        return dict(label=lbl, n_days=0, n_epi=0), pd.DatetimeIndex([]), np.array([])
    epi = declusters(days, min_gap or h, px.index)
    v = ret.loc[epi].values
    s = summarize(v, lbl)
    s["n_days"] = len(days)
    w = int((v > 0).sum())
    s["record"] = f"{w}-{len(v)-w}"
    s["sign_p"] = round(sign_test(w, len(v)), 4)
    return s, epi, v


print("\n\n" + "=" * 78)
print("2. MECHANISM / DOSE RESPONSE on the ^TNX high lookback")
print("   a knife edge at 252 kills the 'differential' story; a plateau supports it")
print("=" * 78)
rows = []
for n in (21, 63, 126, 189, 252):
    m = (dx_r63 <= 20) & at_high(tnx, n)
    for legs, tag in (([("UUP", 1.0)], "UUP"), ([("DX-Y.NYB", 1.0)], "DX")):
        s, _, _ = cell(m, legs, lbl=f"TNX {n}d high x DXr63<=20 [{tag}]")
        rows.append(s)
show(rows, "dose response, high lookback")

print("\n--- continuous mechanism: does fwd dollar return rise with the RATE MOVE? ---")
print("    within DX r63<=20 only, bucket by ^TNX 63d change rank (no high gate)")
tnx_r63 = pct_rank(tnx, 63)
base = (dx_r63 <= 20)
for lo, hi in [(0, 25), (25, 50), (50, 75), (75, 90), (90, 101)]:
    m = base & (tnx_r63 >= lo) & (tnx_r63 < hi)
    s, _, _ = cell(m, [("UUP", 1.0)], lbl=f"TNX r63 [{lo},{hi})")
    rows2 = [s]
    s2, _, _ = cell(m, [("DX-Y.NYB", 1.0)], lbl=f"  same, DX index")
    rows2.append(s2)
    show(rows2, "")

print("\n\n" + "=" * 78)
print("3. GATE ATTRIBUTION + DISCARDED COMPLEMENTS (rule 4)")
print("=" * 78)
gates = {
    "BOTH gates (DEFENDED)": TRIG,
    "  complement of BOTH": ~TRIG,
    "DX r63<=20 ALONE": (dx_r63 <= 20),
    "  its complement DX r63>20": (dx_r63 > 20),
    "TNX 252d high ALONE": tnx_hi252,
    "  its complement not-at-high": (~tnx_hi252) & tnx.notna(),
    "DX r63<=20 & NOT tnx high (discarded by tnx gate)": (dx_r63 <= 20) & (~tnx_hi252),
    "tnx high & DX r63>20 (discarded by dx gate)": tnx_hi252 & (dx_r63 > 20),
}
for legs, tag in (([("UUP", 1.0)], "UUP"), ([("DX-Y.NYB", 1.0)], "DX index")):
    rr = []
    for lbl, m in gates.items():
        s, _, _ = cell(m.fillna(False), legs, lbl=lbl)
        rr.append(s)
    show(rr, f"gate attribution [{tag}] h=5")

print("\n\n" + "=" * 78)
print("4. ERA + EPISODE DATES + MIDTERM SPLIT")
print("=" * 78)
for legs, tag in (([("UUP", 1.0)], "UUP"), ([("DX-Y.NYB", 1.0)], "DX index")):
    s, epi, v = cell(TRIG, legs, lbl=f"DEFENDED [{tag}]")
    print(f"\n[{tag}] episodes N={len(epi)}  mean={s.get('mean_pct', float('nan')):.3f}%  "
          f"record {s.get('record')}  sign p {s.get('sign_p')}")
    for d, x in zip(epi, v):
        yr = d.year
        print(f"    {d.date()}   {100*x:+7.3f}%   {'MIDTERM' if yr % 4 == 2 else '       '}")
    if len(v) >= 2:
        order = np.argsort(-v)
        print(f"    drop-best-1: mean {100*np.delete(v, order[0]).mean():+.3f}% "
              f"(N={len(v)-1}, record {(np.delete(v, order[0])>0).sum()}-"
              f"{(np.delete(v, order[0])<=0).sum()})")
    if len(v) >= 3:
        print(f"    drop-best-2: mean {100*np.delete(v, order[:2]).mean():+.3f}% "
              f"(N={len(v)-2}, record {(np.delete(v, order[:2])>0).sum()}-"
              f"{(np.delete(v, order[:2])<=0).sum()})")
        print(f"    cluster_note: {cluster_note(epi, v)}")
    mid = np.array([d.year % 4 == 2 for d in epi])
    if len(v):
        show([summarize(v[mid], f"MIDTERM years (N={int(mid.sum())})"),
              summarize(v[~mid], f"non-midterm (N={int((~mid).sum())})")],
             f"midterm split [{tag}]")

print("\n\n" + "=" * 78)
print("5. EVENT-CONDITIONING FEASIBILITY (verify the registry claim)")
print("=" * 78)
ev = load_events(["ppi", "cpi", "fomc_decision"])
pos = pd.Series(range(len(px.index)), index=px.index)
state_days = px.index[TRIG.values]
for kind in ("ppi", "cpi", "fomc_decision"):
    e = pd.DatetimeIndex(ev[ev["event"] == kind]["date"])
    epos = set()
    for d in e:
        loc = int(px.index.searchsorted(d))
        if 0 <= loc < len(px.index):
            epos.add(loc)
    n2 = sum(1 for d in state_days if (pos[d] + 2) in epos)
    n1 = sum(1 for d in state_days if (pos[d] + 1) in epos)
    inwin = sum(1 for d in state_days
                if any((pos[d] + k) in epos for k in range(1, H + 2)))
    print(f"  {kind:15s}: state days with the print exactly 2 sessions later = {n2}"
          f" | exactly 1 later = {n1} | anywhere in the h=5 hold = {inwin}"
          f"  (of {len(state_days)} state days)")

print("\n  -- cell behaviour with CPI / FOMC inside the h=5 hold (episodes) --")
for legs, tag in (([("UUP", 1.0)], "UUP"), ([("DX-Y.NYB", 1.0)], "DX index")):
    _, epi, v = cell(TRIG, legs)
    for kinds in (("cpi",), ("fomc_decision",), ("ppi",)):
        fl = event_in_window(epi, px.index, H, 1, kinds)
        show([summarize(v[fl], f"{kinds[0]} IN hold (N={int(fl.sum())})"),
              summarize(v[~fl], f"{kinds[0]} OUT (N={int((~fl).sum())})")],
             f"[{tag}] {kinds[0]}-in-hold")

print("\n\n" + "=" * 78)
print("6. PER-UNIT-OF-RISK: UUP vs DX as the vehicle")
print("=" * 78)
for legs, tag in (([("UUP", 1.0)], "UUP"), ([("DX-Y.NYB", 1.0)], "DX index")):
    s, epi, v = cell(TRIG, legs)
    ret = vehicle_ret(px, legs, H, 1)
    allsd = 100 * ret.dropna().std(ddof=1)
    if len(v):
        print(f"  {tag:9s} episode mean {100*v.mean():+.3f}%  "
              f"uncond 5d sd {allsd:.3f}%  -> mean/sd = {100*v.mean()/allsd:.2f}"
              f"   episode sd {s['sd_pct']:.3f}%  t={s['t']:.2f}")

print("\n\n" + "=" * 78)
print("7. PERMUTATION (rule 1): tested against the DEFENDED cell's episode mean")
print("=" * 78)


def perm_p(mask: pd.Series, legs, h=H, n_perm=4000, seed=7, grid=None):
    """Circular rotation of the trigger mask preserves its clustering.
    Returns (uncharged p on the defended stat, charged max-of-grid p)."""
    rng = np.random.default_rng(seed)
    ret = vehicle_ret(px, legs, h, 1)
    valid = ret.notna().values
    idx = px.index
    m0 = mask.reindex(idx, fill_value=False).values.astype(bool)
    lo = int(np.argmax(valid))                     # first tradeable position
    span = np.zeros(len(idx), bool)
    span[lo:] = valid[lo:]

    def stat(mv):
        d = idx[mv & span]
        if len(d) == 0:
            return -np.inf
        e = declusters(d, h, idx)
        return ret.loc[e].mean()

    obs = stat(m0)
    grid_masks = [g.reindex(idx, fill_value=False).values.astype(bool)
                  for g in (grid or [])]
    obs_grid = max([stat(g) for g in grid_masks], default=obs)
    null_def, null_max = [], []
    n = len(idx)
    for _ in range(n_perm):
        k = int(rng.integers(1, n))
        null_def.append(stat(np.roll(m0, k)))
        if grid_masks:
            null_max.append(max(stat(np.roll(g, k)) for g in grid_masks))
    null_def = np.array(null_def, float)
    p_unch = float((null_def >= obs).mean())
    p_chg = float((np.array(null_max, float) >= obs).mean()) if null_max else np.nan
    return obs, obs_grid, p_unch, p_chg


GRID = [(dx_r63 <= q).fillna(False) & at_high(tnx, n)
        for q in (10, 15, 20, 25, 30) for n in (63, 126, 252)]
for legs, tag in (([("UUP", 1.0)], "UUP"), ([("DX-Y.NYB", 1.0)], "DX index")):
    obs, obsg, pu, pc = perm_p(TRIG, legs, grid=GRID)
    print(f"  [{tag}] statistic tested = DEFENDED cell episode MEAN 5d return "
          f"= {100*obs:+.3f}%")
    print(f"        uncharged p (rotation, defended cell only) = {pu:.4f}")
    print(f"        charged p vs max over the 15-cell grid I walked "
          f"(r63 in 10/15/20/25/30 x TNX high 21..252) = {pc:.4f}"
          f"   [grid max obs = {100*obsg:+.3f}%]")

print("\n\n" + "=" * 78)
print("8. COST (UUP: 0.77%/yr expense = 1.5 bp per 5 sessions, ~2 bp round trip)")
print("=" * 78)
s, epi, v = cell(TRIG, [("UUP", 1.0)])
if len(v):
    edge = 100 * 100 * v.mean()
    rt = 3.5
    print(f"  episode mean {edge:.1f} bps vs {rt} bps all-in -> {edge/rt:.1f}x cost "
          f"(house floor 5x)")
    print(f"  worst episode {100*v.min():+.3f}%  best {100*v.max():+.3f}%")
