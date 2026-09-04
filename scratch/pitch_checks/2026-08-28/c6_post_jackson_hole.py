"""C6 -- the POST-Jackson-Hole anchor, swept across ten asset classes.

Every JH cell in the registry anchors BEFORE the speech and the offset placebo
ladder is 9-for-9 against it. Today the anchor becomes JH+0, so the POST
direction is the last live one. Anchor = the JH session date D; entry MOC at
close D+1 (repo lag=1); exit close D+1+h, h in 1..10.

Kill machinery: 21 vehicles x 10 horizons = 210 cells, so the grid is charged
against a max-|t| permutation over re-drawn late-August anchors.  Anything with
a pulse then owes (a) the placebo offset ladder k=-5..+3, (b) the midterm split
(the registry has reproduced a JH midterm inversion six times), and (c) the
August trading-day-of-month control that turned the 2026-08-13 TLT cell.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from pitch_lab import (load_prices, load_events, fwd_lag, summarize, show,  # noqa: E402
                       sign_test)

CLASSES = {
    "us_large": ["SPY", "QQQ"],
    "us_small": ["IWM"],
    "rates": ["TLT", "IEF", "^TNX"],
    "credit": ["HYG", "LQD"],
    "gold": ["GLD", "GDX"],
    "metals": ["SLV", "XME"],
    "energy": ["USO", "XLE"],
    "dollar_fx": ["UUP", "DX-Y.NYB"],
    "intl": ["EFA", "EEM", "FXI"],
    "vol": ["^VIX", "SVXY"],
}
TICKERS = [t for v in CLASSES.values() for t in v]
HS = tuple(range(1, 11))
ASOF = pd.Timestamp("2026-08-27")

px = load_prices(TICKERS)
S = {t: px[t]["Close"].dropna().loc[:ASOF] for t in px if t in px}
jh = load_events(["jackson_hole"])["date"]
print(f"jackson_hole anchors: {len(jh)}  {jh.min().date()} .. {jh.max().date()}")


def anchor_positions(idx: pd.DatetimeIndex, dates, offset: int = 0) -> list[int]:
    """Trading-day positions of each event date (+offset).

    TWO guards, both of which shipped bugs in this repo:
      (1) loc >= len(idx): an unrealised event resolves to the end of the index
          and mints a spurious recent anchor.
      (2) loc == 0 / d before the instrument's first bar: searchsorted returns 0
          for EVERY pre-inception event, so all 11 pre-2011 Jackson Holes
          collapsed onto SVXY's first sessions and reported n=26 against a real
          history of 14. Found on this run: SVXY's h=7 cell read +11.24% t=4.64
          purely because one early value was counted twelve times.
    """
    out = []
    lo, hi = idx[0], idx[-1]
    for d in pd.DatetimeIndex(dates):
        if d < lo or d > hi:         # guard (2) + (1): outside the real history
            continue
        loc = idx.searchsorted(d)
        if loc >= len(idx):
            continue
        p = loc + offset
        if 0 <= p < len(idx):
            out.append(p)
    return out


_FWD: dict = {}


def _fwd(t: str, h: int, lag: int = 1) -> pd.Series:
    key = (t, h, lag)
    if key not in _FWD:
        _FWD[key] = fwd_lag(S[t], h, lag)
    return _FWD[key]


def cell(s: pd.Series, dates, h: int, offset: int = 0, lag: int = 1, tkr=None):
    """Forward return from entry close D+offset+lag, held h sessions."""
    idx = s.index
    r = _fwd(tkr, h, lag) if tkr else fwd_lag(s, h, lag)
    pos = anchor_positions(idx, dates, offset)
    vals = [r.iloc[p] for p in pos if p < len(r) and not np.isnan(r.iloc[p])]
    return np.asarray(vals, float), [idx[p] for p in pos if p < len(r)
                                     and not np.isnan(r.iloc[p])]


# --------------------------------------------------------------------------
# 1. the full 21 x 10 grid, edge over each instrument's own all-days drift
# --------------------------------------------------------------------------
rows = []
for cls, tks in CLASSES.items():
    for t in tks:
        if t not in S:
            continue
        s = S[t]
        for h in HS:
            v, dts = cell(s, jh, h, tkr=t)
            if len(v) < 5:
                continue
            base = fwd_lag(s, h, 1).dropna()
            # control restricted to the anchor span so era is matched
            base = base.loc[base.index >= min(dts)]
            sd = v.std(ddof=1)
            t_stat = v.mean() / (sd / np.sqrt(len(v))) if sd > 0 else np.nan
            rows.append({
                "class": cls, "tkr": t, "h": h, "n": len(v),
                "mean_pct": 100 * v.mean(),
                "ctrl_pct": 100 * base.mean(),
                "edge_pct": 100 * (v.mean() - base.mean()),
                "hit": 100 * (v > 0).mean(),
                "t": t_stat,
            })
G = pd.DataFrame(rows)
print(f"\ngrid cells computed: {len(G)}  (21 vehicles x 10 horizons, short-history "
      f"vehicles trimmed)")

print("\n=== C6.1  top 12 cells by |t| across the whole grid ===")
top = G.reindex(G["t"].abs().sort_values(ascending=False).index).head(12)
print(top.round(3).to_string(index=False))

print("\n=== C6.2  per-class best cell (by |t|) ===")
best = G.loc[G.groupby("class")["t"].apply(lambda x: x.abs().idxmax())]
print(best.round(3).to_string(index=False))

print("\n=== C6.3  how many cells clear |t|>=2 vs how many you'd expect ===")
n_sig = int((G["t"].abs() >= 2).sum())
print(f"  cells with |t|>=2: {n_sig} of {len(G)} "
      f"({100*n_sig/len(G):.1f}%); iid expectation at 5% = {0.05*len(G):.1f}. "
      f"Horizons overlap heavily so the true expectation is HIGHER than iid.")

# --------------------------------------------------------------------------
# 2. multiplicity: max-|t| permutation over re-drawn late-August anchors
# --------------------------------------------------------------------------
print("\n=== C6.4  max-|t| permutation (anchors re-drawn from August sessions) ===")
rng = np.random.default_rng(42)
spy_idx = S["SPY"].index
aug = spy_idx[(spy_idx.month == 8)]
years = sorted(set(pd.DatetimeIndex(jh).year))
obs_max = float(G["t"].abs().max())
perm_max = []
N_PERM = 400
for _ in range(N_PERM):
    fake = []
    for y in years:
        cand = aug[aug.year == y]
        if len(cand) == 0:
            continue
        fake.append(cand[rng.integers(len(cand))])
    fake = pd.DatetimeIndex(fake)
    mx = 0.0
    for cls, tks in CLASSES.items():
        for t in tks:
            if t not in S:
                continue
            s = S[t]
            for h in HS:
                v, _ = cell(s, fake, h, tkr=t)
                if len(v) < 5:
                    continue
                sd = v.std(ddof=1)
                if sd > 0:
                    tt = abs(v.mean() / (sd / np.sqrt(len(v))))
                    if tt > mx:
                        mx = tt
    perm_max.append(mx)
perm_max = np.asarray(perm_max)
p_fw = float((perm_max >= obs_max).mean())
print(f"  observed max|t| over the grid = {obs_max:.2f}")
print(f"  permutation max|t|: median {np.median(perm_max):.2f}  p90 "
      f"{np.percentile(perm_max, 90):.2f}  p95 {np.percentile(perm_max, 95):.2f}")
print(f"  family-wise P(perm max >= observed) = {p_fw:.3f}  (N_PERM={N_PERM})")

# --------------------------------------------------------------------------
# 3. the placebo offset ladder on the strongest cell of each class
# --------------------------------------------------------------------------
print("\n=== C6.5  placebo offset ladder k=-5..+3 on each class's best cell ===")
for _, r in best.iterrows():
    t, h = r["tkr"], int(r["h"])
    s = S[t]
    line = []
    for k in range(-5, 4):
        v, _ = cell(s, jh, h, offset=k, tkr=t)
        if len(v) < 5:
            line.append(f"k{k:+d}:  n/a ")
            continue
        sd = v.std(ddof=1)
        tt = v.mean() / (sd / np.sqrt(len(v))) if sd > 0 else np.nan
        star = "*" if k == 0 else " "
        line.append(f"k{k:+d}{star}{100*v.mean():+6.2f}%(t{tt:+4.1f})")
    print(f"  {r['class']:<9} {t:<9} h={h:<2} " + " ".join(line))

# --------------------------------------------------------------------------
# 4. midterm split (registry: JH midterm inversion reproduced 6x)
# --------------------------------------------------------------------------
print("\n=== C6.6  midterm split on each class's best cell (year%4==2) ===")
for _, r in best.iterrows():
    t, h = r["tkr"], int(r["h"])
    v, dts = cell(S[t], jh, h, tkr=t)
    yrs = pd.DatetimeIndex(dts).year
    mid = (yrs % 4 == 2)
    a = summarize(v[mid], "midterm")
    b = summarize(v[~mid], "non-mid")
    print(f"  {r['class']:<9} {t:<6} h={h:<2}  midterm n={a.get('n',0)} "
          f"{a.get('mean_pct', float('nan')):+.2f}%  |  non-mid n={b.get('n',0)} "
          f"{b.get('mean_pct', float('nan')):+.2f}%   INVERTED="
          f"{bool(a.get('n',0) and b.get('n',0) and np.sign(a['mean_pct']) != np.sign(b['mean_pct']))}")

# --------------------------------------------------------------------------
# 5. August trading-day-of-month control (what turned the JH TLT cell)
# --------------------------------------------------------------------------
print("\n=== C6.7  August trading-day-of-month control on each class's best cell ===")
print("  (JH sits at a fixed late-August position; if neighbouring August tdom")
print("   positions pay the same, the label 'Jackson Hole' is doing no work.)")
for _, r in best.iterrows():
    t, h = r["tkr"], int(r["h"])
    s = S[t]
    idx = s.index
    tdom = pd.Series(0, index=idx)
    for (y, m), g in pd.Series(idx, index=idx).groupby([idx.year, idx.month]):
        tdom.loc[g.index] = np.arange(1, len(g) + 1)
    jh_pos = []
    for d in pd.DatetimeIndex(jh):
        loc = idx.searchsorted(d)
        if loc >= len(idx):
            continue
        jh_pos.append(int(tdom.iloc[loc]))
    med = int(np.median(jh_pos))
    ret = fwd_lag(s, h, 1)
    aug_mask = (idx.month == 8)
    cells = []
    for k in range(med - 3, med + 4):
        m = aug_mask & (tdom.values == k)
        v = ret[m].dropna().values
        if len(v) < 5:
            continue
        cells.append((k, 100 * v.mean(), len(v)))
    txt = "  ".join(f"td{k}:{mu:+.2f}%(n{n})" for k, mu, n in cells)
    v0, _ = cell(s, jh, h, tkr=t)
    print(f"  {r['class']:<9} {t:<6} h={h:<2} JH tdom median={med}  "
          f"JH cell {100*v0.mean():+.2f}%")
    print(f"      all-August by tdom: {txt}")

print("\n=== C6.8  sign records on each class's best cell ===")
for _, r in best.iterrows():
    v, _ = cell(S[r["tkr"]], jh, int(r["h"]), tkr=r["tkr"])
    w = int((v > 0).sum())
    lo = min(w, len(v) - w)
    hi = max(w, len(v) - w)
    print(f"  {r['class']:<9} {r['tkr']:<6} h={int(r['h']):<2} record {w}-{len(v)-w} "
          f"two-sided sign p = {2*sign_test(hi, len(v)):.3f}")
