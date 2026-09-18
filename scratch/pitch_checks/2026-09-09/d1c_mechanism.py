"""C13 steps 4+5: the COMPLEMENT test (run first) and the MECHANISM test.

Step 5 is decisive: the claim is that equity vol REPRICES UP. Test the direct
object -- forward SPY realized vol over the next 5 and 10 sessions, and forward
VIX change -- conditioned on the state, against the unconditional. If forward
equity vol does not rise, the mechanism is falsified inside its own window
regardless of the return leg.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa
from pitch_lab import _valid_pct_change, rolling_on_valid
import numpy as np, pandas as pd

HERE = Path(__file__).parent
st = pd.read_parquet(HERE / "d1_state.parquet")
px = close_panel(["SPY", "QQQ", "TLT", "SVXY", "^VIX"])
st = st.reindex(px.index)
s, mv_p, rv_p = st["sprA"], st["move_pct"], st["rv_pct"]
THR = 60.0
mask = (s >= THR).fillna(False)

# ===========================================================================
# 4. COMPLEMENT TEST -- run FIRST.  Parent A: MOVE high alone (>=70 lvl pctile,
#    live 72.2).  Parent B: equity vol floor alone (rvol pctile <=10, live 6.3).
#    The conjunction is only interesting if BOTH parents are flat and the
#    intersection is not.
# ===========================================================================
pa = (mv_p >= 70).fillna(False)          # MOVE high alone
pb = (rv_p <= 10).fillna(False)          # equity-vol floor alone
both = (pa & pb)
print("day counts: MOVE>=70 %d | rvol<=10 %d | BOTH %d | spread>=60 %d"
      % (pa.sum(), pb.sum(), both.sum(), mask.sum()))

def cell(m, lbl, h, tkr="SPY"):
    r = fwd_lag(px[tkr], h, 1)
    d = px.index[m.values & r.notna().values]
    epi = declusters(pd.DatetimeIndex(d), h, r.dropna().index)
    v = r.loc[epi].values
    if len(v) < 2:
        return {"label": lbl, "n": len(v)}
    out = summarize(v, lbl)
    base = r.dropna()
    out["edge_pct"] = round(out["mean_pct"] - 100 * base.mean(), 3)
    return out

for h in (5, 10):
    rows = []
    for tkr in ("SPY", "TLT"):
        rows += [cell(pa, f"{tkr} PARENT-A MOVE>=70 only", h, tkr),
                 cell(pb, f"{tkr} PARENT-B rvol<=10 only", h, tkr),
                 cell(both, f"{tkr} BOTH (conjunction)", h, tkr),
                 cell(pa & ~pb, f"{tkr} A ex-B (complement of B)", h, tkr),
                 cell(pb & ~pa, f"{tkr} B ex-A (complement of A)", h, tkr),
                 cell(mask, f"{tkr} spread>={THR:.0f}", h, tkr),
                 cell(~mask & s.notna(), f"{tkr} spread<{THR:.0f} (complement)", h, tkr)]
    show(rows, f"4. COMPLEMENT / gate attribution, h={h} (episodes)")

# ===========================================================================
# 5. MECHANISM TEST -- does equity vol actually reprice?
#    forward realized vol over next 5/10 sessions (measured from close D+1,
#    the entry, forward), vs the unconditional; and forward VIX change.
# ===========================================================================
spy = px["SPY"].dropna()
r1 = spy.pct_change()
def fwd_rvol(h):
    # realized vol of the h returns AFTER the entry close D+1
    fr = r1.shift(-(1 + 1))  # first return after entry close is D+2 vs D+1
    vals = []
    arr = r1.values
    idx = spy.index
    out = pd.Series(np.nan, index=idx)
    for i in range(len(idx)):
        j0, j1 = i + 2, i + 2 + h
        if j1 <= len(arr):
            w = arr[j0:j1]
            if not np.isnan(w).any():
                out.iloc[i] = np.std(w, ddof=1) * np.sqrt(252) * 100
    return out

vix = px["^VIX"].dropna()
rows = []
for h in (5, 10):
    fv = fwd_rvol(h).reindex(px.index)
    d_all = px.index[mask.values & fv.notna().values]
    epi = declusters(pd.DatetimeIndex(d_all), h, fv.dropna().index)
    cur = st["rv_pct"]  # for reference
    cond = fv.loc[epi].values
    base = fv.dropna().values
    # matched control: same-era days
    span = (epi[0], epi[-1])
    insp = fv[(fv.index >= span[0]) & (fv.index <= span[1])].dropna().values
    print(f"\n5. FORWARD SPY REALIZED VOL, next {h} sessions after entry (ann %)")
    print(f"   COND spread>=60 (N={len(cond)}): mean {cond.mean():.2f}  median {np.median(cond):.2f}")
    print(f"   CTRL all days   (N={len(base)}): mean {base.mean():.2f}  median {np.median(base):.2f}")
    print(f"   CTRL same span  (N={len(insp)}): mean {insp.mean():.2f}  median {np.median(insp):.2f}")
    # the honest object: CHANGE vs the rvol at the signal (8.11 live)
    cur21 = (r1.rolling(21).std() * np.sqrt(252) * 100).reindex(px.index)
    ch = (fv - cur21)
    cc, bb = ch.loc[epi].values, ch.dropna().values
    print(f"   CHANGE vs trailing-21d rvol: COND {np.nanmean(cc):+.2f}pp  "
          f"CTRL all {np.nanmean(bb):+.2f}pp   "
          f"COND up-rate {100*np.nanmean(cc>0):.1f}%  CTRL up-rate {100*np.nanmean(bb>0):.1f}%")
    # forward VIX change
    fvx = (vix.shift(-(1 + h)) / vix.shift(-1) - 1.0).reindex(px.index)
    d2 = px.index[mask.values & fvx.notna().values]
    e2 = declusters(pd.DatetimeIndex(d2), h, fvx.dropna().index)
    show([summarize(fvx.loc[e2].values, f"VIX chg COND (N={len(e2)})"),
          summarize(fvx.dropna().values, "VIX chg CTRL all days")],
         f"   forward VIX % change, h={h}")

# ---- and the same mechanism test on the LOW-vol-floor parent for contrast --
print("\n5b. contrast: does the equity-vol FLOOR alone predict a vol rise?")
for h in (5, 10):
    fv = fwd_rvol(h).reindex(px.index)
    cur21 = (r1.rolling(21).std() * np.sqrt(252) * 100).reindex(px.index)
    ch = (fv - cur21)
    for m, lbl in [(pb, "rvol<=10 only"), (pa, "MOVE>=70 only"), (mask, "spread>=60")]:
        d = px.index[m.values & ch.notna().values]
        e = declusters(pd.DatetimeIndex(d), h, ch.dropna().index)
        print(f"   h={h} {lbl:16s} N={len(e):4d}  mean rvol change {np.nanmean(ch.loc[e].values):+.2f}pp"
              f"  up-rate {100*np.nanmean(ch.loc[e].values>0):.1f}%")
    print(f"   h={h} {'ALL DAYS':16s} N={ch.notna().sum():4d}  mean rvol change {np.nanmean(ch.dropna().values):+.2f}pp"
          f"  up-rate {100*np.nanmean(ch.dropna().values>0):.1f}%")
