"""C13 steps 2+3: dose response across the spread, and forward legs at
h=1,2,3,5,10 for SPY QQQ IWM TLT IEF SVXY. Day-level AND declustered episodes.
Both directions are just the sign of the mean, reported once."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa
from pitch_lab import _valid_pct_change, rolling_on_valid
import numpy as np, pandas as pd

HERE = Path(__file__).parent
st = pd.read_parquet(HERE / "d1_state.parquet")
TK = ["SPY", "QQQ", "IWM", "TLT", "IEF", "SVXY"]
px = close_panel(TK + ["^VIX"])
st = st.reindex(px.index)

SPR = "sprA"
s = st[SPR]
LIVE = 65.9

# ---- 2. dose response, fixed buckets over the whole spread range -----------
BINS = [-999, -40, -20, 0, 20, 40, 60, 999]
lab = [f"[{BINS[i]},{BINS[i+1]})" for i in range(len(BINS) - 1)]
buck = pd.cut(s, BINS, right=False, labels=lab)

print("=" * 78)
print("2. DOSE RESPONSE  spread = MOVE lvl pctile - SPY 21d rvol lvl pctile")
print("   live spread 65.9 -> bucket", buck.dropna().iloc[-1])
for h in (5, 10):
    rows = []
    for L in lab:
        m = (buck == L).values
        dts = px.index[m & s.notna().values]
        for tkr in ["SPY", "TLT", "SVXY"]:
            r = fwd_lag(px[tkr], h, 1)
            d = pd.DatetimeIndex(dts).intersection(r.dropna().index)
            epi = declusters(d, h, r.dropna().index)
            v = r.loc[epi].values
            if len(v) == 0:
                continue
            rr = summarize(v, f"{tkr} {L}")
            rr["n_days"] = len(d)
            rows.append(rr)
    show([r for r in rows if r["label"].startswith("SPY")], f"SPY h={h} by bucket (episodes)")
    show([r for r in rows if r["label"].startswith("TLT")], f"TLT h={h} by bucket (episodes)")
    show([r for r in rows if r["label"].startswith("SVXY")], f"SVXY h={h} by bucket (episodes)")

# ---- 3. forward legs on the LIVE cell -------------------------------------
# live cell definition (pre-specified from the live reading, not scanned):
# spread >= 60  (live 65.9, top bucket)
THR = 60.0
mask = (s >= THR)
print("\n" + "=" * 78)
print(f"3. LIVE CELL  spread >= {THR}   day-level N = {int(mask.sum())}")
dts_all = px.index[mask.fillna(False).values]
print("   span", dts_all[0].date(), "..", dts_all[-1].date())
print("   years:", dict(pd.Series(1, index=dts_all).groupby(dts_all.year).sum()))

for h in (1, 2, 3, 5, 10):
    rows = []
    for tkr in TK:
        r = fwd_lag(px[tkr], h, 1)
        d = pd.DatetimeIndex(dts_all).intersection(r.dropna().index)
        epi = declusters(d, h, r.dropna().index)
        v = r.loc[epi].values
        if len(v) < 2:
            rows.append({"label": f"{tkr}", "n": len(v)})
            continue
        rr = summarize(v, tkr)
        base = r.dropna()
        rr["ctrl_all_pct"] = round(100 * base.mean(), 3)
        rr["edge_pct"] = round(rr["mean_pct"] - 100 * base.mean(), 3)
        rr["signp"] = round(sign_test(int((v > 0).sum()), len(v)), 4)
        rows.append(rr)
    show(rows, f"h={h} episodes, spread>={THR}")
