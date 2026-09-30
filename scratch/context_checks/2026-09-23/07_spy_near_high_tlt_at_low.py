"""TLT closed at a 52-week low (and the 10y at a 52w high) with SPY 1.05% under its 52-week high.
How often do bonds break down while stocks sit near a high, and what did SPY/IWM/TLT do next?"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa

px = close_panel(["SPY", "TLT", "IWM", "QQQ", "^TNX", "IEF"])
tlt = px["TLT"].dropna()
idx = tlt.index
spy = px["SPY"].reindex(idx)
iwm = px["IWM"].reindex(idx)
tnx = px["^TNX"].reindex(idx)
tlt_low = tlt <= tlt.rolling(252, min_periods=250).min() + 1e-9
spy_gap = spy / spy.rolling(252, min_periods=250).max() - 1
tnx_hi = tnx >= tnx.rolling(252, min_periods=250).max() - 1e-9
print("today: TLT at low", bool(tlt_low.iloc[-1]), "SPY vs 52w high", round(100 * spy_gap.iloc[-1], 2), "TNX at high", bool(tnx_hi.iloc[-1]))
first_low = tlt_low & ~tlt_low.shift(1, fill_value=False).rolling(21, min_periods=1).max().astype(bool)

def cell(mask, label, gap=21):
    trig = idx[mask.reindex(idx).fillna(False).values]
    trig = trig[trig < idx[-1]]
    epi = declusters(trig, gap, idx)
    ctl = local_control(idx, epi, 126)
    print(f"\n##### {label}: raw {len(trig)}, epi {len(epi)}")
    print("   ", [str(d.date()) for d in epi])
    rows = []
    for nm, s in (("SPY", spy), ("IWM", iwm), ("TLT", tlt)):
        for h in (1, 5, 21, 63):
            f = fwd_ret(s, h)
            e = [d for d in epi if not np.isnan(f.get(d, np.nan))]
            row = summarize(f.loc[e].values, f"{nm} h{h}")
            up = int((f.loc[e] > 0).sum())
            row["rec"] = f"{up}-{len(e) - up}"
            row["local"] = 100 * f.loc[ctl].mean()
            row["all"] = 100 * f.mean()
            rows.append(row)
    show(rows, label)
    for nm, s, h in (("SPY", spy, 21), ("SPY", spy, 63), ("TLT", tlt, 21)):
        f = fwd_ret(s, h).loc[epi].dropna()
        for part in era_split(f.index, f.values):
            print(f"   era {nm} h{h}:", part["label"], part["n"], round(part.get("mean_pct", np.nan), 2), "hit", round(part.get("hit", np.nan), 1))
        print(f"   cluster {nm} h{h}:", cluster_note(f.index, f.values))
    # max drawdown of SPY over next 21 days
    dd = []
    pos = pd.Series(range(len(idx)), index=idx)
    for d in epi:
        p = pos[d]
        path = spy.iloc[p + 1: p + 22] / spy.iloc[p] - 1
        if len(path):
            dd.append(path.min())
    print("   SPY worst close within 21d: median", round(100 * np.median(dd), 2), "share <= -3%", round(np.mean(np.array(dd) <= -0.03), 2))
    return epi

cell(tlt_low & (spy_gap >= -0.02), "TLT at a 52w low, SPY within 2% of its 52w high")
cell(tlt_low & (spy_gap >= -0.015), "TLT at a 52w low, SPY within 1.5% of its 52w high")
cell(tlt_low & (spy_gap < -0.05), "TLT at a 52w low, SPY 5%+ below its high (contrast)")
cell(tlt_low, "TLT at a 52w low, any SPY")
