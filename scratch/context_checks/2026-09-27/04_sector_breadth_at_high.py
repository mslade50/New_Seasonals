"""SPY closed 0.59% under its 52-week high while only 4 of the 9 original SPDR sectors sit above their
200-day averages (XLE, XLF, XLK, XLV; XLB, XLI, XLP, XLU, XLY below). How often does a near-high
S&P come with that little sector participation, and what did SPY do next?
Sector ETFs are the breadth condition only; the subject is SPY."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa

SECT9 = ["XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY"]
px = close_panel(["SPY", "IWM", "QQQ", "TLT"] + SECT9 + ["XLRE", "XLC"])
idx = px["SPY"].dropna().index
px = px.reindex(idx)
spy = px["SPY"]

above = pd.DataFrame({t: px[t] > px[t].rolling(200, min_periods=200).mean() for t in SECT9})
valid = pd.DataFrame({t: px[t].rolling(200, min_periods=200).mean().notna() for t in SECT9}).all(axis=1)
count = above.sum(axis=1).where(valid)
hi = spy.rolling(252, min_periods=200).max()
dist = spy / hi - 1
print("today:", str(idx[-1].date()), "sectors above 200d:", int(count.iloc[-1]), "of 9;", {t: bool(above[t].iloc[-1]) for t in SECT9},
      "SPY vs 252d high", round(100 * dist.iloc[-1], 2))
print("SPY vs 200d", round(100 * (spy.iloc[-1] / spy.tail(200).mean() - 1), 2))

near = (dist >= -0.01) & valid
print(f"\nnear-high days (within 1%): {int(near.sum())}; distribution of the sector count on them:")
print(count[near].value_counts().sort_index().to_string())
print("share of near-high days with count <= 4:", round(100 * (count[near] <= 4).mean(), 1), "%")
print("near-high days with count <= 4 by year:", count[near & (count <= 4)].groupby(count[near & (count <= 4)].index.year).size().to_dict())


def fwd_block(mask, label, gap=21):
    trig = idx[mask.fillna(False).values]
    trig = trig[trig < idx[-1]]
    epi = declusters(trig, gap, idx)
    ctl = local_control(idx, epi, 126)
    print(f"\n##### {label}: raw {len(trig)}, declustered({gap}) {len(epi)}")
    print("   dates:", [str(d.date()) for d in epi])
    out = []
    for t in ["SPY", "IWM"]:
        for h in (5, 21, 63):
            r = fwd_ret(px[t], h)
            v = r.reindex(epi).dropna()
            row = summarize(v.values, f"{t} h{h}")
            row["up"] = f"{int((v > 0).sum())}-{int((v < 0).sum())}"
            row["local"] = 100 * r.reindex(ctl).mean()
            row["all"] = 100 * r.mean()
            # the near-high control: every near-high day, declustered the same way
            out.append(row)
    show(out, label)
    for h in (21, 63):
        v = fwd_ret(spy, h).reindex(epi).dropna()
        print(f"   SPY h{h} era:", [(e["label"], e["n"], round(e.get("mean_pct", np.nan), 2), round(e.get("hit", np.nan), 1)) for e in era_split(v.index, v.values)])
        print(f"   SPY h{h} cluster:", cluster_note(v.index, v.values))
    # drawdown within 21 / 63 sessions
    for h in (21, 63):
        dd = [100 * (spy.loc[d:].iloc[1:h + 1].min() / spy.loc[d] - 1) for d in epi if len(spy.loc[d:]) > h]
        print(f"   SPY worst close within {h}: median {np.median(dd):.2f} mean {np.mean(dd):.2f} share <= -5%: {np.mean(np.array(dd) <= -5) * 100:.0f}%")
    return epi


e_narrow = fwd_block(near & (count <= 4), "SPY within 1% of 52w high, <= 4 of 9 sectors above 200d")
e_mid = fwd_block(near & (count >= 5) & (count <= 6), "SPY within 1% of 52w high, 5-6 of 9 above")
e_broad = fwd_block(near & (count >= 7), "SPY within 1% of 52w high, >= 7 of 9 above")
e_narrow3 = fwd_block(near & (count <= 3), "SPY within 1% of 52w high, <= 3 of 9 above")

# the drop in participation: count 21 sessions ago
print("\ncount 21 sessions ago:", int(count.iloc[-22]), "63 sessions ago:", int(count.iloc[-64]))
print("last 30 sessions count:", count.tail(30).astype(int).tolist())

# 11-sector version since XLC (2018-06)
S11 = SECT9 + ["XLRE", "XLC"]
ab11 = pd.DataFrame({t: px[t] > px[t].rolling(200, min_periods=200).mean() for t in S11})
v11 = pd.DataFrame({t: px[t].rolling(200, min_periods=200).mean().notna() for t in S11}).all(axis=1)
c11 = ab11.sum(axis=1).where(v11)
print("\n11-sector count today:", int(c11.iloc[-1]), "| near-high days since XLC 200d valid:", int((near & v11).sum()),
      "with <= 5 of 11:", int((near & v11 & (c11 <= 5)).sum()))
print("   dates (<=5 of 11, near high):", [str(d.date()) for d in idx[(near & v11 & (c11 <= 5)).values]][-30:])
