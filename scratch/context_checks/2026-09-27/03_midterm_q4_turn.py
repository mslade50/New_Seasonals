"""Pre-specified famous cell: the midterm-year turn into Q4. Anchor = the session before September's
final three (today's analogue, Sep 25 2026). What did the S&P do over the final three, through October,
and to year end, and how deep was the dip first? Midterm years vs the rest, 2000-2025."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa

px = close_panel(["^GSPC", "SPY", "IWM", "QQQ", "TLT"])
idx = px["^GSPC"].dropna().index
px = px.reindex(idx)
g = px["^GSPC"]
MID = {2002, 2006, 2010, 2014, 2018, 2022}

rows = []
for y in range(2000, 2026):
    sep = idx[(idx.year == y) & (idx.month == 9)]
    anc = sep[-4]
    oct_ = idx[(idx.year == y) & (idx.month == 10)]
    dec = idx[(idx.year == y) & (idx.month == 12)]
    a = g.loc[anc]
    path = g.loc[anc:oct_[-1]]
    lo_d = path.idxmin()
    yr = g[idx.year == y]
    rows.append({
        "year": y, "mid": y in MID, "anchor": str(anc.date()),
        "sep_final3": 100 * (g.loc[sep[-1]] / a - 1),
        "oct_first2": 100 * (g.loc[oct_[1]] / g.loc[sep[-1]] - 1),
        "to_oct_end": 100 * (g.loc[oct_[-1]] / a - 1),
        "to_dec_end": 100 * (g.loc[dec[-1]] / a - 1),
        "q4": 100 * (g.loc[dec[-1]] / g.loc[sep[-1]] - 1),
        "dip_by_oct_end": 100 * (path.min() / a - 1),
        "dip_date": str(lo_d.date()),
        "ytd_to_anchor": 100 * (a / g[idx.year == y - 1].iloc[-1] - 1) if y > 2000 else np.nan,
        "year_low_date": str(yr.idxmin().date()),
    })
df = pd.DataFrame(rows).round(2)
print(df.to_string(index=False))

for lab, m in [("midterm", df["mid"]), ("other", ~df["mid"])]:
    d = df[m]
    print(f"\n{lab} (n {len(d)}):")
    for c in ["sep_final3", "oct_first2", "to_oct_end", "to_dec_end", "q4", "dip_by_oct_end"]:
        v = d[c]
        print(f"   {c:14s} mean {v.mean():6.2f} median {v.median():6.2f} up {(v > 0).sum()} of {len(v)}")
up = int((df[df.mid]["to_dec_end"] > 0).sum())
print("\nmidterm to_dec_end sign p:", round(sign_test(up, int(df.mid.sum())), 4),
      "| vs other-year up-rate:", round(sign_test(up, int(df.mid.sum()), float((df[~df.mid]["to_dec_end"] > 0).mean())), 4))
q4up = int((df[df.mid]["q4"] > 0).sum())
print("midterm Q4 sign p:", round(sign_test(q4up, int(df.mid.sum())), 4))

# ex-2018
d = df[df.mid & (df.year != 2018)]
print("midterm ex-2018 to_dec_end:", d["to_dec_end"].round(2).tolist(), "dip:", d["dip_by_oct_end"].round(2).tolist())

# today's anchor position: S&P YTD and distance from high, vs the midterm analogues at their anchors
a = g.iloc[-1]
print("\n2026 anchor:", str(idx[-1].date()), "S&P YTD", round(100 * (a / g[idx.year == 2025].iloc[-1] - 1), 2),
      "vs 252d high", round(100 * (a / g.tail(252).max() - 1), 2))
for y in sorted(MID):
    anc = pd.Timestamp(df.loc[df.year == y, "anchor"].iloc[0])
    hi = g.loc[:anc].tail(252).max()
    print(f"  {y} anchor {anc.date()} vs 252d high {100 * (g.loc[anc] / hi - 1):6.2f}  YTD {df.loc[df.year == y, 'ytd_to_anchor'].iloc[0]:6.2f}")

# near-high midterm-year analogues: within 3% of the 252d high at the anchor, all years
print("\nall years with S&P within 3% of its 252d high at the Sept anchor:")
for _, r in df.iterrows():
    anc = pd.Timestamp(r["anchor"])
    hi = g.loc[:anc].tail(252).max()
    dist = 100 * (g.loc[anc] / hi - 1)
    if dist >= -3:
        print(f"  {r['year']} mid={r['mid']} dist {dist:5.2f} final3 {r['sep_final3']:6.2f} to_oct_end {r['to_oct_end']:6.2f} to_dec {r['to_dec_end']:6.2f} dip {r['dip_by_oct_end']:6.2f}")
