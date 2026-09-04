"""Year-clustered book dead-zone check + combined book+sleeve in dead windows."""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats as sps

ROOT = Path(__file__).resolve().parents[2]
s = pd.read_parquet(ROOT / "scratch" / "ultracode_research" / "gate_ab_series.parquet")
s = s.dropna(subset=["book_pct"])  # 2003-01..2026-06

# --- book by year, midterm vs other (year is the cluster unit) ---
byr_r = s["book_R"].groupby(s.index.year).sum()
byr_p = s["book_pct"].groupby(s.index.year).sum()
mid = (byr_r.index % 4) == 2
print("Book per-year total R (midterm marked *):")
for y in byr_r.index:
    print(f"  {y}{'*' if y % 4 == 2 else ' '}: {byr_r[y]:+7.1f}R   {byr_p[y]*100:+7.1f}% of 750k")
t, p = sps.ttest_ind(byr_r[mid], byr_r[~mid], equal_var=False)
print(f"midterm yrs (N={mid.sum()}) avg {byr_r[mid].mean():+.1f}R/yr vs other (N={(~mid).sum()}) "
      f"{byr_r[~mid].mean():+.1f}R/yr  Welch t={t:+.2f} p={p:.3f}")

# book Jul-Sep, year-clustered: per-year Jul-Sep sum R (26 clusters)
jas = s.index.month.isin([7, 8, 9])
jas_yr = s.loc[jas, "book_R"].groupby(s.index[jas].year).sum()
oth_yr = s.loc[~jas, "book_R"].groupby(s.index[~jas].year).sum() / 3.0  # per-quarter-ish
print(f"\nBook Jul-Sep quarter sum R: avg {jas_yr.mean():+.2f}R vs avg other-quarter {oth_yr.mean():+.2f}R "
      f"(N={len(jas_yr)} years; paired t on year rows: ", end="")
common = jas_yr.index.intersection(oth_yr.index)
d = jas_yr[common] - oth_yr[common]
print(f"t={d.mean()/d.std()*np.sqrt(len(d)):+.2f}, N={len(d)})")

# --- combined book + 1x ex-bonds sleeve in the dead windows (% of 750k) ---
print("\nCombined book + 1.0x EXBONDS sleeve (next-open, net), % of 750k NAV:")
comb = s["book_pct"] + s["exb_open"]
masks = {
    "Jul-Sep": pd.Series(s.index.month.isin([7, 8, 9]), s.index),
    "midterm yrs": pd.Series((s.index.year % 4) == 2, s.index),
    "Jul-Sep x midterm": pd.Series(s.index.month.isin([7, 8, 9]) & ((s.index.year % 4) == 2), s.index),
}
for lbl, m in masks.items():
    b_in, c_in = s.loc[m, "book_pct"], comb[m]
    print(f"  {lbl:<20} N={m.sum():>3}  book {b_in.mean()*100:+.3f}%/mo (hit {(b_in>0).mean()*100:.0f}%)"
          f" -> combined {c_in.mean()*100:+.3f}%/mo (hit {(c_in>0).mean()*100:.0f}%)"
          f"   sleeve adds {s.loc[m,'exb_open'].mean()*100:+.3f}%")

# sleeve dollar fill of the intersection shortfall at 1.0x
short = s.loc[~masks["Jul-Sep x midterm"], "book_pct"].mean() - s.loc[masks["Jul-Sep x midterm"], "book_pct"].mean()
add = s.loc[masks["Jul-Sep x midterm"], "exb_open"].mean()
print(f"\nIntersection: book shortfall vs other months {short*100:.3f}%/mo (~${short*750000:,.0f}/mo); "
      f"sleeve adds {add*100:+.3f}%/mo (~${add*750000:,.0f}/mo) at 1.0x -> fills {add/short*100:.0f}%")

# sleeve corr to book inside dead windows
for lbl, m in masks.items():
    c = s.loc[m, ["exb_open", "book_pct"]].corr().iloc[0, 1]
    print(f"  corr(sleeve, book) in {lbl}: {c:+.3f} (N={m.sum()})")
