"""Exact figures quoted in the brief, recomputed in one place."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa

print("USO Sep Fridays 53 down of 87 decided:", round(sign_test(53, 87), 4))
print("USO pre-2018 35 down of 50:", round(sign_test(35, 50), 4))
print("10y lower a week later 13 of 20:", round(sign_test(13, 20), 4), "| IEF higher 14 of 20:", round(sign_test(14, 20), 4))
print("next session 10y lower 10 of 21:", round(sign_test(11, 21), 4))
print("SPY higher 6 of 7:", round(sign_test(6, 7), 4))
print("HYG-near-high SPY lower a month later 11 of 18:", round(sign_test(11, 18), 4))

hyg = close_panel(["HYG"])["HYG"].dropna()
print("HYG first bar", hyg.index[0].date(), "| first fitted z date", hyg.index[504 + 6].date())

px = close_panel(["SPY", "IEF", "^TNX"])
tnx = px["^TNX"].dropna()
idx = tnx.index
bp2 = tnx.diff(2) * 100
hi = tnx >= tnx.rolling(252, min_periods=200).max() - 1e-9
trig = idx[((hi & (bp2 >= 15)).fillna(False)).values]
trig = trig[trig < idx[-1]]
epi = declusters(trig, 5, idx)
ctl = local_control(idx, epi, 126)
s5 = fwd_ret(px["SPY"].reindex(idx), 5)
v = s5.reindex(epi).dropna()
print(f"headline cell SPY h5: n {len(v)} mean {100 * v.mean():+.2f}% up {int((v > 0).sum())} local {100 * s5.reindex(ctl).mean():+.2f}%")
print("headline cell years:", pd.Series([d.year for d in epi]).value_counts().sort_index().to_dict())
