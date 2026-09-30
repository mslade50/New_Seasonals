"""kB: sanity-check the panel's 252d beta against a direct OLS on the live names."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from kB_common import *  # noqa

P = build_panel()
C, beta = P["C"], P["beta"]
r = C.pct_change(fill_method=None).iloc[-252:]
for t in ["GIS", "PAYX", "CAG", "COST", "MU", "NKE", "AON", "PEP", "MCD", "LOW"]:
    x = r[["SPY", t]].dropna()
    b = np.polyfit(x["SPY"], x[t], 1)[0]
    b63 = np.polyfit(x["SPY"].iloc[-63:], x[t].iloc[-63:], 1)[0]
    print(f"{t:5s} panel beta {beta[t].iloc[-1]:+.2f}  direct 252d {b:+.2f}  direct 63d {b63:+.2f}  "
          f"corr {x.corr().iloc[0, 1]:+.2f}")
# median panel beta through history on LIQ names (should be ~0.8-1.1)
print("median LIQ beta by year:", beta[[t for t in LIQ if t in beta]].median(axis=1)
      .groupby(beta.index.year).median().round(2).to_dict())
