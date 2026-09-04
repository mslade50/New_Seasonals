import numpy as np
import pandas as pd
from scipy import stats

ROOT = r"C:\Users\McKinley Slade\dev\New_Seasonals"
j = pd.read_parquet(ROOT + r"\scratch\ultracode_research\proxy_joined.parquet")
j["Signal Date"] = pd.to_datetime(j["Signal Date"])
j["yr"] = j["Signal Date"].dt.year
fh = j["frag"] >= 50
hi = j[fh].sort_values("rv21_pct", ascending=False)

# top-45 rv within frag-hi
top45 = hi.head(45)
print("frag-hi top-45 by rv21_pct: min rv =", round(top45["rv21_pct"].min(), 3),
      "avgR =", round(top45["R_Multiple"].mean(), 3),
      "win% =", round((top45["R_Multiple"] > 0).mean() * 100),
      "years:", top45["yr"].value_counts().to_dict())
thr = top45["rv21_pct"].min()
lo_above = ((~fh) & (j["rv21_pct"] >= thr)).sum()
print(f"frag-lo trades with rv >= {thr:.3f}: {lo_above} (their table says 200)")

# sweep thresholds: which thr gives cells (694,200,197,45)?
for t in np.arange(0.30, 0.80, 0.025):
    e = j["rv21_pct"] >= t
    print(f"thr={t:.3f}: cells calm-lo={((~fh) & ~e).sum():4d} elev-lo={((~fh) & e).sum():4d} "
          f"calm-hi={(fh & ~e).sum():4d} elev-hi={(fh & e).sum():3d} "
          f"avgR(elev-hi)={j.loc[fh & e, 'R_Multiple'].mean():+.3f}")
