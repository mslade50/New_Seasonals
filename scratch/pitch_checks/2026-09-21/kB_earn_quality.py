"""kB: earnings-calendar quality check before using it as an anchor.
Share of rows landing exactly on a calendar quarter END (fiscal period end
masquerading as an announcement), weekend share, per year 1996+.
Also spot-check NKE / GIS / PAYX / CAG / COST history against price gaps:
the absolute return on the session AFTER date (BMO -> date session, AMC ->
next session) should be large if the date is the real announcement."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa
import numpy as np
import pandas as pd

e = pd.read_parquet(ROOT / "data" / "earnings_calendar.parquet")
e = e[e.date >= "1996-01-01"].copy()
e["qend"] = e.date.dt.is_quarter_end
e["wkend"] = e.date.dt.weekday >= 5
by = e.groupby(e.date.dt.year).agg(n=("ticker", "size"), qend=("qend", "mean"),
                                    wkend=("wkend", "mean"))
print("share of rows on calendar quarter-end / weekend, by year")
print((by.assign(qend=lambda d: (100 * d.qend).round(1),
                 wkend=lambda d: (100 * d.wkend).round(1))).to_string())

# per-ticker: share of quarter-end rows 2010+
x = e[e.date >= "2010-01-01"].groupby("ticker").qend.mean().sort_values(ascending=False)
print("\nTickers with >50% quarter-end dated rows 2010+:", int((x > 0.5).sum()),
      "of", len(x))
print(x.head(25).round(2).to_dict())

# gap test: for each row, max(|ret on date session|, |ret on next session|)
# vs the name's median |daily ret|; real announcements show a spike.
names = ["NKE", "GIS", "PAYX", "CAG", "COST", "MU", "AAPL", "MSFT", "JPM", "PEP"]
px = load_prices(names)
for t in names:
    if t not in px:
        continue
    c = px[t]["Close"]
    r = c.pct_change().abs()
    med = r.rolling(252).median()
    rows = e[(e.ticker == t) & (e.date >= "2005-01-01") & (e.date <= "2026-09-18")]
    idx = c.index
    spikes = []
    for d in rows.date:
        p = idx.searchsorted(d)
        if p + 1 >= len(idx) or p < 1:
            continue
        s = max(r.iloc[p], r.iloc[p + 1]) / med.iloc[p] if med.iloc[p] > 0 else np.nan
        spikes.append((d.date(), round(s, 1), bool(d.is_quarter_end)))
    ratio = np.array([s[1] for s in spikes], float)
    qe = np.array([s[2] for s in spikes])
    print(f"\n{t}: rows {len(spikes)}, median spike ratio {np.nanmedian(ratio):.1f}, "
          f"quarter-end rows {qe.sum()} (median ratio there "
          f"{np.nanmedian(ratio[qe]) if qe.any() else float('nan'):.1f})")
    print("  last 8:", spikes[-8:])
