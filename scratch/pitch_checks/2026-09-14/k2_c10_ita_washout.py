"""C10: ITA r21 <= 2 AND r63 <= 10, long ITA h=5..10. Round 1 + gate attribution +
reference class on 21 sector/industry ETFs (mandatory)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa
sys.path.insert(0, str(Path(__file__).resolve().parent))
from k2_common import cell, fmt, conc, null_maxk, near_high  # noqa

CLASS = ["ITA", "IHI", "IBB", "XBI", "KRE", "XME", "XRT", "XHB", "OIH", "XOP", "ITB",
         "SMH", "IYR", "XLK", "XLF", "XLV", "XLY", "XLP", "XLI", "XLU", "XLB", "XLE"]
px = close_panel(CLASS + ["SPY"])
r21 = {t: pct_rank(px[t], 21) for t in CLASS}
r63 = {t: pct_rank(px[t], 63) for t in CLASS}
mask = (r21["ITA"] <= 2) & (r63["ITA"] <= 10)
print("live ITA r21/r63:", round(r21["ITA"].iloc[-1], 1), round(r63["ITA"].iloc[-1], 1))

battery(px, mask, [("ITA", 1.0)], 10, "C10 long ITA r21<=2 & r63<=10", 5,
        variants={"r21<=5 & r63<=10": (r21["ITA"] <= 5) & (r63["ITA"] <= 10),
                  "r21<=2 & r63<=20": (r21["ITA"] <= 2) & (r63["ITA"] <= 20),
                  "r21<=1 & r63<=5": (r21["ITA"] <= 1) & (r63["ITA"] <= 5),
                  "r21<=10 & r63<=10": (r21["ITA"] <= 10) & (r63["ITA"] <= 10)},
        event_kinds=("fomc_decision",))

print("\n=== gate attribution (episodes) ===")
spy_nh = near_high(px["SPY"])
above200 = px["SPY"] > px["SPY"].rolling(200).mean()
ita200 = px["ITA"] > px["ITA"].rolling(200).mean()
for h in (5, 10):
    L = [("ITA", 1.0)]
    print(f"-- h={h}")
    print(fmt(cell(px, mask, L, h), "joint r21<=2 & r63<=10"))
    print(fmt(cell(px, r21["ITA"] <= 2, L, h), "r21<=2 alone"))
    print(fmt(cell(px, r63["ITA"] <= 10, L, h), "r63<=10 alone"))
    print(fmt(cell(px, (r21["ITA"] <= 2) & (r63["ITA"] > 10), L, h), "r21<=2 & r63>10"))
    print(fmt(cell(px, mask & spy_nh, L, h), "joint & SPY within 2% high (live)"))
    print(fmt(cell(px, mask & ~spy_nh, L, h), "joint & SPY NOT near high"))
    print(fmt(cell(px, mask & ~ita200, L, h), "joint & ITA below 200d (live)"))
    print(fmt(cell(px, mask, [("ITA", 1.0), ("SPY", -1.0)], h), "joint, ITA/SPY pair"))
    c = cell(px, mask, L, h)
    print("  ", conc(c))
    print("  era pre2018:", fmt(cell(px, mask & (px.index < "2018-01-01"), L, h), ""))
    print("  era 2018+ :", fmt(cell(px, mask & (px.index >= "2018-01-01"), L, h), ""))
    mid = pd.Series(px.index.year % 4 == 2, index=px.index)
    print("  midterm   :", fmt(cell(px, mask & mid, L, h), ""))
    print("  non-mid   :", fmt(cell(px, mask & ~mid, L, h), ""))

print("\n=== REFERENCE CLASS: same rule on every member ===")
for h in (5, 10):
    book, rel = {}, {}
    for t in CLASS:
        m = (r21[t] <= 2) & (r63[t] <= 10)
        c = cell(px, m, [(t, 1.0)], h, since="2006-06-01")
        c2 = cell(px, m, [(t, 1.0), ("SPY", -1.0)], h, since="2006-06-01")
        if c.get("n", 0) > 1:
            book[t] = c["ex"]
        if c2.get("n", 0) > 1:
            rel[t] = c2["ex"]
    print(f"\n-- h={h} (2006-06+ for all members)")
    null_maxk(book, "ITA", f"h={h} excess vs own drift")
    null_maxk(rel, "ITA", f"h={h} vs-SPY pair excess")
    allx = np.concatenate(list(book.values()))
    print(f"  pooled class excess {allx.mean():+.3f}pp over {len(allx)} episodes, "
          f"t {allx.mean()/(allx.std(ddof=1)/np.sqrt(len(allx))):+.2f}")
