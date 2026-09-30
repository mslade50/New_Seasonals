import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

REF23 = ["XLE", "VNQ", "SMH", "IYR", "XHB", "XLI", "KRE", "XME", "XLF", "XOP",
         "QQQ", "XRT", "ITB", "OIH", "FXI", "ITA", "XLB", "IBB", "IWM", "XBI",
         "EEM", "GDX", "XLY"]
raw = load_prices(REF23 + ["SPY", "TLT", "USDMXN=X", "^MOVE", "^VIX"])
C = {t: raw[t]["Close"].dropna() for t in raw}

# [11] joint state SPY within 0.5% of 252 high, TLT within 1% of 252 low
sp = C["SPY"] / C["SPY"].rolling(252).max() - 1
tl = C["TLT"] / C["TLT"].rolling(252).min() - 1
j = pd.DataFrame({"spy": sp, "tlt": tl}).dropna()
print("[11] tail:\n", (100 * j[j.index >= "2026-08-12"]).round(2).to_string())

# [66] USDMXN
fx = C["USDMXN=X"]
mv, vx = C["^MOVE"].pct_change(), C["^VIX"].pct_change()
for d in fx.index[-5:]:
    print("[66] USDMXN %s 1d %+.3f%% | MOVE 1d %+.2f%% VIX 1d %+.2f%%" % (
        d.date(), 100 * (fx.loc[d] / fx.shift(1).loc[d] - 1), 100 * mv.get(d, np.nan), 100 * vx.get(d, np.nan)))

# [24] SMH family cell C = r63<=5 & r252>=40% & r5<15, gap 10, signals after 2026-09-15
new = []
for t in REF23:
    s = C[t]
    m = (pct_rank(s, 63) <= 5) & (s / s.shift(252) - 1 >= 0.40) & (pct_rank(s, 5) < 15)
    d = s.index[m.fillna(False).values]
    e = declusters(d, 10, s.index)
    new += [(t, str(x.date())) for x in e if x > pd.Timestamp("2026-09-15")]
    if t == "SMH" or len([x for x in e if x > pd.Timestamp("2026-09-15")]):
        pass
print("[24] cell-C episodes signalled after 09-15:", new)
live = [(t, round(last, 1)) for t in REF23
        for last in [pct_rank(C[t], 63).iloc[-1]] if last <= 5 and C[t].iloc[-1] / C[t].iloc[-253] - 1 >= 0.40]
print("[24] names at r63<=5 & 252d>=40% today:", live)
