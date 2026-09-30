import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

BANKS = ["JPM", "BAC", "C", "WFC", "GS", "MS", "USB", "KEY", "RF", "STT", "SCHW"]
ENERGY = ["XLE", "XOP", "USO", "COP", "CVX", "VLO", "OXY", "SLB", "EOG", "HAL", "WMB"]
FAM29 = ["SPY", "QQQ", "IWM", "DIA", "EFA", "EEM", "EWJ", "FXI", "EWZ",
         "XLK", "XLV", "XLF", "XLI", "XLY", "XLP", "XLU", "XLB", "XLRE", "XLC",
         "SMH", "XBI", "IBB", "KRE", "IHI", "ITB", "XME", "XLE", "XOP", "OIH"]
FAM20 = ["XLK", "XLF", "XLV", "XLY", "XLP", "XLI", "XLE", "XLU", "XLB",
         "SMH", "IBB", "XBI", "IHI", "KRE", "ITA", "XME", "XRT", "XHB", "IYR", "OIH"]
SPDR9 = ["XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY"]
OTHER = ["TLT", "IEF", "LQD", "HYG", "^TNX", "DX-Y.NYB", "UUP", "GLD", "SLV", "GDX",
         "CL=F", "UNG", "NG=F", "^VIX", "^VIX3M", "^MOVE", "^SKEW", "DBC", "MXN=X", "SVXY"]
TK = sorted(set(BANKS + ENERGY + FAM29 + FAM20 + SPDR9 + OTHER))
raw = load_prices(TK)
C = {t: raw[t]["Close"].dropna() for t in raw}
V = {t: raw[t]["Volume"].dropna() for t in raw if "Volume" in raw[t]}
T = pd.Timestamp("2026-09-28")


def last(s, d=T):
    s = s.dropna()
    s = s[s.index <= d]
    return s.iloc[-1] if len(s) else np.nan


def r(t, n, d=T):
    return last(pct_rank(C[t], n), d) if t in C else np.nan


def d1(t, d=T):
    s = C[t][C[t].index <= d]
    return 100 * (s.iloc[-1] / s.iloc[-2] - 1)


def off_hi(t, d=T):
    s = C[t][C[t].index <= d]
    return 100 * (s.iloc[-1] / s.iloc[-252:].max() - 1)


def off_lo(t, d=T):
    s = C[t][C[t].index <= d]
    return 100 * (s.iloc[-1] / s.iloc[-252:].min() - 1)


print("stale tickers (last bar != 09-28):",
      {t: str(C[t].index[-1].date()) for t in C if C[t].index[-1] != T})
print("\n[3] GLD r5 %.1f r63 %.1f off-hi %.2f%% | GDX r5 %.1f" % (r("GLD", 5), r("GLD", 63), off_hi("GLD"), r("GDX", 5)))
print("[8] IHI r21 %.1f off-hi %.2f%%" % (r("IHI", 21), off_hi("IHI")))
e5 = 100 * (C["EEM"].iloc[-1] / C["EEM"].iloc[-6] - 1)
print("[9] FXI r5 %.1f r21 %.1f | EEM 5d %+.2f%%" % (r("FXI", 5), r("FXI", 21), e5))
print("[12] VIX pct_rank21 %.1f, VIX 1d %+.2f%%, SPY 1d %+.2f%%" % (r("^VIX", 21), d1("^VIX"), d1("SPY")))
tnx = C["^TNX"]
print("[13] TNX 21-session change %+.3f pt (lvl %.3f) | DX r21 %.1f | TNX r21 %.1f" % (
    tnx.iloc[-1] - tnx.iloc[-22], tnx.iloc[-1], r("DX-Y.NYB", 21), r("^TNX", 21)))
atr = pd.Series(wilder_atr(raw["SPY"]["High"], raw["SPY"]["Low"], raw["SPY"]["Close"]),
                index=raw["SPY"].index).dropna()
print("[14] XLV1d - XLK1d = %+.2fpp | SPY off-hi %.2f%% | SPY ATR14/px %.2f%%" % (
    d1("XLV") - d1("XLK"), off_hi("SPY"), 100 * atr.iloc[-1] / C["SPY"].iloc[-1]))
R5b = pd.DataFrame({t: pct_rank(C[t], 5) for t in BANKS if t in C})
R63b = pd.DataFrame({t: pct_rank(C[t], 63) for t in BANKS if t in C})
row5, row63 = R5b.loc[T], R63b.loc[T]
print("[17] bank breadth r5<=20: %d/%d = %.0f%% | median r63 %.1f" % (
    (row5 <= 20).sum(), row5.notna().sum(), 100 * (row5 <= 20).sum() / row5.notna().sum(), row63.median()))

# [18] TNX dose episodes
hi = rolling_on_valid(tnx, lambda x: x.rolling(252).max())
chg = (tnx - tnx.shift(252)) * 100
trig = tnx.index[((tnx / hi - 1) >= -0.0025) & (chg >= 78)]
eps = declusters(trig, 10, tnx.index)
print("[18] TNX trig days since 09-01:", [str(d.date()) for d in trig if d >= pd.Timestamp("2026-09-01")])
print("     episodes since 08-01:", [str(d.date()) for d in eps if d >= pd.Timestamp("2026-08-01")],
      "| chg252 %.1f bp | off 252max %.3f%%" % (chg.iloc[-1], 100 * (tnx.iloc[-1] / hi.iloc[-1] - 1)))
cp = pd.DataFrame({"IEF": C["IEF"], "TLT": C["TLT"]}).dropna()
curve = vehicle_ret(cp, [("IEF", 1.0), ("TLT", -0.523)], 8)
for d in [x for x in eps if x >= pd.Timestamp("2026-09-01")]:
    print("     ep %s h=8 curve %s bp" % (d.date(), "%.1f" % (1e4 * curve.get(d)) if pd.notna(curve.get(d)) else "open"))
s = pd.Series(1.0, index=cp.index)
e1 = pd.Timestamp("2026-09-10")
if e1 in cp.index:
    i = cp.index.get_loc(e1)
    x = cp.iloc[i:i + 9]
    print("     ep1 manual 09-10->%s: %.1f bp" % (x.index[-1].date(), 1e4 * ((x.IEF.iloc[-1] / x.IEF.iloc[0] - 1) - 0.523 * (x.TLT.iloc[-1] / x.TLT.iloc[0] - 1))))

Z = {t: zscore(C[t], 10) for t in ENERGY if t in C}
zrow = {t: last(Z[t]) for t in Z}
print("[19] energy z10>=2 count: %d  (%s)" % (sum(v >= 2 for v in zrow.values()),
      ", ".join(f"{k} {v:.2f}" for k, v in sorted(zrow.items(), key=lambda kv: -kv[1])[:4])))
print("[21] SPDR r5<=5 & within 5%% of hi: %s" % [
    (t, round(r(t, 5), 1), round(off_hi(t), 2)) for t in SPDR9 if r(t, 5) <= 5 and off_hi(t) >= -5])
print("     SPDR min r5: %s" % sorted([(round(r(t, 5), 1), t) for t in SPDR9])[:3])
ret252 = 100 * (C["SMH"].iloc[-1] / C["SMH"].iloc[-253] - 1)
print("[24] SMH r63 %.1f r5 %.1f 252d %+.1f%%" % (r("SMH", 63), r("SMH", 5), ret252))
j27 = [(t, round(r(t, 21), 1), round(r(t, 63), 1), round(r(t, 5), 1)) for t in FAM29
       if t in C and r(t, 21) >= 90 and r(t, 63) <= 10]
print("[27] FAM29 holders of r21>=90 & r63<=10 (t, r21, r63, r5):", j27)
print("[28] 1d GLD/SLV/GDX last 8 sessions (F = faithful break, each <= -2%):")
m3 = pd.DataFrame({t: C[t].pct_change() * 100 for t in ["GLD", "SLV", "GDX"]}).dropna()
faith = (m3 <= -2).all(axis=1)
for d in m3.index[-8:]:
    print("     %s %+.2f %+.2f %+.2f %s" % (d.date(), *m3.loc[d].values, "F" if faith[d] else ""))
prior5 = faith.iloc[-6:-1]
print("     today faithful=%s | faithful breaks in prior 5 sessions: %d -> FIRST break=%s" % (
    faith.iloc[-1], prior5.sum(), bool(faith.iloc[-1] and prior5.sum() == 0)))
fb = faith & ~faith.shift(1).rolling(5, min_periods=1).max().fillna(0).astype(bool)
print("     first breaks since 2026-01-01:", [str(d.date()) for d in fb.index[fb.values] if d >= pd.Timestamp("2026-01-01")])
mv = C["^MOVE"]
mp = 100 * (mv.iloc[-252:] <= mv.iloc[-1]).mean()
print("[29] MOVE %.2f trailing-252 level pctile %.1f | TNX off-hi %.3f%%" % (mv.iloc[-1], mp, 100 * (tnx.iloc[-1] / hi.iloc[-1] - 1)))
print("[31] XLE off-hi %.2f%% | SPY 1d %+.2f%%" % (off_hi("XLE"), d1("SPY")))
print("[40] DBC off-hi %.2f%%" % off_hi("DBC"))
ev = load_events()
ppi = ev[(ev.event == "ppi") & (ev.date >= "2026-09-01") & (ev.date <= "2026-10-31")]
print("[42] PPI dates:", [str(d.date()) for d in ppi.date])
for d in ppi.date:
    if d <= T:
        tl = C["TLT"]
        i = tl.index.get_loc(d)
        th = hi.get(d)
        print("     PPI %s TNX %.3f vs 252max %.3f (off %.3f%%); TLT %s->%s h=3 %+.2f%%" % (
            d.date(), tnx.get(d), th, 100 * (tnx.get(d) / th - 1), d.date(),
            tl.index[min(i + 3, len(tl) - 1)].date(), 100 * (tl.iloc[min(i + 3, len(tl) - 1)] / tl.iloc[i] - 1)))
fo = set(ev[ev.event == "fomc_decision"].date)
vx = set(ev[ev.event == "vix_expiry"].date)
col = sorted(d for d in fo & vx if pd.Timestamp("2026-01-01") <= d <= pd.Timestamp("2027-12-31"))
print("[44/46] FOMC x VIX-expiry collisions 2026-27:", [str(d.date()) for d in col])
for d in col:
    if d <= T and d in C["SVXY"].index:
        print("     SVXY on %s: %+.2f%%" % (d.date(), d1("SVXY", d)))
j47 = [(t, round(r(t, 5), 1), round(r(t, 63), 1)) for t in FAM20 if t in C and r(t, 5) <= 2 and r(t, 63) >= 90]
print("[47] FAM20 r5<=2 & r63>=90:", j47, "| near (r5<=5 & r63>=85):",
      [(t, round(r(t, 5), 1), round(r(t, 63), 1)) for t in FAM20 if t in C and r(t, 5) <= 5 and r(t, 63) >= 85])
print("[48] r5 XLV %.1f IBB %.1f XBI %.1f IHI %.1f" % tuple(r(t, 5) for t in ["XLV", "IBB", "XBI", "IHI"]))
print("[53] HYG pitch_lab z10 %.2f | IEF r5 %.1f | HYG off-hi %.2f%%" % (last(zscore(C["HYG"], 10)), r("IEF", 5), off_hi("HYG")))
print("[54] DX r5 %.1f" % r("DX-Y.NYB", 5))
xr, tr = pct_rank(C["XLU"], 21), pct_rank(C["TLT"], 21)
jt = pd.concat([xr, tr], axis=1).dropna()
jt = jt[(jt.iloc[:, 0] <= 5) & (jt.iloc[:, 1] < 25)]
print("[61] XLU r21 %.1f TLT r21 %.1f | joint days since 09-10: %s" % (
    last(xr), last(tr), [str(d.date()) for d in jt.index if d >= pd.Timestamp("2026-09-10")]))
u5, u21 = pct_rank(C["USO"], 5), pct_rank(C["USO"], 21)
print("[63] USO r5 %.1f r21 %.1f | CL=F r5 %.1f | USO r21>=90 dates since 09-01: %s" % (
    last(u5), last(u21), r("CL=F", 5), [str(d.date()) for d in u21.dropna().index if d >= pd.Timestamp("2026-09-01") and u21[d] >= 90]))
mvc = mv.pct_change().dropna()
m18 = mvc[mvc.index >= "2018-03-01"]
print("[64] MOVE 1d %+.3f%% (pctile of 2018-03+ daily moves %.2f; 90th %+.3f%% 97th %+.3f%%) | SPY 1d %+.3f%%" % (
    100 * mvc.iloc[-1], 100 * (m18 <= mvc.iloc[-1]).mean(), 100 * m18.quantile(.9), 100 * m18.quantile(.97), d1("SPY")))
m18x = m18.iloc[:-1]
print("     ex-today: pctile %.2f, 90th %+.3f%% | MOVE last 3 closes %s | last bar %s" % (
    100 * (m18x <= mvc.iloc[-1]).mean(), 100 * m18x.quantile(.9), mv.iloc[-3:].round(2).tolist(), mv.index[-1].date()))
vr = V["TLT"].iloc[-1] / V["TLT"].iloc[-64:-1].mean()
print("[65] TLT 1d %+.2f%% vol %.2fx 63d | off-lo %.2f%% | MOVE 1d %+.2f%%" % (d1("TLT"), vr, off_lo("TLT"), 100 * mvc.iloc[-1]))
if "MXN=X" in C:
    for d in C["MXN=X"].index[-4:]:
        print("[66] USDMXN %s 1d %+.3f%%" % (d.date(), d1("MXN=X", d)))
for d in C["UNG"].index[-4:]:
    i = V["UNG"].index.get_loc(d)
    print("[67] UNG %s 1d %+.2f%% vol %.2fx 63d" % (d.date(), d1("UNG", d), V["UNG"].iloc[i] / V["UNG"].iloc[i - 63:i].mean()))
sp_hi = C["SPY"] / rolling_on_valid(C["SPY"], lambda x: x.rolling(252).max()) - 1
tl_lo = C["TLT"] / rolling_on_valid(C["TLT"], lambda x: x.rolling(252).min()) - 1
j11 = pd.concat([sp_hi, tl_lo], axis=1).dropna()
j11 = j11[(j11.iloc[:, 0] >= -0.005) & (j11.iloc[:, 1] <= 0.01)]
print("[11] SPY off-hi %.2f%% TLT off-lo %.2f%% | joint days since 08-17: %s" % (
    off_hi("SPY"), off_lo("TLT"), [str(d.date()) for d in j11.index if d >= pd.Timestamp("2026-08-17")]))
print("[5/25] off-lo TLT %.2f IEF %.2f LQD %.2f | HYG off-hi %.2f" % (off_lo("TLT"), off_lo("IEF"), off_lo("LQD"), off_hi("HYG")))
print("[50] VIX/VIX3M %.3f" % (C["^VIX"].iloc[-1] / C["^VIX3M"].iloc[-1]) if "^VIX3M" in C else "[50] ^VIX3M missing")
print("[32/34] VIX rel-range pctile (d1b relpct) %.1f" % last(relpct := (lambda v: rolling_on_valid(
    (rolling_on_valid(v, lambda x: x.rolling(21).max()) - rolling_on_valid(v, lambda x: x.rolling(21).min()))
    / rolling_on_valid(v, lambda x: x.rolling(21).mean()), lambda x: x.rolling(252).rank(pct=True) * 100))(C["^VIX"])))
print("[6] SKEW r5 %.1f | [33/59] SPY vs 200d %+.2f%%" % (r("^SKEW", 5), 100 * (C["SPY"].iloc[-1] / C["SPY"].iloc[-200:].mean() - 1)))
print("[20] SPY off-hi %.2f%%; [4] USO 1d %+.2f%%; [7] USO r5 %.1f r63 %.1f; [16] TLT 1d %+.2f%%" % (
    off_hi("SPY"), d1("USO"), r("USO", 5), r("USO", 63), d1("TLT")))

# ---- added 2026-09-29
print("[12] VIX r21 %.1f | VIX 1d %+.2f%% | SPY 1d %+.3f%% (needs r21<=25, VIX>=+5%%, SPY down < 0.75%%)" % (
    r("^VIX", 21), d1("^VIX"), d1("SPY")))
if len(eps) and eps[-1] >= pd.Timestamp("2026-09-15"):
    i0 = cp.index.get_loc(eps[-1]) + 1
    x = cp.iloc[i0:]
    print("[18] ep %s running mark %s->%s: %.1f bp" % (eps[-1].date(), x.index[0].date(), x.index[-1].date(),
          1e4 * ((x.IEF.iloc[-1] / x.IEF.iloc[0] - 1) - 0.523 * (x.TLT.iloc[-1] / x.TLT.iloc[0] - 1))))
px61 = pd.DataFrame({"XLU": C["XLU"], "TLT": C["TLT"]}).dropna()
w61 = pct_rank(px61["XLU"], 21) <= 5
j61 = w61 & (pct_rank(px61["TLT"], 21) < 25)
x5 = vehicle_ret(px61, [("XLU", 1.0)], 5)
for lbl, m in [("joint", j61), ("XLU-only washout", w61)]:
    e = declusters(px61.index[m.fillna(False).values], 21, px61.index)
    print("[61] %s episodes since 2026-08-01: %s" % (lbl, [(str(d.date()), "%+.2f%%" % (100 * x5.get(d)) if pd.notna(x5.get(d)) else "open")
          for d in e if d >= pd.Timestamp("2026-08-01")]))
print("[61] joint days since 09-15: %s" % [str(d.date()) for d in j61.index[j61.fillna(False).values] if d >= pd.Timestamp("2026-09-15")])
dx, uup = C["DX-Y.NYB"], C["UUP"]
print("[68] TNX %.3f vs 252max %.3f (off %.3f%%) | DX 09-28 close %.3f (1d %+.2f%%) | UUP %.2f | DX r21 %.1f" % (
    tnx.iloc[-1], hi.iloc[-1], 100 * (tnx.iloc[-1] / hi.iloc[-1] - 1), dx.iloc[-1], d1("DX-Y.NYB"), uup.iloc[-1], r("DX-Y.NYB", 21)))
g = C["GLD"]
g200 = g.iloc[-200:].mean()
print("[69] GLD %.2f vs 200d %.2f (%+.2f%%) | off 252 hi %.2f%% | DX r21 %.1f" % (
    g.iloc[-1], g200, 100 * (g.iloc[-1] / g200 - 1), off_hi("GLD"), r("DX-Y.NYB", 21)))
print("[1/5/25] LQD off-lo %.2f | IEF off-lo %.2f | TLT off-lo %.2f" % (off_lo("LQD"), off_lo("IEF"), off_lo("TLT")))
print("[45/58] CL=F 1d %+.2f%% | UNG 1d %+.2f%% | NG=F 1d %+.2f%%" % (d1("CL=F"), d1("UNG"), d1("NG=F")))
