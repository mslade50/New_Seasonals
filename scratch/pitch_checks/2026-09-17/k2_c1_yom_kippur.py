"""K2 c1 round 1: short SPY from YK-2 close to the Yom Kippur close.

Dates COMPUTED: Gauss's Passover formula (15 Nisan) + 163 days = 1 Tishrei
(Rosh Hashanah); Yom Kippur = RH + 9. Cross-checked against the Calendrical
Calculations molad arithmetic and validated against the brief's 11 known dates
before any statistic is computed.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa
import datetime as dt
from math import floor

OUT = Path(__file__).resolve().parent


def gauss_passover(Y: int) -> dt.date:
    a = (12 * Y + 12) % 19
    b = Y % 4
    Q = 20.095587638424 + 1.554241796621 * a + 0.25 * b - 0.003177794022 * Y
    M = floor(Q)
    m = Q - M
    c = (M + 3 * Y + 5 * b + 1) % 7
    if c in (2, 4, 6):
        M += 1
    elif c == 1 and a > 6 and m >= 0.632870370:
        M += 2
    elif c == 0 and a > 11 and m >= 0.897723765:
        M += 1
    s = Y // 100 - Y // 400 - 2  # Julian -> Gregorian
    return dt.date(Y, 3, 1) + dt.timedelta(days=M + s - 1)


def cc_new_year(Y: int) -> dt.date:
    def elapsed(h):
        months = (235 * h - 234) // 19
        parts = 12084 + 13753 * months
        days = 29 * months + parts // 25920
        return days + 1 if (3 * (days + 1)) % 7 < 3 else days

    def corr(h):
        if elapsed(h + 1) - elapsed(h) == 356:
            return 2
        if elapsed(h) - elapsed(h - 1) == 382:
            return 1
        return 0
    h = Y + 3761
    return dt.date.fromordinal(-1373427 + elapsed(h) + corr(h))


def rosh_hashanah(Y):
    return gauss_passover(Y) + dt.timedelta(days=163)


KNOWN_RH = {2023: "2023-09-16", 2024: "2024-10-03", 2025: "2025-09-23", 2026: "2026-09-12"}
KNOWN_YK = {2020: "2020-09-28", 2021: "2021-09-16", 2022: "2022-10-05", 2023: "2023-09-25",
            2024: "2024-10-12", 2025: "2025-10-02", 2026: "2026-09-21"}
bad = 0
for y, d in KNOWN_RH.items():
    ok = str(rosh_hashanah(y)) == d
    bad += not ok
    print(f"RH {y}: gauss {rosh_hashanah(y)} cc {cc_new_year(y)} known {d} {'OK' if ok else 'FAIL'}")
for y, d in KNOWN_YK.items():
    g = rosh_hashanah(y) + dt.timedelta(days=9)
    ok = str(g) == d
    bad += not ok
    print(f"YK {y}: gauss {g} known {d} {'OK' if ok else 'FAIL'}")
mism = [y for y in range(1995, 2031) if rosh_hashanah(y) != cc_new_year(y)]
print(f"gauss vs CC mismatches 1995-2030: {mism}")
assert bad == 0 and not mism, "DATE VALIDATION FAILED"

TK = ["SPY", "IWM", "TLT", "^VIX"]
P = load_prices(TK)
spy = P["SPY"]
idx = spy.index
print(f"SPY {idx[0].date()} .. {idx[-1].date()}")
pos = pd.Series(range(len(idx)), index=idx)
ev = load_events(["quad_witching", "fomc_decision", "opex"])

rows = []
for Y in range(1999, 2027):
    rh = pd.Timestamp(rosh_hashanah(Y))
    yk = rh + pd.Timedelta(days=9)
    y = int(idx.searchsorted(yk))          # first session on/after YK
    r = int(idx.searchsorted(rh)) - 1      # last session before RH
    if y >= len(idx) or r < 0:
        rows.append({"year": Y, "rh": rh.date(), "yk": yk.date(), "live": True})
        continue
    e = y - 2
    rec = {"year": Y, "rh": rh.date(), "yk": yk.date(), "yk_dow": yk.day_name()[:3],
           "yk_sess": idx[y].date(), "entry": idx[e].date(), "pre_rh": idx[r].date(),
           "yk_on_session": idx[y] == yk}
    lo, hi = idx[e], idx[y]
    evw = ev[(ev["date"] > lo) & (ev["date"] <= hi)]["event"].tolist()
    evf = ev[(ev["date"] > idx[r]) & (ev["date"] <= hi)]["event"].tolist()
    rec["quad_in_rung"] = "quad_witching" in evw
    rec["fomc_in_rung"] = "fomc_decision" in evw
    rec["quad_in_full"] = "quad_witching" in evf
    rec["fomc_in_full"] = "fomc_decision" in evf
    for t in TK:
        c = P[t]["Close"].reindex(idx)
        rec[f"{t}_rung"] = c.iloc[y] / c.iloc[e] - 1
        rec[f"{t}_full"] = c.iloc[y] / c.iloc[r] - 1
        for hh in (5, 10):
            rec[f"{t}_post{hh}"] = (c.iloc[y + hh] / c.iloc[y] - 1) if y + hh < len(idx) else np.nan
    for k in range(-6, 7):
        if y + k < len(idx) and e + k >= 0:
            rec[f"lad{k}"] = spy["Close"].iloc[y + k] / spy["Close"].iloc[e + k] - 1
    # tdom-matched same-year control (Sep/Oct, same tdom as entry, no overlap with [r, y])
    ent = idx[e]
    mon_idx = idx[(idx.year == Y) & (idx.month == ent.month)]
    tdom = int(mon_idx.get_loc(ent))
    ctl = []
    for mth in (9, 10):
        mi = idx[(idx.year == Y) & (idx.month == mth)]
        if tdom < len(mi):
            p = int(pos[mi[tdom]])
            if p + 2 < len(idx) and (p + 2 < r or p > y):
                ctl.append(spy["Close"].iloc[p + 2] / spy["Close"].iloc[p] - 1)
    rec["tdom"] = tdom + 1
    rec["ctl_tdom_same_year"] = np.mean(ctl) if ctl else np.nan
    rows.append(rec)

D = pd.DataFrame(rows)
hist = D[D["live"] != True].copy() if "live" in D else D.copy()
hist = hist.dropna(subset=["SPY_rung"])
hist["midterm"] = hist["year"] % 4 == 2
for _c in ("quad_in_rung", "fomc_in_rung", "quad_in_full", "fomc_in_full", "yk_on_session"):
    hist[_c] = hist[_c].astype(bool)
hist["year"] = hist["year"].astype(int)
pd.set_option("display.width", 250)
print(D[["year", "rh", "yk", "yk_dow", "pre_rh", "entry", "yk_sess", "tdom", "quad_in_rung",
         "fomc_in_rung", "quad_in_full", "SPY_rung", "SPY_full"]].to_string(index=False))

# calendar-slot control from OTHER years: the 2-session hold entered at the same month/day
sc = spy["Close"]
def slot_ret(Y, month, day, hold=2):
    p = int(idx.searchsorted(pd.Timestamp(Y, month, day)))
    if p + hold >= len(idx):
        return np.nan
    return sc.iloc[p + hold] / sc.iloc[p] - 1
years = hist["year"].tolist()
slot = []
for _, rw in hist.iterrows():
    ent = pd.Timestamp(rw["entry"])
    slot.append(np.nanmean([slot_ret(Yo, ent.month, ent.day) for Yo in years if Yo != rw["year"]]))
hist["ctl_slot_other_years"] = slot

# pooled Sep1-Oct20 2-session windows excluding [pre_rh, yk_sess]
r2 = sc.shift(-2) / sc - 1
pool = []
for Y in years:
    rw = hist[hist["year"] == Y].iloc[0]
    m = (idx >= pd.Timestamp(Y, 9, 1)) & (idx <= pd.Timestamp(Y, 10, 20))
    for d in idx[m]:
        p = int(pos[d])
        if p + 2 > int(pos[pd.Timestamp(rw["pre_rh"])]) and p <= int(pos[pd.Timestamp(rw["yk_sess"])]):
            continue
        pool.append(r2.iloc[p])
pool = np.array(pool)


def short_stats(v, label):
    s = summarize(-np.asarray(v), label)  # SHORT P&L
    if s["n"]:
        w = int((-np.asarray(v)[~np.isnan(v)] > 0).sum())
        s["rec"] = f"{w}-{s['n']-w}"
        s["sign_p"] = sign_test(w, s["n"])
    return s

rows = [short_stats(hist["SPY_rung"], "SHORT SPY YK-2->YK (live rung)"),
        short_stats(hist["SPY_full"], "SHORT SPY preRH->YK (full window)"),
        short_stats(hist["ctl_tdom_same_year"].dropna(), "CTRL tdom-matched Sep/Oct same yr"),
        short_stats(hist["ctl_slot_other_years"], "CTRL same calendar slot other yrs"),
        short_stats(pool, "CTRL all 2-sess Sep1-Oct20 ex window"),
        short_stats(r2.dropna().values, "CTRL all 2-sess days SPY 1999+")]
for t in ("IWM", "TLT", "^VIX"):
    rows.append(short_stats(hist[f"{t}_rung"], f"SHORT {t} rung"))
    rows.append(short_stats(hist[f"{t}_full"], f"SHORT {t} full window"))
show(rows, "(a)(b)(d) short P&L, percent (positive = short wins)")
dpair = -(hist["SPY_rung"] - hist["ctl_slot_other_years"])
print(f"paired rung minus slot control (short P&L): mean {100*dpair.mean():+.3f}pp, "
      f"{int((dpair>0).sum())}-{int((dpair<=0).sum())}, sign p {sign_test(int((dpair>0).sum()), len(dpair)):.4f}")
dp2 = -(hist["SPY_rung"] - hist["ctl_tdom_same_year"]).dropna()
print(f"paired rung minus tdom same-year control: mean {100*dp2.mean():+.3f}pp, "
      f"{int((dp2>0).sum())}-{int((dp2<=0).sum())} (n={len(dp2)})")
print(f"per-year rung cost 1.5bp RT: short mean edge {-100*hist['SPY_rung'].mean()*100:.1f} bp")
print("concentration (short P&L):", cluster_note(pd.DatetimeIndex(pd.to_datetime(hist['entry'])), -hist["SPY_rung"].values))

# (c) placebo ladder
lad = []
for k in range(-6, 7):
    v = hist[f"lad{k}"].dropna()
    s = short_stats(v, f"k={k:+d}")
    lad.append(s)
L = pd.DataFrame(lad)
L["rank_short"] = L["mean_pct"].rank(ascending=False).astype(int)
show(L.to_dict("records"), "(c) placebo ladder, 2-session SHORT shifted k sessions (k=0 live rung)")
print(f"k=0 ranks {int(L.loc[L['label']=='k=+0','rank_short'].iloc[0])} of {len(L)} (1 = best short)")

# (e) collisions (f) eras
def split(mask, lab):
    return [short_stats(hist.loc[mask, "SPY_rung"], f"{lab} IN"),
            short_stats(hist.loc[~mask, "SPY_rung"], f"{lab} OUT")]
er = []
er += split(hist["quad_in_rung"], "quad in rung")
er += split(hist["fomc_in_rung"], "fomc in rung")
er += split(hist["quad_in_full"], "quad in full window")
er += split(hist["year"] < 2013, "pre-2013")
er += split(hist["year"] >= 2018, "2018+")
er += split(hist["midterm"], "midterm")
er += split(hist["yk_on_session"].astype(bool), "YK on a weekday session")
show(er, "(e)(f) splits, live rung short P&L")
ef = []
ef += [short_stats(hist.loc[hist["quad_in_full"], "SPY_full"], "full: quad IN"),
       short_stats(hist.loc[~hist["quad_in_full"], "SPY_full"], "full: quad OUT"),
       short_stats(hist.loc[hist["year"] < 2013, "SPY_full"], "full: pre-2013"),
       short_stats(hist.loc[hist["year"] >= 2013, "SPY_full"], "full: 2013+"),
       short_stats(hist.loc[hist["year"] >= 2018, "SPY_full"], "full: 2018+"),
       short_stats(hist.loc[hist["midterm"], "SPY_full"], "full: midterm")]
show(ef, "(e)(f) splits, full window short P&L")

# (g) buy Yom Kippur mirror (LONG P&L)
g = []
for t in TK:
    for hh in (5, 10):
        v = hist[f"{t}_post{hh}"].dropna().values
        s = summarize(v, f"LONG {t} YK close -> +{hh}")
        w = int((v > 0).sum())
        s["rec"] = f"{w}-{len(v)-w}"
        s["sign_p"] = sign_test(w, len(v))
        g.append(s)
for hh in (5, 10):
    rr = (sc.shift(-hh) / sc - 1)
    g.append(summarize(rr[(idx.month.isin([9, 10]))].dropna().values, f"CTRL SPY all Sep/Oct days +{hh}"))
show(g, "(g) buy Yom Kippur mirror")

# mechanism: SPY volume on RH/YK sessions vs surrounding sessions
vol = spy["Volume"]
vr = []
for Y in years:
    rh = pd.Timestamp(rosh_hashanah(Y))
    yk = rh + pd.Timedelta(days=9)
    for lab, d in (("RH1", rh), ("RH2", rh + pd.Timedelta(days=1)), ("YK", yk)):
        if d in pos.index:
            p = int(pos[d])
            nb = list(range(max(0, p - 10), p - 1)) + list(range(p + 2, min(len(idx), p + 11)))
            base = vol.iloc[nb]
            vr.append({"year": Y, "kind": lab, "ratio": vol.iloc[p] / base.median()})
V = pd.DataFrame(vr)
V["era"] = np.where(V["year"] < 2013, "pre2013", "2013+")
print("\n=== mechanism: SPY volume / median(+-10 sessions ex +-1), holiday sessions on weekdays ===")
print(V.groupby(["kind", "era"])["ratio"].agg(["count", "median", "mean"]).round(3).to_string())
print(V.groupby("kind")["ratio"].agg(["count", "median", "mean", lambda x: (x < 1).mean()]).round(3).to_string())
# placebo for volume: same statistic on every session
allr = []
for p in range(10, len(idx) - 11, 1):
    nb = list(range(p - 10, p - 1)) + list(range(p + 2, p + 11))
    allr.append(vol.iloc[p] / vol.iloc[nb].median())
allr = np.array(allr)
print(f"all sessions ratio median {np.median(allr):.3f}, share<1 {(allr<1).mean():.3f}")
hist.to_csv(OUT / "k2_c1_anchors.csv", index=False)
