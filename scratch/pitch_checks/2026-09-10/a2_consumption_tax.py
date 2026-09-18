"""a2 — ADVERSARIAL round 1 on CANDIDATE A2.

"The consumption tax": LONG XLY / SHORT XLE, equal dollar weight, entry
lag=1 MOC, when the 21-day return spread XLE minus XLY sits at or above its
trailing-252 95th percentile.

Order convention (rule 7, fixed and stated): FILTER first, then DECLUSTER
(min_gap = h). Cost: 2 bp per leg per side => 8 bp for the pair.

The kill this script is built around: is XLE-vs-XLY the generic 21-day
cross-sectional reversal factor wearing a sector label?
"""
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa
from pitch_lab import _valid_pct_change  # noqa

H = 5
COST_LEG_BPS = 4.0
SPDR = ["XLB", "XLC", "XLE", "XLF", "XLI", "XLK", "XLP", "XLRE", "XLU",
        "XLV", "XLY"]
TICKERS = SPDR + ["SPY", "USO"]

raw = close_panel(TICKERS)
CAL = raw["SPY"].dropna().index
px = raw.reindex(CAL)
POS = pd.Series(range(len(CAL)), index=CAL)
print("panel:", CAL[0].date(), "..", CAL[-1].date(), len(CAL), "sessions")

R21 = {t: _valid_pct_change(px[t], 21) for t in SPDR}
R21["USO"] = _valid_pct_change(px["USO"], 21)


def spread_trigger(leader: str, laggard: str, thr: float = 95.0) -> pd.Series:
    """21d return spread leader - laggard at/above its trailing-252 pctile."""
    sp = R21[leader] - R21[laggard]
    rk = rolling_on_valid(sp, lambda x: x.rolling(252).rank(pct=True) * 100.0)
    return (rk >= thr) & sp.notna()


def episodes(mask: pd.Series, ret: pd.Series, h: int = H) -> pd.DatetimeIndex:
    days = CAL[mask.reindex(CAL, fill_value=False).values & ret.notna().values]
    return declusters(days, h, CAL)


def pair_ret(long_t: str, short_t: str, h: int = H, lag: int = 1) -> pd.Series:
    return vehicle_ret(px, [(long_t, 1.0), (short_t, -1.0)], h, lag)


MASK = spread_trigger("XLE", "XLY")
RET = pair_ret("XLY", "XLE")
sp_now = (R21["XLE"] - R21["XLY"]).iloc[-1]
rk_now = rolling_on_valid(R21["XLE"] - R21["XLY"],
                          lambda x: x.rolling(252).rank(pct=True) * 100.0).iloc[-1]
print(f"today: XLE-XLY 21d spread {100*sp_now:+.2f}pp, pctile {rk_now:.1f}, "
      f"trigger live = {bool(MASK.iloc[-1])}")
print(f"trigger days (full history): {int((MASK & RET.notna()).sum())}")


# ==========================================================================
# 1. battery on the defended cell
# ==========================================================================
variants = {f"pctile>={t}": spread_trigger("XLE", "XLY", t)
            for t in (90, 92.5, 95, 97.5, 99)}
battery(px, MASK, [("XLY", 1.0), ("XLE", -1.0)], H,
        "A2 DEFENDED  LONG XLY / SHORT XLE (XLE-XLY 21d spread >= 95th pct)",
        COST_LEG_BPS, variants=variants, min_gap=H, event_kinds=("cpi",))


# ==========================================================================
# 2. leg attribution on the SAME episodes
# ==========================================================================
EPI = episodes(MASK, RET, H)
span = (CAL >= EPI[0]) & (CAL <= EPI[-1])
rl, rs, rspy = fwd_lag(px["XLY"], H), fwd_lag(px["XLE"], H), fwd_lag(px["SPY"], H)
show([
    summarize(RET.loc[EPI].values, f"PAIR XLY-XLE (N={len(EPI)})"),
    summarize(rl.loc[EPI].values, "  LONG leg XLY on same episodes"),
    summarize(rl[span & rl.notna().values].values, "  CTRL XLY own drift, same span"),
    summarize(-rs.loc[EPI].values, "  SHORT leg XLE contribution (-1x)"),
    summarize(-rs[span & rs.notna().values].values, "  CTRL short-XLE contrib, same span"),
    summarize((rl - rspy).loc[EPI].values, "  XLY minus SPY on same episodes"),
    summarize((rl - rspy)[span & (rl - rspy).notna().values].values,
              "  CTRL XLY-SPY own drift, same span"),
    summarize((-rs + rspy).loc[EPI].values, "  SPY minus XLE on same episodes"),
    summarize((-rs + rspy)[span & (rs).notna().values].values,
              "  CTRL SPY-XLE own drift, same span"),
], "LEG ATTRIBUTION — A2")


# ==========================================================================
# 3. THE LIKELY KILLER: generic cross-sectional 21d reversal control
# ==========================================================================
def generic_reversal(dates: pd.DatetimeIndex, k: int, h: int = H,
                     lag: int = 1) -> tuple[np.ndarray, list]:
    """On each episode date, LONG the k worst 21d SPDR sectors / SHORT the k
    best, equal dollar weight. Returns (rets, composition log)."""
    out, log = [], []
    C = {t: px[t].values for t in SPDR}
    for d in dates:
        p = int(POS[d])
        if p + lag + h >= len(CAL):
            continue
        avail = [t for t in SPDR
                 if not np.isnan(R21[t].iloc[p])
                 and not np.isnan(C[t][p + lag]) and not np.isnan(C[t][p + lag + h])]
        if len(avail) < 2 * k + 1:
            continue
        order = sorted(avail, key=lambda t: R21[t].iloc[p])
        longs, shorts = order[:k], order[-k:]
        rl_ = np.mean([C[t][p + lag + h] / C[t][p + lag] - 1 for t in longs])
        rs_ = np.mean([C[t][p + lag + h] / C[t][p + lag] - 1 for t in shorts])
        out.append(rl_ - rs_)
        log.append((str(d.date()), longs, shorts))
    return np.asarray(out), log


for k in (1, 2, 3):
    g, log = generic_reversal(EPI, k)
    show([summarize(RET.loc[EPI].values, f"DEFENDED XLY-XLE (N={len(EPI)})"),
          summarize(g, f"GENERIC long {k} worst / short {k} best (N={len(g)})")],
         f"3. generic 21d reversal control, k={k}, SAME episode dates")
    if k == 2:
        print("  composition on each episode (date, longs, shorts):")
        for row in log:
            print("   ", row)
        # paired difference
        v = RET.loc[EPI].values[:len(g)]
        d = v - g
        w = int((d > 0).sum())
        print(f"  PAIRED diff (defended - generic k=2): mean {100*d.mean():+.3f}%  "
              f"t {d.mean()/(d.std(ddof=1)/np.sqrt(len(d))):+.2f}  "
              f"record {w}-{len(d)-w}  sign p {sign_test(w, len(d)):.4f}")

# is XLE actually the best / XLY the worst on those days?
print("\n  how often is XLE the #1 21d leader and XLY the #1 laggard on trigger days?")
n_lead = n_lag = 0
for d in EPI:
    p = int(POS[d])
    avail = [t for t in SPDR if not np.isnan(R21[t].iloc[p])]
    order = sorted(avail, key=lambda t: R21[t].iloc[p])
    n_lead += order[-1] == "XLE"
    n_lag += order[0] == "XLY"
print(f"   XLE = top leader on {n_lead}/{len(EPI)} episodes; "
      f"XLY = bottom laggard on {n_lag}/{len(EPI)} episodes")


# ==========================================================================
# 4. crude-leg attribution: does USO 21d >= +10% add anything?
# ==========================================================================
uso_hot = R21["USO"] >= 0.10
ok = RET.notna().values
m = MASK.reindex(CAL, fill_value=False).values & ok
u = uso_hot.reindex(CAL, fill_value=False).values
show([
    summarize(RET.values[m], f"bare trigger (day level) N={int(m.sum())}"),
    summarize(RET.values[m & u], f"  + USO 21d>=+10% N={int((m & u).sum())}"),
    summarize(RET.values[m & ~u],
              f"  DISCARDED complement (USO cool) N={int((m & ~u).sum())}"),
    summarize(RET.values[u & ok], f"USO hot ALONE N={int((u & ok).sum())}"),
    summarize(RET.values[ok], f"ALL DAYS N={int(ok.sum())}"),
], "4. crude-leg gate attribution (day level)")
epi_u = episodes(MASK & uso_hot, RET, H)
epi_nu = episodes(MASK & ~uso_hot, RET, H)
show([summarize(RET.loc[EPI].values, f"bare episodes N={len(EPI)}"),
      summarize(RET.loc[epi_u].values, f"+USO hot episodes N={len(epi_u)}"),
      summarize(RET.loc[epi_nu].values, f"discarded (USO cool) N={len(epi_nu)}")],
     "4b. same, episode level")


# ==========================================================================
# 5. reference class — all 55 unordered SPDR pairs, identical construction
# ==========================================================================
print("\n--- REFERENCE CLASS: 55 unordered SPDR pairs, LONG laggard / SHORT leader")
DEF_STAT = float(np.nanmean(RET.loc[EPI].values))
print(f"DEFENDED statistic = episode mean of LONG XLY/SHORT XLE, h={H}, "
      f"N={len(EPI)} = {100*DEF_STAT:+.3f}%")

rows, sib = [], {}
for a, b in combinations(SPDR, 2):
    for lag_t, lead_t in ((a, b), (b, a)):
        mk = spread_trigger(lead_t, lag_t)
        r = pair_ret(lag_t, lead_t)
        e = episodes(mk, r, H)
        if len(e) < 5:
            continue
        d = summarize(r.loc[e].values, f"LONG {lag_t} / SHORT {lead_t}")
        rows.append(d)
        sib[(lag_t, lead_t)] = (r.values, np.asarray([POS[x] for x in e], int))
df = pd.DataFrame(rows).sort_values("mean_pct", ascending=False)
print(df.round(3).to_string(index=False))
better = int((df["mean_pct"] > 100 * DEF_STAT).sum())
print(f"\nobserved rank of LONG XLY / SHORT XLE: {better + 1} of {len(df)} "
      f"directed pairs with N>=5")

# max-of-K permutation against the DEFENDED statistic
rng = np.random.default_rng(42)
n = len(CAL)
NP = 3000
ge_max = ge_def = 0
def_pos = np.asarray([POS[d] for d in EPI], int)
def_arr = RET.values
for _ in range(NP):
    off = int(rng.integers(1, n))
    best = -1e9
    for key, (v, p) in sib.items():
        mm = np.nanmean(v[(p + off) % n])
        if not np.isnan(mm):
            best = max(best, mm)
    if best >= DEF_STAT:
        ge_max += 1
    md = np.nanmean(def_arr[(def_pos + off) % n])
    if not np.isnan(md) and md >= DEF_STAT:
        ge_def += 1
print(f"  permutation statistic = EPISODE MEAN of LONG XLY / SHORT XLE (h={H})")
print(f"  UNCHARGED p (defended cell, own circular-shift null) = {(ge_def+1)/(NP+1):.4f}")
print(f"  CHARGED   p (max over K={len(sib)} directed pairs >= defended stat) "
      f"= {(ge_max+1)/(NP+1):.4f}")


# ==========================================================================
# 6. era / midterm / inflation-shock splits, concentration, direction honesty
# ==========================================================================
v = RET.loc[EPI].values
yrs = pd.DatetimeIndex(EPI).year
print(f"\n--- CONCENTRATION: {cluster_note(EPI, v)}")
order = np.argsort(-v)
keep = np.ones(len(v), bool)
keep[order[:2]] = False
show([summarize(v, "all episodes"), summarize(v[keep], "drop best 2")], "drop-best-2")
by = pd.Series(v).groupby(yrs.values).agg(["count", "mean", "sum"])
by["mean"] = (100 * by["mean"]).round(3)
by["sum"] = (100 * by["sum"]).round(3)
print("by year:\n", by.to_string())
bestyr = by["sum"].idxmax()
show([summarize(v[yrs != bestyr], f"drop best year {bestyr}")], "drop-best-year")
show(era_split(EPI, v), "era split")
mid = (yrs % 4) == 2
show([summarize(v[mid], "midterm years"), summarize(v[~mid], "non-midterm")],
     "midterm split")
infl = np.isin(yrs, [2007, 2008, 2021, 2022])
show([summarize(v[infl], "2007/2008/2021/2022"), summarize(v[~infl], "all other years")],
     "inflation-shock years")

print("\n--- DIRECTION HONESTY (momentum side = LONG XLE / SHORT XLY)")
w = int((v > 0).sum())
print(f"  fade mean {100*v.mean():+.3f}% record {w}-{len(v)-w} sign p "
      f"{sign_test(w, len(v)):.4f}  |  momentum mean {-100*v.mean():+.3f}% "
      f"record {len(v)-w}-{w} sign p {sign_test(len(v)-w, len(v)):.4f}")

print("\n--- EVENT IN WINDOW (episodes)")
for kind in (("cpi",), ("fomc_decision",), ("ppi",)):
    fl = event_in_window(EPI, CAL, H, 1, kind)
    show([summarize(v[fl], f"{kind[0]} IN (N={int(fl.sum())})"),
          summarize(v[~fl], f"{kind[0]} OUT (N={int((~fl).sum())})")],
         f"{kind[0]} in hold")

print(f"\n--- COST: 8 bps. episode mean {100*DEF_STAT*100:.1f} bps "
      f"= {100*DEF_STAT*100/8:.1f}x cost (need >=5x)")
print("episode dates:", ", ".join(str(d.date()) for d in EPI))
