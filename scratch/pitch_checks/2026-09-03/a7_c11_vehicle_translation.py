"""C11 -- which vehicle expresses C1.

WHOLE variants over the SAME gated anchor set. No marginal-fill decomposition
(memory rule: "new fills only" streams are selection-biased by construction).

Candidates:
  long SVXY   -- the pitched vehicle. -0.5x since 2018-02-28 (was -1x).
  short UVXY  -- +1.5x since 2018-02-28 (was +2x). SAME break date, so an era
                 split is required on both, not just SVXY.
  short VXX   -- NOT IN THE CACHE. Verified below rather than assumed.
  spot ^VIX   -- not tradeable. Carried only as the statistical leg.
  short VIX futures / options -- out of scope: the pitch grammar's closed
                 vocabulary is MOO | MOC | LIMIT(anchor, k ATR) on cash
                 instruments, and a futures leg marks the row Manual_Only.

Cost is stated as an assumption, not measured (the cache has no quotes):
  SVXY  round trip ~8 bps  (spread + slippage, $63 name, decent book)
  UVXY  round trip ~10 bps (cheaper price, wider relative spread)
  UVXY  BORROW: a hard-to-borrow inverse-of-inverse; 10-30% annualised is
        routine, so a one-session hold costs ~4-12 bps on top, and the locate
        can simply fail. Priced at 8 bps here and flagged as the soft cost.
Small-account constraint: a SHORT cannot be sized freely in the PA (~$30k live
cap, and UVXY borrow is worse there), whereas long SVXY is a plain cash buy.
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa: F401,F403,E402
from pitch_lab import (close_panel, load_prices, fwd_lag, summarize, sign_test,
                       load_events, rolling_on_valid, show, anchor_positions,
                       bootstrap_p_le0)

warnings.filterwarnings("ignore")
pd.set_option("display.width", 250)

WANT = ["SVXY", "UVXY", "VXX", "VIXY", "VIXM", "SVIX", "UVIX", "^VIX", "SPY"]
RAW = load_prices(WANT)
print("=" * 118)
print("0. VEHICLE AVAILABILITY IN THE CACHE (measured, not assumed)")
for t in WANT:
    if t in RAW:
        s = RAW[t]["Close"].dropna()
        print(f"   {t:6s} PRESENT  n={len(s):5d}  {s.index[0].date()}.."
              f"{s.index[-1].date()}  last {s.iloc[-1]:.2f}")
    else:
        print(f"   {t:6s} ABSENT -- cannot be a variant this morning")
print("=" * 118)

px = close_panel([t for t in WANT if t in RAW])
cal = px["SPY"].dropna().index
vix = px["^VIX"]
rng21 = (rolling_on_valid(vix, lambda x: x.rolling(21).max())
         - rolling_on_valid(vix, lambda x: x.rolling(21).min()))
REL = rolling_on_valid(rng21 / rolling_on_valid(vix, lambda x: x.rolling(21).mean()),
                       lambda x: x.rolling(252).rank(pct=True) * 100)

KINDS = ("nfp", "cpi", "ppi", "fomc_decision")
EV = {k: load_events([k])["date"] for k in KINDS}
ALL_PRINTS = pd.DatetimeIndex(sorted(pd.concat(list(EV.values())).unique()))
pos = pd.Series(range(len(cal)), index=cal)
rows = []
for kind in KINDS:
    p, kept = anchor_positions(cal, EV[kind], -2)
    for i, ap in enumerate(p):
        d0 = kept[i]
        nxt = ALL_PRINTS[ALL_PRINTS > d0]
        rw = 99 if len(nxt) == 0 else int(
            pos.get(nxt[0], int(cal.searchsorted(nxt[0])))
            - pos.get(d0, int(cal.searchsorted(d0))))
        rows.append({"anchor": cal[ap], "kind": kind, "runway_td": rw})
F = pd.DataFrame(rows).set_index("anchor").sort_index()
g = F.groupby(level=0)
F = F[~F.index.duplicated(keep="first")].assign(runway_td=g["runway_td"].min(),
                                                kind=g["kind"].apply(lambda x: "+".join(sorted(set(x)))))
F["rel"] = REL.reindex(F.index).values
POOL = F[(F["rel"] <= 15) & (F["runway_td"] >= 3)].index
NFPA = F[(F["rel"] <= 15) & (F["runway_td"] >= 3)
         & (F["kind"].str.contains("nfp"))].index
BREAK = pd.Timestamp("2018-02-28")

VARIANTS = [("long SVXY", [("SVXY", 1.0)], 8.0),
            ("short UVXY", [("UVXY", -1.0)], 18.0),   # 10 spread + 8 borrow
            ("short ^VIX (NOT TRADEABLE)", [("^VIX", -1.0)], 0.0)]


def run(anchors, label):
    print(f"\n########## {label}  (n_anchors={len(anchors)}) ##########")
    out = []
    for nm, legs, cost in VARIANTS:
        r = None
        for t, w in legs:
            if t not in px:
                r = None
                break
            f = w * fwd_lag(px[t].dropna(), 1, lag=1)
            r = f if r is None else r + f
        if r is None:
            out.append({"label": nm, "n": 0})
            continue
        v = r.reindex(anchors).dropna()
        if len(v) == 0:
            out.append({"label": nm, "n": 0}); continue
        st = summarize(v.values, nm)
        st["signp"] = round(sign_test(int((v > 0).sum()), len(v)), 4)
        st["cost_bps"] = cost
        st["net_bps"] = round(100 * 100 * v.mean() - cost, 1)
        st["cost_mult"] = round((100 * 100 * v.mean()) / cost, 1) if cost else np.nan
        st["bootP"] = round(bootstrap_p_le0(v.values), 4)
        out.append(st)
    show(out, f"{label}: whole variants, h=1, entry lag=1, gross then net of cost")
    # era split -- BOTH SVXY and UVXY re-levered on 2018-02-28
    out = []
    for nm, legs, cost in VARIANTS:
        if any(t not in px for t, _ in legs):
            continue
        r = None
        for t, w in legs:
            f = w * fwd_lag(px[t].dropna(), 1, lag=1)
            r = f if r is None else r + f
        v = r.reindex(anchors).dropna()
        for lbl, m in (("pre 2018-02-28", v.index < BREAK),
                       ("post 2018-02-28", v.index >= BREAK)):
            st = summarize(v.values[m], f"{nm} | {lbl}")
            if st["n"]:
                st["signp"] = round(sign_test(int((v.values[m] > 0).sum()),
                                              int(m.sum())), 4)
            out.append(st)
    show(out, f"{label}: leverage-break era split (SVXY -1x->-0.5x, UVXY 2x->1.5x)")


run(POOL, "POOLED clear-calendar gated anchors")
run(NFPA, "NFP-only clear-calendar gated anchors")

# ---------------------------------------------------------------------------
print("\n" + "=" * 118)
print("1. PAIRWISE, ON THE SAME DAYS ONLY (the honest comparison -- SVXY and")
print("   UVXY both start 2011-10-04 so the sets already match, verified here)")
print("=" * 118)
sv = fwd_lag(px["SVXY"].dropna(), 1, lag=1).reindex(POOL).dropna()
uv = -fwd_lag(px["UVXY"].dropna(), 1, lag=1).reindex(POOL).dropna()
both = sv.index.intersection(uv.index)
print(f"   SVXY-covered anchors {len(sv)}, short-UVXY-covered {len(uv)}, "
      f"both {len(both)}")
d = pd.DataFrame({"svxy": sv.reindex(both), "shortuvxy": uv.reindex(both)})
d["diff"] = d["svxy"] - d["shortuvxy"]
print(f"   mean long SVXY {100*d['svxy'].mean():+.3f}%   "
      f"mean short UVXY {100*d['shortuvxy'].mean():+.3f}%   "
      f"paired diff {100*d['diff'].mean():+.3f}pp "
      f"(SVXY wins {int((d['diff']>0).sum())} of {len(d)})")
print(f"   corr {d['svxy'].corr(d['shortuvxy']):+.4f}; "
      f"sd SVXY {100*d['svxy'].std():.3f}pp vs short UVXY "
      f"{100*d['shortuvxy'].std():.3f}pp "
      f"-> UVXY carries {d['shortuvxy'].std()/d['svxy'].std():.2f}x the variance")
print(f"   worst: SVXY {100*d['svxy'].min():+.2f}% on {d['svxy'].idxmin().date()}, "
      f"short UVXY {100*d['shortuvxy'].min():+.2f}% on {d['shortuvxy'].idxmin().date()}")
print("   RISK-ADJUSTED (mean / sd, one session):")
print(f"     long SVXY  {d['svxy'].mean()/d['svxy'].std():+.4f}")
print(f"     short UVXY {d['shortuvxy'].mean()/d['shortuvxy'].std():+.4f}")
print("   per-1-ATR-risk comparison is what actually sizes the pitch:")
for t, sgn in (("SVXY", 1), ("UVXY", -1)):
    hi, lo, cl = RAW[t]["High"], RAW[t]["Low"], RAW[t]["Close"]
    a = pd.Series(np.asarray(wilder_atr(hi, lo, cl, 14)).ravel(), index=RAW[t].index)
    apct = (a / cl).reindex(both)
    r = (sgn * fwd_lag(px[t].dropna(), 1, lag=1)).reindex(both)
    print(f"     {t}: live ATR% {100*float(a.iloc[-1]/cl.iloc[-1]):.2f}%, "
          f"mean move in ATRs {(r/apct).mean():+.3f}, "
          f"worst {(r/apct).min():+.2f} ATR")

print("\n2. THE LEVERAGE ASYMMETRY THAT DECIDES IT")
print("   long SVXY: worst possible outcome is -100% (bounded), plain cash buy,")
print("     placeable in either account, no borrow, no locate risk.")
print("   short UVXY: unbounded left tail on a vol explosion, needs a locate,")
print("     pays borrow every night, and the small account's live cap makes the")
print("     size non-comparable. On this cell it also carries "
      f"{d['shortuvxy'].std()/d['svxy'].std():.2f}x the one-session variance")
print("     for a similar gross edge, before any borrow is charged.")
print("\n3. VXX / SVIX / UVIX: absent from master_prices (section 0), so a variant")
print("   on them cannot be measured this morning and must not be pitched.")
