"""C3 KILL CHECK -- long duration into the payrolls print, ^TNX at a 252d yield high.

Live geometry: today 2026-09-03 is the last session before the 2026-09-04 print.
That is the k=-1 entry off a k=-2 anchor dated 2026-09-02, i.e.
    anchor_positions(cal, nfp, -2) + lag=1 + h=1  ==  MOC the session before the
    print, out MOC at the print close.

Attacks in order (each one able to kill on its own):
  1. battery: parent vs own drift / all days / local +/-126td, era, cost
  2. TRADING-DAY-OF-MONTH matched control      <- registry 2026-08-11/-08-13
  3. MONTH-OF-YEAR control (September is TLT's second-worst month)
  4. midterm split (watchlist 0 is midterm-dead; C3 owes its own)
  5. yield-gate attribution: does "^TNX at its 252d max" filter or just subtract N?
  6. duration-neutral read (TLT residual on IEF, beta reported)
  7. placebo anchor ladder k=-8..+8 (four-for-four as a killer in the registry)
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa
from pitch_lab import (load_prices, load_events, anchor_positions, battery,
                       summarize, show, sign_test, bootstrap_p_le0, declusters,
                       rolling_on_valid)

warnings.filterwarnings("ignore")
pd.set_option("display.width", 220)

TK = ["TLT", "IEF", "SPY", "LQD", "HYG", "^TNX"]
raw = load_prices(TK)
px = pd.DataFrame({t: raw[t]["Close"] for t in TK}).dropna(subset=["TLT", "IEF"])
cal = px.index
tnx = raw["^TNX"]["Close"].dropna()
tnx_max = tnx.rolling(252).max()
prox = (tnx / tnx_max).reindex(cal).ffill(limit=3)          # 1.000 = AT the max
tnx_lvl_rank = rolling_on_valid(
    tnx, lambda x: x.rolling(252).rank(pct=True) * 100).reindex(cal).ffill(limit=3)

nfp = load_events(["nfp"])["date"]
pos, kept = anchor_positions(cal, nfp, -2)
anchor = pd.DatetimeIndex([cal[i] for i in pos])
entry = pd.DatetimeIndex([cal[i + 1] for i in pos if i + 1 < len(cal)])
print("cal %s .. %s   NFP k=-2 anchors constructed: %d  (last 3 %s)"
      % (cal[0].date(), cal[-1].date(), len(anchor),
         [str(d.date()) for d in anchor[-3:]]))
print("live check -- prox to 252d TNX max on 2026-09-02: %.4f   level rank %.1f"
      % (prox.loc["2026-09-02"], tnx_lvl_rank.loc["2026-09-02"]))

# --------------------------------------------------------------------------
# trading day of month, on the ENTRY session (today = tdom 3)
# --------------------------------------------------------------------------
_one = pd.Series(1, index=cal)
tdom = pd.Series(_one.groupby([cal.year, cal.month]).cumcount().values + 1, index=cal)
print("entry-session tdom distribution:",
      dict(pd.Series(tdom.loc[entry]).value_counts().sort_index()))


def ret_from_entry(s: pd.Series, h: int) -> pd.Series:
    """h-session close-to-close return measured FROM the entry close."""
    return s.shift(-h) / s - 1.0


def gate_mask(thr: float) -> pd.Series:
    m = pd.Series(False, index=cal)
    ok = anchor[(prox.loc[anchor] >= thr).fillna(False).values]
    m.loc[ok] = True
    return m


# ==========================================================================
print("\n" + "=" * 78)
print("ATTACK 1 -- battery on the gated cell, TLT then IEF, h=1 (through the print)")
print("=" * 78)
GATE = 0.99          # within 1% of the trailing-252 yield max; today = 1.000
m_gate = gate_mask(GATE)
print("gated anchors: %d of %d   dates %s"
      % (int(m_gate.sum()), len(anchor),
         ", ".join(str(d.date()) for d in cal[m_gate.values][-6:])))
variants = {"prox>=1.000 (AT max)": gate_mask(1.0),
            "prox>=0.98": gate_mask(0.98),
            "prox>=0.95": gate_mask(0.95),
            "gate OFF (all NFP)": pd.Series(cal.isin(anchor), index=cal)}
for h in (1, 2, 3):
    for tkr, cost in (("TLT", 4.0), ("IEF", 3.0)):
        battery(px, m_gate, [(tkr, 1.0)], h=h,
                title=f"C3 {tkr} long, NFP k=-2 anchor, TNX prox>={GATE}",
                cost_bps=cost, variants=variants if h == 1 else None,
                lag=1, min_gap=5, event_kinds=("nfp",))

# ==========================================================================
print("\n" + "=" * 78)
print("ATTACK 2 -- trading-day-of-month matched control (the central attack)")
print("=" * 78)
tdom_set = sorted(set(tdom.loc[entry].values))
for h in (1, 2, 3):
    rows = []
    for tkr in ("TLT", "IEF"):
        r = ret_from_entry(px[tkr], h)
        cond_all = r.loc[entry].dropna()
        gent = pd.DatetimeIndex([cal[i + 1] for i in pos
                                 if i + 1 < len(cal) and prox.iloc[i] >= GATE])
        cond_g = r.loc[gent].dropna()
        ctl_tdom = r[tdom.isin(tdom_set).values].dropna()
        ctl_t3 = r[(tdom == 3).values].dropna()
        ctl_all = r.dropna()
        rows += [summarize(cond_all.values, f"{tkr} h={h} NFP entry, ALL"),
                 summarize(cond_g.values, f"{tkr} h={h} NFP entry, GATED"),
                 summarize(ctl_tdom.values, f"{tkr} h={h} CTRL tdom in {tdom_set}"),
                 summarize(ctl_t3.values, f"{tkr} h={h} CTRL tdom==3 (today)"),
                 summarize(ctl_all.values, f"{tkr} h={h} CTRL all days")]
    show(rows, f"tdom-matched control, h={h}")
    for tkr in ("TLT", "IEF"):
        r = ret_from_entry(px[tkr], h)
        c = r.loc[entry].dropna()
        b = r[tdom.isin(tdom_set).values].dropna()
        print("  %s h=%d  excess over tdom-matched = %+.3fpp   (all-days excess %+.3fpp)"
              % (tkr, h, 100 * (c.mean() - b.mean()),
                 100 * (c.mean() - r.dropna().mean())))

# within-month paired excess: each anchor against its OWN month's other sessions
print("\n  within-month paired excess (anchor minus its own month's other starts):")
for h in (1, 2, 3):
    for tkr in ("TLT", "IEF"):
        r = ret_from_entry(px[tkr], h)
        d = []
        for e in entry:
            if pd.isna(r.get(e, np.nan)):
                continue
            same = r[(cal.year == e.year) & (cal.month == e.month)].dropna()
            same = same.drop(index=e, errors="ignore")
            if len(same) < 5:
                continue
            d.append(r.loc[e] - same.mean())
        d = np.asarray(d)
        w = int((d > 0).sum())
        print("    %s h=%d  paired excess %+.3fpp  n=%d  t=%+.2f  record %d-%d  sign p %.4f"
              % (tkr, h, 100 * d.mean(), len(d),
                 d.mean() / (d.std(ddof=1) / np.sqrt(len(d))), w, len(d) - w,
                 sign_test(w, len(d))))

# ==========================================================================
print("\n" + "=" * 78)
print("ATTACK 3 -- month-of-year control (September is TLT's second-worst month)")
print("=" * 78)
for h in (1, 2, 3):
    for tkr in ("TLT", "IEF"):
        r = ret_from_entry(px[tkr], h)
        by_m = r.dropna().groupby(r.dropna().index.month).mean() * 100
        print("  %s h=%d unconditional by month (%%): %s"
              % (tkr, h, {int(k): round(v, 3) for k, v in by_m.items()}))
        sep = pd.DatetimeIndex([e for e in entry if e.month == 9])
        c = r.loc[sep].dropna()
        base = r[(cal.month == 9)].dropna()
        w = int((c > 0).sum())
        print("     SEPTEMBER NFP entries only: n=%d mean %+.3f%% hit %.1f%% record %d-%d "
              "sign p %.4f   vs Sept all-days %+.3f%% -> excess %+.3fpp"
              % (len(c), 100 * c.mean(), 100 * (c > 0).mean(), w, len(c) - w,
                 sign_test(w, len(c)), 100 * base.mean(),
                 100 * (c.mean() - base.mean())))

# ==========================================================================
print("\n" + "=" * 78)
print("ATTACK 4 -- midterm split (watchlist 0's kill; C3 owes its own)")
print("=" * 78)
for h in (1, 2, 3):
    rows = []
    for tkr in ("TLT", "IEF"):
        r = ret_from_entry(px[tkr], h)
        for lbl, sel in (("ALL", entry),
                         ("gated", pd.DatetimeIndex(
                             [cal[i + 1] for i in pos
                              if i + 1 < len(cal) and prox.iloc[i] >= GATE]))):
            c = r.loc[sel].dropna()
            mid = c[[d.year % 4 == 2 for d in c.index]]
            non = c[[d.year % 4 != 2 for d in c.index]]
            for nm, v in (("MIDTERM", mid), ("non-midterm", non)):
                s = summarize(v.values, f"{tkr} h={h} {lbl} {nm}")
                if s["n"]:
                    s["sign_p"] = round(sign_test(int((v > 0).sum()), len(v)), 4)
                rows.append(s)
    show(rows, f"midterm split, h={h}")

# ==========================================================================
print("\n" + "=" * 78)
print("ATTACK 5 -- yield-gate attribution: does the gate filter, or just cut N?")
print("=" * 78)
for h in (1, 2, 3):
    rows = []
    for tkr in ("TLT", "IEF"):
        r = ret_from_entry(px[tkr], h)
        for thr in (0.90, 0.95, 0.98, 0.99, 1.0):
            sel = pd.DatetimeIndex([cal[i + 1] for i in pos
                                    if i + 1 < len(cal) and prox.iloc[i] >= thr])
            disc = pd.DatetimeIndex([cal[i + 1] for i in pos
                                     if i + 1 < len(cal) and prox.iloc[i] < thr])
            c, d = r.loc[sel].dropna(), r.loc[disc].dropna()
            rows.append({"leg": tkr, "h": h, "prox>=": thr, "n_kept": len(c),
                         "kept_pct": round(100 * c.mean(), 3) if len(c) else np.nan,
                         "n_disc": len(d),
                         "disc_pct": round(100 * d.mean(), 3) if len(d) else np.nan,
                         "gate_value_pp": round(100 * (c.mean() - r.loc[entry].dropna().mean()), 3)
                         if len(c) else np.nan})
    show(rows, f"gate ladder, h={h}  (gate_value = gated minus all-NFP)")

# ==========================================================================
print("\n" + "=" * 78)
print("ATTACK 6 -- duration-neutral read: TLT residual on IEF (beta reported)")
print("=" * 78)
for h in (1, 2, 3):
    rt = ret_from_entry(px["TLT"], h)
    ri = ret_from_entry(px["IEF"], h)
    both = pd.concat([rt, ri], axis=1).dropna()
    both.columns = ["TLT", "IEF"]
    beta = np.polyfit(both["IEF"], both["TLT"], 1)[0]
    resid = both["TLT"] - beta * both["IEF"]
    for lbl, sel in (("all NFP entries", entry),
                     ("gated", pd.DatetimeIndex(
                         [cal[i + 1] for i in pos
                          if i + 1 < len(cal) and prox.iloc[i] >= GATE]))):
        v = resid.reindex(sel).dropna()
        w = int((v > 0).sum())
        print("  h=%d beta(TLT~IEF)=%.3f  %-16s residual %+.4fpp n=%d record %d-%d sign p %.4f"
              % (h, beta, lbl, 100 * v.mean(), len(v), w, len(v) - w,
                 sign_test(w, len(v))))
    # LQD/HYG carried for the C8 cross-read
    for extra in ("LQD", "HYG", "SPY"):
        r = ret_from_entry(px[extra], h)
        c = r.loc[entry].dropna()
        print("     %s h=%d NFP-entry %+.3f%% (n=%d) vs own all-days %+.3f%% -> %+.3fpp"
              % (extra, h, 100 * c.mean(), len(c), 100 * r.dropna().mean(),
                 100 * (c.mean() - r.dropna().mean())))

# ==========================================================================
print("\n" + "=" * 78)
print("ATTACK 7 -- placebo anchor ladder k=-8..+8 (is the print the anchor at all?)")
print("=" * 78)
for h in (1, 3):
    rows = []
    for k in range(-8, 9):
        p2, _ = anchor_positions(cal, nfp, k)
        ent = pd.DatetimeIndex([cal[i + 1] for i in p2 if i + 1 < len(cal)])
        for tkr in ("TLT", "IEF"):
            r = ret_from_entry(px[tkr], h)
            c = r.loc[ent].dropna()
            rows.append({"k": k, "leg": tkr, "h": h, "n": len(c),
                         "mean_pct": round(100 * c.mean(), 3),
                         "hit": round(100 * (c > 0).mean(), 1)})
    df = pd.DataFrame(rows)
    for tkr in ("TLT", "IEF"):
        sub = df[df.leg == tkr].sort_values("mean_pct", ascending=False)
        print(f"\n  {tkr} h={h} anchor ladder, best first (live config is k=-2):")
        print(sub.to_string(index=False))
print("\nDONE.")
