"""ROUND 2 on the September VIX-settle SVXY carry.

Round 1 (07_svxy_sep_expiry.py) established:
  - the candidate's 14 anchors are a 100% subset of the 2026-08-07 corpse
    "pre-expiry short-vol carry (long SVXY into VIX expiry)" (same k, same
    exit), and September ranks 2 of 12 in that parent's month scan;
  - on the tradeable -0.5x vehicle the cell is +1.335% (7-1), not +4.05%;
  - a trading-day-of-month-matched SEPTEMBER control that is NOT an expiry
    anchor pays MORE in that era: +1.551% (24-8);
  - the pass-through ratio is 1.09x against a 0.68x horizon-matched baseline,
    so the mechanism DOES reach the vehicle. That attack failed.

Round 2 asks the four questions round 1 left open, all on the -0.5x vehicle:
  L. the anchor placebo ladder RESTRICTED TO THE TRADEABLE ERA (the screen ran
     it on the pooled 14, which is two securities)
  M. concentration / drop-best, and the year histogram
  N. the live conditioner stack, one rung at a time, priced against cost
  O. the September ROUND TRIP: does the pre-settle gain survive the rest of
     the month, i.e. is this a timing artifact inside one month? The repo's
     own V4_POSTOPEX_VOL excludes September because the post side pays
     -1.535% at 0-for-8.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa: F401,F403
from pitch_lab import (
    anchor_positions, bootstrap_p_le0, load_events, load_prices,
    rolling_on_valid, sign_test,
)

pd.set_option("display.width", 220)
BREAK = pd.Timestamp("2018-02-28")
K, H = 6, 5
COST = 20.0     # bps round trip, SVXY

px = load_prices(["SVXY", "^VIX", "SPY"])
sv = px["SVXY"]["Close"].dropna()
vix = px["^VIX"]["Close"].dropna()
spy = px["SPY"]["Close"].dropna()
cal = sv.index
pos = pd.Series(range(len(cal)), index=cal)
ev = load_events()
VXP = pd.DatetimeIndex(sorted(ev.loc[ev.event == "vix_expiry", "date"]))
OPEX = pd.DatetimeIndex(sorted(ev.loc[ev.event == "opex", "date"]))
FOMC = set(pd.DatetimeIndex(ev.loc[ev.event == "fomc_decision", "date"]))
SEP = VXP[VXP.month == 9]


def rets(expiries, k=K, h=H, offset=0):
    p, kept = anchor_positions(cal, expiries, offset=0)
    out = {}
    for pp, d in zip(p, kept):
        a, b = pp - k + offset, pp - k + offset + h
        if a < 0 or b >= len(cal):
            continue
        out[cal[a]] = float(sv.iloc[b] / sv.iloc[a] - 1.0)
    return pd.Series(out).sort_index()


def rec(v):
    v = np.asarray(v, float)
    w = int((v > 0).sum())
    return w, len(v) - w


def line(label, v, extra=""):
    v = np.asarray(v, float)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        print("  {:<44s}  n=0".format(label))
        return
    w, l = rec(v)
    print("  {:<44s}  n={:>3d}  mean={:+7.3f}%  med={:+7.3f}%  {}-{}  "
          "sign_p={:.4f}  worst={:+7.2f}%  {:.1f}x cost {}".format(
              label, len(v), 100 * v.mean(), 100 * np.median(v), w, l,
              sign_test(w, len(v)), 100 * v.min(),
              100 * v.mean() / (COST / 100), extra))


base = rets(SEP)
era = np.asarray(base.index >= BREAK)

# ===========================================================================
print("=" * 84)
print("L. ANCHOR PLACEBO LADDER, offsets -10..+10 around the entry")
print("   offset 0 = the pitched entry (expiry_pos-6). Window length fixed at")
print("   h=5. The screen ran this on the pooled 14 = TWO securities.")
print("=" * 84)
rows = []
for j in range(-10, 11):
    r = rets(SEP, offset=j)
    m = np.asarray(r.index >= BREAK)
    w_a, l_a = rec(r.values)
    w_e, l_e = rec(r.values[m])
    rows.append({"offset": j, "n_all": len(r),
                 "all_mean": round(100 * r.values.mean(), 3),
                 "all_rec": "{}-{}".format(w_a, l_a),
                 "n_05x": int(m.sum()),
                 "e05_mean": round(100 * r.values[m].mean(), 3),
                 "e05_rec": "{}-{}".format(w_e, l_e),
                 "e05_med": round(100 * float(np.median(r.values[m])), 3)})
L = pd.DataFrame(rows)
print(L.to_string(index=False))
tr = L[L.offset == 0].iloc[0]
print("\n  POOLED (both securities): offset 0 ranks {} of 21 on |mean|, "
      "{} of 21 on mean".format(
          int(L["all_mean"].abs().rank(ascending=False).loc[L.offset == 0].iloc[0]),
          int(L["all_mean"].rank(ascending=False).loc[L.offset == 0].iloc[0])))
print("  -0.5x ERA ONLY  : offset 0 ranks {} of 21 on |mean|, {} of 21 on mean"
      .format(int(L["e05_mean"].abs().rank(ascending=False).loc[L.offset == 0].iloc[0]),
              int(L["e05_mean"].rank(ascending=False).loc[L.offset == 0].iloc[0])))
print("  -0.5x era: offsets beating the true anchor's mean: {} of 20"
      .format(int((L.loc[L.offset != 0, "e05_mean"] > tr["e05_mean"]).sum())))
print("  -0.5x era: offsets with a record at least as good ({}): {} of 20"
      .format(tr["e05_rec"],
              int((L.loc[L.offset != 0, "e05_rec"]
                   .str.split("-").str[0].astype(int) >= 7).sum())))
print("  -0.5x era mean over ALL 21 offsets = {:+.3f}%  (the true anchor is "
      "{:+.3f}%)".format(L["e05_mean"].mean(), tr["e05_mean"]))

# ===========================================================================
print("\n" + "=" * 84)
print("M. CONCENTRATION on the tradeable vehicle")
print("=" * 84)
e = base.values[era]
d = base.index[era]
print("  -0.5x era instances: " + ", ".join(
    "{}:{:+.2f}%".format(x.year, 100 * y) for x, y in zip(d, e)))
line("-0.5x era, all 8", e)
order = np.argsort(-e)
for k in (1, 2, 3):
    keep = np.ones(len(e), bool)
    keep[order[:k]] = False
    line("-0.5x era, drop-best-{}".format(k), e[keep])
tot = e.sum()
print("  best single episode {} = {:+.2f}pp of {:+.2f}pp total ({:.0f}%)".format(
    d[order[0]].date(), 100 * e[order[0]], 100 * tot,
    100 * e[order[0]] / tot))
print("  best TWO = {:.0f}% of total".format(100 * e[order[:2]].sum() / tot))
print("  bootstrap P(mean<=0) on the 8: {:.4f};  on drop-best-1: {:.4f}".format(
    bootstrap_p_le0(e), bootstrap_p_le0(e[np.arange(len(e)) != order[0]])))

# ===========================================================================
print("\n" + "=" * 84)
print("N. THE LIVE CONDITIONER STACK, one rung at a time (-0.5x era)")
print("=" * 84)
mx = rolling_on_valid(vix, lambda x: x.rolling(21).max())
mn = rolling_on_valid(vix, lambda x: x.rolling(21).min())
mu = rolling_on_valid(vix, lambda x: x.rolling(21).mean())
relp = rolling_on_valid((mx - mn) / mu,
                        lambda x: x.rolling(252).rank(pct=True) * 100)
frag = pd.read_parquet(Path(__file__).resolve().parents[3] / "data" /
                       "rd2_fragility.parquet")
ma = frag["63d"].rolling(10).mean().dropna()
live_relp = float(relp.dropna().iloc[-1])
live_dial = float(ma.iloc[-1])
print("  LIVE state: midterm=True  coincident_FOMC=True  vehicle=-0.5x  "
      "rel-range pctile={:.2f}  dial={:.1f}".format(live_relp, live_dial))
B = pd.DataFrame({"ret": base})
B["midterm"] = B.index.year % 4 == 2
B["coinc"] = [pd.Timestamp(x) in FOMC for x in
              [VXP[VXP.searchsorted(i)] for i in B.index]]
# rebuild coincidence from the expiry each anchor belongs to
co = []
for eidx in B.index:
    p = pos[eidx] + K
    co.append(cal[p] in FOMC if p < len(cal) else False)
B["coinc"] = co
B["relp"] = relp.reindex(B.index).values
B["dial"] = ma.reindex(B.index).values
B["era05"] = era
line("rung 0  -0.5x era, no other condition", B["ret"].values[era])
for nm, m in (("rung 1  + midterm", era & B["midterm"].values),
              ("rung 2  + coincident settle/FOMC",
               era & B["midterm"].values & np.asarray(B["coinc"], bool)),
              ("rung 3  + rel-range pctile < 15",
               era & (B["relp"].values < 15)),
              ("rung 4  + dial >= 50",
               era & (B["dial"].values >= 50))):
    line(nm, B["ret"].values[m])
print()
print("  the rel-range gradient across ALL 14 (OLS on percentile):")
b, a = np.polyfit(B["relp"].values, B["ret"].values, 1)
print("    ret = {:+.3f}% {:+.4f}%/pctile-pt -> fitted at live {:.1f} = {:+.3f}%"
      .format(100 * a, 100 * b, live_relp, 100 * (a + b * live_relp)))
b2, a2 = np.polyfit(B["relp"].values[era], B["ret"].values[era], 1)
print("    -0.5x era only: {:+.3f}% {:+.4f}%/pt -> fitted at {:.1f} = {:+.3f}%"
      .format(100 * a2, 100 * b2, live_relp, 100 * (a2 + b2 * live_relp)))
lo3 = B.sort_values("relp").head(3)
print("    three lowest-compression instances: " + ", ".join(
    "{}:{:.0f}p/{:+.2f}%".format(i.year, r, 100 * v)
    for i, r, v in zip(lo3.index, lo3["relp"], lo3["ret"])) +
    "  mean {:+.3f}%".format(100 * lo3["ret"].mean()))
hi3 = B.sort_values("relp").tail(3)
print("    three highest:                      " + ", ".join(
    "{}:{:.0f}p/{:+.2f}%".format(i.year, r, 100 * v)
    for i, r, v in zip(hi3.index, hi3["relp"], hi3["ret"])) +
    "  mean {:+.3f}%".format(100 * hi3["ret"].mean()))

# ===========================================================================
print("\n" + "=" * 84)
print("O. THE SEPTEMBER ROUND TRIP  (is the pre-settle gain kept?)")
print("   repo prereg: V4_POSTOPEX_VOL EXCLUDES September, 'September pays")
print("   -1.535% over 8 post-break anchors at a 0% hit rate'.")
print("=" * 84)


def fwd_from(anchor_dates, off_start, h):
    out = {}
    for dt in anchor_dates:
        p = pos.get(dt)
        if p is None:
            continue
        a, b = p + off_start, p + off_start + h
        if a < 0 or b >= len(cal):
            continue
        out[dt] = float(sv.iloc[b] / sv.iloc[a] - 1.0)
    return pd.Series(out).sort_index()


exits = pd.DatetimeIndex([cal[pos[i] + H] for i in base.index])
for h2 in (3, 5, 10):
    r = fwd_from(exits, 0, h2)
    m = np.asarray(r.index >= BREAK)
    line("AFTER the settle-eve exit, +{} td (all)".format(h2), r.values)
    line("AFTER the settle-eve exit, +{} td (-0.5x)".format(h2), r.values[m])
print()
# round trip: entry -> exit+10
rt = {}
for i in base.index:
    p = pos[i]
    if p + H + 10 < len(cal):
        rt[i] = float(sv.iloc[p + H + 10] / sv.iloc[p] - 1.0)
rt = pd.Series(rt).sort_index()
mrt = np.asarray(rt.index >= BREAK)
line("ROUND TRIP entry -> exit+10 td (all)", rt.values)
line("ROUND TRIP entry -> exit+10 td (-0.5x)", rt.values[mrt])
# the repo's own V4 September window, reproduced
sep_opex = OPEX[OPEX.month == 9]
v4 = fwd_from(pd.DatetimeIndex([cal[min(pos.get(d, len(cal) - 1), len(cal) - 1)]
                                for d in sep_opex if d in pos.index]), 0, 3)
m4 = np.asarray(v4.index >= BREAK)
line("V4 shape on Sep opex, +3 td (all eras)", v4.values)
line("V4 shape on Sep opex, +3 td (-0.5x)", v4.values[m4])

# ===========================================================================
print("\n" + "=" * 84)
print("P. THE LOAD-BEARING TEST: same 8 Septembers, anchor vs its own")
print("   non-anchor neighbours at the SAME trading-day-of-month slots")
print("=" * 84)
tvals = pd.Series(cal, index=cal).groupby([cal.year, cal.month]).cumcount() + 1
tdom = pd.Series(tvals.values, index=cal)
f5 = sv.pct_change(H).shift(-H)          # entry close -> +5 sessions
slots = sorted({int(tdom[i]) for i in base.index})
print("  anchor tdom slots: {}".format(slots))
rows = []
for i in base.index:
    yr = i.year
    m = ((cal.year == yr) & (cal.month == 9) & tdom.isin(slots).values
         & (cal != i) & f5.notna().values)
    others = f5[m]
    if len(others) == 0:
        continue
    rows.append({"year": yr, "era": "-0.5x" if i >= BREAK else "-1.0x",
                 "anchor_pct": round(100 * base[i], 2),
                 "n_other": len(others),
                 "other_mean_pct": round(100 * float(others.mean()), 2),
                 "diff_pp": round(100 * (base[i] - float(others.mean())), 2)})
Pd = pd.DataFrame(rows)
print(Pd.to_string(index=False))
for nm, mm2 in (("ALL 14 Septembers", np.ones(len(Pd), bool)),
                ("-0.5x era <- tradeable", (Pd["era"] == "-0.5x").values)):
    v = Pd["diff_pp"].values[mm2] / 100.0
    line("paired anchor-minus-neighbours, " + nm, v)

print("\n" + "=" * 84)
print("Q. EXIT SENSITIVITY from the same entry (-0.5x era)")
print("=" * 84)
for h2 in range(2, 10):
    r = rets(SEP, h=h2)
    m = np.asarray(r.index >= BREAK)
    line("h={} (5 is the pitched hold)".format(h2), r.values[m])

print("\n" + "=" * 84)
print("SUMMARY NUMBERS")
print("=" * 84)
print("  tradeable-era cell                  +1.335%  7-1   6.7x cost")
print("  tdom-matched Sep control, same era   +1.551%  24-8  (control WINS by "
      "-0.22pp)")
print("  -0.5x era mean over 21 placebo offsets {:+.3f}%".format(L["e05_mean"].mean()))
print("  drop-best-1 / drop-best-2 (-0.5x)   see M")
print("  live rel-range pctile {:.2f}, live dial {:.1f}".format(live_relp, live_dial))
print("\nDONE")
