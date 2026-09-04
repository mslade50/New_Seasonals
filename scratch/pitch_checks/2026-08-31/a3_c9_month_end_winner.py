"""C9 -- buy the month's LARGEST 21-day winner at the month-end close.

Pension-rebalance-reversal, tested directly rather than as the plain month
turn. A FIXED cross-asset basket is ranked by trailing 21-day return as of
the ME-1 close; the top name is bought MOC at the ME-0 close and held h
sessions into the new month. Signal ME-1, entry ME-0 close: lag=1.

The basket is DECLARED HERE and not tuned:
  SPY QQQ IWM TLT HYG GLD GDX SLV USO XLE EEM EFA FXI UUP XLK XLF XLV XLU

Kill tests:
 1. rank-1 winner vs (a) equal-weight basket same window, (b) the winner's
    own unconditional forward return, (c) THE SAME RANK-1 TRADE ON EVERY
    NON-MONTH-END SESSION -- the control that decides whether "month end"
    does anything at all
 2. rank ladder 1 / 2 / 3 / middle / worst (a rebalance-reversal story
    predicts a monotone gradient)
 3. the INTO-the-print leg (ME-1c -> ME-0c) and the short side
 4. offset placebo ladder ME-5..ME+5 on the entry
 5. era split pre/post 2013 + midterm split
 6. cost 4 bps one leg / 8 bps two legs, need >= 5x
 7. concentration + sign test
 8. selection honesty: drop the metals (GLD, GDX, SLV) entirely
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa

import numpy as np
import pandas as pd

BASKET = ["SPY", "QQQ", "IWM", "TLT", "HYG", "GLD", "GDX", "SLV", "USO",
          "XLE", "EEM", "EFA", "FXI", "UUP", "XLK", "XLF", "XLV", "XLU"]
METALS = ["GLD", "GDX", "SLV"]
HS = (1, 2, 3, 5, 10)
LOOKBACK = 21

px = load_prices(BASKET)
cal = px["SPY"].index                                  # master NYSE calendar
panel = pd.DataFrame({t: px[t]["Close"].reindex(cal) for t in BASKET})
V = panel.values
COLS = list(panel.columns)
N = len(cal)

is_me = np.zeros(N, dtype=bool)
per = pd.Series(cal.to_period("M"), index=range(N))
me_pos = [int(g.index.max()) for _, g in per.groupby(per.values)][:-1]  # complete months only
is_me[me_pos] = True


def ranked_trade(sub_cols, h, rank_slot):
    """For every session p with a valid rank at p-1 and an exit at p+h,
    return (position, chosen ticker, forward return close[p]->close[p+h]).
    rank_slot: 0 = best 21d, 1 = 2nd, ... , -1 = worst, 'mid' = median."""
    ci = [COLS.index(c) for c in sub_cols]
    out_p, out_t, out_r, out_n = [], [], [], []
    for p in range(LOOKBACK + 1, N - h):
        past = V[p - 1, ci] / V[p - 1 - LOOKBACK, ci] - 1.0
        fwd = V[p + h, ci] / V[p, ci] - 1.0
        ok = ~np.isnan(past) & ~np.isnan(fwd) & ~np.isnan(V[p, ci])
        if ok.sum() < 6:
            continue
        idx_ok = np.where(ok)[0]
        order = idx_ok[np.argsort(-past[idx_ok])]        # best first
        if rank_slot == "mid":
            j = order[len(order) // 2]
        else:
            if abs(rank_slot) >= len(order):
                continue
            j = order[rank_slot]
        out_p.append(p)
        out_t.append(sub_cols[j])
        out_r.append(fwd[j])
        out_n.append(len(order))
    return (np.asarray(out_p), np.asarray(out_t, dtype=object),
            np.asarray(out_r, float), np.asarray(out_n))


def ew_basket(sub_cols, h):
    ci = [COLS.index(c) for c in sub_cols]
    out_p, out_r = [], []
    for p in range(LOOKBACK + 1, N - h):
        fwd = V[p + h, ci] / V[p, ci] - 1.0
        ok = ~np.isnan(fwd)
        if ok.sum() < 6:
            continue
        out_p.append(p)
        out_r.append(float(np.nanmean(fwd[ok])))
    return np.asarray(out_p), np.asarray(out_r, float)


print("=" * 78)
print("MEMBERSHIP (honest report of the changing basket)")
print("=" * 78)
first = {t: panel[t].first_valid_index() for t in BASKET}
for t in BASKET:
    print("  %-4s first bar %s" % (t, first[t].date()))
p_, t_, r_, n_ = ranked_trade(BASKET, 3, 0)
mem = pd.Series(n_, index=cal[p_])
print("  basket size over time: %s" % mem.resample("YE").last().astype(int).to_dict())

print()
print("=" * 78)
print("1. RANK-1 WINNER AT THE MONTH-END CLOSE vs THE THREE CONTROLS")
print("=" * 78)
head_rows = []
for h in HS:
    p1, t1, r1, _ = ranked_trade(BASKET, h, 0)
    me_mask = is_me[p1]
    pe, re = ew_basket(BASKET, h)
    ew_map = dict(zip(pe, re))
    ew_at_me = np.array([ew_map.get(p, np.nan) for p in p1[me_mask]])
    d_me = cal[p1[me_mask]]
    a = r1[me_mask]
    b = r1[~me_mask]
    rows = [summarize(a, "COND rank-1 at ME-0 (N=%d)" % len(a)),
            summarize(ew_at_me, "CTRL-a equal-weight basket, same windows"),
            summarize(b, "CTRL-c rank-1 on EVERY NON-month-end session"),
            summarize(r1, "CTRL rank-1 on all sessions")]
    show(rows, "h=%d" % h)
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    sew = np.sqrt(a.var(ddof=1) / len(a) + np.nanvar(ew_at_me, ddof=1) / len(a))
    w = int((a > 0).sum())
    print("  ME-vs-nonME diff = %+.3fpp  welch t = %+.2f | ME-vs-EWbasket diff = %+.3fpp  welch t = %+.2f"
          % (100 * (a.mean() - b.mean()), (a.mean() - b.mean()) / se,
             100 * (a.mean() - np.nanmean(ew_at_me)),
             (a.mean() - np.nanmean(ew_at_me)) / sew))
    print("  record %d-%d, sign p = %.4f, bootstrap P(mean<=0) = %.3f"
          % (w, len(a) - w, sign_test(w, len(a)), bootstrap_p_le0(a)))
    print("  concentration: %s" % cluster_note(d_me, a, k=3))
    print("  most-chosen names: %s"
          % pd.Series(t1[me_mask]).value_counts().head(6).to_dict())
    head_rows.append({"h": h, "n": len(a), "me_pct": round(100 * a.mean(), 3),
                      "nonme_pct": round(100 * b.mean(), 3),
                      "ew_pct": round(100 * np.nanmean(ew_at_me), 3),
                      "diff_vs_nonme_pp": round(100 * (a.mean() - b.mean()), 3),
                      "t_vs_nonme": round((a.mean() - b.mean()) / se, 2),
                      "hit": round(100 * (a > 0).mean(), 1)})
show(head_rows, "SUMMARY 1: month-end rank-1 vs every non-month-end rank-1")

print()
print("=" * 78)
print("1b. CTRL-b the WINNER'S OWN unconditional forward return")
print("=" * 78)
for h in HS:
    p1, t1, r1, _ = ranked_trade(BASKET, h, 0)
    me_mask = is_me[p1]
    own = []
    for tk in pd.unique(t1[me_mask]):
        s = panel[tk].dropna()
        f = (s.shift(-h) / s - 1.0).dropna()
        own.append((tk, 100 * f.mean(), int((t1[me_mask] == tk).sum())))
    tot = sum(c for _, _, c in own)
    blend = sum(m * c for _, m, c in own) / tot
    print("  h=%2d: chosen-name blended unconditional fwd = %+.3f%%  vs the cell %+.3f%%"
          % (h, blend, 100 * r1[me_mask].mean()))

print()
print("=" * 78)
print("2. RANK LADDER (a rebalance-reversal story predicts monotone)")
print("=" * 78)
for h in HS:
    rows = []
    for lbl, slot in [("rank 1 (winner)", 0), ("rank 2", 1), ("rank 3", 2),
                      ("middle", "mid"), ("rank -2", -2), ("rank -1 (loser)", -1)]:
        p, t, r, _ = ranked_trade(BASKET, h, slot)
        m = is_me[p]
        rr = summarize(r[m], lbl)
        rr["nonME_pct"] = round(100 * r[~m].mean(), 3)
        rr["ME_minus_nonME_pp"] = round(rr["mean_pct"] - rr["nonME_pct"], 3)
        rows.append(rr)
    show(rows, "h=%d rank ladder at the month-end close" % h)

print()
print("=" * 78)
print("3. THE INTO-THE-PRINT LEG (ME-1c -> ME-0c) -- which side does the")
print("   mechanism actually imply?")
print("=" * 78)
for h in HS[:3]:
    ci = [COLS.index(c) for c in BASKET]
    into, after, dts = [], [], []
    for p in me_pos:
        if p - 1 - LOOKBACK < 0 or p + h >= N:
            continue
        past = V[p - 1, ci] / V[p - 1 - LOOKBACK, ci] - 1.0
        ok = ~np.isnan(past) & ~np.isnan(V[p, ci]) & ~np.isnan(V[p + h, ci]) \
            & ~np.isnan(V[p - 1, ci])
        if ok.sum() < 6:
            continue
        j = np.where(ok)[0][np.argmax(past[np.where(ok)[0]])]
        into.append(V[p, ci][j] / V[p - 1, ci][j] - 1.0)
        after.append(V[p + h, ci][j] / V[p, ci][j] - 1.0)
        dts.append(cal[p])
    into, after = np.asarray(into), np.asarray(after)
    show([summarize(into, "winner INTO the print (ME-1c->ME-0c)"),
          summarize(after, "winner AFTER (ME-0c->ME+%d c) LONG" % h),
          summarize(-after, "winner AFTER, SHORT side")],
         "h=%d" % h)
    print("  corr(into, after) = %+.3f  (a reversal story wants this negative)"
          % float(np.corrcoef(into, after)[0, 1]))

print()
print("=" * 78)
print("4. OFFSET PLACEBO LADDER ME-5..ME+5 (entry session shifted)")
print("=" * 78)
for h in (3, 5):
    row = {"h": h}
    for k in range(-5, 6):
        vals = []
        ci = [COLS.index(c) for c in BASKET]
        for p0 in me_pos:
            p = p0 + k
            if p - 1 - LOOKBACK < 0 or p + h >= N:
                continue
            past = V[p - 1, ci] / V[p - 1 - LOOKBACK, ci] - 1.0
            fwd = V[p + h, ci] / V[p, ci] - 1.0
            ok = ~np.isnan(past) & ~np.isnan(fwd)
            if ok.sum() < 6:
                continue
            j = np.where(ok)[0][np.argmax(past[np.where(ok)[0]])]
            vals.append(fwd[j])
        row["ME%+d" % k] = round(100 * float(np.mean(vals)), 3)
    show([row], "rank-1 winner mean %% by entry offset, h=%d" % h)
    v = {k: row[k] for k in row if k != "h"}
    order = sorted(v.items(), key=lambda kv: -kv[1])
    print("  ME+0 ranks %d of 11; top3 %s"
          % ([k for k, _ in order].index("ME+0") + 1, order[:3]))

print()
print("=" * 78)
print("5. ERA SPLIT (pre/post 2013) + MIDTERM SPLIT")
print("=" * 78)
for h in HS:
    p1, t1, r1, _ = ranked_trade(BASKET, h, 0)
    m = is_me[p1]
    d, a = cal[p1[m]], r1[m]
    mid = np.array([x.year % 4 == 2 for x in d])
    aug = np.array([x.month == 8 for x in d])
    rows = era_split(d, a, "2013-01-01") + era_split(d, a, "2018-01-01") + [
        summarize(a[mid], "MIDTERM (today)"), summarize(a[~mid], "non-midterm"),
        summarize(a[aug], "AUGUST turn (today)"), summarize(a[~aug], "non-August")]
    show(rows, "h=%d" % h)

print()
print("=" * 78)
print("6. COST (1 leg 4 bps outright, 2 legs 8 bps vs the basket) need >= 5x")
print("=" * 78)
for h in HS:
    p1, t1, r1, _ = ranked_trade(BASKET, h, 0)
    m = is_me[p1]
    pe, re = ew_basket(BASKET, h)
    ew_map = dict(zip(pe, re))
    ew_at = np.array([ew_map.get(p, np.nan) for p in p1[m]])
    a = r1[m]
    b = r1[~m]
    print("  h=%2d: outright %+.2f bps -> %.2fx (4 bps) | vs EW basket %+.2f bps -> %.2fx (8 bps)"
          " | vs non-ME rank-1 %+.2f bps -> %.2fx (4 bps)"
          % (h, 1e4 * a.mean(), 1e4 * a.mean() / 4.0,
             1e4 * (a.mean() - np.nanmean(ew_at)), 1e4 * (a.mean() - np.nanmean(ew_at)) / 8.0,
             1e4 * (a.mean() - b.mean()), 1e4 * (a.mean() - b.mean()) / 4.0))

print()
print("=" * 78)
print("8. SELECTION HONESTY: drop GLD/GDX/SLV from the basket entirely")
print("=" * 78)
NOMET = [t for t in BASKET if t not in METALS]
for h in HS:
    p1, t1, r1, _ = ranked_trade(BASKET, h, 0)
    p2, t2, r2, _ = ranked_trade(NOMET, h, 0)
    m1, m2 = is_me[p1], is_me[p2]
    rows = [summarize(r1[m1], "full basket, ME-0"),
            summarize(r2[m2], "NO METALS, ME-0"),
            summarize(r1[~m1], "full basket, non-ME"),
            summarize(r2[~m2], "NO METALS, non-ME")]
    show(rows, "h=%d" % h)
    print("  metals share of month-end picks: %.1f%%  (%s)"
          % (100 * np.isin(t1[m1], METALS).mean(),
             pd.Series(t1[m1][np.isin(t1[m1], METALS)]).value_counts().to_dict()))
