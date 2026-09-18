"""c1 — SHORT SPY on macro-event DENSITY (>=4 scheduled events in the forward
6 trading sessions). Pre-specified: direction SHORT, vehicle SPY, entry lag=1
MOC, time exit, no stop.

Everything the brief demands, in order:
  0. premise / rarity: density distribution, episode count
  1. the state, day-level + episode-level, vs the three controls
  2. the confound battery: month-matched, tdom-matched, quad_witching removed,
     non-quarterly months only
  3. dose response >=4 / ==3 / ==2 / <=1  (monotone supports, spike-at-4 kills)
  4. placebo ladder k = -5..+5 on the anchor
  5. MECHANISM: does forward ^VIX actually rise, does forward realised vol rise
  6. LONG side reported plainly
  7. era split pre/post-2018, midterm split
  8. concentration: cluster_note, drop-best-2, drop-best-year
  9. cost at ~2 bp all-in (short SPY pays borrow)
 10. tail: worst episode, largest adverse DAILY move inside the window
 11. definition fragility: window length 5/6/7/8 sessions
 12. permutation, CHARGED and UNCHARGED, naming the defended statistic
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

EVENT_KINDS = ["nfp", "cpi", "ppi", "fomc_decision", "opex",
               "quad_witching", "vix_expiry"]
WIN = 6            # forward trading sessions D+1 .. D+WIN
THRESH = 4         # pre-specified density threshold
PITCH_H = 6        # placeholder; the horizon scan below decides
COST_BPS = 2.0     # SPY 1bp round trip + borrow on the short leg

px = close_panel(["SPY", "^VIX"])
spy = px["SPY"].dropna()
idx = spy.index                                   # SPY's calendar is THE calendar
vix = px["^VIX"].reindex(idx).ffill()             # rule 8: reindex ^VIX to SPY
pos = pd.Series(range(len(idx)), index=idx)

ev = load_events(EVENT_KINDS)
ev = ev[(ev["date"] >= idx[0]) & (ev["date"] <= idx[-1])]


def density(kinds, win=WIN, count_dates=False):
    """# of event instances (or distinct dates) landing in sessions D+1..D+win."""
    e = ev[ev["event"].isin(kinds)]
    # map each event date to the first trading session on/after it
    locs = idx.searchsorted(pd.DatetimeIndex(e["date"]))
    ok = locs < len(idx)
    locs = locs[ok]
    edates = pd.DatetimeIndex(e["date"].values[ok])
    per_sess = np.zeros(len(idx))
    per_sess_dates = [set() for _ in range(len(idx))]
    for L, d in zip(locs, edates):
        per_sess[L] += 1
        per_sess_dates[L].add(d)
    if count_dates:
        cnt = np.array([len(s) for s in per_sess_dates], dtype=float)
    else:
        cnt = per_sess
    fwd = np.full(len(idx), np.nan)
    cs = np.concatenate([[0.0], np.cumsum(cnt)])
    for p in range(len(idx)):
        hi = min(len(idx), p + win + 1)
        if p + win >= len(idx):
            continue
        fwd[p] = cs[hi] - cs[p + 1]
    return pd.Series(fwd, index=idx)


dens = density(EVENT_KINDS)
dens_dates = density(EVENT_KINDS, count_dates=True)
dens_noquad = density([k for k in EVENT_KINDS if k != "quad_witching"])

print("=" * 80)
print(f"0. PREMISE: how rare is 'density >= {THRESH} in the forward {WIN} sessions'?")
vc = dens.dropna().value_counts().sort_index()
tot = int(vc.sum())
print("   density distribution over all sessions "
      f"{idx[0].date()}..{idx[-1].date()} (N={tot}):")
for k, v in vc.items():
    print(f"     {int(k)} events: {v:5d} days ({100*v/tot:5.2f}%)"
          + ("   <-- STATE" if k >= THRESH else ""))
state_days = int((dens >= THRESH).sum())
print(f"   ** density >= {THRESH}: {state_days} days = "
      f"{100*state_days/tot:.2f}% of all sessions **")
print(f"   distinct-DATE version >= {THRESH}: "
      f"{int((dens_dates >= THRESH).sum())} days "
      f"({100*(dens_dates >= THRESH).sum()/tot:.2f}%)")

# today's reading
last = idx[-1]
print(f"\n   today's anchor {last.date()}: density(D+1..D+{WIN}) = "
      f"{dens.get(last, float('nan'))} (nan = window runs past cache end)")
fut = ev[ev["date"] > last].head(10)
print("   next scheduled events in the cache:")
for _, r in fut.iterrows():
    print(f"     {r['date'].date()}  {r['event']}")

mask = (dens >= THRESH).fillna(False)
sig_all = idx[mask.values]
print(f"\n   trigger days {len(sig_all)}, span {sig_all[0].date()} .. "
      f"{sig_all[-1].date()}")
by_month = pd.Series(1, index=sig_all).groupby(sig_all.month).sum()
print("   trigger days by calendar month:")
print("   " + "  ".join(f"{m}:{by_month.get(m, 0)}" for m in range(1, 13)))
print("   -> quarterly months (3,6,9,12) share = "
      f"{100*by_month.reindex([3,6,9,12]).fillna(0).sum()/len(sig_all):.1f}%")

# ------------------------------------------------------------------ 1. horizon
print("\n" + "=" * 80)
print("1. HORIZON SCAN, SHORT SPY, episode level (min_gap = h)")
epi_all = {}
rows = []
for h in range(1, 11):
    ret = vehicle_ret(px, [("SPY", -1.0)], h, 1)
    valid = ret.dropna().index
    t = pd.DatetimeIndex(sig_all).intersection(valid)
    epi = declusters(t, max(h, WIN), valid)   # fixed decluster gap >= window
    epi_all[h] = epi
    v = ret.loc[epi].values
    base = ret.loc[valid]
    s = summarize(v, f"h={h}")
    s["base_pct"] = round(100 * base.mean(), 3)
    s["edge_pct"] = round(s["mean_pct"] - 100 * base.mean(), 3)
    s["signp"] = round(sign_test(int((v > 0).sum()), len(v)), 4)
    rows.append(s)
show(rows, "SHORT SPY by horizon (decluster gap = max(h, 6))")

print("\n   LONG side (direction honesty) — same cells, sign flipped:")
for r in rows:
    print(f"     h={r['label'][2:]:>2s}  LONG SPY mean {-r['mean_pct']:+.3f}%  "
          f"hit {100-r['hit']:.1f}%  t {-r['t']:+.2f}  N={r['n']}")

# pick the horizon that carries past the cluster: window ends D+6, entry D+1,
# so exit D+1+h >= D+6 needs h >= 5. Report both 5 and 6.
for H in (5, 6):
    print(f"\n{'='*80}\n   ---- DETAIL AT h={H} (carries entry D+1 to exit D+{1+H}) ----")
    ret = vehicle_ret(px, [("SPY", -1.0)], H, 1)
    valid = ret.dropna().index
    t = pd.DatetimeIndex(sig_all).intersection(valid)
    epi = declusters(t, max(H, WIN), valid)
    v = ret.loc[epi].values
    in_span = (valid >= t[0]) & (valid <= t[-1])
    loc = local_control(valid, t)
    show([summarize(ret.loc[t].values, f"COND day-level N={len(t)}"),
          summarize(v, f"COND episodes N={len(epi)}"),
          summarize(ret.loc[valid][in_span].values, "CTRL-a own drift same span"),
          summarize(ret.loc[valid].values, "CTRL-b all days"),
          summarize(ret.loc[loc].values, "CTRL-c local +/-126td ex-trigger")],
         f"conditional vs controls, h={H}")
    ctrl = ret.loc[valid][in_span].values
    se = np.sqrt(v.var(ddof=1) / len(v) + ctrl.var(ddof=1) / len(ctrl))
    w = int((v > 0).sum())
    print(f"   diff vs own-drift {100*(v.mean()-ctrl.mean()):+.3f}%  welch t "
          f"{(v.mean()-ctrl.mean())/se:+.2f}  boot P(mean<=0) "
          f"{bootstrap_p_le0(v):.3f}  record {w}-{len(v)-w} sign p "
          f"{sign_test(w, len(v)):.4f}")
    print(f"   concentration: {cluster_note(epi, v)}")
    o = np.argsort(-v)
    print(f"   drop-best-2: mean {100*np.delete(v, o[:2]).mean():+.3f}% "
          f"(N={len(v)-2})")
    yrs = pd.DatetimeIndex(epi).year
    by_y = pd.Series(v).groupby(yrs.values).sum()
    worst_y = by_y.idxmax()
    keep = yrs != worst_y
    print(f"   drop-best-year ({worst_y}, contributed "
          f"{100*by_y.max():+.2f}pp): mean {100*v[keep].mean():+.3f}% "
          f"(N={int(keep.sum())})")
    show(era_split(epi, v), f"era split h={H}")
    mid = pd.DatetimeIndex(epi).year % 4 == 2
    show([summarize(v[mid], f"MIDTERM yrs N={int(mid.sum())}"),
          summarize(v[~mid], f"non-midterm N={int((~mid).sum())}")],
         f"midterm split h={H}")
    edge_bps = 100 * 100 * v.mean()
    print(f"   cost: {edge_bps:+.1f} bp gross vs {COST_BPS} bp all-in -> "
          f"{edge_bps/COST_BPS:+.1f}x cost (need >= 5x)")
    # tail
    print(f"   TAIL: worst episode {100*v.min():+.2f}% on "
          f"{epi[int(np.argmin(v))].date()}")
    adv = []
    for d in epi:
        p = pos.get(d)
        seg = spy.values[p + 1: p + 1 + H + 1]
        r = seg[1:] / seg[:-1] - 1.0
        adv.append(100 * r.max())       # worst move AGAINST a short = biggest up day
    adv = np.array(adv)
    print(f"   largest adverse DAILY move inside window (short pays on up days): "
          f"median {np.median(adv):+.2f}%  p90 {np.percentile(adv, 90):+.2f}%  "
          f"max {adv.max():+.2f}%")
    if H == 6:
        DEF = dict(h=H, epi=epi, v=v, ret=ret, valid=valid)

# ------------------------------------------------------- 2. confound battery
H = DEF["h"]
ret, valid, epi, v = DEF["ret"], DEF["valid"], DEF["epi"], DEF["v"]
print("\n" + "=" * 80)
print(f"2. CONFOUND BATTERY at h={H}")

# (a) month-matched control
r_valid = ret.loc[valid]
trig_set = set(pd.DatetimeIndex(sig_all))
nontrig = r_valid[~r_valid.index.isin(trig_set)]
m_mean = nontrig.groupby(nontrig.index.month).mean()
matched = np.array([m_mean.get(d.month, np.nan) for d in epi])
diff_m = v - matched
print(f"   (a) MONTH-matched: episode mean {100*v.mean():+.3f}%  "
      f"month-matched control {100*np.nanmean(matched):+.3f}%  "
      f"diff {100*np.nanmean(diff_m):+.3f}%  "
      f"t {np.nanmean(diff_m)/(np.nanstd(diff_m, ddof=1)/np.sqrt(len(diff_m))):+.2f}")

# (b) trading-day-of-month matched
tdom = pd.Series(idx, index=idx).groupby([idx.year, idx.month]).rank().astype(int)
tdom = pd.Series(tdom.values, index=idx)
nt_tdom = tdom.reindex(nontrig.index)
d_mean = nontrig.groupby(nt_tdom.values).mean()
matched2 = np.array([d_mean.get(tdom.get(d), np.nan) for d in epi])
diff_d = v - matched2
print(f"   (b) TDOM-matched : episode mean {100*v.mean():+.3f}%  "
      f"tdom-matched control {100*np.nanmean(matched2):+.3f}%  "
      f"diff {100*np.nanmean(diff_d):+.3f}%  "
      f"t {np.nanmean(diff_d)/(np.nanstd(diff_d, ddof=1)/np.sqrt(len(diff_d))):+.2f}")

# (c) quad_witching removed from the event set
m_nq = (dens_noquad >= THRESH).fillna(False)
t_nq = idx[m_nq.values]
t_nq = pd.DatetimeIndex(t_nq).intersection(valid)
epi_nq = declusters(t_nq, max(H, WIN), valid)
show([summarize(ret.loc[epi_nq].values,
                f"quad_witching REMOVED from set (N={len(epi_nq)})"),
      summarize(v, f"original set (N={len(epi)})")],
     "   (c) event-set sensitivity")

# (d) non-quarterly months only
qmask = pd.DatetimeIndex(epi).month.isin([3, 6, 9, 12])
show([summarize(v[qmask], f"quarterly months Mar/Jun/Sep/Dec (N={int(qmask.sum())})"),
      summarize(v[~qmask], f"NON-quarterly months (N={int((~qmask).sum())})")],
     "   (d) quarterly vs non-quarterly")

# ---------------------------------------------------------- 3. dose response
print("\n" + "=" * 80)
print(f"3. DOSE RESPONSE at h={H} (gate attribution + DISCARDED COMPLEMENT)")
rows = []
for lbl, m in [(">=5", dens >= 5), ("==4", dens == 4), ("==3", dens == 3),
               ("==2", dens == 2), ("==1", dens == 1), ("==0", dens == 0),
               (">=4 (PITCHED)", dens >= THRESH),
               ("<=3 (COMPLEMENT)", dens <= 3),
               (">=3", dens >= 3), (">=2", dens >= 2)]:
    t = pd.DatetimeIndex(idx[m.fillna(False).values]).intersection(valid)
    if len(t) == 0:
        rows.append({"label": lbl, "n": 0})
        continue
    e = declusters(t, max(H, WIN), valid)
    s = summarize(ret.loc[e].values, lbl)
    s["n_days"] = len(t)
    rows.append(s)
show(rows, f"   density dose response, SHORT SPY, h={H}")

# --------------------------------------------------------- 4. placebo ladder
print("\n" + "=" * 80)
print(f"4. PLACEBO LADDER k=-5..+5 (anchor shifted k sessions), h={H}")
rows = []
for k in range(-5, 6):
    shifted = []
    for d in pd.DatetimeIndex(sig_all):
        p = pos.get(d)
        if p is None:
            continue
        q = p + k
        if 0 <= q < len(idx):
            shifted.append(idx[q])
    t = pd.DatetimeIndex(sorted(set(shifted))).intersection(valid)
    e = declusters(t, max(H, WIN), valid)
    s = summarize(ret.loc[e].values, f"k={k:+d}" + ("  <-- TRUE" if k == 0 else ""))
    rows.append(s)
show(rows, "   placebo ladder")
true_m = [r for r in rows if "TRUE" in r["label"]][0]["mean_pct"]
allm = [r["mean_pct"] for r in rows]
rank = 1 + sum(1 for x in allm if x > true_m)
print(f"   TRUE anchor mean {true_m:+.3f}% ranks {rank}/11 among its own placebos")

# --------------------------------------------------------- 5. THE MECHANISM
print("\n" + "=" * 80)
print("5. MECHANISM TEST: does ^VIX actually RISE across the dense window?")
vv = vix.reindex(idx)
rows = []
for lbl, dates in [(f"DENSE >= {THRESH}", epi),
                   ("all days", valid)]:
    ch, rv_f, rv_p = [], [], []
    lr = np.log(spy).diff()
    for d in pd.DatetimeIndex(dates):
        p = pos.get(d)
        if p is None or p + 1 + H >= len(idx) or p - 21 < 0:
            continue
        ch.append(vv.values[p + 1 + H] / vv.values[p + 1] - 1.0)
        fw = lr.values[p + 2: p + 2 + H]
        pr = lr.values[p - 20: p + 1]
        if np.isfinite(fw).all() and np.isfinite(pr).all():
            rv_f.append(np.std(fw, ddof=1) * np.sqrt(252))
            rv_p.append(np.std(pr, ddof=1) * np.sqrt(252))
    ch, rv_f, rv_p = np.array(ch), np.array(rv_f), np.array(rv_p)
    rows.append({"label": lbl, "n": len(ch),
                 "vix_chg_pct": round(100 * ch.mean(), 2),
                 "vix_up_share": round(100 * (ch > 0).mean(), 1),
                 "fwd_rv_ann_pct": round(100 * rv_f.mean(), 2),
                 "prior21_rv_ann_pct": round(100 * rv_p.mean(), 2),
                 "rv_delta_pp": round(100 * (rv_f - rv_p).mean(), 2)})
show(rows, f"   forward ^VIX change and realised vol across h={H}")
print("   mechanism claim: dense calendar -> hedging bid -> vol RISES, drift")
print("   suppressed. If ^VIX FALLS across the window the mechanism runs")
print("   backwards and any return leg is an unexplained cell.")

# ------------------------------------------- 11. definition fragility (window)
print("\n" + "=" * 80)
print(f"11. DEFINITION FRAGILITY: window length, h={H}")
rows = []
for w in (4, 5, 6, 7, 8, 10):
    dw = density(EVENT_KINDS, win=w)
    t = pd.DatetimeIndex(idx[(dw >= THRESH).fillna(False).values]).intersection(valid)
    if len(t) == 0:
        rows.append({"label": f"win={w}", "n": 0})
        continue
    e = declusters(t, max(H, w), valid)
    s = summarize(ret.loc[e].values, f"win={w}")
    s["n_days"] = len(t)
    s["pct_of_days"] = round(100 * len(t) / len(valid), 1)
    rows.append(s)
show(rows, "   forward-window length sensitivity (threshold fixed at 4)")

rows = []
dd = density(EVENT_KINDS, count_dates=True)
for th in (3, 4, 5):
    t = pd.DatetimeIndex(idx[(dd >= th).fillna(False).values]).intersection(valid)
    if len(t) == 0:
        continue
    e = declusters(t, max(H, WIN), valid)
    s = summarize(ret.loc[e].values, f"distinct-DATES >= {th}")
    s["n_days"] = len(t)
    rows.append(s)
show(rows, "   counting distinct event DATES instead of instances")

# ------------------------------------------------------------ 12. permutation
print("\n" + "=" * 80)
print(f"12. PERMUTATION at h={H}")
print("   DEFENDED STATISTIC: episode-level mean of SHORT SPY, density>=4,")
print(f"   window=6, h={H}, decluster gap 6.  (Pre-specified: rule 2 says no")
print("   search charge is owed. The CHARGED number below prices the grids")
print("   this script actually walked: 10 horizons x 6 windows = 60 cells.)")
rng = np.random.default_rng(42)
rv = ret.loc[valid]
n_valid = len(rv)
trig_pos = np.array([valid.get_loc(d) for d in pd.DatetimeIndex(sig_all)
                     if d in valid])
obs = v.mean()
null_true, null_max = [], []
for _ in range(2000):
    sh = rng.integers(1, n_valid)
    pp = (trig_pos + sh) % n_valid
    dts = valid[np.sort(np.unique(pp))]
    e = declusters(dts, max(H, WIN), valid)
    null_true.append(rv.loc[e].mean())
    # charged: max over the horizon grid for this same shifted mask
    best = -9e9
    for hh in (3, 5, 6, 8, 10):
        r2 = vehicle_ret(px, [("SPY", -1.0)], hh, 1).loc[valid]
        e2 = declusters(dts, max(hh, WIN), valid)
        best = max(best, r2.loc[e2].dropna().mean())
    null_max.append(best)
null_true, null_max = np.array(null_true), np.array(null_max)
print(f"   observed episode mean = {100*obs:+.3f}%")
print(f"   UNCHARGED p (obs vs shifted-mask null of the SAME cell) = "
      f"{(null_true >= obs).mean():.4f}")
print(f"   CHARGED  p (obs vs max-over-5-horizon null)            = "
      f"{(null_max >= obs).mean():.4f}")
