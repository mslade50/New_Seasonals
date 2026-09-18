"""10 -- ADVERSARIAL ATTACK on "long SPY / short IWM while the fragility dial
is in its top decile" (S6b survivor, h=5 MOC 2026-09-08 -> 2026-09-15).

Four attacks, run in kill order:

  A1  Is the dial a LABEL on a state a plain variable already describes?
      Rebuild the SAME SPY-minus-IWM cell on five cheap substitutes at a
      MATCHED ON-RATE (same number of trigger days over the same window,
      picked in-sample = generous to the substitute). If any of them
      reproduces the edge, the dial is decoration.
        (a) SPY distance above its 200d SMA          (corr +0.398 w/ dial)
        (b) days since SPY's last 5% drawdown        (corr +0.555)
        (c) SPY nearness to its 252d high            (corr +0.285)
        (d) SPY 21d realised vol, BOTTOM decile      (corr -0.231)
        (e) VIX level, BOTTOM decile                 (corr -0.011)

  A2  Is it just "SPY beats IWM", i.e. a static factor tilt with a timing
      story bolted on? Unconditional SPY-minus-IWM drift over the dial era,
      year by year, against the conditional cell.

  A3  Vintage. rd2_fragility.parquet is true point-in-time only from
      2026-07-02; earlier rows are a recompute vintage that drifted up to ~7
      dial points. (i) the cell with 2021 removed, (ii) how much of the mask
      is within 7 points of its own cut, i.e. could flip under the other
      vintage, and what the cell does when the cut is stressed +/-3.5 and
      +/-7 points.

  A4  Today vs the support. The live dial is 87.96 against a top-decile cut
      near 56; use the distance-from-extreme GRADIENT (regress the episode
      return on the dial reading, read the fit at 88) rather than an
      empty-bucket argument. Plus the Event Sleeve T2/T3 wash arithmetic.

Conventions inherited from pitch_lab: fractions in, percent out; entry lag=1
(state on close D, MOC on close D+1); episodes declustered at h td.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _survey_lib import (  # noqa: E402
    align, bootstrap_p_le0, cluster_note, declusters, fwd_lag, load_prices,
    local_control, np, pd, roll_max, show, sign_test, sma, summarize,
)

ROOT = Path(__file__).resolve().parents[3]
FRAG = pd.read_parquet(ROOT / "data" / "rd2_fragility.parquet")
PX = load_prices(["SPY", "IWM", "^VIX"])
IDX = PX["SPY"].index
C = {t: PX[t]["Close"] for t in PX}

H_MAIN = 5


def pair(h):
    """SPY minus IWM, dollar-neutral, entry MOC D+1, exit MOC D+1+h."""
    return align(fwd_lag(C["SPY"], h, 1), IDX) - align(fwd_lag(C["IWM"], h, 1), IDX)


# ---------------------------------------------------------------- the dial mask
ma = FRAG["63d"].rolling(10).mean()
q90_exp = ma.expanding(252).quantile(0.90)
m_dial = align((ma >= q90_exp).fillna(False), IDX).fillna(0).astype(bool)
ma_on_idx = align(ma, IDX)
cut_on_idx = align(q90_exp, IDX)

# The window every comparison lives in: days where the dial mask is COMPUTABLE.
WIN = IDX[align(q90_exp.notna(), IDX).fillna(0).astype(bool).values]
WIN = WIN[WIN >= FRAG.index[0]]

print("=" * 78)
print("10  ADVERSARIAL: long SPY / short IWM at the fragility-dial top decile")
print("=" * 78)
print(f"dial series {FRAG.index[0].date()} .. {FRAG.index[-1].date()}   "
      f"live 10d-MA-63d = {ma.iloc[-1]:.2f}   expanding q90 cut = {q90_exp.iloc[-1]:.2f}")
print(f"comparison window (dial mask computable): {WIN[0].date()} .. {WIN[-1].date()}"
      f"   {len(WIN)} sessions")

trig_dial = IDX[m_dial.values]
trig_dial = pd.DatetimeIndex(trig_dial).intersection(WIN)
ON_RATE = len(trig_dial) / len(WIN)
print(f"dial trigger days in window: {len(trig_dial)}  ->  ON-rate {100*ON_RATE:.2f}%")


def cellstats(ret, trig, h, label, win=None):
    """One row: episodes, mean, edge vs the SAME-WINDOW unconditional, record."""
    valid = ret.dropna().index
    t = pd.DatetimeIndex(trig).intersection(valid)
    if win is not None:
        t = t.intersection(win)
    if len(t) == 0:
        return {"label": label, "n": 0}
    epi = declusters(t, h, valid)
    ep = ret.loc[epi].values
    base_win = valid if win is None else valid.intersection(win)
    base = float(ret.loc[base_win].mean())
    w = int((ep > 0).sum())
    return {"label": label, "n_days": len(t), "n": len(epi),
            "mean_pct": round(100 * ep.mean(), 3),
            "hit": round(100 * (ep > 0).mean(), 1),
            "base_win_pct": round(100 * base, 3),
            "edge_pct": round(100 * (ep.mean() - base), 3),
            "rec": f"{w}-{len(epi)-w}",
            "sign_p": round(sign_test(w, len(epi)), 4),
            "worst_pct": round(100 * ep.min(), 2)}


# reproduce the headline first so the attack is measured against the same cell
print("\n" + "-" * 78)
print("REPRODUCTION of the pitched cell (must match +0.517% / 47-24 at h=5)")
print("-" * 78)
rows = [cellstats(pair(h), trig_dial, h, f"DIAL top decile  h={h}", WIN)
        for h in (1, 2, 3, 5, 10)]
show(rows)
r5 = pair(H_MAIN)
epi5 = declusters(pd.DatetimeIndex(trig_dial).intersection(r5.dropna().index),
                  H_MAIN, r5.dropna().index)
ep5 = r5.loc[epi5].values
print(f"  bootstrap P(mean<=0) at h=5 = {bootstrap_p_le0(ep5):.4f}   "
      f"{cluster_note(epi5, ep5)}")

# =============================================================== ATTACK 1
print("\n" + "=" * 78)
print("ATTACK 1 -- does a CHEAP SUBSTITUTE reproduce it at a matched ON-rate?")
print("=" * 78)

spy = C["SPY"]
vix = align(C["^VIX"], IDX)

# (a) distance above the 200d SMA
sub_a = (spy / sma(spy, 200) - 1.0)

# (b) days since SPY's last 5% drawdown from its trailing 252d high
dd = spy / roll_max(spy, 252) - 1.0
flag = (dd <= -0.05).fillna(False)
days_since = np.zeros(len(spy))
cnt = 0
for i, f in enumerate(flag.values):
    cnt = 0 if f else cnt + 1
    days_since[i] = cnt
sub_b = pd.Series(days_since, index=spy.index)
sub_b[roll_max(spy, 252).isna()] = np.nan

# (c) nearness to the 252d high (higher = nearer)
sub_c = spy / roll_max(spy, 252)

# (d) 21d realised vol -> BOTTOM decile, so negate to keep "top-k" uniform
lr = np.log(spy).diff()
sub_d = -(lr.rolling(21).std() * np.sqrt(252))

# (e) VIX level -> BOTTOM decile, negate
sub_e = -vix

SUBS = {
    "(a) SPY dist above 200d SMA  (top)": sub_a,
    "(b) days since SPY -5% DD    (top)": sub_b,
    "(c) SPY nearness to 252d high(top)": sub_c,
    "(d) SPY 21d realised vol     (bot)": sub_d,
    "(e) VIX level                (bot)": sub_e,
}

pos_win = pd.Series(range(len(WIN)), index=WIN)
rows, ovl = [], []
rows.append(cellstats(pair(H_MAIN), trig_dial, H_MAIN, "DIAL (the candidate)", WIN))
sub_trigs = {}
for name, s in SUBS.items():
    sv = align(s, IDX).reindex(WIN).dropna()
    k = min(len(trig_dial), len(sv))
    top = sv.sort_values(ascending=False).index[:k]
    top = pd.DatetimeIndex(sorted(top))
    sub_trigs[name] = top
    rows.append(cellstats(pair(H_MAIN), top, H_MAIN, name, WIN))
    inter = len(set(top) & set(trig_dial))
    ovl.append({"substitute": name, "n_days": len(top),
                "overlap_with_dial_days": inter,
                "jaccard": round(inter / len(set(top) | set(trig_dial)), 3)})
print(f"\nh={H_MAIN}, matched ON-rate ({len(trig_dial)} trigger days each, "
      f"substitutes ranked IN-SAMPLE = generous to the substitute):")
show(rows)
print("\nhow much of the same tape are they actually selecting?")
show(ovl)

print(f"\nh=10 replication of the same table:")
rows10 = [cellstats(pair(10), trig_dial, 10, "DIAL (the candidate)", WIN)]
for name, top in sub_trigs.items():
    rows10.append(cellstats(pair(10), top, 10, name, WIN))
show(rows10)

# the decisive read: does the dial add anything ON TOP of the best substitute?
best = max((r for r in rows[1:] if r.get("n")), key=lambda r: r["mean_pct"])
print(f"\nbest substitute at h=5: {best['label']}  {best['mean_pct']:+.3f}%  "
      f"{best['rec']}  sign p={best['sign_p']}")
bt = sub_trigs[best["label"]]
only_dial = pd.DatetimeIndex(sorted(set(trig_dial) - set(bt)))
only_sub = pd.DatetimeIndex(sorted(set(bt) - set(trig_dial)))
both = pd.DatetimeIndex(sorted(set(bt) & set(trig_dial)))
show([cellstats(pair(H_MAIN), both, H_MAIN, "BOTH dial and best sub", WIN),
      cellstats(pair(H_MAIN), only_dial, H_MAIN, "DIAL only (sub says no)", WIN),
      cellstats(pair(H_MAIN), only_sub, H_MAIN, "SUB only (dial says no)", WIN)],
     "incremental content: where do dial and the best substitute disagree?")

# =============================================================== ATTACK 2
print("\n" + "=" * 78)
print("ATTACK 2 -- is it just the static SPY-over-IWM tilt?")
print("=" * 78)
r = pair(H_MAIN)
valid = r.dropna().index
win_valid = valid.intersection(WIN)
loc = local_control(valid, pd.DatetimeIndex(trig_dial))
show([summarize(r.loc[pd.DatetimeIndex(trig_dial)].values, "COND day-level"),
      summarize(ep5, f"COND episodes (N={len(ep5)})"),
      summarize(r.loc[win_valid].values, "UNCOND, dial-era window (the honest control)"),
      summarize(r.loc[valid].values, "UNCOND, all days full history"),
      summarize(r.loc[loc].values, "UNCOND, local +/-126td ex-trigger"),
      summarize(r.loc[valid.difference(WIN)].values, "UNCOND, PRE-dial-era only")],
     f"SPY minus IWM, h={H_MAIN}, lag=1")

print("\nunconditional SPY-minus-IWM 5d drift, YEAR BY YEAR over the dial era")
yr = []
for y in sorted(set(win_valid.year)):
    d = win_valid[win_valid.year == y]
    tt = pd.DatetimeIndex(trig_dial).intersection(d)
    ee = declusters(tt, H_MAIN, valid) if len(tt) else pd.DatetimeIndex([])
    yr.append({"year": y, "uncond_days": len(d),
               "uncond_mean_pct": round(100 * float(r.loc[d].mean()), 3),
               "trig_days": len(tt), "epi": len(ee),
               "cond_mean_pct": round(100 * float(r.loc[ee].mean()), 3) if len(ee) else np.nan,
               "cond_minus_uncond_pp": round(100 * (float(r.loc[ee].mean()) - float(r.loc[d].mean())), 3) if len(ee) else np.nan})
print(pd.DataFrame(yr).to_string(index=False))
share = 100 * float(r.loc[win_valid].mean()) / (100 * ep5.mean())
print(f"\n  unconditional dial-era drift explains {100*share:.1f}% of the "
      f"conditional +{100*ep5.mean():.3f}%")

# =============================================================== ATTACK 3
print("\n" + "=" * 78)
print("ATTACK 3 -- vintage and 2021 concentration")
print("=" * 78)
by_year = pd.Series(ep5, index=epi5).groupby(epi5.year)
print("episode count and mean by year (h=5):")
print(pd.DataFrame({"n": by_year.size(),
                    "mean_pct": (100 * by_year.mean()).round(3),
                    "wins": by_year.apply(lambda s: int((s > 0).sum()))}).to_string())

drop_rows = []
for y in sorted(set(epi5.year)):
    keep = epi5[epi5.year != y]
    kv = r5.loc[keep].values
    w = int((kv > 0).sum())
    drop_rows.append({"drop_year": y, "n": len(kv),
                      "mean_pct": round(100 * kv.mean(), 3),
                      "rec": f"{w}-{len(kv)-w}",
                      "sign_p": round(sign_test(w, len(kv)), 4)})
show(drop_rows, "LOYO on the episodes (drop each year in turn)")

pit_cut = pd.Timestamp("2026-07-02")
for lbl, sel in (("RECOMPUTE vintage (< 2026-07-02)", epi5[epi5 < pit_cut]),
                 ("TRUE PIT (>= 2026-07-02)", epi5[epi5 >= pit_cut])):
    v = r5.loc[sel].values
    if len(v):
        w = int((v > 0).sum())
        print(f"  {lbl}: n={len(v)}  mean {100*v.mean():+.3f}%  {w}-{len(v)-w}  "
              f"sign p={sign_test(w, len(v)):.4f}")

# mask fragility: how close is each trigger to its own cut?
marg = (ma_on_idx - cut_on_idx)
print("\nmask fragility: distance of each dial reading from its expanding q90 cut")
d_trig = marg.loc[pd.DatetimeIndex(trig_dial)].dropna()
d_epi = marg.loc[epi5].dropna()
for lbl, d in (("all trigger days", d_trig), ("h=5 episodes", d_epi)):
    print(f"  {lbl}: n={len(d)}  median +{d.median():.2f} pts above cut; "
          f"within 7 pts of the cut: {int((d <= 7).sum())} "
          f"({100*float((d <= 7).mean()):.1f}%);  within 3.5: "
          f"{int((d <= 3.5).sum())} ({100*float((d <= 3.5).mean()):.1f}%)")
non = marg.loc[WIN.difference(pd.DatetimeIndex(trig_dial))].dropna()
print(f"  NON-trigger days within 7 pts BELOW the cut: {int(((non < 0) & (non >= -7)).sum())}"
      f"  (would flip IN under an upward vintage drift)")

print("\ncut stress: shift the top-decile threshold by +/-3.5 and +/-7 dial points")
srows = []
for shift in (-7.0, -3.5, 0.0, 3.5, 7.0):
    m = align(((ma - cut_on_idx.reindex(ma.index) - shift) >= 0).fillna(False), IDX)
    m = m.fillna(0).astype(bool)
    t = pd.DatetimeIndex(IDX[m.values]).intersection(WIN)
    srows.append(dict(cellstats(pair(H_MAIN), t, H_MAIN, f"cut {shift:+.1f} pts", WIN)))
show(srows)

# =============================================================== ATTACK 4
print("\n" + "=" * 78)
print("ATTACK 4 -- today's 87.96 versus where the support actually sits")
print("=" * 78)
dv = ma_on_idx.loc[epi5].values
print(f"episode dial readings: min {dv.min():.1f}  med {np.median(dv):.1f}  "
      f"max {dv.max():.1f}   LIVE = {ma.iloc[-1]:.2f}")
for thr in (70, 75, 80, 85, 87.96):
    sel = epi5[dv >= thr]
    v = r5.loc[sel].values
    if len(v):
        w = int((v > 0).sum())
        print(f"  episodes with dial >= {thr:>5}: n={len(v):3d}  mean {100*v.mean():+.3f}%"
              f"  {w}-{len(v)-w}  sign p={sign_test(w, len(v)):.4f}")
    else:
        print(f"  episodes with dial >= {thr:>5}: n=0  (EMPTY)")

slope, intercept = np.polyfit(dv, ep5, 1)
fit88 = intercept + slope * float(ma.iloc[-1])
resid = ep5 - (intercept + slope * dv)
r2 = 1 - resid.var() / ep5.var()
se_slope = np.sqrt(resid.var(ddof=2) / ((dv - dv.mean()) ** 2).sum())
print(f"\n  GRADIENT: episode return = {100*intercept:+.3f}% + "
      f"{100*slope:+.4f}pp per dial point   R2={r2:.3f}  slope t={slope/se_slope:+.2f}")
print(f"  fitted value at the live dial {ma.iloc[-1]:.2f}: {100*fit88:+.3f}%  "
      f"(vs cell mean {100*ep5.mean():+.3f}%)")

print("\n  EVENT SLEEVE WASH (arithmetic, no data):")
acct, atr_risk_bps = 750_000.0, 30.0
spy_px, spy_atr = 770.19, 6.221
risk_dollars = acct * atr_risk_bps / 1e4
# pitch sizing convention: risk = ATR-denominated, notional = risk/ATR * price
shares = risk_dollars / spy_atr
notional = shares * spy_px
t2 = 0.10 * acct
print(f"    pitch SPY leg at {atr_risk_bps:.0f} bps of ATR risk = ${risk_dollars:,.0f} risk"
      f"  -> {shares:.0f} sh  -> ${notional:,.0f} notional")
print(f"    Event Sleeve T2 shorts SPY 10% of the fixed $750k book = ${t2:,.0f} notional,"
      f"  MOC 2026-09-10 -> MOO 2026-09-16 (INSIDE the 09-08..09-15 hold)")
print(f"    T2 cancels {100*t2/notional:.0f}% of the pitch's SPY leg for 3 of 5 sessions")
print(f"    T3 shorts IWM 15% = ${0.15*acct:,.0f} from 2026-09-18 -- AFTER the exit, no overlap")
print("=" * 78)

# ------------------------------------------------------- A4b: name the support
print("\nA4b  the episodes that actually sit where the live reading sits (dial >= 80)")
hi = epi5[dv >= 80]
for d in hi:
    print(f"    {d.date()}  dial {ma_on_idx.loc[d]:6.2f}  ret {100*r5.loc[d]:+6.2f}%")
mid = epi5[(dv >= 70) & (dv < 80)]
mv = r5.loc[mid].values
w = int((mv > 0).sum())
print(f"    band [70,80): n={len(mv)} mean {100*mv.mean():+.3f}%  {w}-{len(mv)-w}  "
      f"sign p={sign_test(w, len(mv)):.4f}   <-- the weak pocket")
lo = epi5[dv < 70]
lv = r5.loc[lo].values
w = int((lv > 0).sum())
print(f"    band [cut,70): n={len(lv)} mean {100*lv.mean():+.3f}%  {w}-{len(lv)-w}  "
      f"sign p={sign_test(w, len(lv)):.4f}   <-- where the cell's weight lives")
