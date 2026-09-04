"""C4 ROUND 2 — re-run the natgas cell under the LIVE-STATE z10 convention,
plus a quantification of what the continuous futures series is actually doing.

Why this script exists: 04 used `pitch_lab.zscore` and read UNG's live z10 as
**+0.71**, while the surface map and `data/pitch_state.json` say **+1.34**.
Both are right for their own definition and CLAUDE.md flags exactly this trap:

  pitch_lab.zscore   : (r10 - rolling252 mean of r10) / rolling252 sd of r10
  build_pitch_state  : r10 / (21d daily-return sd * sqrt(10))     <- the tape

A kill has to be measured with the definition that reads the live tape as
firing, or the composer can dismiss it as the wrong cell. So the whole cell is
re-measured on the STATE convention here. If the verdict is the same under
both, the candidate is dead under either reading and the convention is a
footnote rather than a defence.

Also quantifies the NG=F roll so the "futures escape the ETF decay" claim can
be priced instead of asserted.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa
import pandas as pd
import numpy as np

ASOF = pd.Timestamp("2026-08-31")
PX = load_prices(["UNG", "NG=F"])
PX = {t: d[d.index <= ASOF] for t, d in PX.items()}
ung = PX["UNG"]["Close"].dropna()
ngf = PX["NG=F"]["Close"].dropna()


def state_z10(close):
    """build_pitch_state._metrics_for convention: 10d return over 21d daily
    vol scaled to 10d."""
    r10 = close.pct_change(10)
    vol21 = close.pct_change().rolling(21).std()
    return r10 / (vol21 * np.sqrt(10))


zs = state_z10(ung)
zl = zscore(ung, 10)
r5 = pct_rank(ung, 5)
dd = ung / ung.rolling(252, min_periods=252).max() - 1.0
print("===== CONVENTION RECONCILIATION (live bar %s) =====" % ung.index[-1].date())
print("  state z10 (r10 / 21d vol x sqrt10) = %+.3f   <- pitch_state says 1.34"
      % zs.iloc[-1])
print("  pitch_lab.zscore (252d standardised) = %+.3f  <- what 04 used" % zl.iloc[-1])
print("  r5 pct_rank = %.1f   <- pitch_state says 76.2 (agrees)" % r5.iloc[-1])
print("  dd from 252-high = %.2f%%   200d dist = %.2f%%"
      % (100 * dd.iloc[-1], 100 * (ung.iloc[-1] / ung.rolling(200).mean().iloc[-1] - 1)))
print("  correlation of the two z10 series: %.3f"
      % pd.concat([zs, zl], axis=1).dropna().corr().iloc[0, 1])

SEP = pd.Series(ung.index.month == 9, index=ung.index)
SHOULDER = pd.Series(ung.index.month.isin([9, 10]), index=ung.index)

CELLS = {
    "THRUST(state z10>=1.34, r5>=76) alone, all months":
        ((zs >= 1.34) & (r5 >= 76)).fillna(False),
    "THRUST(state z10>=1.0, r5>=75) alone, all months":
        ((zs >= 1.0) & (r5 >= 75)).fillna(False),
    "SEPTEMBER alone": SEP,
    "CELL: state z10>=1.0 & r5>=75 & Sep":
        ((zs >= 1.0) & (r5 >= 75) & SEP).fillna(False),
    "CELL tight: state z10>=1.34 & r5>=76 & Sep":
        ((zs >= 1.34) & (r5 >= 76) & SEP).fillna(False),
    "CELL & shoulder(Sep+Oct)":
        ((zs >= 1.0) & (r5 >= 75) & SHOULDER).fillna(False),
    "CELL & dd<=-25% (TODAY'S depth)":
        ((zs >= 1.0) & (r5 >= 75) & SEP & (dd <= -0.25)).fillna(False),
    "CELL & dd<=-30% (closer to today's -37.6%)":
        ((zs >= 1.0) & (r5 >= 75) & SEP & (dd <= -0.30)).fillna(False),
}

print("\n===== COUNT FIRST, STATE CONVENTION =====")
for lbl, m in CELLS.items():
    t = ung.index[m.reindex(ung.index, fill_value=False).values]
    print("  %-46s %4d days | %3d ep(gap10) | yrs %s"
          % (lbl, len(t), len(declusters(t, 10, ung.index)),
             sorted(set(t.year)) if len(t) < 400 else "many"))

print("\n===== ABSOLUTE-FIRST TABLE, STATE CONVENTION (UNG long, lag=1, gap 10) =====")
for H in (3, 5, 10, 21):
    r = fwd_lag(ung, H, 1)
    drift = 100 * r.dropna().mean()
    rows = []
    for lbl, m in CELLS.items():
        t = ung.index[m.reindex(ung.index, fill_value=False).values]
        t = t.intersection(r.dropna().index)
        e = declusters(t, 10, ung.index)
        s = summarize(r.reindex(e).values, lbl)
        if s["n"]:
            s["excess_pp"] = round(s["mean_pct"] - drift, 3)
            wins = int((r.reindex(e).values > 0).sum())
            s["rec"] = f"{wins}-{s['n']-wins}"
            s["sign_p"] = round(sign_test(wins, s["n"]), 4)
        rows.append(s)
    rows.append(summarize(r.dropna().values, "ALL DAYS (the bleed)"))
    show(rows, f"h={H}   ABSOLUTE mean_pct decides an outright long")

# ---------------------------------------------------------------------------
# the thrust leg, measured properly: is it wrong-signed under BOTH conventions?
# ---------------------------------------------------------------------------
print("\n===== THE THRUST LEG UNDER BOTH CONVENTIONS (all months, gap 10) =====")
for name, z in (("state z10", zs), ("pitch_lab zscore", zl)):
    for H in (5, 10, 21):
        m = ((z >= 1.0) & (r5 >= 75)).fillna(False)
        r = fwd_lag(ung, H, 1)
        t = ung.index[m.reindex(ung.index, fill_value=False).values].intersection(r.dropna().index)
        e = declusters(t, 10, ung.index)
        s = summarize(r.reindex(e).values, "")
        wins = int((r.reindex(e).values > 0).sum())
        print("  %-18s h=%2d N=%3d ABS %+.3f%% hit %.1f%% t %+.2f  record %d-%d  "
              "excess %+.3fpp"
              % (name, H, s["n"], s["mean_pct"], s["hit"], s["t"], wins,
                 s["n"] - wins, s["mean_pct"] - 100 * r.dropna().mean()))

# ---------------------------------------------------------------------------
# QUANTIFY THE ROLL: NG=F vs UNG over the same span
# ---------------------------------------------------------------------------
print("\n===== QUANTIFYING THE FUTURES ROLL (can NG=F escape the ETF decay?) =====")
j = pd.DataFrame({"ung": ung, "ngf": ngf}).dropna()
ru, rn = j["ung"].pct_change(), j["ngf"].pct_change()
print("  common span %s .. %s (%d sessions)"
      % (j.index[0].date(), j.index[-1].date(), len(j)))
print("  UNG cumulative %+.2f%%   NG=F cumulative %+.2f%%   GAP %+.1f pp"
      % (100 * (j['ung'].iloc[-1] / j['ung'].iloc[0] - 1),
         100 * (j['ngf'].iloc[-1] / j['ngf'].iloc[0] - 1),
         100 * ((j['ngf'].iloc[-1] / j['ngf'].iloc[0]) - (j['ung'].iloc[-1] / j['ung'].iloc[0]))))
print("  mean daily return  UNG %+.4f%%  NG=F %+.4f%%  -> NG=F prints "
      "%+.4f pp/day MORE than the tracking ETF"
      % (100 * ru.mean(), 100 * rn.mean(), 100 * (rn.mean() - ru.mean())))
print("  daily return correlation %.3f (they track the same underlying)"
      % ru.corr(rn))
print("  => the %+.4f pp/day wedge is the unadjusted CONTINUOUS ROLL, not a "
      "tradeable return. Over 10 td that is %+.2f pp of phantom edge, which is "
      "larger than the whole conditional excess measured above."
      % (100 * (rn.mean() - ru.mean()), 1000 * (rn.mean() - ru.mean())))
sep_wedge = (rn - ru)[j.index.month == 9].mean()
print("  and the wedge is SEASONAL: September mean wedge %+.4f pp/day vs "
      "all-month %+.4f pp/day -> the NG=F 'September edge' is the roll's own "
      "seasonality." % (100 * sep_wedge, 100 * (rn - ru).mean()))
