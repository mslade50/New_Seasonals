"""C5 round 1+2: SHORT USO / DBC / XLE across the PPI print session.

Claim under test (from 01_event_class_recon): at the anchor 2 sessions before a
PPI release -- entry MOC on the eve (= tonight's close), exit MOC on the print
(h=1, lag=1) -- commodities are the most negative class on the board.
USO -0.222% excess over own same-span drift at a 46% hit, N=242.

Kill angles this script runs:
  A. the MANDATED placebo anchor ladder k=-8..+12. A nonsense anchor beating
     the real one is the kill (registry: short UNG through a CPI window).
  B. the CPI confound. PPI and CPI sit adjacent in the calendar; if the PPI eve
     IS usually a CPI session then this cell is the CPI-day cell wearing a hat.
  C. calendar-matched controls the own-drift control cannot see: trading-day-of
     -month matched and day-of-week matched.
  D. era split, episode year histogram, midterm split, concentration.
  E. cost: 1 leg short an ETF, borrow + spread.
  F. mechanism: is the sign stable in the two sub-eras of PPI reporting, and is
     it there in the crude FUTURES-tracking name only or across the complex?
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

TK = ["USO", "DBC", "XLE", "XOP"]
px = close_panel(TK)
idx = px.index
ev = load_events(["ppi", "cpi"])
PPI = pd.DatetimeIndex(sorted(ev.loc[ev.event == "ppi", "date"].unique()))
CPI = pd.DatetimeIndex(sorted(ev.loc[ev.event == "cpi", "date"].unique()))


def anchors(dates, k):
    """Sessions exactly k td before each event (k=0 -> the event session)."""
    out = []
    for d in dates:
        loc = idx.searchsorted(pd.Timestamp(d))
        if loc >= len(idx):
            continue
        p = loc - k
        if 0 <= p < len(idx):
            out.append(idx[p])
    return pd.DatetimeIndex(sorted(set(out)))


def mask_from(dts):
    return pd.Series(True, index=dts).reindex(idx, fill_value=False)


A2 = anchors(PPI, 2)          # the live anchor: entry = anchor+1 = PPI eve
print(f"PPI events {len(PPI)}  anchors(k=2) inside price index: {len(A2)}")
print(f"today's analogue: anchor 2026-08-11, entry 2026-08-12 close, "
      f"exit 2026-08-13 (PPI) close\n")

# ---------------------------------------------------------------- B. confound
pos = pd.Series(range(len(idx)), index=idx)
entries = pd.DatetimeIndex([idx[pos[d] + 1] for d in A2 if pos[d] + 1 < len(idx)])
is_cpi_entry = entries.isin(CPI)
print("=== B. CPI confound on the ENTRY session (the PPI eve) ===")
print(f"  PPI-eve entries that are ALSO a CPI print session: "
      f"{is_cpi_entry.sum()} / {len(entries)} = {100*is_cpi_entry.mean():.1f}%")
# and the reverse: is the PPI print session itself a CPI session?
exits = pd.DatetimeIndex([idx[pos[d] + 2] for d in A2 if pos[d] + 2 < len(idx)])
print(f"  PPI print sessions that are ALSO a CPI print session: "
      f"{exits.isin(CPI).sum()} / {len(exits)}")
# how often does CPI lead PPI by exactly 1 session (today's configuration)?
lead = []
for d in PPI:
    loc = idx.searchsorted(pd.Timestamp(d))
    if loc >= len(idx) or loc == 0:
        continue
    lead.append(idx[loc - 1] in set(CPI))
print(f"  PPI preceded by CPI on the immediately prior session: "
      f"{sum(lead)} / {len(lead)} = {100*np.mean(lead):.1f}%  "
      f"(today IS this configuration)\n")

# ---------------------------------------------------------------- A. placebo
print("=== A. PLACEBO ANCHOR LADDER k=-8..+12 (excess over own same-span "
      "drift, h=1 lag=1, LONG basis; the short pays the negative) ===")
lad = []
for k in range(-8, 13):
    a = anchors(PPI, k)
    row = {"k": k, "real": "<<<< REAL" if k == 2 else ""}
    for t in TK:
        r = fwd_lag(px[t], 1, 1)
        v = r.loc[r.index.intersection(a)].dropna()
        if len(v) < 10:
            row[t] = np.nan
            continue
        span = (idx >= v.index[0]) & (idx <= v.index[-1])
        row[t] = round(100 * (v.mean() - r[span].dropna().mean()), 3)
        row[f"{t}_n"] = len(v)
    lad.append(row)
show(lad, "placebo ladder")
for t in TK:
    col = pd.Series({r["k"]: r.get(t) for r in lad}).dropna()
    real = col.get(2, np.nan)
    better = col[col < real]
    print(f"  {t}: real k=2 excess {real:+.3f}%.  nonsense anchors MORE "
          f"negative: {len(better)} of {len(col)-1}  -> {dict(better.round(3))}")

# ------------------------------------------------------- C. matched controls
print("\n=== C. calendar-matched controls (LONG basis) ===")
tdom = pd.Series(idx, index=idx).groupby([idx.year, idx.month]).rank().values
dow = idx.dayofweek
tdom_s = pd.Series(tdom, index=idx)
dow_s = pd.Series(dow, index=idx)
ent_tdom = tdom_s.loc[entries]
ent_dow = dow_s.loc[entries]
print(f"  entry tdom distribution: {ent_tdom.value_counts().head(6).to_dict()}")
print(f"  entry dow distribution : {ent_dow.value_counts().to_dict()} "
      f"(0=Mon)")
for t in TK:
    r = fwd_lag(px[t], 1, 1)
    v = r.loc[r.index.intersection(A2)].dropna()
    if len(v) < 10:
        continue
    span = (idx >= v.index[0]) & (idx <= v.index[-1])
    base = r[span].dropna()
    # tdom-matched: anchor days whose ENTRY tdom is in the observed set
    keep_t = tdom_s.reindex(base.index).isin(set(ent_tdom.values))
    keep_d = dow_s.reindex(base.index).isin(set(ent_dow.values))
    print(f"  {t}: cond {100*v.mean():+.3f}%  own-drift {100*base.mean():+.3f}%"
          f"  tdom-matched {100*base[keep_t.values].mean():+.3f}%"
          f"  dow-matched {100*base[keep_d.values].mean():+.3f}%"
          f"  -> excess vs tdom {100*(v.mean()-base[keep_t.values].mean()):+.3f}%")

# ---------------------------------------------------------------- battery
for t in ["USO", "DBC", "XLE"]:
    variants = {f"k={k} anchor": mask_from(anchors(PPI, k)) for k in (1, 2, 3)}
    variants["k=2, CPI not on entry"] = mask_from(
        pd.DatetimeIndex([d for d in A2
                          if idx[pos[d] + 1] not in set(CPI)]))
    variants["k=2, CPI IS on entry"] = mask_from(
        pd.DatetimeIndex([d for d in A2 if idx[pos[d] + 1] in set(CPI)]))
    battery(px, mask_from(A2), [(t, -1.0)], 1,
            f"C5 SHORT {t} anchor k=2 -> exit on the PPI print",
            cost_bps=8.0, variants=variants, min_gap=5, event_kinds=("cpi",))

# --------------------------------------------------- D. eras / midterm split
print("\n=== D. era, midterm and year histogram (SHORT USO, episodes) ===")
r = vehicle_ret(px, [("USO", -1.0)], 1, 1)
epi = declusters(A2, 5, idx)
epi = pd.DatetimeIndex([d for d in epi if not np.isnan(r.get(d, np.nan))])
vals = r.loc[epi].values
show(era_split(epi, vals), "era")
mid = np.array([y % 4 == 2 for y in epi.year])
show([summarize(vals[mid], f"midterm yrs (N={mid.sum()})"),
      summarize(vals[~mid], f"non-midterm (N={(~mid).sum()})")], "midterm split")
yr = pd.Series(100 * vals, index=epi).groupby(epi.year).agg(["sum", "count"])
print("\nyear histogram (sum pp, n episodes):")
print(yr.round(2).to_string())
pos_yr = (yr["sum"] > 0).sum()
print(f"  positive years {pos_yr}/{len(yr)}")
print(f"  concentration: {cluster_note(epi, vals, k=3)}")
