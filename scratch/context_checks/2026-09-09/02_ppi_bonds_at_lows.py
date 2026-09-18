"""Drill 02 (2026-09-09): does the PPI-day bond bid survive when bonds go into
the print ALREADY AT the lows?

Base cell from the sweep: anchor 1 td before each PPI, h=1 lag=0, IEF n=287,
mean +0.062%, hit 58.2%, t=2.67, record 167-117, sign p 0.0033, era-stable,
passes Benjamini-Hochberg. TLT the same effect at n=287, t=2.42, sign p 0.009.

Tonight's state (2026-09-09 close): ^TNX exactly AT its 52w high, IEF 0.05%
above its 52w LOW, LQD +0.09%, TLT +0.85%. So the live case is the
"bonds already at the lows" bucket, which is what this drill prices.

Conventions: fwd_ret lag=0 close-to-close (CONTEXT, not an entry). All state is
read on the ANCHOR close (the session BEFORE the PPI), so nothing here peeks at
the print. Rolling windows run on each ticker's OWN dropna'd series.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

SUBJECTS = ["IEF", "TLT", "^TNX", "LQD"]
ERA_CUT = "2018-01-01"
NEAR = 0.01          # "within 1%" band
LOOKBACK = 252       # trailing sessions

px = load_prices(SUBJECTS)
CLOSE = {t: px[t]["Close"].dropna() for t in SUBJECTS}

ppi_dates = pd.DatetimeIndex(
    load_events(["ppi"]).loc[lambda d: d["event"] == "ppi", "date"])


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def cell(vals, label):
    v = np.asarray(vals, dtype=float)
    v = v[~np.isnan(v)]
    s = summarize(v, label)
    if s["n"] == 0:
        return s
    w = int((v > 0).sum())
    losses = int((v < 0).sum())
    flat = int((v == 0).sum())
    s["record"] = f"{w}-{losses}" + (f" ({flat} flat)" if flat else "")
    s["sign_p"] = sign_test(w, len(v))
    return s


def report(dates, vals, label, indent="  "):
    d = pd.DatetimeIndex(dates)
    v = np.asarray(vals, dtype=float)
    s = cell(v, label)
    if s["n"] == 0:
        print(f"{indent}{label}: EMPTY")
        return s
    small = "   *** n<15, SMALL ***" if s["n"] < 15 else ""
    print(f"{indent}{label}: n={s['n']}  mean={s['mean_pct']:+.4f}%  "
          f"median={s['median_pct']:+.4f}%  hit={s['hit']:.1f}%  "
          f"t={s['t']:+.3f}  record {s['record']}  sign_p={s['sign_p']:.4f}  "
          f"sd={s['sd_pct']:.3f}%  worst={s['worst_pct']:+.2f}%  "
          f"best={s['best_pct']:+.2f}%{small}")
    for e in era_split(d, v, ERA_CUT):
        if e["n"]:
            print(f"{indent}    era {e['label']}: n={e['n']} "
                  f"mean={e['mean_pct']:+.4f}% hit={e['hit']:.1f}% "
                  f"t={e['t']:+.3f}")
        else:
            print(f"{indent}    era {e['label']}: n=0")
    print(f"{indent}    cluster: {cluster_note(d, v)}")
    return s


def base_cell(tkr, h=1, offset=-1):
    """(ppi_dates_kept, anchor_dates, values) for the PPI session's own move."""
    s = CLOSE[tkr]
    idx = s.index
    pos, kept = anchor_positions(idx, ppi_dates, offset)
    r = fwd_ret(s, h)
    vals = r.iloc[pos].to_numpy()
    anch = pd.DatetimeIndex([idx[p] for p in pos])
    m = ~np.isnan(vals)
    return kept[m], anch[m], vals[m]


def dist_to_low(tkr, lookback=LOOKBACK):
    """Close / trailing-`lookback` MIN - 1, inclusive of the current bar.
    0.0 = sitting exactly on the 52w low. Computed on the dropna'd series."""
    s = CLOSE[tkr]
    lo = s.rolling(lookback, min_periods=lookback).min()
    return s / lo - 1.0


def dist_to_high(tkr, lookback=LOOKBACK):
    """Close / trailing-`lookback` MAX - 1, inclusive. 0.0 = exactly at the
    52w high, -0.01 = 1% below it."""
    s = CLOSE[tkr]
    hi = s.rolling(lookback, min_periods=lookback).max()
    return s / hi - 1.0


def welch(a, b):
    a = np.asarray(a, float); a = a[~np.isnan(a)]
    b = np.asarray(b, float); b = b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return np.nan
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return (a.mean() - b.mean()) / se if se > 0 else np.nan


def split(subject, state, state_name, true_lbl, false_lbl):
    """Split the subject's base PPI cell by a boolean state read on the ANCHOR
    close. `state` is a Series indexed by date (may live on another ticker's
    calendar); anchors with no state reading are DROPPED and counted."""
    ev, anch, vals = base_cell(subject)
    st = state.reindex(anch)
    undef = int(st.isna().sum())
    m = st.notna().to_numpy()
    ev_d, anch_d, v = ev[m], anch[m], vals[m]
    b = st[m].to_numpy().astype(bool)
    print(f"\n  --- {subject} base cell split by {state_name} ---")
    print(f"      anchors with a defined state: {int(m.sum())} of {len(vals)} "
          f"({undef} dropped: state undefined, i.e. inside the first "
          f"{LOOKBACK}-session warmup)")
    a = report(ev_d[b], v[b], f"{true_lbl:<34s}")
    c = report(ev_d[~b], v[~b], f"{false_lbl:<34s}")
    if a["n"] and c["n"]:
        print(f"      DIFF (true - false): mean "
              f"{a['mean_pct'] - c['mean_pct']:+.4f}pp   "
              f"hit {a['hit'] - c['hit']:+.1f}pp   "
              f"welch t {welch(v[b], v[~b]):+.3f}")
    return a, c


print("=" * 78)
print("DRILL 02 -- the PPI-day bond bid, conditioned on bonds ALREADY AT LOWS")
print("=" * 78)
for t in SUBJECTS:
    print(f"  {t}: {len(CLOSE[t])} sessions "
          f"{CLOSE[t].index[0].date()}..{CLOSE[t].index[-1].date()}")
print(f"  PPI calendar: {len(ppi_dates)} dates "
      f"{ppi_dates.min().date()}..{ppi_dates.max().date()}")

# ---------------------------------------------------------------------------
# 1. reproduce the base cell
# ---------------------------------------------------------------------------
print("\n--- 1. BASE CELL sanity check (anchor = 1 td before PPI, h=1, lag=0) ---")
print("    sweep reported: IEF n=287 mean +0.062% hit 58.2% t=2.67 "
      "record 167-117 sign p 0.0033 ; TLT n=287 t=2.42 sign p 0.009")
base = {}
for tkr in ["IEF", "TLT"]:
    ev, anch, v = base_cell(tkr)
    base[tkr] = (ev, anch, v)
    report(ev, v, f"{tkr} PPI-session h=1 (FULL HISTORY)")

ief_s = cell(base["IEF"][2], "IEF")
tlt_s = cell(base["TLT"][2], "TLT")
print("\n    REPRODUCTION VERDICT:")
print(f"      IEF got n={ief_s['n']} mean={ief_s['mean_pct']:+.4f}% "
      f"hit={ief_s['hit']:.1f}% t={ief_s['t']:+.3f} record {ief_s['record']} "
      f"sign_p={ief_s['sign_p']:.4f}")
print(f"      TLT got n={tlt_s['n']} mean={tlt_s['mean_pct']:+.4f}% "
      f"hit={tlt_s['hit']:.1f}% t={tlt_s['t']:+.3f} record {tlt_s['record']} "
      f"sign_p={tlt_s['sign_p']:.4f}")
ok = (ief_s["n"] == 287 and abs(ief_s["mean_pct"] - 0.062) < 0.005
      and abs(ief_s["t"] - 2.67) < 0.15)
print("      -> MATCHES the sweep." if ok else
      "      -> *** DOES NOT MATCH THE SWEEP -- read the note below. ***")

# ---------------------------------------------------------------------------
# 2. the three splits
# ---------------------------------------------------------------------------
print("\n--- 2. splits on the ANCHOR-day state ---")
print(f'    "within 1% of the 252d LOW"  == Close / rolling-252-min  - 1 <= '
      f'{NEAR:.2f}  (0.0 = sitting exactly on the low; window INCLUDES the '
      f'anchor bar)')
print(f'    "within 1% of the 252d HIGH" == Close / rolling-252-max  - 1 >= '
      f'{-NEAR:.2f}')
print(f"    rank rule                    == pct_rank(Close, n=21, "
      f"lookback=252) < 25")

ief_low = dist_to_low("IEF")
tnx_high = dist_to_high("^TNX")
ief_r21 = pct_rank(CLOSE["IEF"], 21, LOOKBACK)
tlt_low = dist_to_low("TLT")

print("\n  ===== (a) IEF within 1% of its trailing-252d LOW on the anchor close"
      " =====")
split("IEF", ief_low <= NEAR, "IEF dist-to-252d-low <= 1%",
      "IEF AT LOWS (<=1% above low)", "IEF not at lows")

print("\n  ===== (b) ^TNX within 1% of its trailing-252d HIGH on the anchor "
      "close =====")
split("IEF", tnx_high >= -NEAR, "^TNX dist-to-252d-high >= -1%",
      "^TNX AT HIGHS (within 1%)", "^TNX not at highs")

print("\n  ===== (c) IEF 21d-return percentile rank (252d lookback) < 25 "
      "=====")
split("IEF", ief_r21 < 25, "IEF 21d-return rank < 25",
      "IEF 21d rank < 25 (weak)", "IEF 21d rank >= 25")

# --- 2d. threshold sensitivity + the actual episodes -----------------------
print("\n  ===== (d) threshold sensitivity: is the at-lows result a band "
      "artifact? =====")
ev_i, anch_i, v_i = base_cell("IEF")
for band in (0.005, 0.01, 0.02, 0.03, 0.05):
    st = (ief_low <= band).reindex(anch_i).fillna(False).to_numpy().astype(bool)
    s = cell(v_i[st], f"IEF within {100*band:.1f}% of 252d low")
    if s["n"]:
        print(f"    IEF low-band <= {100*band:4.1f}% : n={s['n']:3d} "
              f"mean={s['mean_pct']:+.4f}% hit={s['hit']:5.1f}% "
              f"t={s['t']:+.3f} record {s['record']} "
              f"sign_p={s['sign_p']:.4f}")
for band in (0.005, 0.01, 0.02, 0.03):
    st = (tnx_high >= -band).reindex(anch_i).fillna(False).to_numpy().astype(bool)
    s = cell(v_i[st], "x")
    if s["n"]:
        print(f"    ^TNX high-band >= -{100*band:4.1f}%: n={s['n']:3d} "
              f"mean={s['mean_pct']:+.4f}% hit={s['hit']:5.1f}% "
              f"t={s['t']:+.3f} record {s['record']} "
              f"sign_p={s['sign_p']:.4f}   (subject = IEF)")
for cutv in (15, 20, 25, 30, 35):
    st = (ief_r21 < cutv).reindex(anch_i).fillna(False).to_numpy().astype(bool)
    s = cell(v_i[st], "x")
    if s["n"]:
        print(f"    IEF 21d rank <  {cutv:3d}   : n={s['n']:3d} "
              f"mean={s['mean_pct']:+.4f}% hit={s['hit']:5.1f}% "
              f"t={s['t']:+.3f} record {s['record']} "
              f"sign_p={s['sign_p']:.4f}")

print("\n  ===== (e) the episodes behind the small buckets (IEF PPI-session "
      "move %) =====")
for lbl, state in [("IEF AT LOWS (<=1% above 252d low)", ief_low <= NEAR),
                   ("^TNX AT HIGHS (within 1% of 252d high)",
                    tnx_high >= -NEAR)]:
    st = state.reindex(anch_i).fillna(False).to_numpy().astype(bool)
    print(f"    {lbl}: n={int(st.sum())}")
    for d, a, val in zip(ev_i[st], anch_i[st], v_i[st]):
        print(f"      PPI {d.date()} (anchor {a.date()})  "
              f"IEF {100*val:+7.4f}%")

# ---------------------------------------------------------------------------
# 3. current readings
# ---------------------------------------------------------------------------
print("\n--- 3. CURRENT readings on the 2026-09-09 close (which bucket is "
      "tonight in?) ---")
asof = pd.Timestamp("2026-09-09")
rows = [
    ("IEF  dist to 252d LOW ", ief_low, f"<= +{100*NEAR:.0f}%", lambda x: x <= NEAR),
    ("TLT  dist to 252d LOW ", tlt_low, f"<= +{100*NEAR:.0f}%", lambda x: x <= NEAR),
    ("^TNX dist to 252d HIGH", tnx_high, f">= {-100*NEAR:.0f}%",
     lambda x: x >= -NEAR),
]
for lbl, s, rule, fn in rows:
    val = s.get(asof, np.nan)
    print(f"    {lbl} on {asof.date()} = {100*val:+.4f}%   "
          f"rule {rule}  ->  {'IN the at-extreme bucket' if fn(val) else 'OUT'}")
r_now = ief_r21.get(asof, np.nan)
print(f"    IEF 21d-return rank (252d)   on {asof.date()} = {r_now:.2f}   "
      f"rule < 25  ->  "
      f"{'IN the weak bucket' if r_now < 25 else 'OUT'}")
print(f"    raw levels: IEF close {CLOSE['IEF'].get(asof):.4f} vs 252d min "
      f"{CLOSE['IEF'].rolling(LOOKBACK).min().get(asof):.4f}; "
      f"TLT close {CLOSE['TLT'].get(asof):.4f} vs 252d min "
      f"{CLOSE['TLT'].rolling(LOOKBACK).min().get(asof):.4f}; "
      f"^TNX close {CLOSE['^TNX'].get(asof):.4f} vs 252d max "
      f"{CLOSE['^TNX'].rolling(LOOKBACK).max().get(asof):.4f}")
lqd_low = dist_to_low("LQD")
print(f"    (reference) LQD dist to 252d LOW = "
      f"{100*lqd_low.get(asof, np.nan):+.4f}%")

# ---------------------------------------------------------------------------
# 4. the same (a) split on TLT
# ---------------------------------------------------------------------------
print("\n--- 4. (a) repeated on TLT: TLT within 1% of its OWN 252d LOW ---")
split("TLT", tlt_low <= NEAR, "TLT dist-to-252d-low <= 1%",
      "TLT AT LOWS (<=1% above low)", "TLT not at lows")

# ---------------------------------------------------------------------------
# 5. local control for IEF
# ---------------------------------------------------------------------------
print("\n--- 5. base rate: local +/-126td control for IEF ---")
s = CLOSE["IEF"]
r1 = fwd_ret(s, 1)
valid = r1.dropna().index
trig = pd.DatetimeIndex(base["IEF"][1])          # anchor dates
loc = local_control(valid, trig, 126)
report(loc, r1.loc[loc].to_numpy(), "CTRL-c local +/-126td ex-anchor")
report(valid, r1.loc[valid].to_numpy(), "CTRL-b IEF all sessions        ")
ctrl_mean = 100 * r1.loc[loc].mean()
print(f"\n    EDGE vs local control: base PPI cell "
      f"{ief_s['mean_pct']:+.4f}% - local {ctrl_mean:+.4f}% = "
      f"{ief_s['mean_pct'] - ctrl_mean:+.4f}pp")

# at-lows bucket vs the same local control
ev, anch, v = base_cell("IEF")
st = (ief_low <= NEAR).reindex(anch)
m = st.notna().to_numpy() & st.fillna(False).to_numpy().astype(bool)
if m.sum():
    al = 100 * np.nanmean(v[m])
    print(f"    EDGE vs local control: AT-LOWS bucket {al:+.4f}% - local "
          f"{ctrl_mean:+.4f}% = {al - ctrl_mean:+.4f}pp  (n={int(m.sum())})")

print("\ndone.")
