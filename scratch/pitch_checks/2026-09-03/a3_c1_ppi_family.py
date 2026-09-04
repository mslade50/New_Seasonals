"""C1 DEBT 2 -- "it may not be payrolls".

Family permutation across NFP/CPI/PPI/FOMC gives family-wise P 0.2766. SVXY by
kind at k=-2, gate ON, h=1: NFP +1.313% (n=21), CPI +0.527% (n=28), FOMC
+0.718% (n=17), PPI **-1.023%** (n=28). One of four inverts, which is exactly
what an arbitrary label looks like.

The structural hypothesis on offer: the mechanism is EVENT RISK RESOLVING, not
"payrolls". A print that is immediately followed by another print does not
resolve anything -- the option market simply rolls its bid forward -- so the
h=1 crush should be absent there. PPI is, by construction, the print that
usually sits one or two sessions in front of CPI. If PPI's inversion is
entirely "another print is queued right behind it", the family is COHERENT and
the multiplicity charge weakens. If it is not, the event label is arbitrary and
that is a kill (a filter that does not filter / mechanism falsified in-window).

Every return below is on the PITCHED side: long SVXY, and short ^VIX reported
as -(VIX return). A "hit" is therefore a win for the pitch, per the 2026-08-10
registry warning about reading a long-side hit column for a short.

Today's live runway, stated so the split can be applied: NFP prints 2026-09-04;
the next scheduled prints are PPI 2026-09-10 and CPI 2026-09-11, i.e. 3+
sessions of clear calendar behind the print.
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa: F401,F403,E402
from pitch_lab import (close_panel, fwd_lag, summarize, sign_test, load_events,
                       rolling_on_valid, show, anchor_positions, bootstrap_p_le0,
                       declusters)

warnings.filterwarnings("ignore")
pd.set_option("display.width", 250)

px = close_panel(["^VIX", "^VIX3M", "SVXY", "UVXY", "SPY"])
cal = px["SPY"].dropna().index
vix = px["^VIX"]
rng21 = (rolling_on_valid(vix, lambda x: x.rolling(21).max())
         - rolling_on_valid(vix, lambda x: x.rolling(21).min()))
REL = rolling_on_valid(rng21 / rolling_on_valid(vix, lambda x: x.rolling(21).mean()),
                       lambda x: x.rolling(252).rank(pct=True) * 100)
G15 = REL <= 15.0

KINDS = ("nfp", "cpi", "ppi", "fomc_decision")
EV = {k: load_events([k])["date"] for k in KINDS}
ALL_PRINTS = pd.DatetimeIndex(sorted(pd.concat(list(EV.values())).unique()))

svxy_h1 = fwd_lag(px["SVXY"].dropna(), 1, lag=1)
vix_h1 = -fwd_lag(px["^VIX"].dropna(), 1, lag=1)
pos = pd.Series(range(len(cal)), index=cal)


def build(kind, k=-2, gate=G15):
    """Anchors of `kind` with the gate applied, plus the runway to the NEXT
    scheduled print of a DIFFERENT kind, measured in trading sessions from
    the print date itself."""
    p, kept = anchor_positions(cal, EV[kind], k)
    rows = []
    for i, anc_pos in enumerate(p):
        anc = cal[anc_pos]
        print_date = kept[i]
        others = ALL_PRINTS[(ALL_PRINTS > print_date)]
        # sessions between the print and the next print (calendar -> sessions)
        if len(others) == 0:
            runway = 99
        else:
            nxt = others[0]
            pp, pn = pos.get(print_date), pos.get(nxt)
            if pp is None:
                pp = int(cal.searchsorted(print_date))
            if pn is None:
                pn = int(cal.searchsorted(nxt))
            runway = int(pn - pp)
        # also: is another print on the SAME day as this one?
        same = int(((ALL_PRINTS == print_date).sum()) - 1)
        rows.append({"anchor": anc, "print": print_date, "kind": kind,
                     "runway_td": runway, "same_day_others": same})
    df = pd.DataFrame(rows).set_index("anchor")
    df["gate"] = gate.reindex(df.index).fillna(False).values
    df["svxy"] = svxy_h1.reindex(df.index).values
    df["vix"] = vix_h1.reindex(df.index).values
    df["rel"] = REL.reindex(df.index).values
    return df


ALL = pd.concat([build(k) for k in KINDS]).sort_index()
G = ALL[ALL["gate"]].copy()

print("=" * 118)
print("0. RUNWAY BY KIND -- how many sessions until the NEXT scheduled print?")
print("   (the structural claim about PPI, measured before it is used)")
print("=" * 118)
r = ALL.groupby("kind")["runway_td"].describe()[["count", "mean", "25%", "50%", "75%"]]
r["pct_runway<=1"] = ALL.groupby("kind")["runway_td"].apply(lambda x: 100 * (x <= 1).mean())
r["pct_runway<=2"] = ALL.groupby("kind")["runway_td"].apply(lambda x: 100 * (x <= 2).mean())
r["pct_runway>=3"] = ALL.groupby("kind")["runway_td"].apply(lambda x: 100 * (x >= 3).mean())
print(r.round(2).to_string())
print("\n  -> PPI's median runway is the number the whole structural story rests on.")

# ---------------------------------------------------------------------------
print("\n" + "=" * 118)
print("1. THE PPI CELL SPLIT ON ITS RUNWAY (gate ON, k=-2, h=1)")
print("=" * 118)
ppi = G[G["kind"] == "ppi"]
print(f"   gated PPI anchors: {len(ppi)}  (SVXY-covered {int(ppi['svxy'].notna().sum())})")
rows = []
for lbl, m in (("runway <= 1 td (print queued right behind)", ppi["runway_td"] <= 1),
               ("runway == 2 td", ppi["runway_td"] == 2),
               ("runway >= 3 td (clear calendar)", ppi["runway_td"] >= 3)):
    v = ppi.loc[m, "svxy"].dropna()
    st = summarize(v.values, f"SVXY | PPI, {lbl}")
    if st["n"]:
        st["signp"] = round(sign_test(int((v > 0).sum()), len(v)), 4)
    rows.append(st)
rows.append(summarize(ppi["svxy"].dropna().values, "SVXY | PPI, all gated"))
show(rows, "long SVXY h=1, gated PPI anchors by runway")
rows = []
for lbl, m in (("runway <= 1", ppi["runway_td"] <= 1),
               ("runway == 2", ppi["runway_td"] == 2),
               ("runway >= 3", ppi["runway_td"] >= 3)):
    v = ppi.loc[m, "vix"].dropna()
    rows.append(summarize(v.values, f"-^VIX | PPI, {lbl}"))
rows.append(summarize(ppi["vix"].dropna().values, "-^VIX | PPI, all gated"))
show(rows, "short ^VIX h=1, gated PPI anchors by runway")
print("   per-anchor PPI detail (gated):")
print(ppi.assign(svxy=(100 * ppi["svxy"]).round(2),
                 vix=(100 * ppi["vix"]).round(2),
                 rel=ppi["rel"].round(1))
      [["print", "runway_td", "same_day_others", "rel", "svxy", "vix"]].to_string())

# ---------------------------------------------------------------------------
print("\n" + "=" * 118)
print("2. THE SAME SPLIT ON EVERY KIND -- if 'clear calendar' is the mechanism it")
print("   must work on NFP, CPI and FOMC too, not only rescue PPI.")
print("=" * 118)
for side, col in (("long SVXY", "svxy"), ("short ^VIX", "vix")):
    rows = []
    for kind in KINDS:
        sub = G[G["kind"] == kind]
        for lbl, m in (("runway<=1", sub["runway_td"] <= 1),
                       ("runway>=3", sub["runway_td"] >= 3)):
            v = sub.loc[m, col].dropna()
            st = summarize(v.values, f"{kind} | {lbl}")
            if st["n"]:
                st["signp"] = round(sign_test(int((v > 0).sum()), len(v)), 4)
            rows.append(st)
    show(rows, f"{side} h=1, gated, by kind x runway")

# ---------------------------------------------------------------------------
print("\n" + "=" * 118)
print("3. POOLED CELLS -- all four kinds, and the 'coherent' clear-calendar subset")
print("=" * 118)
for side, col in (("long SVXY", "svxy"), ("short ^VIX", "vix")):
    rows = []
    v = G[col].dropna()
    st = summarize(v.values, "POOLED all 4 kinds, gate ON")
    st["signp"] = round(sign_test(int((v > 0).sum()), len(v)), 4)
    rows.append(st)
    for lo in (2, 3, 4):
        m = G["runway_td"] >= lo
        v = G.loc[m, col].dropna()
        st = summarize(v.values, f"POOLED, runway >= {lo} td")
        st["signp"] = round(sign_test(int((v > 0).sum()), len(v)), 4)
        rows.append(st)
    m = G["runway_td"] <= 1
    v = G.loc[m, col].dropna()
    st = summarize(v.values, "POOLED, runway <= 1 td")
    st["signp"] = round(sign_test(int((v > 0).sum()), len(v)), 4)
    rows.append(st)
    # ungated control for the pooled coherent set
    m = (ALL["runway_td"] >= 3)
    v = ALL.loc[m, col].dropna()
    rows.append(summarize(v.values, "CTRL ungated, runway >= 3"))
    v = px["SVXY"].dropna() if col == "svxy" else px["^VIX"].dropna()
    base = (svxy_h1 if col == "svxy" else vix_h1).dropna()
    rows.append(summarize(base.values, "CTRL all days, full history"))
    show(rows, f"{side} h=1 pooled")

print("\n3b. is NFP still special INSIDE the clear-calendar subset, or is the")
print("    payrolls label doing no work once runway is controlled?")
sub = G[G["runway_td"] >= 3]
rows = []
for kind in KINDS:
    v = sub.loc[sub["kind"] == kind, "svxy"].dropna()
    st = summarize(v.values, f"SVXY | {kind}, runway>=3")
    if st["n"]:
        st["signp"] = round(sign_test(int((v > 0).sum()), len(v)), 4)
    rows.append(st)
show(rows, "clear-calendar cell by kind")
# permutation across kinds INSIDE the clear-calendar subset, SVXY
vals = {k: sub.loc[sub["kind"] == k, "svxy"].dropna().values for k in KINDS}
ns = {k: len(v) for k, v in vals.items()}
pool = np.concatenate(list(vals.values()))
rng_ = np.random.default_rng(11)
obs = vals["nfp"].mean()
mx = []
for _ in range(20000):
    perm = rng_.permutation(pool)
    i, mm = 0, []
    for k, n in ns.items():
        mm.append(perm[i:i + n].mean()); i += n
    mx.append(max(mm))
print(f"    permutation max-of-4 (SVXY, clear-calendar subset): NFP observed "
      f"{100*obs:+.3f}%, family-wise P = {float((np.array(mx) >= obs).mean()):.4f}"
      f"  (N per kind {ns})")
print(f"    pooled clear-calendar SVXY bootstrap P(mean<=0) = "
      f"{bootstrap_p_le0(sub['svxy'].dropna().values):.4f}  "
      f"(day-level; anchors are >=2 weeks apart by construction so overlap is low)")
epi = declusters(sub.index, 5, cal)
print(f"    declustered at 5 td: {len(epi)} of {len(sub)} anchors survive")

print("\n3c. LIVE APPLICABILITY: today's NFP prints 2026-09-04; next scheduled")
print("    prints are PPI 2026-09-10 and CPI 2026-09-11.")
nxt = ALL_PRINTS[ALL_PRINTS > pd.Timestamp("2026-09-04")]
print(f"    next print after the live NFP: {nxt[0].date()} "
      f"-> runway is >= 3 sessions, i.e. the COHERENT side of the split.")

print("\n4. SANITY: the ungated versions, so the runway split is not just the gate")
for side, col in (("long SVXY", "svxy"), ("short ^VIX", "vix")):
    rows = []
    for kind in KINDS:
        sub2 = ALL[ALL["kind"] == kind]
        for lbl, m in (("runway<=1", sub2["runway_td"] <= 1),
                       ("runway>=3", sub2["runway_td"] >= 3)):
            v = sub2.loc[m, col].dropna()
            rows.append(summarize(v.values, f"{kind} | {lbl} (UNGATED)"))
    show(rows, f"{side} h=1, UNGATED anchors, by kind x runway")
