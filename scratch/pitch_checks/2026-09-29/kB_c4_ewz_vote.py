"""C4 round 1: EWZ into and across the Brazilian first-round vote (Sunday).

Live map: entry 09-29 close = k=-4 (4th session before the vote Sunday 10-04);
Friday close 10-02 = h=3 (run-in); first post-vote close 10-05 = h=4 (across).
Forms: EWZ (USD), ^BVSP (BRL, own B3 calendar), EWZ - EEM (and EWZ - 1.056*EEM).
Controls: EWZ all-days drift at h=3/h=4; the SAME window around the first Sunday
of October in every non-election year (placebo years); runoffs as replication.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from kB_common import *  # noqa

FIRST = ["2002-10-06", "2006-10-01", "2010-10-03", "2014-10-05", "2018-10-07", "2022-10-02"]
RUNOFF = ["2002-10-27", "2006-10-29", "2010-10-31", "2014-10-26", "2018-10-28", "2022-10-30"]
# first-round result (general knowledge): leader, runner-up, did the market-favoured side beat polls
WHO = {2002: "Lula 46.4 / Serra 23.2 (Lula led; market feared him; forced to runoff)",
       2006: "Lula 48.6 / Alckmin 41.6 (Lula led; Alckmin beat polls, runoff forced)",
       2010: "Dilma 46.9 / Serra 32.6 (Dilma led; forced to runoff)",
       2014: "Dilma 41.6 / Aecio 33.6 (Dilma led; Aecio surged past Marina, market-friendly surprise)",
       2018: "Bolsonaro 46.0 / Haddad 29.3 (market favourite led and beat polls)",
       2022: "Lula 48.4 / Bolsonaro 43.2 (Lula led; Bolsonaro beat polls, market-friendly surprise)"}

px = nyse_panel(["EWZ", "EEM"])
idx = px.index
E, M = px["EWZ"].values, px["EEM"].values
bv = close_panel(["^BVSP"])["^BVSP"].dropna().loc[:BAR]
bidx, B = bv.index, bv.values
print(f"LIVE {idx[-1].date()}: EWZ 5d {100*(E[-1]/E[-6]-1):+.2f}%  21d {100*(E[-1]/E[-22]-1):+.2f}%  "
      f"63d {100*(E[-1]/E[-64]-1):+.2f}%;  ^BVSP 5d {100*(B[-1]/B[-6]-1):+.2f}%")


def windows(sunday: str, k: int = -4) -> dict:
    """k = entry session relative to the vote (-1 = Friday)."""
    f = pos_before(idx, sunday)      # Friday (k=-1)
    m = pos_after(idx, sunday)       # Monday (+1)
    e = f + (k + 1)                  # k=-4 -> f-3
    fb = pos_before(bidx, sunday)
    mb = pos_after(bidx, sunday)
    eb = fb + (k + 1)
    ewz_in, ewz_x = span_ret(E, e, f), span_ret(E, e, m)
    eem_in, eem_x = span_ret(M, e, f), span_ret(M, e, m)
    return {"vote": sunday, "entry": idx[e].date(), "fri": idx[f].date(), "mon": idx[m].date() if m < len(idx) else None,
            "EWZ_runin": ewz_in, "EWZ_across": ewz_x, "EWZ_react": span_ret(E, f, m),
            "EWZ_post3": span_ret(E, m, m + 3),
            "BVSP_runin": span_ret(B, eb, fb), "BVSP_across": span_ret(B, eb, mb),
            "BVSP_react": span_ret(B, fb, mb),
            "EEM_runin": eem_in, "EEM_across": eem_x,
            "EWZ-EEM_runin": ewz_in - eem_in, "EWZ-EEM_across": ewz_x - eem_x,
            "EWZ-EEM_react": span_ret(E, f, m) - span_ret(M, f, m)}


def first_sunday_oct(y: int) -> pd.Timestamp:
    d = pd.Timestamp(f"{y}-10-01")
    return d + pd.Timedelta(days=(6 - d.weekday()) % 7)


def last_sunday_oct(y: int) -> pd.Timestamp:
    d = pd.Timestamp(f"{y}-10-31")
    return d - pd.Timedelta(days=(d.weekday() - 6) % 7)


W = pd.DataFrame([windows(v) for v in FIRST])
pd.set_option("display.width", 250)
cols = ["vote", "entry", "fri", "mon", "EWZ_runin", "EWZ_across", "EWZ_react", "BVSP_runin", "BVSP_across",
        "BVSP_react", "EWZ-EEM_runin", "EWZ-EEM_across", "EWZ_post3"]
print("\n=== per-election returns (%), entry k=-4 close ===")
print((W[cols].set_index("vote").apply(lambda c: c * 100 if c.dtype.kind == "f" else c)).round(2).to_string())
for y in [2002, 2006, 2010, 2014, 2018, 2022]:
    print(f"  {y}: {WHO[y]}")

rows = []
for c in ["EWZ_runin", "EWZ_across", "EWZ_react", "BVSP_runin", "BVSP_across", "BVSP_react",
          "EWZ-EEM_runin", "EWZ-EEM_across", "EWZ-EEM_react", "EWZ_post3"]:
    rows.append(cell(W[c].values, f"first round {c}"))
show(rows, "first-round cells (N=6; EEM pair N=5, no EEM in 2002)")

# controls: all-days drift
r3 = fwd_ret(px["EWZ"], 3).dropna()
r4 = fwd_ret(px["EWZ"], 4).dropna()
r1 = fwd_ret(px["EWZ"], 1).dropna()
span = (r3.index >= "2002-01-01") & (r3.index <= "2022-12-31")
show([cell(r3.values, "EWZ all days h=3 (full)"), cell(r4.values, "EWZ all days h=4 (full)"),
      cell(r3[span].values, "EWZ all days h=3, 2002-2022"), cell(r1.values, "EWZ all days h=1 (for the Fri->Mon react)")],
     "CTRL: EWZ own drift")

# placebo years: same window around the first Sunday of October, non-election years
P = []
for y in range(2001, 2026):
    if y in (2002, 2006, 2010, 2014, 2018, 2022):
        continue
    w = windows(str(first_sunday_oct(y).date()))
    w["year"] = y
    P.append(w)
P = pd.DataFrame(P)
show([cell(P["EWZ_runin"].values, "placebo yrs EWZ run-in k=-4->Fri"),
      cell(P["EWZ_across"].values, "placebo yrs EWZ across k=-4->Mon"),
      cell(P["EWZ_react"].values, "placebo yrs EWZ Fri->Mon"),
      cell(P["BVSP_across"].values, "placebo yrs BVSP across"),
      cell(P["EWZ-EEM_across"].values, "placebo yrs EWZ-EEM across")],
     f"CTRL: same first-Sunday-of-October window, {len(P)} non-election years 2001-2025")
d, t = welch(W["EWZ_across"], P["EWZ_across"])
print(f"  election minus placebo-year EWZ across: {d:+.3f}pp (welch t {t:+.2f})")
d, t = welch(W["EWZ_runin"], P["EWZ_runin"])
print(f"  election minus placebo-year EWZ run-in: {d:+.3f}pp (welch t {t:+.2f})")
# rank of the election mean among all 25 years
allY = pd.concat([P.assign(elec=False), W.assign(year=[2002, 2006, 2010, 2014, 2018, 2022], elec=True)])
for c in ["EWZ_runin", "EWZ_across", "EWZ_react"]:
    s = allY.sort_values(c, ascending=False).reset_index(drop=True)
    ranks = [int(i) + 1 for i in s.index[s.elec]]
    print(f"  {c}: election-year ranks among {len(s)} years (1 = best): {ranks}")

# replication: runoffs
R = pd.DataFrame([windows(v) for v in RUNOFF])
print("\n=== runoffs (replication set for uncertainty resolution) ===")
print((R[cols].set_index("vote").apply(lambda c: c * 100 if c.dtype.kind == "f" else c)).round(2).to_string())
show([cell(R[c].values, f"runoff {c}") for c in ["EWZ_runin", "EWZ_across", "EWZ_react", "BVSP_across", "EWZ-EEM_across"]],
     "runoff cells")
PR = []
for y in range(2001, 2026):
    if y in (2002, 2006, 2010, 2014, 2018, 2022):
        continue
    PR.append(windows(str(last_sunday_oct(y).date())))
PR = pd.DataFrame(PR)
show([cell(PR["EWZ_across"].values, "placebo last-Sunday-Oct EWZ across")], "runoff placebo")

# entry neighbours (definition fragility preview)
nb = []
for k in (-5, -4, -3, -2):
    Wk = pd.DataFrame([windows(v, k) for v in FIRST])
    Pk = pd.DataFrame([windows(str(first_sunday_oct(y).date()), k) for y in range(2001, 2026)
                       if y not in (2002, 2006, 2010, 2014, 2018, 2022)])
    for c in ("EWZ_runin", "EWZ_across"):
        r = cell(Wk[c].values, f"k={k} {c}")
        r["placebo_mean"] = round(100 * Pk[c].mean(), 3)
        nb.append(r)
show(nb, "entry neighbours k=-5..-2 (election vs placebo-year mean)")

# cost
print(f"\ncost: EWZ ~5 bp round trip. first-round across mean {100*W['EWZ_across'].mean():+.3f}% "
      f"= {1e4*W['EWZ_across'].mean():.0f} bp")
# NFP inside the window
nfp = load_events(["nfp"])["date"]
for v in FIRST:
    f = pos_before(idx, v)
    e = f - 3
    m = pos_after(idx, v)
    inside = nfp[(nfp > idx[e]) & (nfp <= idx[m])]
    print(f"  {v}: NFP inside k=-4 -> Mon window: {[str(x.date()) for x in inside]}")
