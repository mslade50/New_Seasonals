"""C5 round 2: the mechanism test the brief demanded -- explain it or kill it.

If commodities fall BECAUSE of the PPI release, the move must sit in the
OVERNIGHT GAP. PPI prints at 08:30 ET, an hour before the cash open, so the
release is fully in the prior-close -> open segment. Any effect that lives in
the 09:30 -> 16:00 intraday segment instead is not the release; it is an
unexplained calendar coincidence, and a calendar coincidence that a placebo
anchor already matches.

Also here: the ladder rank as an empirical p, and an honest cost build.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

TK = ["USO", "DBC", "XLE", "XOP"]
mp = pd.read_parquet(PRICES_PATH)
mp["date"] = pd.to_datetime(mp["date"])
bars = {t: mp[mp.ticker == t].set_index("date").sort_index() for t in TK}
idx = bars["XLE"].index
ev = load_events(["ppi"])
PPI = pd.DatetimeIndex(sorted(ev.loc[ev.event == "ppi", "date"].unique()))

print("=== MECHANISM: where inside the PPI print session does the move sit? ===")
print("PPI prints 08:30 ET. The release is entirely inside the prior-close -> "
      "09:30-open GAP.")
print("A release-driven effect CANNOT live in the 09:30->16:00 intraday leg.\n")

for t in TK:
    b = bars[t]
    bi = b.index
    prev_close = b["Close"].shift(1)
    gap = b["Open"] / prev_close - 1.0            # overnight, holds the print
    intr = b["Close"] / b["Open"] - 1.0           # session, after the print
    full = b["Close"] / prev_close - 1.0
    print_days = pd.DatetimeIndex([d for d in PPI if d in set(bi)])
    if len(print_days) < 20:
        continue
    span = (bi >= print_days[0]) & (bi <= print_days[-1])
    rows = []
    for lbl, s in [("GAP prior-close->open (holds the 08:30 print)", gap),
                   ("INTRADAY open->close (after the print)", intr),
                   ("FULL session close->close (the traded cell)", full)]:
        v = s.loc[print_days].dropna()
        base = s[span].dropna()
        rows.append({"segment": lbl, "n": len(v),
                     "cond_pct": round(100*v.mean(), 4),
                     "drift_pct": round(100*base.mean(), 4),
                     "excess_pct": round(100*(v.mean()-base.mean()), 4),
                     "hit": round(100*(v > 0).mean(), 1)})
    show(rows, f"{t} on PPI print sessions")
    g = rows[0]["excess_pct"]
    i = rows[1]["excess_pct"]
    f = rows[2]["excess_pct"]
    print(f"  -> share of the traded excess that sits in the RELEASE gap: "
          f"{100*g/f:.0f}%   in the post-release intraday leg: {100*i/f:.0f}%")

# ------------------------------------------------------------- ladder rank
print("\n=== ladder rank as an empirical p (21 anchors, k=-8..+12) ===")
px = close_panel(TK)
pidx = px.index


def anch(k):
    out = []
    for d in PPI:
        loc = pidx.searchsorted(pd.Timestamp(d))
        if loc >= len(pidx):
            continue
        p = loc - k
        if 0 <= p < len(pidx):
            out.append(pidx[p])
    return pd.DatetimeIndex(sorted(set(out)))


for t in TK:
    r = fwd_lag(px[t], 1, 1)
    vals = {}
    for k in range(-8, 13):
        a = anch(k)
        v = r.loc[r.index.intersection(a)].dropna()
        if len(v) < 10:
            continue
        span = (pidx >= v.index[0]) & (pidx <= v.index[-1])
        vals[k] = 100 * (v.mean() - r[span].dropna().mean())
    s = pd.Series(vals)
    rank = int((s <= s[2]).sum())
    print(f"  {t}: real k=2 = {s[2]:+.3f}%, rank {rank}/{len(s)} most negative "
          f"-> empirical p = {rank/len(s):.3f};  ladder sd = {s.std():.3f}%, "
          f"real is {(s[2]-s.mean())/s.std():+.2f} sd from the ladder mean")

# ------------------------------------------------------------------- cost
print("\n=== honest cost of the 1-session short ===")
print("  USO: spread ~1.5 bps + commission ~1 bp, each way   = ~5 bps round trip")
print("  short borrow on a commodity ETF ~1-3%/yr over 1 night = ~0.4-1.2 bps")
print("  slippage on a MOC in a name that just ran 10% in 5d  = ~2-4 bps")
print("  -> realistic 7-10 bps round trip against a 22.4 bps episode mean")
print("     = 2.2x-3.2x cost, versus the 5x bar.")
r = fwd_lag(px["USO"], 1, 1)
a2 = anch(2)
v = r.loc[r.index.intersection(a2)].dropna()
short = -v
for c in (0.0005, 0.0008, 0.0010):
    net = short - c
    w = int((net > 0).sum())
    print(f"  net of {100*c:.2f}% cost: mean {100*net.mean():+.3f}%  "
          f"record {w}-{len(net)-w}  sign p = {sign_test(w, len(net)):.4f}")
