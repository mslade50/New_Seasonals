"""C4 round 2: placebo offset ladder + reference-class vehicles.

Ladder: shift the entry k sessions relative to the TRUE print anchor,
k = -10 .. +5. If the true offset (k=0) does not stand out, the "NVDA print"
is decoration on late-August position and the anchor is dead.

Also: does the story survive the reference class (equal-weight semi
components, NVDA excluded and included)? SOXX is NOT in master_prices, so
the ETF reference class is SMH only; the component basket is the substitute.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

SEMI = ["SMH", "NVDA", "AMD", "AVGO", "MU", "INTC", "TXN", "ADI", "AMAT",
        "LRCX", "KLAC", "QCOM", "ASML", "TSM", "MRVL", "ON", "MCHP", "SWKS",
        "TER", "MPWR", "NXPI"]
px = close_panel(SEMI)
idx = px.index

ec = pd.read_parquet("data/earnings_calendar.parquet")
nv = pd.DatetimeIndex(pd.to_datetime(ec[ec["ticker"] == "NVDA"]["date"])
                      .sort_values().unique())
aug = nv[nv.month == 8]
ENTRY_LEAD, H = 5, 7
r63 = pct_rank(px["SMH"].dropna(), 63).reindex(idx)


def anchors(prints, k=0):
    out = []
    for p in prints:
        loc = idx.searchsorted(p)
        if loc >= len(idx):
            continue
        d = loc - (ENTRY_LEAD + 1) + k
        if 0 <= d < len(idx):
            out.append(idx[d])
    return pd.DatetimeIndex(out)


ret = vehicle_ret(px, [("SMH", -1.0)], H, 1)

print("=" * 78)
print("PLACEBO OFFSET LADDER  (short SMH, h=7, entry lag 1)")
print("k = sessions the entry is shifted vs the true print anchor;")
print("k=0 is the real trade (entry 5 sessions before the print).")
print("=" * 78)
rows = []
for k in range(-10, 6):
    a = anchors(aug, k)
    a = a[ret.reindex(a).notna().values]
    g = a[(r63.reindex(a) <= 25).fillna(False).values]
    rows.append({
        "k": k,
        "n_all": len(a),
        "mean_all_pct": round(100 * ret.loc[a].mean(), 3),
        "hit_all": round(100 * (ret.loc[a] > 0).mean(), 1),
        "n_gate": len(g),
        "mean_gate_pct": round(100 * ret.loc[g].mean(), 3) if len(g) else np.nan,
        "hit_gate": round(100 * (ret.loc[g] > 0).mean(), 1) if len(g) else np.nan,
    })
lad = pd.DataFrame(rows)
print(lad.to_string(index=False))

for col, lbl in [("mean_all_pct", "UNGATED August prints"),
                 ("mean_gate_pct", "GATED (SMH r63rank<=25)")]:
    s = lad.set_index("k")[col].dropna()
    if len(s) == 0:
        continue
    order = s.rank(ascending=False)
    true = s.get(0, np.nan)
    print(f"\n{lbl}: true k=0 mean {true:+.3f}%, ranks "
          f"{int(order.get(0, np.nan))} of {len(s)} offsets "
          f"(best {s.max():+.3f}% at k={int(s.idxmax())}, "
          f"worst {s.min():+.3f}% at k={int(s.idxmin())}, "
          f"ladder mean {s.mean():+.3f}%)")

# ---------------------------------------------------------- reference class
print("\n" + "=" * 78)
print("REFERENCE CLASS: does the sign hold across semi vehicles?")
print("(gated cell, August prints, SMH r63rank<=25, short, h=7)")
print("=" * 78)
a = anchors(aug, 0)
gate = a[(r63.reindex(a) <= 25).fillna(False).values]
rows = []
for t in SEMI:
    r = vehicle_ret(px, [(t, -1.0)], H, 1)
    d = gate[r.reindex(gate).notna().values]
    if len(d) < 3:
        rows.append({"label": f"short {t}", "n": len(d)})
        continue
    s = summarize(r.loc[d].values, f"short {t}")
    w = int((r.loc[d] > 0).sum())
    s["record"] = f"{w}-{len(d)-w}"
    s["sign_p"] = round(sign_test(w, len(d)), 4)
    rows.append(s)
# equal-weight component basket, NVDA excluded (NVDA is the event, not the read)
comp = [t for t in SEMI if t not in ("SMH", "NVDA")]
w_ = -1.0 / len(comp)
r = vehicle_ret(px, [(t, w_) for t in comp], H, 1)
d = gate[r.reindex(gate).notna().values]
if len(d):
    s = summarize(r.loc[d].values, f"short EW basket ex-NVDA ({len(comp)} names)")
    ww = int((r.loc[d] > 0).sum())
    s["record"] = f"{ww}-{len(d)-ww}"
    s["sign_p"] = round(sign_test(ww, len(d)), 4)
    rows.append(s)
show(rows, "per-vehicle, gated August cell")
pos_frac = np.mean([r_["mean_pct"] > 0 for r_ in rows if r_.get("n", 0) >= 3])
print(f"\nfraction of vehicles with a POSITIVE short mean: {100*pos_frac:.1f}%")

# ---------------------------------------------- the contradicting relative
print("\n" + "=" * 78)
print("THE CELL THAT CONTRADICTS THE CANDIDATE")
print("=" * 78)
a_all = anchors(nv, 0)
g_all = a_all[(r63.reindex(a_all) <= 25).fillna(False).values]
g_all = g_all[ret.reindex(g_all).notna().values]
epi = declusters(g_all, 40, idx)
v = ret.loc[epi].values
w = int((v > 0).sum())
print(f"ALL NVDA prints (any month) + SMH r63rank<=25, SHORT SMH h=7:")
print(f"  episodes N={len(v)} mean {100*v.mean():+.3f}% t={summarize(v)['t']:+.2f} "
      f"record {w}-{len(v)-w} sign p={sign_test(w, len(v)):.4f}")
print(f"  i.e. the LONG in that cell pays {-100*v.mean():+.3f}%.")
print("  The candidate's own most-populated relative points the OTHER WAY.")
for dd, vv in zip(epi, v):
    print(f"    {dd.date()}  short {100*vv:+6.2f}%  (r63rank {r63.get(dd, np.nan):5.1f})")
