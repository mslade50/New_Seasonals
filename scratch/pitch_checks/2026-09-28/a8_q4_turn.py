import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa

px = close_panel(["IWM", "SPY", "^RUT"])
px = px[px["IWM"].notna()].copy()
idx = px.index
r63 = pct_rank(px["IWM"], 63)
r63_rut = pct_rank(px["^RUT"], 63)
print(f"last bar {idx[-1].date()}  IWM r63 {r63.iloc[-1]:.2f}  ^RUT r63 {r63_rut.iloc[-1]:.2f}")

s = pd.Series(idx, index=idx)
me = s.groupby([idx.year, idx.month]).max().values  # last session of each month
me = pd.DatetimeIndex(me)
me = me[me < idx[-1]]  # completed months only (Sept 2026 ME not yet printed)
pos = pd.Series(range(len(idx)), index=idx)
OFF = 3  # signal at ME-3 -> entry close ME-2 (lag 1)
sig = pd.DatetimeIndex([idx[pos[d] - OFF] for d in me])
is_q = np.array([d.month in (3, 6, 9, 12) for d in me])
midterm = np.array([d.year % 4 == 2 for d in me])
print(f"today's analogue signal: 2026-09-25 = QE-3 (QE 2026-09-30); months measured {len(me)}, quarters {is_q.sum()}")

rows = []
for h in (3, 4, 5):
    iwm = fwd_lag(px["IWM"], h)
    res = fwd_lag(px["IWM"], h) - fwd_lag(px["SPY"], h)
    drift = iwm.dropna().mean()
    gate_all = iwm[(r63 <= 5) & iwm.notna()]
    gate_all_ep = iwm.loc[declusters(gate_all.index, h, idx)]
    for nm, ser in (("IWM", iwm), ("IWM-SPY", res)):
        g = (r63.reindex(sig).values <= 5)
        cells = {"QE all": is_q, "QE r63<=5": is_q & g, "QE r63>5": is_q & ~g,
                 "nonQE ME all": ~is_q, "nonQE ME r63<=5": ~is_q & g,
                 "all ME r63<=5": g, "QE r63<=10": is_q & (r63.reindex(sig).values <= 10),
                 "QE r63<=20": is_q & (r63.reindex(sig).values <= 20),
                 "QE r63<=5 midterm": is_q & g & midterm, "QE r63<=5 non-mid": is_q & g & ~midterm,
                 "Sep QE r63<=5": g & np.array([d.month == 9 for d in me])}
        for lbl, m in cells.items():
            v = ser.reindex(sig[m]).values
            v = v[~np.isnan(v)]
            w = int((v > 0).sum())
            rows.append({"h": h, "veh": nm, "cell": lbl, "n": len(v), "rec": f"{w}-{len(v)-w}",
                         "sign_p": round(sign_test(w, len(v)), 3) if len(v) else np.nan,
                         "mean": round(100 * v.mean(), 3) if len(v) else np.nan,
                         "xs_drift": round(100 * (v.mean() - ser.dropna().mean()), 3) if len(v) else np.nan})
        if nm == "IWM":
            rows.append({"h": h, "veh": nm, "cell": "ALL DAYS drift", "n": int(iwm.notna().sum()),
                         "mean": round(100 * drift, 3)})
            v = gate_all_ep.values
            w = int((v > 0).sum())
            rows.append({"h": h, "veh": nm, "cell": "gate alone, any day (episodes)", "n": len(v),
                         "rec": f"{w}-{len(v)-w}", "sign_p": round(sign_test(w, len(v)), 3),
                         "mean": round(100 * v.mean(), 3), "xs_drift": round(100 * (v.mean() - drift), 3)})
pd.set_option("display.width", 250)
print(pd.DataFrame(rows).to_string(index=False))

h = 4
iwm = fwd_lag(px["IWM"], h)
g = r63.reindex(sig).values <= 5
q = sig[is_q & g]
print("\nQE r63<=5 episodes (signal date, QE, IWM h=3/4/5, IWM-SPY h=4, r63):")
for d in q:
    print(f"  {d.date()}  QE {idx[pos[d]+OFF].date()}  "
          f"{100*fwd_lag(px['IWM'],3)[d]:+.2f} {100*fwd_lag(px['IWM'],4)[d]:+.2f} {100*fwd_lag(px['IWM'],5)[d]:+.2f}  "
          f"res {100*(fwd_lag(px['IWM'],4)-fwd_lag(px['SPY'],4))[d]:+.2f}  r63 {r63[d]:.1f}")
m = sig[~is_q & g]
v = iwm.reindex(m).values
print(f"\nnonQE ME r63<=5 h=4: era pre-2018 {100*np.nanmean(v[m<'2018']):+.3f} (n={int((m<'2018').sum())}) "
      f"2018+ {100*np.nanmean(v[m>='2018']):+.3f} (n={int((m>='2018').sum())})")
v = iwm.reindex(q).values
print(f"QE r63<=5 h=4: pre-2018 {100*np.nanmean(v[q<'2018']):+.3f} (n={int((q<'2018').sum())}) "
      f"2018+ {100*np.nanmean(v[q>='2018']):+.3f} (n={int((q>='2018').sum())})")
print("cluster:", cluster_note(q, v))
