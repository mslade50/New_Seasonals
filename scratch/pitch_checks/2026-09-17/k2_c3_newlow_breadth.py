"""K2 c3 round 1: new-LOW breadth under a near-high index, SPY h=1..10.

Share of a universe within X% of its trailing-252 CLOSE low (tape definition:
build_pitch_state close.rolling(252).min()), ranked against its own trailing
252/504 distribution (PIT), gated on SPY within 5%/3% of its trailing-252 high.
Three universes: tape single stocks (today's survivors), full cache (all names,
today's survivors, broader), and a sector+industry ETF universe.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa
import json

OUT = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[3]

NON_STOCK = set("""CEF DBC DIA DX-Y.NYB EEM EFA EWJ EWZ FXI GDX GLD HYG IBB IEF IHI ITA ITB
IWM IYR KRE LQD OIH QQQ SLV SMH SPY SVXY TLT UNG USO UUP UVXY VNQ XBI XHB XLB XLC XLE
XLF XLI XLK XLP XLRE XLU XLV XLY XME XOP XRT ^GSPC ^MOVE ^NDX ^SKEW ^TNX ^VIX
^VIX3M""".split())
ETF_CAND = """XLB XLE XLF XLI XLK XLP XLU XLV XLY XLC XLRE IBB IHI ITA ITB IYR KRE OIH SMH
XBI XHB XME XOP XRT VNQ KBE KIE XAR XTN XSD XPH XHE XES XSW IYT IGV SOXX IYW IYF IYH
IYE IYJ IYM IYK IYC IDU IYZ PBJ PPH TAN ICLN FDN IGE IYG IAT IEZ IHF IEO""".split()

tape = json.load(open(ROOT / "data" / "pitch_tape.json"))
TAPE_STOCKS = sorted(t for t in tape["tickers"] if t not in NON_STOCK)

mp = pd.read_parquet(PRICES_PATH, columns=["ticker", "date", "Close"])
mp["date"] = pd.to_datetime(mp["date"])
mp = mp.drop_duplicates(["ticker", "date"], keep="last")
W = mp.pivot(index="date", columns="ticker", values="Close").sort_index()
del mp
spy_idx = W["SPY"].dropna().index
W = W.loc[W.index >= spy_idx[0]]
print(f"panel {W.shape}, SPY from {spy_idx[0].date()}, last {W.index[-1].date()}")

ALL = [c for c in W.columns if not c.startswith("^") and "=" not in c
       and not c.endswith(".NYB")]
ETFS = [t for t in ETF_CAND if t in W.columns]
print(f"tape stocks {len(TAPE_STOCKS)}, full cache names {len(ALL)}, ETFs {len(ETFS)}: {ETFS}")


def dist_frames(cols):
    lo, hi = {}, {}
    for c in cols:
        s = W[c].dropna()
        if len(s) < 253:
            continue
        mn = s.rolling(252).min()
        mx = s.rolling(252).max()
        lo[c] = (s / mn - 1.0).reindex(W.index)
        hi[c] = (s / mx - 1.0).reindex(W.index)
    return pd.DataFrame(lo), pd.DataFrame(hi)


def shares(cols, tag):
    dlo, dhi = dist_frames(cols)
    nval = dlo.notna().sum(axis=1)
    out = pd.DataFrame(index=W.index)
    out["n"] = nval
    for x in (0.005, 0.01, 0.02):
        out[f"lo{x}"] = (dlo <= x).sum(axis=1) / nval.replace(0, np.nan)
        out[f"hi{x}"] = (dhi >= -x).sum(axis=1) / nval.replace(0, np.nan)
    out = out[out["n"] >= max(8, int(0.5 * len(cols)) if tag == "etf" else 30)]
    return out


def pit_pct(s, lb):
    return rolling_on_valid(s, lambda x: x.rolling(lb).rank(pct=True) * 100.0)


U = {"tape": shares(TAPE_STOCKS, "tape"), "full": shares(ALL, "full"),
     "etf": shares(ETFS, "etf")}

spy = W["SPY"]
spy_dd = rolling_on_valid(spy, lambda x: x / x.rolling(252).max() - 1.0)
px = W[["SPY"]].loc[spy_idx].copy()
px["IWM"] = W["IWM"]

print("\n=== TODAY'S READING (2026-09-16) and PIT percentiles ===")
print(f"SPY dist to 252 high: {100*spy_dd.iloc[-1]:.2f}%")
for k, df in U.items():
    last = df.index[-1]
    row = df.loc[last]
    msg = [f"{k}: date {last.date()} n={int(row['n'])}"]
    for x in (0.005, 0.01, 0.02):
        s = df[f"lo{x}"]
        msg.append(f"lo{x}={100*row[f'lo{x}']:.2f}% pit252={pit_pct(s,252).iloc[-1]:.1f} "
                   f"pit504={pit_pct(s,504).iloc[-1]:.1f} full={100*(s<=s.iloc[-1]).mean():.1f}")
    msg.append(f"hi0.01={100*row['hi0.01']:.2f}% pit252={pit_pct(df['hi0.01'],252).iloc[-1]:.1f}")
    print("  " + " | ".join(msg))

# save the share series for round 2
pd.concat({k: v for k, v in U.items()}, axis=1).to_parquet(OUT / "k2_c3_shares.parquet")

H = (1, 2, 3, 5, 10)
rows = []
for k, df in U.items():
    for lb in (252, 504):
        pct = pit_pct(df["lo0.01"], lb).reindex(px.index)
        dd = spy_dd.reindex(px.index)
        for thr in (90, 95):
            for idx_d in (-0.05, -0.03):
                near = dd >= idx_d
                cell = near & (pct >= thr)
                comp = near & (pct < thr) & pct.notna()
                breadth_only = (pct >= thr)
                for h in (1, 5, 10):
                    r = vehicle_ret(px, [("SPY", 1.0)], h, 1)
                    v = r.notna()
                    ep = declusters(px.index[cell.fillna(False) & v], h, px.index)
                    rc = summarize(r.loc[ep].values)
                    rows.append({"univ": k, "lb": lb, "thr": thr, "idx": idx_d, "h": h,
                                 "N_days": int((cell.fillna(False) & v).sum()), "N_ep": rc["n"],
                                 "cell_ep": rc.get("mean_pct"), "hit": rc.get("hit"),
                                 "cell_day": 100*r[cell.fillna(False) & v].mean(),
                                 "nearhigh_nobreadth": 100*r[comp.fillna(False) & v].mean(),
                                 "breadth_only": 100*r[breadth_only.fillna(False) & v].mean(),
                                 "nearhigh_all": 100*r[near.fillna(False) & v].mean(),
                                 "all_days": 100*r[v].mean()})
g = pd.DataFrame(rows)
pd.set_option("display.width", 250)
print("\n=== GRID: cell vs near-high-without-breadth (decisive) ===")
print(g.round(3).to_string(index=False))

# canonical cell: tape, lb252, thr90, idx -5%
for k in ("tape", "full", "etf"):
    df = U[k]
    pct = pit_pct(df["lo0.01"], 252).reindex(px.index)
    mask = (spy_dd.reindex(px.index) >= -0.05) & (pct >= 90)
    var = {"pct95": (spy_dd.reindex(px.index) >= -0.05) & (pct >= 95),
           "idx3": (spy_dd.reindex(px.index) >= -0.03) & (pct >= 90),
           "no_index_gate": (pct >= 90)}
    battery(px, mask.fillna(False), [("SPY", 1.0)], 5,
            f"c3 [{k}] low-share pit252>=90 & SPY within 5% of high, LONG SPY", 1.5,
            variants={a: b.fillna(False) for a, b in var.items()}, event_kinds=("fomc_decision",))
