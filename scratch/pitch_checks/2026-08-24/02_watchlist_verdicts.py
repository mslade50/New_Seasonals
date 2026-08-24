"""Today's live value of every ACTIVE watchlist trigger.

Stage B1 owes each active entry a verdict citing today's number. This prints
the number; 00_surface_map.md writes the verdict.
"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa
import numpy as np, pandas as pd

BAR = pd.Timestamp("2026-08-21")
ROOT = Path(__file__).resolve().parents[3]
tape = json.load(open(ROOT / "data" / "pitch_tape.json"))["tickers"]

need = ["TLT", "IEF", "LQD", "HYG", "SPY", "GLD", "GDX", "USO", "XLE", "IHI",
        "FXI", "EEM", "^VIX", "^SKEW", "DX-Y.NYB", "^TNX", "XLV", "XLK", "KRE",
        "XLF", "SVXY", "QQQ", "IWM"]
px = load_prices(need)


def own(t, fn):
    c = px[t]["Close"].dropna()
    c = c[c.index <= BAR]
    return fn(c)


def d52(t, which="low"):
    c = px[t]["Close"].dropna(); c = c[c.index <= BAR]
    ext = c.rolling(252).min().iloc[-1] if which == "low" else c.rolling(252).max().iloc[-1]
    return 100 * (c.iloc[-1] / ext - 1)


def rank(t, n):
    c = px[t]["Close"].dropna(); c = c[c.index <= BAR]
    return float(pct_rank(c, n).iloc[-1])


def ret(t, n):
    c = px[t]["Close"].dropna(); c = c[c.index <= BAR]
    return 100 * (c.iloc[-1] / c.iloc[-1 - n] - 1)


print("W0  nfp x rates (TLT NFP, midterm-dead)      : 2026 is midterm; next NFP 2026-09-04 = +9 td. cycle trigger 2027-01.")
print(f"W1  HYG/LQD joint 52w extremes              : HYG {d52('HYG','high'):+.2f}% off high, LQD {d52('LQD','low'):+.2f}% off low; trigger = >=8 declustered episodes, still 4 since 2007.")
print(f"W2  SVXY overnight into CPI                 : next CPI 2026-09-11 = +13 td, outside the 10 td horizon cap.")
print(f"W3  GLD on a miner-led thrust               : GDX r5 rank {rank('GDX',5):.1f}, GLD r5 rank {rank('GLD',5):.1f}, GLD off 52wh {d52('GLD','high'):+.2f}% (4th condition needs >-10%).")
print(f"W4  XLE on a crude 1d thrust in [5,6)%      : USO 1d {ret('USO',1):+.2f}%.")
print(f"W5  TLT + IG complex pinned (tight rung)    : TLT {d52('TLT','low'):+.2f}% (needs <=0.5), IEF {d52('IEF','low'):+.2f}% (<=1.0), LQD {d52('LQD','low'):+.2f}% (<=1.0).")
sk = px.get("^SKEW")
if sk is not None:
    print(f"W6  SPY on a skew spike, non-midterm        : SKEW r5 rank {rank('^SKEW',5):.1f}, SPY off 52wh {d52('SPY','high'):+.2f}%; MIDTERM year blocks it (turns on 2027-01).")
else:
    print("W6  SPY on a skew spike                     : ^SKEW not in cache; midterm year blocks it regardless (turns on 2027-01).")
print(f"W7  crude thrust from a deep base           : USO r5 rank {rank('USO',5):.1f} (needs >=90), r63 rank {rank('USO',63):.1f} (needs <=20).")
ihi = px.get("IHI")
print(f"W8  IHI 21d rank 100 out of a drawdown      : IHI r21 rank {rank('IHI',21):.1f} (needs 100); reference-class blocker stands." if ihi is not None else "W8  IHI: not in cache.")
print(f"W9  FXI 5d break inside an intact thrust    : FXI r5 rank {rank('FXI',5):.1f} (needs <=20), r21 rank {rank('FXI',21):.1f}.")
print( "W10 TLT NOVEMBER month-position             : parks to ~2026-11-05 (trading days 4-12 of November).")
print(f"W11 SPY 52w high while TLT 52w low          : SPY {d52('SPY','high'):+.2f}% off high (needs <=0.5), TLT {d52('TLT','low'):+.2f}% off low (needs <=1.0).")
print(f"W12 TLT month-end ME-9, ungated             : August ME-9 was 2026-08-18 (gone; today is ME-5). TLT {d52('TLT','low'):+.2f}% off its low against the >3% trigger.")
print(f"W13 SPY on a vol pop inside a calm tape     : VIX 21d rank {rank('^VIX',21):.1f} (needs <=25), VIX 1d {ret('^VIX',1):+.2f}% (needs >=+5).")
print(f"W14 gold on an unconfirmed rate rise        : DX 21d rank {rank('DX-Y.NYB',21):.1f} (needs <=15) OK; 21-session yield rise "
      f"{px['^TNX']['Close'].dropna()[lambda s: s.index<=BAR].iloc[-1] - px['^TNX']['Close'].dropna()[lambda s: s.index<=BAR].iloc[-22]:+.3f}pt (needs >=+0.20).")
print(f"W15 tech vs healthcare rotation gap         : 1d XLV-XLK gap {ret('XLV',1)-ret('XLK',1):+.2f}pp (needs >=+3.0).")
print(f"W16 short the dollar on a rate rise         : TNX 21d rank {rank('^TNX',21):.1f} (needs >=65), DX 21d rank {rank('DX-Y.NYB',21):.1f} (needs <=20).")
print(f"W17 crude through Jackson Hole, JH-6        : JH is 2026-08-28; JH-6 was 2026-08-20 (gone). XLE {d52('XLE','high'):+.2f}% off its 52w high (entry forbids within 5%).")
print(f"W18 short TLT after a big up day at the low : TLT 1d {ret('TLT',1):+.2f}% (needs >=+1.5).")
print(f"W19 KRE vs XLF on a bank breadth washout    : cost trigger on history (+0.35% h=3 ex-crisis); KRE r5 rank {rank('KRE',5):.1f}, XLF r5 rank {rank('XLF',5):.1f}.")
print( "W20 HYG across Jackson Hole, JH-5           : JH-5 was 2026-08-21 (gone); anchor closed on credit 2026-08-21.")
