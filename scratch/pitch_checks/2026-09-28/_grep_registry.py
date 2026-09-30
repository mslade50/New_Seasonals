import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
lines = (ROOT / "data/pitch_negative_registry.md").read_text(encoding="utf-8").splitlines()
TOPICS = {
    "C1 quarter-end rebalance": r"rebalanc|pension|quarter-end|quarter end|QE-",
    "C2 size spread": r"IWM.{0,40}QQQ|QQQ.{0,40}IWM|size spread|small.cap.{0,30}large|RSP",
    "C3 nfp": r"\bNFP\b|payroll",
    "C4 move/vix": r"MOVE.{0,60}VIX|VIX.{0,60}MOVE|bond vol",
    "C5 credit/equity": r"HYG.{0,80}SPY|credit.{0,40}confirm|non-confirm",
    "C6 gold/silver": r"gold.silver|SLV.{0,40}GLD|GLD.{0,40}SLV|silver",
    "C7 xle/uso": r"XLE.{0,50}USO|USO.{0,50}XLE|energy equit",
    "C8 analogue": r"analogue|nearest.neighbo",
    "C9 MU/earnings sector": r"\bMU\b|bellwether|SMH.{0,40}earn",
    "C10 52w low into print": r"52w low.{0,60}earn|earn.{0,60}52w low|into (its|the) print|pre-print|announcement premium",
}
which = sys.argv[1:] or list(TOPICS)
for k in TOPICS:
    if not any(w in k for w in which):
        continue
    pat = re.compile(TOPICS[k], re.I)
    hits = [(i + 1, l) for i, l in enumerate(lines) if pat.search(l)]
    print(f"\n==== {k}: {len(hits)} hits")
    for n, l in hits[:40]:
        print(f"{n}: {l[:230]}")
