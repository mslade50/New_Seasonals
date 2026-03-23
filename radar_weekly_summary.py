"""
radar_weekly_summary.py

Reads the last 7 days of equity radar briefs, pulls live market data for
every ticker mentioned, and pipes everything to Claude Code with a rigorous
PM-style framework to produce a weekly "best of" digest.

Runs locally Sunday ~8:30 AM ET via Task Scheduler.
Output: data/radar_weekly_summary.md (committed + pushed for weekly rundown).

Author: McKinley
"""

import os
import sys
import re
import json
import subprocess
import logging
from datetime import datetime, timedelta
from pathlib import Path

import yfinance as yf
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
RADAR_BRIEFS_DIR = Path(r"C:\Users\mckin\projects\last30days-radar\output\briefs")
RADAR_JSON_DIR = Path(r"C:\Users\mckin\projects\last30days-radar\output\json")
OUTPUT_PATH = PROJECT_ROOT / "data" / "radar_weekly_summary.md"


# ---------------------------------------------------------------------------
# 1. Collect the week's briefs
# ---------------------------------------------------------------------------

def collect_briefs(days_back=7):
    """Read equity_radar briefs from the last N days. Returns list of (date_str, text)."""
    cutoff = datetime.now() - timedelta(days=days_back)
    briefs = []

    for f in sorted(RADAR_BRIEFS_DIR.glob("equity_radar_*.md")):
        # Parse date from filename: equity_radar_20260323_073309.md
        match = re.search(r"equity_radar_(\d{8})_", f.name)
        if not match:
            continue
        file_date = datetime.strptime(match.group(1), "%Y%m%d")
        if file_date >= cutoff:
            text = f.read_text(encoding="utf-8")
            date_str = file_date.strftime("%Y-%m-%d")
            briefs.append((date_str, text))

    logger.info(f"Collected {len(briefs)} briefs from last {days_back} days")
    return briefs


# ---------------------------------------------------------------------------
# 2. Extract unique tickers
# ---------------------------------------------------------------------------

def extract_tickers(briefs):
    """Pull unique tickers from brief markdown tables and catalyst headers."""
    tickers = set()
    for _, text in briefs:
        # Table rows: | TICKER | ...
        for m in re.finditer(r"^\|\s*([A-Z]{1,5})\s*\|", text, re.MULTILINE):
            candidate = m.group(1)
            # Skip table header words
            if candidate not in {"Ticker", "Score", "Type", "Why"}:
                tickers.add(candidate)
        # Catalyst headers: ### TICKER [category]
        for m in re.finditer(r"^###\s+([A-Z_]{1,20})\s+\[", text, re.MULTILINE):
            candidate = m.group(1)
            # Skip basket/theme names with underscores for now,
            # but keep single tickers
            if "_" not in candidate:
                tickers.add(candidate)

    logger.info(f"Extracted {len(tickers)} unique tickers")
    return sorted(tickers)


# ---------------------------------------------------------------------------
# 3. Pull market data snapshot
# ---------------------------------------------------------------------------

def pull_market_data(tickers):
    """Pull yfinance snapshot for each ticker. Returns dict of ticker -> data."""
    snapshots = {}

    for ticker in tickers:
        try:
            tk = yf.Ticker(ticker)
            info = tk.info or {}

            # Price history for MAs
            hist = tk.history(period="1y")
            if hist.empty:
                logger.warning(f"  {ticker}: no price history, skipping")
                continue

            close = hist["Close"]
            current_price = float(close.iloc[-1])
            sma_50 = float(close.rolling(50).mean().iloc[-1]) if len(close) >= 50 else None
            sma_200 = float(close.rolling(200).mean().iloc[-1]) if len(close) >= 200 else None

            # Volume
            vol_recent = float(hist["Volume"].iloc[-5:].mean()) if len(hist) >= 5 else None
            vol_20d = float(hist["Volume"].iloc[-20:].mean()) if len(hist) >= 20 else None
            vol_ratio = round(vol_recent / vol_20d, 2) if vol_recent and vol_20d and vol_20d > 0 else None

            # RSI 14
            rsi = None
            if len(close) >= 15:
                delta = close.diff()
                gain = delta.where(delta > 0, 0).rolling(14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
                rs = gain.iloc[-1] / loss.iloc[-1] if loss.iloc[-1] > 0 else 100
                rsi = round(100 - (100 / (1 + rs)), 1)

            # 52w range
            high_52w = float(info.get("fiftyTwoWeekHigh", close.max()))
            low_52w = float(info.get("fiftyTwoWeekLow", close.min()))
            pct_from_52w_high = round((current_price / high_52w - 1) * 100, 1) if high_52w else None

            # Trend summary
            trend = "N/A"
            if sma_50 and sma_200:
                if current_price > sma_50 > sma_200:
                    trend = "Strong uptrend (price > 50d > 200d)"
                elif current_price > sma_50 and sma_50 < sma_200:
                    trend = "Recovering (price > 50d, 50d < 200d)"
                elif current_price < sma_50 < sma_200:
                    trend = "Downtrend (price < 50d < 200d)"
                elif current_price < sma_50 and sma_50 > sma_200:
                    trend = "Weakening (price < 50d, 50d > 200d)"
                else:
                    trend = "Mixed"
            elif sma_50:
                trend = f"{'Above' if current_price > sma_50 else 'Below'} 50d SMA"

            snapshots[ticker] = {
                "price": round(current_price, 2),
                "sma_50": round(sma_50, 2) if sma_50 else None,
                "sma_200": round(sma_200, 2) if sma_200 else None,
                "trend": trend,
                "rsi_14": rsi,
                "52w_high": round(high_52w, 2),
                "52w_low": round(low_52w, 2),
                "pct_from_52w_high": pct_from_52w_high,
                "volume_ratio_5d_vs_20d": vol_ratio,
                "market_cap": info.get("marketCap"),
                "trailing_pe": info.get("trailingPE"),
                "forward_pe": info.get("forwardPE"),
                "price_to_sales": info.get("priceToSalesTrailing12Months"),
                "price_to_book": info.get("priceToBook"),
                "ev_to_ebitda": info.get("enterpriseToEbitda"),
                "free_cash_flow_yield": (
                    round(info["freeCashflow"] / info["marketCap"] * 100, 2)
                    if info.get("freeCashflow") and info.get("marketCap")
                    else None
                ),
                "revenue_growth": info.get("revenueGrowth"),
                "earnings_growth": info.get("earningsGrowth"),
            }
            logger.info(f"  {ticker}: ${current_price:.2f}, {trend}")

        except Exception as e:
            logger.warning(f"  {ticker}: failed ({e})")

    logger.info(f"Got market data for {len(snapshots)}/{len(tickers)} tickers")
    return snapshots


# ---------------------------------------------------------------------------
# 4. Build the Claude prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a senior portfolio manager at a multi-strategy hedge fund.
You are reviewing this week's daily equity radar briefings alongside live market data.
Your job is to distill the week into the highest-signal ideas — things that would
actually change how you allocate capital.

## How to think about this

You are not applying a checklist. You are reasoning about each situation on its own
terms. Sometimes valuation is everything. Sometimes a single catalyst is so large it
overwhelms all other factors. Sometimes a trend break after a long base is the entire
signal. Context determines which factors matter.

Consider these lenses, but weight them as the situation demands:

**Catalyst magnitude**: Is this a marginal news item or does it structurally change
the earnings power of the business?

**Valuation vs forward reality**: Current multiples reflect trailing or consensus
expectations. If a catalyst meaningfully changes forward cash flows, the question is
whether the market has re-priced to reflect that yet. A stock up 30% can still be cheap
if the world just changed by more than 30%.

**Trend and market structure**: A stock in a protracted downtrend absorbs positive news
differently than one basing or in an uptrend — trapped holders create overhead supply.
But this is not a hard rule. Sometimes the catalyst is so large it doesn't matter
(NVDA May 2023). Judge when trend matters and when it doesn't.

**Persistence**: Did this show up once this week or every day? Repetition suggests a
durable theme, not noise.

**Crowd positioning**: When retail, analysts, and institutional flow all agree, the risk
is the crowded unwind, not the upside.

## The two questions that matter most

**1. Variant Perception (REQUIRED)**

For every idea you surface, you MUST articulate what you specifically believe that the
market's current pricing does not reflect. This is not "the company is good" or "the
catalyst is positive" — the market may already agree with that. The variant perception
is the SPECIFIC disagreement with consensus pricing.

Example: "The market is treating the Hormuz disruption as a 3-month event that
mean-reverts. I believe the structural rerouting of LNG contracts makes this a
multi-year earnings reset that current multiples don't capture."

If you cannot articulate a clear variant perception — if your thesis IS the consensus —
kill the idea. It doesn't matter how good the setup looks. No variant perception, no
edge, no inclusion.

**2. Why does this opportunity exist? Who is on the other side?**

If an idea is obviously great, you need to explain why you're able to buy it at this
price. Someone is selling to you. Why?

- **Forced/mechanical selling** (index rebalance, fund redemptions, margin liquidation,
  ESG exclusion) — This is a gift. The seller has no view, they are compelled to sell.
  This is the cleanest setup.
- **Informed disagreement** — Someone smart is on the other side. You need to understand
  their bear case cold, not just acknowledge it exists. Articulate their strongest
  argument, then explain specifically why you think they're wrong.
- **Neglect/indifference** — Nobody is looking. No coverage, small cap, boring sector.
  Edge through attention asymmetry. But verify it's actually neglected vs neglected for
  good reason.

If you can't identify why the opportunity exists, you are likely the patsy. Say so and
move on.

## Output format

Include everything that clears the bar. Exclude everything that doesn't. Some weeks
that's several ideas, some weeks it's zero. The number itself is informative.

For each idea that clears the bar:
- **Ticker/theme** and one-line thesis
- **Variant perception**: What you believe that consensus doesn't
- **Who's on the other side**: Why this opportunity exists
- **The data**: Quote the specific numbers (price, valuation, trend, catalyst)
- **What would change your mind**: The specific thing that would invalidate this

Be direct. Take a view. If something is noise, say so. If something is the most
important thing in the market right now, say that. Don't hedge everything.

If nothing clears the bar this week, say "Nothing cleared the bar this week" and
briefly explain why (e.g., "everything interesting was already consensus" or
"catalysts were real but fully priced").

Output as clean markdown. Use ## headers for each idea."""


def build_prompt(briefs, market_data):
    """Assemble the full prompt with briefs + market data."""
    parts = [SYSTEM_PROMPT, "\n---\n"]

    # Briefs
    parts.append("# This Week's Radar Briefs\n")
    for date_str, text in briefs:
        parts.append(f"## {date_str}\n{text}\n")

    # Market data
    parts.append("\n---\n")
    parts.append("# Live Market Data (as of today)\n")
    parts.append("```json\n")
    parts.append(json.dumps(market_data, indent=2, default=str))
    parts.append("\n```\n")

    prompt = "\n".join(parts)
    logger.info(f"Prompt assembled: {len(prompt)} chars")
    return prompt


# ---------------------------------------------------------------------------
# 5. Invoke Claude Code
# ---------------------------------------------------------------------------

def invoke_claude(prompt, timeout_seconds=600):
    """Call Claude Code via subprocess, piping prompt on stdin."""
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env.pop("CLAUDECODE", None)

    claude_cmd = os.environ.get("CLAUDE_CMD", "claude")
    cmd = [claude_cmd, "-p", "--no-session-persistence",
           "--dangerously-skip-permissions"]

    logger.info(f"Invoking Claude Code (timeout={timeout_seconds}s, prompt={len(prompt)} chars)...")

    try:
        result = subprocess.run(
            cmd,
            input=prompt,
            capture_output=True,
            text=True,
            encoding="utf-8",
            env=env,
            timeout=timeout_seconds,
            cwd=str(PROJECT_ROOT),
            shell=True,
        )
        if result.returncode != 0:
            logger.warning(f"Claude returned exit code {result.returncode}")
            if result.stderr:
                logger.warning(f"stderr: {result.stderr[:500]}")
        return result.stdout.strip()
    except subprocess.TimeoutExpired:
        logger.error(f"Claude timed out after {timeout_seconds}s")
        return ""
    except FileNotFoundError:
        logger.error("'claude' command not found")
        return ""


# ---------------------------------------------------------------------------
# 6. Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("RADAR WEEKLY SUMMARY")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # 1. Collect briefs
    print("\n[1/4] Collecting briefs...")
    briefs = collect_briefs(days_back=7)
    if not briefs:
        print("  No briefs found. Exiting.")
        return

    # 2. Extract tickers and pull market data
    print("[2/4] Pulling market data...")
    tickers = extract_tickers(briefs)
    print(f"  Tickers: {', '.join(tickers)}")
    market_data = pull_market_data(tickers)

    # 3. Build prompt and invoke Claude
    print("[3/4] Running through Claude...")
    prompt = build_prompt(briefs, market_data)
    summary = invoke_claude(prompt, timeout_seconds=900)

    if not summary:
        print("  Empty response from Claude. Exiting.")
        return

    # 4. Write output
    print("[4/4] Writing summary...")
    date_str = datetime.now().strftime("%Y-%m-%d")
    header = f"# Radar Weekly Digest — {date_str}\n\n"
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(header + summary, encoding="utf-8")
    print(f"  Written to {OUTPUT_PATH}")
    print(f"  Length: {len(summary)} chars")

    print("\nDone.")


if __name__ == "__main__":
    main()
