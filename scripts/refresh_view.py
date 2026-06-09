"""
refresh_view.py — one-command daily refresh of the trade ledger + HTML view.

The full rebuild is fast (~50s warm, ~100s cold), so there's no need for
incremental machinery. This wrapper just runs the two steps in order and
(optionally) opens the result.

  python scripts/refresh_view.py            # rebuild ledger + regen HTML
  python scripts/refresh_view.py --open     # ... then open it in the browser
  python scripts/refresh_view.py --html-only  # skip the ~50s rebuild, just regen HTML (~10s)

Data freshness: by default this force-pulls master_prices + earnings from R2
first (bypassing data_provider's 18h staleness window) so the view always
includes the latest close. Pass --no-pull to skip and use the local copies.
"""
import argparse
import os
import subprocess
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
HTML = os.path.join(_ROOT, "reports", "portfolio", "backtester_view.html")


def run(script):
    t0 = time.time()
    print(f"\n=== {script} ===", flush=True)
    r = subprocess.run([sys.executable, os.path.join(_HERE, script)],
                       capture_output=True, text=True)
    dt = time.time() - t0
    # surface the meaningful tail lines, suppress streamlit/bare-mode noise
    for line in r.stdout.splitlines():
        low = line.lower()
        if any(k in low for k in ("trades", "wrote", "loaded", "error", "fail", "candidate")):
            print("  " + line.strip())
    if r.returncode != 0:
        print(r.stderr[-2000:])
        raise SystemExit(f"{script} failed (exit {r.returncode})")
    print(f"  [{dt:.1f}s]", flush=True)
    return dt


def pull_fresh_data():
    """Force a fresh master_prices + earnings pull from R2 so the nightly view
    includes today's close (bypasses data_provider's 18h staleness window).
    Resilient: any failure falls back to whatever local copy exists."""
    print("\n=== pull fresh data (R2) ===", flush=True)
    try:
        sys.path.insert(0, _ROOT)
        from cache_io import download_to_local, is_configured
        if not is_configured():
            print("  R2 not configured - using local data as-is", flush=True)
            return
        for key in ("master_prices.parquet", "earnings_calendar.parquet"):
            dest = os.path.join(_ROOT, "data", key)
            ok = download_to_local(key, dest)
            print(f"  pulled {key}: {'ok' if ok else 'FAILED (using local)'}", flush=True)
    except Exception as e:
        print(f"  data pull skipped ({e}); using local copies", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--open", action="store_true", help="open the HTML when done")
    ap.add_argument("--html-only", action="store_true", help="skip ledger rebuild")
    ap.add_argument("--no-pull", action="store_true", help="skip the R2 master_prices/earnings refresh")
    args = ap.parse_args()

    t0 = time.time()
    if not args.html_only:
        if not args.no_pull:
            pull_fresh_data()
        run("build_trade_ledger.py")
    run("backtester_html_report.py")
    print(f"\nTotal: {time.time()-t0:.1f}s  ->  {HTML}", flush=True)

    if args.open:
        try:
            os.startfile(HTML)  # Windows
        except AttributeError:
            subprocess.run(["xdg-open", HTML])


if __name__ == "__main__":
    main()
