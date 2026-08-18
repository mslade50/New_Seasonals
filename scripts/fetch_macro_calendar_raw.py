"""Fetch + cache raw source pages for the macro event calendar.

Sources:
- federalreserve.gov: fomchistorical{2000..2020}.htm + fomccalendars.htm
  (direct — the Fed does not block plain HTTP clients)
- bls.gov: /schedule/{year}/home.htm per-year release schedules, fetched
  through the Wayback Machine because bls.gov 403s non-browser clients.

Everything lands in artifacts/macro_calendar_raw/ and is never refetched if
the file already exists (delete a file to force a refresh).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "artifacts" / "macro_calendar_raw"

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/126.0 Safari/537.36")

FED_HIST_YEARS = range(2000, 2021)   # 2021+ live on fomccalendars.htm
BLS_YEARS = range(2000, 2026)        # 2025 page exists on wayback by now


def fetch(url: str, dest: Path, retries: int = 3, timeout: int = 90) -> bool:
    if dest.exists() and dest.stat().st_size > 5000:
        print(f"  cached  {dest.name}")
        return True
    last_err = ""
    for attempt in range(retries):
        try:
            r = requests.get(url, headers={"User-Agent": UA}, timeout=timeout)
            if r.status_code == 200 and len(r.content) > 5000:
                dest.write_bytes(r.content)
                print(f"  fetched {dest.name} ({len(r.content)} bytes)")
                return True
            last_err = f"HTTP {r.status_code} len={len(r.content)}"
        except requests.RequestException as e:
            last_err = str(e)
        time.sleep(3 * (attempt + 1))
    print(f"  FAILED  {dest.name}: {last_err}")
    return False


def main() -> int:
    CACHE.mkdir(parents=True, exist_ok=True)
    failures = []

    print("Fed FOMC pages (direct):")
    for y in FED_HIST_YEARS:
        url = f"https://www.federalreserve.gov/monetarypolicy/fomchistorical{y}.htm"
        if not fetch(url, CACHE / f"fomchistorical{y}.htm"):
            failures.append(f"fed:{y}")
    if not fetch("https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm",
                 CACHE / "fomccalendars.htm"):
        failures.append("fed:calendars")

    print("BLS schedule pages (via Wayback Machine):")
    for y in BLS_YEARS:
        url = (f"https://web.archive.org/web/2026/"
               f"https://www.bls.gov/schedule/{y}/home.htm")
        if not fetch(url, CACHE / f"bls_sched_{y}.htm"):
            failures.append(f"bls:{y}")
        time.sleep(1)

    # 2026 schedule lives on the current-year news_release pages
    for name in ("cpi", "empsit", "ppi"):
        url = (f"https://web.archive.org/web/2026/"
               f"https://www.bls.gov/schedule/news_release/{name}.htm")
        if not fetch(url, CACHE / f"bls_current_{name}.htm"):
            failures.append(f"bls:current:{name}")
        time.sleep(1)

    if failures:
        print(f"\n{len(failures)} failures: {failures}")
        return 1
    print("\nAll sources cached.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
