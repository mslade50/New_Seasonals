"""
DEPRECATED — merged into daily_scan.py on 2026-04-30.

The local overflow scan is now invoked as:
    python daily_scan.py --scope=overflow

That command runs the 6 overflow-eligible strategies against
CSV_UNIVERSE − LIQUID_PLUS_COMMODITIES with the OLV bps override (35→25),
applies the OVS earnings blackout, stages to the Overflow Google Sheets tab,
and sends the standard daily_scan email — same outputs as the legacy
local_overflow_scan.py, but through a single unified codebase.

This stub forwards any direct invocation to daily_scan.py for backwards
compatibility with the old Task Scheduler entry. The PowerShell wrapper
(scripts/run_overflow_scan.ps1) was also updated to call daily_scan.py
directly. Remove this file once the wrapper has run cleanly for a few days.
"""

import os
import sys

if __name__ == "__main__":
    print("⚠️ local_overflow_scan.py is deprecated. Forwarding to "
          "`python daily_scan.py --scope=overflow`.")
    here = os.path.dirname(os.path.abspath(__file__))
    sys.exit(os.system(f'python "{os.path.join(here, "daily_scan.py")}" --scope=overflow'))
