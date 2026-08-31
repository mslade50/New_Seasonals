# DISASTER RECOVERY RUNBOOK
### "The System is Down — What Do I Do?"

**Print this page. Keep it at your desk. It should NOT live only in the GitHub repo.**

Last updated: 2026-08-31

---

## SCENARIO 1: GitHub is Unavailable

**Symptoms:** Can't push/pull, GitHub Actions don't run, can't access repo.

**Impact:** Local data/scan/report jobs continue. GitHub-hosted backup and
private/shared site deploys are unavailable until GitHub recovers; the prior
site deployment remains live.

**Recovery:**
1. Check the `New Seasonals Local v3 - *` Task Scheduler entries and pinned-runtime
   logs under `New_Seasonals-automation-runtime-v3\artifacts\automation\logs`.
2. Do not repoint tasks to the development checkout. The pinned operational
   worktree is fully functional without GitHub after installation.
3. Signals still email and write Google Sheets as long as R2, Google, SMTP, and
   market-data endpoints are reachable. Site builds queue only after GitHub returns.
4. If the local machine also fails, skip new entries and follow the complete
   system-failure protocol below; the GitHub backup cannot run during the outage.
5. **Manual scanning fallback:** Open TradingView or your broker platform. Check the 5 core filters for each strategy by hand:
   - Performance rank (2d/5d/10d/21d return percentile)
   - ATR % range
   - Price vs 200 SMA
   - Volume vs 63-day average
   - Seasonal rank

**Prevention:** Keep the pinned runtime/task set healthy and retain the immutable
fallback tag. Scheduled production code never `git pull`s or self-upgrades.

---

## SCENARIO 2: Google Sheets API Credentials Expire

**Symptoms:** `daily_scan.py` runs but throws `gspread` auth errors. Orders don't stage.

**Impact:** Signals are detected but NOT written to the staging sheet.

**Recovery:**
1. Signals are still sent via **email**. Check your inbox for the daily signal email.
2. Stage orders manually in the Google Sheet by copying signal details from the email.
3. To fix credentials:
   - Go to Google Cloud Console → APIs & Services → Credentials
   - Download a new service account JSON key
   - Replace the external key file referenced by `LOCAL_AUTOMATION_GCP_JSON_PATH`
   - Update the GitHub backup `GCP_JSON` secret in repo Settings → Secrets
4. Test with: `python -c "import gspread; gc = gspread.service_account(); print('OK')"`

**Prevention:** Set a calendar reminder to check credential expiry every 6 months. Google service account keys don't expire by default, but org policies may rotate them.

---

## SCENARIO 3: Email Alerts Stop Working

**Symptoms:** No morning/afternoon signal emails. Script may or may not error.

**Impact:** You won't see signals unless you check Google Sheets or run the scan manually.

**Recovery:**
1. Check Google Sheets directly — orders may still be staged even if email failed.
2. Common causes:
   - Gmail app password was revoked → Generate new one at myaccount.google.com → Security → App Passwords
   - Gmail daily send limit hit (500/day for regular, 2000/day for Workspace)
   - SMTP port blocked by network
3. Quick test: `python -c "import smtplib; s=smtplib.SMTP('smtp.gmail.com',587); s.starttls(); s.login('your@email.com','app_password'); print('OK')"`
4. **Temporary workaround:** Run `daily_scan.py` locally and just read the console output.

**Prevention:** Strict automation treats a missing/failed email as non-success;
check the component's R2 receipt and pinned-runtime log.

---

## SCENARIO 4: yfinance Returns Bad/Empty Data

**Symptoms:** Script runs but produces zero signals, or signals on unusual tickers.

**Impact:** False negatives (missed signals) or false positives (bad signals).

**Recovery:**
1. Check yfinance directly: `python -c "import yfinance as yf; print(yf.download('SPY', period='5d'))"`
2. If yfinance is down:
   - Yahoo Finance API may be rate-limited or changed. This happens periodically.
   - Wait 30 minutes and retry
   - Check https://github.com/ranaroussi/yfinance/issues for known outages
3. **Alternative data source (manual):**
   - Download CSV from Yahoo Finance website or your broker
   - Or use TradingView for quick manual checks of your top strategies
4. If the data looks weird (e.g., unadjusted prices, missing recent days), inspect
   the yfinance cache and producer log before any cleanup; do not delete caches
   as a first-line fix.

**Prevention:** The scan should validate that SPY's last trading date matches yesterday (or today if run intraday). Add a sanity check if not already present.

---

## SCENARIO 5: Complete System Failure (Multiple Things Down)

**Symptoms:** Can't access repo, can't run scripts, everything is broken.

**Manual Trading Protocol:**
1. Open your broker platform (TWS/IBKR web portal)
2. Open the **"Active Positions"** tab in your Google Sheet (accessible from any browser)
3. For each active position:
   - Check if today is the scheduled exit date → if yes, place MOC sell order
   - Check if price has breached the stop level → if yes, place market sell order
4. For new entries: SKIP THEM. One day without new entries won't hurt the edge. Protecting existing positions is the priority.
5. After the crisis passes, run `reconcile_positions.py` to sync everything back up.

---

## KEY INFORMATION (Fill These In)

| Item | Value |
|------|-------|
| Local repo path | `~/___________________________` |
| Google Sheet URL | `https://docs.google.com/spreadsheets/d/___________________________` |
| Google credentials file | `~/___________________________` |
| Email account for alerts | `___________________________@gmail.com` |
| IBKR gateway port (paper) | `7497` |
| IBKR gateway port (live) | `7496` |
| Broker support phone | `___________________________` |
| Strategy count (current) | `___` active strategies |
| Max positions (current) | `___` concurrent positions |

---

## DAILY CHECKLIST (When Automated System is Running)

- [ ] **8:30 AM** — Run `reconcile_positions.py`. Confirm positions match.
- [ ] **3:00 PM** — Check for MOC signal email. Review and execute manually if needed.
- [ ] **4:05 PM** — Check for post-close signal email. Stage limit orders for tomorrow.
- [ ] **4:30 PM** — Verify staged orders appeared in Google Sheet.

## WEEKLY CHECKLIST

- [ ] Run `python trade_journal.py --report 7` — review slippage trends
- [ ] Verify the pinned runtime SHA/tag and review the weekday health-task result
- [ ] Spot-check one strategy's signals against TradingView
