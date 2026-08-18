"""Build data/macro_events.csv — the historical macro event calendar.

Event types produced:
- fomc_decision      scheduled FOMC meeting decision days (2000+), date = last
                     day of the meeting (statement day)
- fomc_intermeeting  famous unscheduled policy actions (hardcoded, announcement
                     dates: 2001 cuts, 2008 cuts, 2020 emergency actions)
- fomc_minutes       actual minutes release dates parsed from the Fed pages
- cpi / nfp / ppi    BLS release dates (2000+), 8:30 AM ET, actual dates
                     (shutdown-delayed releases overridden to their real dates,
                     canceled releases dropped)
- jackson_hole       one row per year, date = chair-keynote day (Friday of the
                     symposium; Thursday in 2020), detail carries the full range
- opex               monthly options expiration (3rd Friday, Good Friday -> Thu)
- quad_witching      Mar/Jun/Sep/Dec opex
- election           US general election days (even years)

Sources are cached raw pages in artifacts/macro_calendar_raw/ (see
scripts/fetch_macro_calendar_raw.py). Federal Reserve pages are fetched
directly; BLS pages come via the Wayback Machine (bls.gov blocks bots).

The per-year BLS schedule pages are EX-ANTE schedules. Two shutdowns moved
releases after those schedules were published (Oct 2013, Oct 2025 - Feb 2026);
both are corrected here from the official BLS revision pages. Rows for the
canceled October 2025 CPI/NFP/PPI releases are dropped entirely.
"""
from __future__ import annotations

import re
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
from dateutil.easter import easter

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "artifacts" / "macro_calendar_raw"
LEGACY_CACHE = ROOT / "scratch" / "macro_calendar_raw"
OUT = ROOT / "data" / "macro_events.csv"

MONTHS = {m: i + 1 for i, m in enumerate(
    ["January", "February", "March", "April", "May", "June", "July",
     "August", "September", "October", "November", "December"])}
MONTH_ABBR = {k[:3].lower(): v for k, v in MONTHS.items()}

BLS_RELEASES = {
    "Employment Situation": "nfp",
    "Consumer Price Index": "cpi",
    "Consumer Price Indexes": "cpi",
    "Producer Price Index": "ppi",
    "Producer Price Indexes": "ppi",
}


def _read(name: str) -> str:
    path = CACHE / name
    if not path.exists():
        # Preserve rebuild compatibility with caches created before the
        # workspace-hygiene migration. New downloads always use artifacts/.
        path = LEGACY_CACHE / name
    return path.read_text(encoding="utf-8", errors="ignore")


# ---------------------------------------------------------------------------
# FOMC
# ---------------------------------------------------------------------------

def _parse_meeting_heading(text: str, year: int) -> tuple[date, bool] | None:
    """'January 29-30 Meeting - 2008' -> (decision date, scheduled)."""
    t = text.strip()
    if "Conference Call" in t or "cancelled" in t or "notation vote" in t:
        return None
    unscheduled = "(unscheduled)" in t
    t = t.replace("(unscheduled)", "")
    # Formats: "January 29-30 Meeting", "Jan/Feb 31-1 Meeting",
    # "July 31-August 1  Meeting", "March 18 Meeting"
    m = re.match(
        r"(\w+)(?:/(\w+))?\s+(\d+)(?:\s*-\s*(?:(\w+)\s+)?(\d+))?\s+Meeting", t)
    if not m:
        return None
    mon1, mon_slash, d1, mon_dash, d2 = m.groups()
    end_mon = mon_dash or mon_slash or mon1
    month = MONTH_ABBR[end_mon[:3].lower()]
    day = int(d2) if d2 else int(d1)
    return date(year, month, day), not unscheduled


def parse_fed_historical(year: int) -> list[dict]:
    """fomchistorical{year}.htm -> fomc_decision + fomc_minutes rows."""
    html = _read(f"fomchistorical{year}.htm")
    rows: list[dict] = []
    heads = list(re.finditer(r"<h5[^>]*>([^<]+)</h5>", html))
    for i, h in enumerate(heads):
        parsed = _parse_meeting_heading(h.group(1), year)
        if parsed is None:
            continue
        decision, scheduled = parsed
        if not scheduled:
            continue  # unscheduled actions come from the hardcoded list
        seg = html[h.end(): heads[i + 1].start() if i + 1 < len(heads)
                   else len(html)]
        rows.append({"date": decision, "event": "fomc_decision",
                     "detail": h.group(1).strip(), "ref_period": "",
                     "time_et": "14:15" if year < 2013 else "14:00",
                     "source": f"fed:fomchistorical{year}"})
        rel = re.search(r"\(Released\s+([A-Z][a-z]+)\.?\s+(\d+),\s+(\d{4})\)",
                        seg)
        if rel:
            mon = MONTH_ABBR[rel.group(1)[:3].lower()]
            rows.append({"date": date(int(rel.group(3)), mon,
                                      int(rel.group(2))),
                         "event": "fomc_minutes",
                         "detail": f"minutes of {decision:%Y-%m-%d} meeting",
                         "ref_period": "", "time_et": "14:00",
                         "source": f"fed:fomchistorical{year}"})
    return rows


def parse_fed_calendars() -> list[dict]:
    """fomccalendars.htm (2021+) -> fomc_decision + fomc_minutes rows."""
    html = _read("fomccalendars.htm")
    rows: list[dict] = []
    panels = list(re.finditer(r"(20\d\d) FOMC Meetings", html))
    for i, p in enumerate(panels):
        year = int(p.group(1))
        seg = html[p.end(): panels[i + 1].start() if i + 1 < len(panels)
                   else len(html)]
        meetings = list(re.finditer(
            r'fomc-meeting__month[^>]*>\s*<strong>([^<]+)</strong>\s*</div>\s*'
            r'<div class="fomc-meeting__date[^>]*>([^<]+)</div>', seg))
        for j, mm in enumerate(meetings):
            mon_txt, day_txt = mm.group(1).strip(), mm.group(2).strip()
            if any(k in day_txt for k in
                   ("notation vote", "cancelled", "unscheduled")):
                continue
            mseg = seg[mm.end(): meetings[j + 1].start() if j + 1 < len(meetings)
                       else len(seg)]
            stmt = re.search(r"monetary(\d{8})a", mseg)
            if stmt:
                decision = pd.to_datetime(stmt.group(1)).date()
            else:  # future meeting — no statement yet, parse the label
                end_mon = mon_txt.split("/")[-1].strip()
                mon_num = MONTH_ABBR[end_mon[:3].lower()]
                days = re.findall(r"\d+", day_txt)
                if not days:
                    continue
                decision = date(year, mon_num, int(days[-1]))
            rows.append({"date": decision, "event": "fomc_decision",
                         "detail": f"{mon_txt} {day_txt} Meeting - {year}",
                         "ref_period": "", "time_et": "14:00",
                         "source": "fed:fomccalendars"})
            rel = re.search(
                r"\(Released\s+([A-Z][a-z]+)\.?\s+(\d+),\s+(\d{4})\)", mseg)
            if rel:
                mon = MONTH_ABBR[rel.group(1)[:3].lower()]
                rows.append({"date": date(int(rel.group(3)), mon,
                                          int(rel.group(2))),
                             "event": "fomc_minutes",
                             "detail": f"minutes of {decision:%Y-%m-%d} meeting",
                             "ref_period": "", "time_et": "14:00",
                             "source": "fed:fomccalendars"})
    return rows


# Announcement dates of the famous intermeeting policy actions. These are the
# statement dates (the tradeable event), which can differ from the meeting/call
# date by a day (e.g. the Jan 21 2008 call -> Jan 22 pre-open cut).
FOMC_INTERMEETING = [
    ("2001-01-03", "surprise 50bp cut"),
    ("2001-04-18", "surprise 50bp cut"),
    ("2001-09-17", "50bp cut, post-9/11 market reopen"),
    ("2007-08-17", "discount rate cut statement"),
    ("2008-01-22", "surprise 75bp cut (pre-open)"),
    ("2008-10-08", "coordinated global 50bp cut"),
    ("2020-03-03", "emergency 50bp cut"),
    ("2020-03-15", "emergency cut to zero + QE (Sunday)"),
    ("2020-03-23", "QE-unlimited announcement"),
]


# ---------------------------------------------------------------------------
# BLS (CPI / NFP / PPI)
# ---------------------------------------------------------------------------

def parse_bls_pre(year: int) -> list[dict]:
    """2000-2007 <pre>-style pages."""
    text = re.sub(r"<[^>]+>", "", _read(f"bls_sched_{year}.htm"))
    names = "|".join(sorted(BLS_RELEASES, key=len, reverse=True))
    pat = re.compile(
        rf"(?:The\s+)?({names})\s*,\s+(January|February|March|April|May|June|"
        rf"July|August|September|October|November|December)\s+(\d{{4}})\s+"
        rf"([A-Z][a-z]+)\.?\s+(\d{{1,2}})\s*\*{{0,4}}(?:,\s*(\d{{4}}))?\s+"
        rf"(\d{{1,2}}:\d{{2}})\s*(am|pm)")
    rows = []
    for m in pat.finditer(text):
        name, ref_mon, ref_year, rel_mon, rel_day, rel_year, t, ampm = m.groups()
        rel = date(int(rel_year) if rel_year else year,
                   MONTH_ABBR[rel_mon[:3].lower()], int(rel_day))
        hh, mm_ = t.split(":")
        hh = int(hh) % 12 + (12 if ampm == "pm" else 0)
        rows.append({"date": rel, "event": BLS_RELEASES[name],
                     "detail": f"{name} release",
                     "ref_period": f"{ref_mon} {ref_year}",
                     "time_et": f"{hh:02d}:{mm_}",
                     "source": f"bls:sched{year}"})
    return rows


def parse_bls_table(year: int) -> list[dict]:
    """2008+ release-list table pages."""
    html = _read(f"bls_sched_{year}.htm")
    pat = re.compile(
        r'<td class="date-cell"><p>\w+,\s+(\w+)\s+(\d+),\s+(\d{4})</p></td>\s*'
        r'<td class="time-cell"><p>(\d{2}):(\d{2})\s*([AP]M)</p></td>\s*'
        r'<td class="desc-cell"><p><strong>([^<]+)</strong>\s*for\s+([^<]+)</p>')
    rows = []
    for m in pat.finditer(html):
        mon, day, yr, hh, mm_, ampm, name, ref = m.groups()
        name = name.strip()
        if name not in BLS_RELEASES:
            continue
        hh = int(hh) % 12 + (12 if ampm == "PM" else 0)
        rows.append({"date": date(int(yr), MONTHS[mon], int(day)),
                     "event": BLS_RELEASES[name],
                     "detail": f"{name} release",
                     "ref_period": ref.strip(),
                     "time_et": f"{hh:02d}:{mm_}",
                     "source": f"bls:sched{year}"})
    return rows


def parse_bls_current(name: str, event: str) -> list[dict]:
    """Current-year schedule pages (ref month | release date | time)."""
    html = _read(f"bls_current_{name}.htm")
    pat = re.compile(
        r"<td>(\w+)\s+(\d{4})</td>\s*<td>([A-Z][a-z]+)\.?\s+(\d+),\s+(\d{4})"
        r"</td>\s*<td>(\d{2}):(\d{2})\s*([AP]M)</td>")
    rows = []
    for m in pat.finditer(html):
        ref_mon, ref_yr, rel_mon, rel_day, rel_yr, hh, mm_, ampm = m.groups()
        hh = int(hh) % 12 + (12 if ampm == "PM" else 0)
        rows.append({"date": date(int(rel_yr), MONTH_ABBR[rel_mon[:3].lower()],
                                  int(rel_day)),
                     "event": event, "detail": f"{event.upper()} release",
                     "ref_period": f"{ref_mon} {ref_yr}",
                     "time_et": f"{hh:02d}:{mm_}",
                     "source": f"bls:current:{name}"})
    return rows


# Shutdown corrections, keyed (event, ref_period) -> actual date or None
# (None = release canceled, row dropped).
# 2013: BLS "Updated Schedule of BLS News Releases" (post Oct 1-16 shutdown).
# 2025-26: BLS "Revised news release dates following the 2025 and 2026 lapses
# in appropriations" (bls.gov/bls/2025-lapse-revised-release-dates.htm).
BLS_OVERRIDES: dict[tuple[str, str], date | None] = {
    ("nfp", "September 2013"): date(2013, 10, 22),
    ("cpi", "September 2013"): date(2013, 10, 30),
    ("ppi", "September 2013"): date(2013, 10, 29),
    ("nfp", "October 2013"): date(2013, 11, 8),
    ("cpi", "October 2013"): date(2013, 11, 20),
    ("ppi", "October 2013"): date(2013, 11, 21),
    ("nfp", "September 2025"): date(2025, 11, 20),
    ("cpi", "September 2025"): date(2025, 10, 24),
    ("ppi", "September 2025"): date(2025, 11, 25),
    ("nfp", "October 2025"): None,
    ("cpi", "October 2025"): None,
    ("ppi", "October 2025"): None,
    ("nfp", "November 2025"): date(2025, 12, 16),
    ("cpi", "November 2025"): date(2025, 12, 18),
    ("ppi", "November 2025"): date(2026, 1, 14),
    ("ppi", "December 2025"): date(2026, 1, 30),
    ("nfp", "January 2026"): date(2026, 2, 11),
    ("cpi", "January 2026"): date(2026, 2, 13),
    ("ppi", "January 2026"): date(2026, 2, 27),
    ("ppi", "February 2026"): date(2026, 3, 18),
}


# ---------------------------------------------------------------------------
# Jackson Hole (KC Fed symposium; anchor = chair-keynote day)
# ---------------------------------------------------------------------------

# year -> (anchor date, full range label). Anchor is the Friday of the
# symposium (the chair keynote slot) except 2020 (virtual, Powell spoke
# Thursday Aug 27). 2013/2015 had no chair speech; anchor stays the Friday.
JACKSON_HOLE = {
    2000: ("2000-08-25", "Aug 24-26"),
    2001: ("2001-08-31", "Aug 30 - Sep 1"),
    2002: ("2002-08-30", "Aug 29-31"),
    2003: ("2003-08-29", "Aug 28-30"),
    2004: ("2004-08-27", "Aug 26-28"),
    2005: ("2005-08-26", "Aug 25-27"),
    2006: ("2006-08-25", "Aug 24-26"),
    2007: ("2007-08-31", "Aug 30 - Sep 1"),
    2008: ("2008-08-22", "Aug 21-23"),
    2009: ("2009-08-21", "Aug 20-22"),
    2010: ("2010-08-27", "Aug 26-28"),
    2011: ("2011-08-26", "Aug 25-27"),
    2012: ("2012-08-31", "Aug 30 - Sep 1"),
    2013: ("2013-08-23", "Aug 22-24 (no chair speech)"),
    2014: ("2014-08-22", "Aug 21-23"),
    2015: ("2015-08-28", "Aug 27-29 (no chair speech)"),
    2016: ("2016-08-26", "Aug 25-27"),
    2017: ("2017-08-25", "Aug 24-26"),
    2018: ("2018-08-24", "Aug 23-25"),
    2019: ("2019-08-23", "Aug 22-24"),
    2020: ("2020-08-27", "Aug 27-28 (virtual, keynote Thu)"),
    2021: ("2021-08-27", "Aug 27 (virtual)"),
    2022: ("2022-08-26", "Aug 25-27"),
    2023: ("2023-08-25", "Aug 24-26"),
    2024: ("2024-08-23", "Aug 22-24"),
    2025: ("2025-08-22", "Aug 21-23"),
    2026: ("2026-08-28", "Aug 27-29"),
}


# ---------------------------------------------------------------------------
# Computed events
# ---------------------------------------------------------------------------

def third_friday(year: int, month: int) -> date:
    d = date(year, month, 1)
    d += timedelta(days=(4 - d.weekday()) % 7)  # first Friday
    return d + timedelta(days=14)


def vix_expiry(year: int, month: int) -> date:
    """VIX futures/options expiration for the given month: 30 calendar days
    before the FOLLOWING month's SPX opex (normally a Wednesday; shifts
    with a Good-Friday-moved opex)."""
    ny, nm = (year + 1, 1) if month == 12 else (year, month + 1)
    exp = third_friday(ny, nm)
    if exp == easter(ny) - timedelta(days=2):  # Good Friday opex -> Thursday
        exp -= timedelta(days=1)
    return exp - timedelta(days=30)


def computed_rows(y0: int = 2000, y1: int = 2027) -> list[dict]:
    rows = []
    for y in range(y0, y1 + 1):
        gf = easter(y) - timedelta(days=2)  # Good Friday
        for mo in range(1, 13):
            d = third_friday(y, mo)
            if d == gf:
                d -= timedelta(days=1)
            quad = mo in (3, 6, 9, 12)
            rows.append({"date": d, "event": "opex",
                         "detail": "quad witching" if quad
                                   else "monthly options expiration",
                         "ref_period": "", "time_et": "16:00",
                         "source": "computed"})
            if quad:
                rows.append({"date": d, "event": "quad_witching",
                             "detail": "quad witching", "ref_period": "",
                             "time_et": "16:00", "source": "computed"})
            rows.append({"date": vix_expiry(y, mo), "event": "vix_expiry",
                         "detail": "VIX futures/options expiration (SOQ at "
                                   "the open)",
                         "ref_period": "", "time_et": "09:30",
                         "source": "computed"})
        if y % 2 == 0:  # first Tuesday after first Monday in November
            d = date(y, 11, 1)
            d += timedelta(days=(0 - d.weekday()) % 7)  # first Monday
            d += timedelta(days=1)
            kind = "presidential" if y % 4 == 0 else "midterm"
            rows.append({"date": d, "event": "election",
                         "detail": f"US general election ({kind})",
                         "ref_period": "", "time_et": "",
                         "source": "computed"})
    return rows


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

def build() -> pd.DataFrame:
    rows: list[dict] = []
    for y in range(2000, 2021):
        rows.extend(parse_fed_historical(y))
    rows.extend(parse_fed_calendars())
    for dt, note in FOMC_INTERMEETING:
        rows.append({"date": pd.to_datetime(dt).date(),
                     "event": "fomc_intermeeting", "detail": note,
                     "ref_period": "", "time_et": "", "source": "manual"})

    bls: list[dict] = []
    for y in range(2000, 2008):
        bls.extend(parse_bls_pre(y))
    for y in range(2008, 2026):
        bls.extend(parse_bls_table(y))
    for name, ev in (("cpi", "cpi"), ("empsit", "nfp"), ("ppi", "ppi")):
        bls.extend(parse_bls_current(name, ev))

    # Dedupe BLS by (event, ref_period): the current-year pages overlap the
    # per-year schedule pages; keep the current-page row (it reflects revised
    # dates), then apply the shutdown override table on top of everything.
    seen: dict[tuple[str, str], dict] = {}
    for r in bls:
        key = (r["event"], r["ref_period"])
        if key not in seen or r["source"].startswith("bls:current"):
            seen[key] = r
    for key, override in BLS_OVERRIDES.items():
        if key in seen:
            if override is None:
                seen.pop(key)
            else:
                seen[key]["date"] = override
                seen[key]["source"] += "+revised"
        elif override is not None:
            ev, ref = key
            seen[key] = {"date": override, "event": ev,
                         "detail": f"{ev.upper()} release (revised)",
                         "ref_period": ref, "time_et": "08:30",
                         "source": "bls:revised"}
    rows.extend(seen.values())

    for y, (anchor, rng) in JACKSON_HOLE.items():
        d = pd.to_datetime(anchor).date()
        assert d.weekday() == 4 or y == 2020, (y, anchor)
        rows.append({"date": d, "event": "jackson_hole",
                     "detail": f"Jackson Hole symposium {rng}, {y}",
                     "ref_period": str(y), "time_et": "10:00",
                     "source": "manual:kcfed"})

    rows.extend(computed_rows())

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df = (df.drop_duplicates(subset=["date", "event"])
            .sort_values(["date", "event"]).reset_index(drop=True))

    # A two-day meeting occasionally appears as two one-day h5 entries
    # (Sep 15 + Sep 16, 2003). Keep only the statement day (the later one).
    dec = set(df.loc[df["event"] == "fomc_decision", "date"])
    drop = df[(df["event"] == "fomc_decision")
              & (df["date"] + pd.Timedelta(days=1)).isin(dec)].index
    return df.drop(drop).reset_index(drop=True)


def validate(df: pd.DataFrame) -> list[str]:
    problems = []
    fomc = df[df["event"] == "fomc_decision"].set_index("date")
    for y in range(2000, 2027):
        n = len(fomc[fomc.index.year == y])
        expected = 7 if y == 2020 else 8  # March 2020 meeting was cancelled
        if n != expected:
            problems.append(f"fomc_decision {y}: {n} != {expected}")
    for ev in ("cpi", "nfp", "ppi"):
        sub = df[df["event"] == ev]
        for y in range(2000, 2026):
            n = (sub["date"].dt.year == y).sum()
            if not 10 <= n <= 13:
                problems.append(f"{ev} {y}: {n} releases")
    if df.duplicated(subset=["date", "event"]).any():
        problems.append("duplicate (date, event) rows")
    return problems


def main() -> int:
    df = build()
    problems = validate(df)
    for p in problems:
        print(f"WARN: {p}")
    df.to_csv(OUT, index=False, date_format="%Y-%m-%d")
    print(f"\nWrote {OUT} — {len(df)} rows, "
          f"{df['date'].min():%Y-%m-%d} .. {df['date'].max():%Y-%m-%d}")
    print(df["event"].value_counts().to_string())
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
