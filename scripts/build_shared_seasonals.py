"""Build the isolated, read-only teammate analytics site.

Only the approved Seasonality (Lab + Macro), Heatmap, and redacted Risk
shells plus their adjusted price snapshot, macro snapshot and book-free risk
payload are emitted.
The portfolio, execution, order, signal, and research payloads from the main
private site are intentionally outside this builder's allow-list.

Console output stays ASCII so a Windows cp1252 terminal never mangles it.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_risk_json import assert_shared_payload_clean
from scripts.macro_site_data import export_macro_snapshot
from scripts.seasonality_site_data import export_seasonality_snapshot


SHARED_SOURCE = ROOT / "shared_site"
ASSET_SOURCE = ROOT / "site" / "assets"
DEFAULT_PRICES = ROOT / "data" / "master_prices.parquet"
DEFAULT_RANKS = ROOT / "atr_seasonal_ranks.parquet"
DEFAULT_OUTPUT = ROOT / "dist-shared"
DEFAULT_RISK_PAYLOAD = ROOT / "data" / "site_risk_shared.json"

# A regime read older than a long weekend plus a holiday is worse than no
# regime read: the tab renders its "no payload" state instead.
RISK_PAYLOAD_MAX_AGE_DAYS = 5

STATIC_FILES = {
    SHARED_SOURCE / "index.html": Path("index.html"),
    SHARED_SOURCE / "heatmaps.html": Path("heatmaps.html"),
    SHARED_SOURCE / "correlations.html": Path("correlations.html"),
    SHARED_SOURCE / "risk.html": Path("risk.html"),
    SHARED_SOURCE / "_headers": Path("_headers"),
    ASSET_SOURCE / "common.js": Path("assets/common.js"),
    ASSET_SOURCE / "seasonality.js": Path("assets/seasonality.js"),
    ASSET_SOURCE / "macro_seasonal.js": Path("assets/macro_seasonal.js"),
    ASSET_SOURCE / "heatmaps.js": Path("assets/heatmaps.js"),
    ASSET_SOURCE / "risk.js": Path("assets/risk.js"),
    ASSET_SOURCE / "style.css": Path("assets/style.css"),
}
# Owned by the shared-site frontend and optional by design: the nav dropdown
# degrades to plain links when its helper is absent, so a missing file is a
# warning rather than a broken deploy.
OPTIONAL_STATIC_FILES = {
    ASSET_SOURCE / "shared_nav.js": Path("assets/shared_nav.js"),
}
ALLOWED_ROOT_FILES = {
    "index.html", "heatmaps.html", "correlations.html", "risk.html", "_headers",
}
ALLOWED_ASSETS = {
    "common.js", "seasonality.js", "macro_seasonal.js", "heatmaps.js",
    "risk.js", "shared_nav.js", "style.css",
}
SCANNED_PAGES = ("index.html", "risk.html")


def _copy_static(output: Path, cache_bust: str) -> None:
    for source, relative in STATIC_FILES.items():
        if not source.is_file():
            raise FileNotFoundError(f"required shared-site file is missing: {source}")
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)

    for source, relative in OPTIONAL_STATIC_FILES.items():
        if not source.is_file():
            print(f"shared site: optional file absent, skipping -> {source}")
            continue
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)

    for html_path in output.glob("*.html"):
        html = html_path.read_text(encoding="utf-8")
        html = re.sub(
            r'(assets/[\w.-]+\.(?:js|css))(?:\?v=\d+)?',
            rf"\1?v={cache_bust}",
            html,
        )
        html_path.write_text(html, encoding="utf-8")


def _payload_age_days(asof: object) -> int | None:
    try:
        stamped = dt.date.fromisoformat(str(asof)[:10])
    except (TypeError, ValueError):
        return None
    return (dt.datetime.now(dt.timezone.utc).date() - stamped).days


def copy_risk_payload(
    output: Path,
    source: Path = DEFAULT_RISK_PAYLOAD,
    max_age_days: int = RISK_PAYLOAD_MAX_AGE_DAYS,
) -> bool:
    """Ship the redacted regime read, or ship nothing.

    Two independent gates.  The clean assertion is FAIL CLOSED — a payload
    that still names strategies or keeps sizing policy raises and takes the
    whole shared build down with it.  Staleness only declines to copy: the
    tab then renders its "no payload" state, which beats a month-old dial
    presented as today's.
    """
    if not source.is_file():
        print(f"shared site: no redacted risk payload at {source} - risk tab ships empty")
        return False

    payload = json.loads(source.read_text(encoding="utf-8"))
    assert_shared_payload_clean(payload)

    age = _payload_age_days(payload.get("asof"))
    if age is None:
        print(f"shared site: risk payload has no usable asof ({payload.get('asof')!r}) - not shipped")
        return False
    if age > max_age_days:
        print(
            f"shared site: WARNING risk payload asof {payload['asof']} is {age}d old "
            f"(limit {max_age_days}d) - not shipped, risk tab renders its no-payload state"
        )
        return False

    destination = output / "data" / "risk.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    print(f"shared site: risk payload asof {payload['asof']} ({age}d old) -> data/risk.json")
    return True


def export_macro(output: Path, prices: Path, ranks: Path = DEFAULT_RANKS) -> bool:
    """Macro sub-tab payload beside the Seasonality Lab's per-ticker binaries.

    Best effort: the ATR seasonal-rank parquet is a 50 MB canonical R2 object
    and the shared workflow pulls it separately.  Without it the Lab still
    ships and the Macro sub-tab shows its no-data state, which beats failing
    the whole shared deploy over a secondary table.
    """
    destination = output / "data" / "seasonality" / "macro.json"
    try:
        payload = export_macro_snapshot(prices, ranks, destination)
    except Exception as exc:
        print(f"shared site: WARNING macro snapshot failed ({exc}) - Macro sub-tab ships empty")
        return False
    print(
        f"shared site: macro payload {len(payload['rows'])} tickers, "
        f"asof {payload['asof']}, seasonal ranks "
        f"{'present' if payload['sznl_available'] else 'ABSENT'} -> data/seasonality/macro.json"
    )
    return True


def validate_shared_output(output: Path) -> None:
    """Fail closed if anything outside the explicit share boundary is present."""
    if not (output / "index.html").is_file():
        raise ValueError("shared site is missing index.html")
    for page in ("heatmaps.html", "correlations.html", "risk.html"):
        if not (output / page).is_file():
            raise ValueError(f"shared site is missing {page}")
    if not (output / "data/seasonality/manifest.json").is_file():
        raise ValueError("shared site is missing the seasonality manifest")

    for path in output.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(output)
        parts = relative.parts
        allowed = (
            (len(parts) == 1 and parts[0] in ALLOWED_ROOT_FILES)
            or (len(parts) == 2 and parts[0] == "assets" and parts[1] in ALLOWED_ASSETS)
            or (len(parts) >= 3 and parts[0] == "data" and parts[1] == "seasonality")
            or (relative.as_posix() == "data/risk.json")
        )
        if not allowed:
            raise ValueError(f"non-shareable file escaped into shared output: {relative}")

    # Re-assert on the bytes actually sitting in the deploy directory, not on
    # whatever was checked upstream: this is the last gate before wrangler.
    risk_json = output / "data" / "risk.json"
    if risk_json.is_file():
        assert_shared_payload_clean(json.loads(risk_json.read_text(encoding="utf-8")))

    forbidden = ("execution.html", "orders.html", "signals.html", "portfolio", "/exec-book")
    for page in SCANNED_PAGES:
        page_path = output / page
        if not page_path.is_file():
            continue
        html = page_path.read_text(encoding="utf-8").lower()
        matches = [value for value in forbidden if value in html]
        if matches:
            raise ValueError(
                f"{page} references private-site surfaces: {', '.join(matches)}"
            )


def build_shared_site(
    prices: Path,
    output: Path,
    min_year: int = 2000,
    risk_payload: Path = DEFAULT_RISK_PAYLOAD,
    ranks: Path = DEFAULT_RANKS,
) -> dict:
    prices = prices.resolve()
    output = output.resolve()
    if not prices.is_file():
        raise FileNotFoundError(f"master price snapshot not found: {prices}")
    if output.exists():
        raise FileExistsError(
            f"refusing to replace existing output: {output}. Use a fresh path or remove it explicitly."
        )

    output.mkdir(parents=True)
    cache_bust = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d%H%M")
    _copy_static(output, cache_bust)
    manifest = export_seasonality_snapshot(
        prices, output / "data" / "seasonality", min_year=min_year
    )
    manifest["macro_payload"] = export_macro(output, prices, ranks)
    manifest["risk_payload"] = copy_risk_payload(output, risk_payload)
    validate_shared_output(output)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prices", type=Path, default=DEFAULT_PRICES)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--min-year", type=int, default=2000)
    parser.add_argument("--risk-payload", type=Path, default=DEFAULT_RISK_PAYLOAD)
    parser.add_argument("--ranks", type=Path, default=DEFAULT_RANKS)
    args = parser.parse_args()

    manifest = build_shared_site(
        args.prices,
        args.out,
        min_year=args.min_year,
        risk_payload=args.risk_payload,
        ranks=args.ranks,
    )
    print(
        f"Shared seasonality site: {manifest['ticker_count']:,} tickers, "
        f"{manifest['row_count']:,} rows through {manifest['asof']} -> {args.out} "
        f"(macro: {'shipped' if manifest['macro_payload'] else 'absent'}, "
        f"risk payload: {'shipped' if manifest['risk_payload'] else 'absent'})"
    )


if __name__ == "__main__":
    main()
