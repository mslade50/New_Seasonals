"""Contract metadata repair without changing the held instrument."""
import copy
from datetime import datetime, timedelta


def qualify_held(ib, contract):
    wanted = int(getattr(contract, "conId", 0) or 0)
    if wanted <= 0:
        raise ValueError("held contract needs an exact conId")
    candidate = copy.deepcopy(contract)
    if candidate.secType == "STK":
        candidate.exchange = "SMART"
    matches = ib.qualifyContracts(candidate)
    if len(matches) != 1 or int(matches[0].conId or 0) != wanted:
        raise ValueError("held contract qualification changed the exact instrument")
    return matches[0]


def qualify_position(ib, position):
    contract = qualify_held(ib, position.contract)
    if hasattr(position, "_replace"):
        return position._replace(contract=contract)
    candidate = copy.copy(position)
    candidate.contract = contract
    return candidate


def select_front_details(details, today, buffer_days):
    """Select by last trading day, but return the delivery contract month.

    Energy contracts can stop trading in the month before their named month.
    Truncating lastTradeDate would therefore select the wrong MCL contract.
    """
    start = datetime.strptime(today, "%Y%m%d")
    threshold = (start + timedelta(days=buffer_days)).strftime("%Y%m%d")
    rows = []
    for detail in details:
        last = str(getattr(detail, "realExpirationDate", "") or
                   detail.contract.lastTradeDateOrContractMonth or "")[:8]
        month = str(getattr(detail, "contractMonth", "") or "")
        if len(last) != 8 or len(month) != 6:
            continue
        try:
            datetime.strptime(last, "%Y%m%d")
            datetime.strptime(month, "%Y%m")
        except ValueError:
            continue
        if last >= today and int(detail.contract.conId or 0) > 0:
            rows.append((last, month, detail))
    rows.sort(key=lambda row: (row[0], row[1]))
    usable = [row for row in rows if row[0] >= threshold] or rows
    if not usable:
        raise ValueError("no unexpired contract with verified delivery month")
    last, month, detail = usable[0]
    upcoming = list(dict.fromkeys(row[1] for row in usable))[:6]
    return detail, month, upcoming, last
