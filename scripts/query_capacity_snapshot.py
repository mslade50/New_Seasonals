"""Bound one existing read-only Primary query for conservative sizing proof."""
import datetime as dt
import json
import sys
from query_inventory_snapshot import query


def collect(snapshot_path):
    started = dt.datetime.now(dt.timezone.utc).isoformat()
    book = query(snapshot_path)
    book['capacity_query_started_at'] = started
    book['capacity_query_completed_at'] = dt.datetime.now(dt.timezone.utc).isoformat()
    return book


if __name__ == '__main__':
    print(json.dumps(collect(sys.argv[1])))
