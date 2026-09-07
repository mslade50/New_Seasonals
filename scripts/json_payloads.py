"""Strict JSON for generated display payloads; absent numbers stay absent."""
import json
import math
from numbers import Real


def finite_json(value):
    if isinstance(value, dict):
        return {key: finite_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite_json(item) for item in value]
    if isinstance(value, Real) and not math.isfinite(value):
        return None
    return value


def dumps_payload(value):
    return json.dumps(finite_json(value), separators=(",", ":"), ensure_ascii=False, allow_nan=False)
