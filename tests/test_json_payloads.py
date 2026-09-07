import json

import pytest

from scripts.json_payloads import dumps_payload


def test_nested_unavailable_numbers_are_json_null():
    result = dumps_payload({'price': float('nan'), 'series': [1, float('inf'), -float('inf')], 'ok': True})
    assert json.loads(result) == {'price': None, 'series': [1, None, None], 'ok': True}
    assert 'NaN' not in result and 'Infinity' not in result


def test_unsupported_objects_are_not_silently_stringified():
    with pytest.raises(TypeError):
        dumps_payload({'price': object()})
