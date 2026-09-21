from argparse import Namespace

import pytest

from scripts.databento_futures import (
    SUBMIT_CONFIRMATION,
    _expand_dbn_inputs,
    build_spec,
    format_bytes,
    get_quote,
    safe_job_summary,
    store_api_key,
    store_api_key_in_env_file,
    validate_api_key,
    validate_submit,
)


def test_build_spec_uses_volume_continuous_front_contracts():
    args = Namespace(
        dataset="GLBX.MDP3",
        roots=["es", "NQ", "rty"],
        schema="ohlcv-1m",
        start="2010-06-06",
        end="2026-09-01",
        roll_rule="v",
    )

    spec = build_spec(args)

    assert spec.symbols == ("ES.v.0", "NQ.v.0", "RTY.v.0")
    assert spec.stype_in == "continuous"


def test_quote_calls_only_metadata_endpoints():
    class Metadata:
        def __init__(self):
            self.calls = []

        def get_cost(self, **kwargs):
            self.calls.append(("cost", kwargs))
            return 12.345

        def get_billable_size(self, **kwargs):
            self.calls.append(("size", kwargs))
            return 123_000_000

    class Client:
        metadata = Metadata()

    args = Namespace(
        dataset="GLBX.MDP3",
        roots=["ES"],
        schema="ohlcv-1m",
        start="2020-01-01",
        end="2021-01-01",
        roll_rule="v",
    )
    spec = build_spec(args)

    assert get_quote(Client(), spec) == (12.345, 123_000_000)
    assert [name for name, _ in Client.metadata.calls] == ["cost", "size"]


def test_submit_guard_requires_confirmation_and_cost_ceiling():
    with pytest.raises(ValueError, match="Refusing to submit"):
        validate_submit(10.0, 20.0, "")
    with pytest.raises(ValueError, match="without --max-cost-usd"):
        validate_submit(10.0, None, SUBMIT_CONFIRMATION)
    with pytest.raises(ValueError, match="exceeds"):
        validate_submit(20.01, 20.0, SUBMIT_CONFIRMATION)

    validate_submit(20.0, 20.0, SUBMIT_CONFIRMATION)


def test_job_summary_never_includes_credentials_or_user_id():
    summary = safe_job_summary(
        {
            "id": "job-123",
            "state": "done",
            "api_key": "prod-001",
            "user_id": "private-user",
            "cost_usd": 1.25,
        }
    )

    assert summary == {"id": "job-123", "state": "done", "cost_usd": 1.25}


def test_format_bytes_uses_decimal_units():
    assert format_bytes(996_200_000) == "996.2 MB"


def test_api_key_validation_and_secure_storage():
    class Keyring:
        stored = None

        @classmethod
        def set_password(cls, service, username, value):
            cls.stored = (service, username, value)

    key = "db-" + "x" * 29
    assert validate_api_key(f" {key} ") == key
    store_api_key(key, Keyring)
    assert Keyring.stored[-1] == key

    with pytest.raises(ValueError, match="32 characters"):
        validate_api_key("db-too-short")


def test_env_file_storage_preserves_other_entries_and_replaces_key(tmp_path):
    env_path = tmp_path / ".env"
    env_path.write_text("OTHER_SECRET=leave-me\nDATABENTO_API_KEY=db-" + "a" * 29 + "\n")
    new_key = "db-" + "b" * 29

    store_api_key_in_env_file(new_key, env_path)

    assert env_path.read_text() == f"OTHER_SECRET=leave-me\nDATABENTO_API_KEY={new_key}\n"


def test_expand_dbn_inputs_accepts_directory(tmp_path):
    first = tmp_path / "a.dbn.zst"
    second = tmp_path / "b.dbn"
    ignored = tmp_path / "metadata.json"
    first.touch()
    second.touch()
    ignored.touch()

    assert _expand_dbn_inputs([tmp_path]) == [first, second]
