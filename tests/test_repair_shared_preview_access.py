import copy
import pytest

from scripts.repair_shared_preview_access import LEGACY_NAME, repair
from tests.test_configure_shared_access import FakeClient


class RepairClient(FakeClient):
    account_id = "account"

    def __init__(self):
        super().__init__()
        self.calls = []
        self.policies["preview-app"] = [
            *self.policies["preview-app"],
            {"id": "legacy", "name": LEGACY_NAME, "decision": "allow",
             "include": [{"cloudflare_account_member": {"account_id": "account"}}]},
        ]

    def list_policies(self, app_id):
        return copy.deepcopy(super().list_policies(app_id))

    def request(self, method, path):
        assert (method, path) == ("DELETE", "/accounts/account/access/apps/preview-app/policies/legacy")
        self.calls.append((method, path))
        self.policies["preview-app"] = [p for p in self.policies["preview-app"] if p["id"] != "legacy"]


def test_default_is_read_only():
    client = RepairClient()
    result = repair(client, backup=lambda _: pytest.fail("No backup needed for a plan"))
    assert result["status"] == "planned"
    assert client.calls == []


def test_backup_precedes_exact_preview_only_deletion():
    client = RepairClient()
    production = copy.deepcopy(client.policies["target-app"])
    backups = []

    def backup(payload):
        assert not client.calls
        backups.append(payload)
        return "private/backup.json"

    assert repair(client, apply=True, backup=backup)["status"] == "repaired"
    assert backups[0]["policy"]["id"] == "legacy"
    assert client.policies["target-app"] == production
    assert client.policies["preview-app"] == production
    assert repair(client, apply=True, backup=backup)["status"] == "already_correct"
    assert len(client.calls) == len(backups) == 1


def test_missing_team_policy_refuses_before_backup():
    client = RepairClient()
    client.policies["preview-app"] = [client.policies["preview-app"][-1]]
    with pytest.raises(RuntimeError, match="Denali Team"):
        repair(client, apply=True, backup=lambda _: pytest.fail("Unsafe repair"))
    assert not client.calls


def test_failed_backup_prevents_deletion():
    client = RepairClient()

    def backup(_):
        raise OSError("Storage unavailable")

    with pytest.raises(OSError):
        repair(client, apply=True, backup=backup)
    assert not client.calls


def test_concurrent_policy_change_prevents_deletion():
    client = RepairClient()

    def backup(_):
        client.policies["preview-app"][-1]["name"] = "Changed by owner"
        return "private/backup.json"

    with pytest.raises(RuntimeError, match="changed during repair"):
        repair(client, apply=True, backup=backup)
    assert not client.calls
