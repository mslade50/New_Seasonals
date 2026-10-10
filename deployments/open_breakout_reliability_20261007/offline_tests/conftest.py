"""Offline test harness: prevent owner handoff sockets and runtime registration.

Existing service/session tests already inject simulated broker transports and
alerts. This stub replaces their unrelated local OwnerServer startup only;
it does not qualify a production owner adapter or any close operation.
"""
import sys
from types import ModuleType

import pytest


@pytest.fixture(autouse=True)
def inert_owner_handoff(monkeypatch):
    class OwnerServer:
        def __init__(self, transport, operator_callback):
            self.transport = transport
            self.operator_callback = operator_callback

        async def start(self):
            return None

        async def close(self):
            return None

    package = ModuleType('broker_runtime')
    owner = ModuleType('broker_runtime.owner_connection')
    owner.OwnerServer = OwnerServer
    monkeypatch.setitem(sys.modules, 'broker_runtime', package)
    monkeypatch.setitem(sys.modules, 'broker_runtime.owner_connection', owner)
