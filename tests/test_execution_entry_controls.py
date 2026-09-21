"""Test actual executor gates without importing a live module or contacting IBKR."""
import ast
import copy
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from broker_runtime.prepare_entry_controls import patch as patch_source

SOURCE = Path(os.environ.get("EXEC_CONTROL_TEST_SOURCE", ""))


def node_named(tree, name):
    return next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)


def load_nodes(tree, names, namespace):
    selected = [node for node in tree.body if (
        isinstance(node, ast.FunctionDef) and node.name in names
    ) or (isinstance(node, ast.Assign) and any(
        isinstance(target, ast.Name) and target.id in names for target in node.targets))]
    exec(compile(ast.Module(body=selected, type_ignores=[]), "<isolated-executor-gates>", "exec"), namespace)


def evaluate_gate(node, namespace):
    fn = ast.parse("def gate():\n    pass\n").body[0]
    fn.body = [copy.deepcopy(node), ast.Return(value=ast.Constant(None))]
    module = ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[]))
    exec(compile(module, "<actual-cap-branch>", "exec"), namespace)
    return namespace["gate"]()


@unittest.skipUnless((SOURCE / "exec_agent.py.original").exists(), "Set EXEC_CONTROL_TEST_SOURCE to a prepared candidate")
class EntryControlsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original = {}
        cls.candidate = {}
        for name in ("exec_agent.py", "execute_order.py"):
            text = (SOURCE / (name + ".original")).read_text(encoding="utf-8-sig")
            cls.original[name] = ast.parse(text)
            candidate = patch_source(text, executor=name == "execute_order.py")
            cls.candidate[name] = ast.parse(candidate)
            if candidate != (SOURCE / name).read_text(encoding="utf-8"):
                raise AssertionError("Prepared candidate differs from tested patch")

    def namespace(self, tree):
        ns = {"os": os, "UNCAPPED_FUTURES_ACCOUNTS": {"primary"},
              "_max_notional": lambda account: 30000, "_acct_max_notional": lambda account: 30000,
              "_out": lambda ok, state, detail, **kw: {"ok": ok, "state": state, "detail": detail, **kw}}
        load_nodes(tree, {"_uncapped_futures", "_futures_notional_exempt"}, ns)
        return ns

    def notional_gate(self, tree, executor, account, sec_type, value):
        fn = node_named(tree, "_do_entry_bracket" if executor else "_validate")
        gate = next(n for n in ast.walk(fn) if isinstance(n, ast.If)
                    and ast.unparse(n.test).startswith("sec_type != 'CASH' and"))
        ns = self.namespace(tree)
        ns.update(acct=account, sec_type=sec_type, notional=value, reasons=[])
        result = evaluate_gate(gate, ns)
        return result if executor else ns["reasons"]

    def test_original_rejects_one_mes_at_7700_both_layers(self):
        with patch.dict(os.environ, {"LIVE_FUTURES_NOTIONAL_EXEMPT_ACCOUNTS": "pa"}):
            for name, tree in self.original.items():
                with self.subTest(layer=name):
                    result = self.notional_gate(tree, name == "execute_order.py", "pa", "FUT", 38500)
                    self.assertTrue(result)
                    self.assertIn("notional", str(result))

    def test_notional_exemption_is_explicit_and_account_instrument_scoped(self):
        for name, tree in self.candidate.items():
            for setting, account, sec_type, value, blocked in [
                ("", "pa", "FUT", 38500, True),
                ("pa", "pa", "FUT", 38500, False),
                (" PA ", "pa", "FUT", 38500, False),
                ("pa", "pa", "STK", 38500, True),
                ("pa", "other", "FUT", 38500, True),
                ("pa", "primary", "FUT", 38500, False),
                ("", "pa", "FUT", 30000, False),
                ("", "pa", "FUT", 30001, True),
            ]:
                with self.subTest(layer=name, setting=setting, account=account, sec_type=sec_type, value=value):
                    with patch.dict(os.environ, {"LIVE_FUTURES_NOTIONAL_EXEMPT_ACCOUNTS": setting}):
                        result = self.notional_gate(tree, name == "execute_order.py", account, sec_type, value)
                        self.assertEqual(bool(result), blocked)

    def test_pa_quantity_cap_remains(self):
        tree = self.candidate["execute_order.py"]
        fn = node_named(tree, "_do_entry_bracket")
        gate = next(n for n in fn.body if isinstance(n, ast.If)
                    and "qty > LIVE_MAX_FUT_CONTRACTS" in ast.unparse(n))
        ns = self.namespace(tree)
        ns.update(acct="pa", sec_type="FUT", qty=4, LIVE_MAX_FUT_CONTRACTS=3, LIVE_MAX_QTY=0)
        with patch.dict(os.environ, {"LIVE_FUTURES_NOTIONAL_EXEMPT_ACCOUNTS": "pa"}):
            result = evaluate_gate(gate, ns)
        self.assertIn("LIVE_MAX_FUT_CONTRACTS", result["detail"])

    def test_pa_stop_risk_cap_remains(self):
        tree = self.candidate["exec_agent.py"]
        fn = node_named(tree, "_validate")
        gate = next(n for n in ast.walk(fn) if isinstance(n, ast.If)
                    and "risk > MAX_RISK_PCT * nlv" in ast.unparse(n.test))
        ns = self.namespace(tree)
        ns.update(acct="pa", sec_type="FUT", stop=7600, risk=5001, nlv=100000,
                  MAX_RISK_PCT=0.05, reasons=[])
        with patch.dict(os.environ, {"LIVE_FUTURES_NOTIONAL_EXEMPT_ACCOUNTS": "pa"}):
            evaluate_gate(gate, ns)
        self.assertIn("exceeds 5% of NLV", ns["reasons"][0])

    def dispatch(self, tree, account, *, armed=True, enabled=True):
        events = []
        class FakeEvent:
            def __iadd__(self, callback): return self
        class FakeIB:
            errorEvent = FakeEvent()
            def connect(self, *args, **kwargs): events.append("connect")
            def disconnect(self): events.append("disconnect")
        ns = {"json": json, "sys": SimpleNamespace(argv=["executor", json.dumps({
                  "id": "test-command", "type": "add_to_position", "account": account,
                  "payload": {"con_id": 42, "qty": 500}})]),
              "LIVE_ENABLED": enabled, "LIVE_ACCOUNTS": {"primary", "pa"},
              "LIVE_TYPES": {"add_to_position"} if armed else set(),
              "IB": FakeIB, "PORTS": {"primary": ("fake", 0, 1), "pa": ("fake", 0, 2)},
              "_on_err": lambda *args: None,
              "_resolve_broker_account": lambda *args: "test-account",
              "_out": lambda ok, state, detail, **kw: {"ok": ok, "state": state, "detail": detail},
              "_do_add_to_position": lambda *args: events.append("add_handler") or {"state": "test_dispatch"}}
        load_nodes(tree, {"SUPPORTED", "DISABLED_UNSAFE_MUTATIONS", "main"}, ns)
        return ns["main"](), events

    def test_original_add_dispatch_reproduces_this_mornings_rejection(self):
        result, events = self.dispatch(self.original["execute_order.py"], "primary")
        self.assertIn("not supported live", result["detail"])
        self.assertEqual(events, [])

    def test_primary_add_dispatch_reaches_existing_handler(self):
        result, events = self.dispatch(self.candidate["execute_order.py"], "primary")
        self.assertEqual(result["state"], "test_dispatch")
        self.assertEqual(events, ["connect", "add_handler", "disconnect"])

    def test_pa_add_and_unarmed_add_stay_blocked(self):
        for account, armed, enabled in [("pa", True, True), ("primary", False, True), ("primary", True, False)]:
            with self.subTest(account=account, armed=armed, enabled=enabled):
                result, events = self.dispatch(self.candidate["execute_order.py"], account, armed=armed, enabled=enabled)
                self.assertEqual(result["state"], "rejected")
                self.assertEqual(events, [])

    def test_only_expected_ast_changes(self):
        for name, tree in self.candidate.items():
            restored = copy.deepcopy(tree)
            restored.body = [n for n in restored.body if not (isinstance(n, ast.FunctionDef) and n.name == "_futures_notional_exempt")]
            for node in ast.walk(restored):
                if isinstance(node, ast.Name) and node.id == "_futures_notional_exempt":
                    node.id = "_uncapped_futures"
            if name == "execute_order.py":
                supported = next(n for n in restored.body if isinstance(n, ast.Assign)
                                 and any(isinstance(t, ast.Name) and t.id == "SUPPORTED" for t in n.targets))
                supported.value.elts = [n for n in supported.value.elts if not (isinstance(n, ast.Constant) and n.value == "add_to_position")]
            self.assertEqual(ast.dump(restored), ast.dump(self.original[name]))


if __name__ == "__main__":
    unittest.main()
