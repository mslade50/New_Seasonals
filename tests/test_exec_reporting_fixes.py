"""Guards for the execution-reporting fixes (brief items 1-5), 2026-09-21.

Follows the candidate-directory pattern of
`tests/test_execution_entry_controls.py` but with its OWN environment variable:
that module gates on `EXEC_CONTROL_TEST_SOURCE` and only checks that the path
holds an `exec_agent.py.original`, which THIS candidate also has, so sharing the
name made the entry-controls guards run against the wrong candidate and error.
The tests read the CANDIDATE built by
`broker_runtime/prepare_exec_reporting_fixes.py` and never import, patch or
touch the live trading directory. `position_actions.py` is imported by path
against a fake broker; `execute_order.py` is too large and side-effecting to
import, so its gates are exec'd node by node out of the AST, exactly as the
entry-controls tests do.

Set `EXEC_REPORTING_TEST_SOURCE` to the candidate directory, e.g.

    python broker_runtime/prepare_exec_reporting_fixes.py \\
        --source "<OneDrive>/trading_ibkr" \\
        --output artifacts/exec_reporting_candidate_20260921
    EXEC_REPORTING_TEST_SOURCE=artifacts/exec_reporting_candidate_20260921 \\
        python -m pytest tests/test_exec_reporting_fixes.py -q

Without it the whole module skips, which is what CI does.
"""
from __future__ import annotations

import ast
import copy
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from broker_runtime.prepare_exec_reporting_fixes import (  # noqa: E402
    patch_exec_agent, patch_execute_order, patch_olv_test, patch_position_actions,
)

SOURCE = Path(os.environ.get("EXEC_REPORTING_TEST_SOURCE", ""))
HAVE_CANDIDATE = bool(str(SOURCE)) and (SOURCE / "manifest.json").exists()
PATCHERS = {"test_olv_exits.py": patch_olv_test,
            "execute_order.py": patch_execute_order,
            "position_actions.py": patch_position_actions,
            "exec_agent.py": patch_exec_agent}


def load_nodes(tree, names, namespace):
    """Exec only the named top-level defs/assignments into `namespace`."""
    selected = [node for node in tree.body if (
        isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names
    ) or (isinstance(node, ast.Assign) and any(
        isinstance(target, ast.Name) and target.id in names for target in node.targets))]
    exec(compile(ast.Module(body=selected, type_ignores=[]), "<isolated>", "exec"), namespace)
    return namespace


def evaluate(node, namespace):
    """Run one statement in isolation and return what it returns."""
    fn = ast.parse("def gate():\n    pass\n").body[0]
    fn.body = [copy.deepcopy(node), ast.Return(value=ast.Constant(None))]
    module = ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[]))
    exec(compile(module, "<isolated-branch>", "exec"), namespace)
    return namespace["gate"]()


# --------------------------------------------------------------------------
# Fakes. Nothing here connects; every method is a recorded no-op.
# --------------------------------------------------------------------------
class FakeOrderStatus(SimpleNamespace):
    pass


class FakeTrade:
    def __init__(self, *, account, con_id, symbol="TEST", client_id=98, order_id=1,
                 perm_id=1000, action="SELL", order_type="LMT", qty=100, filled=0,
                 status="PreSubmitted", order_ref=""):
        self.contract = SimpleNamespace(conId=con_id, symbol=symbol)
        self.order = SimpleNamespace(account=account, clientId=client_id, orderId=order_id,
                                     permId=perm_id, action=action, orderType=order_type,
                                     totalQuantity=qty, orderRef=order_ref)
        self.orderStatus = FakeOrderStatus(status=status, filled=filled)


class FakeBroker:
    """A broker that can only be READ. Any mutation attempt fails the test."""

    def __init__(self, positions=(), trades=()):
        self._positions = list(positions)
        self._trades = list(trades)
        self.calls = []

    # reads
    def reqPositions(self):
        self.calls.append("reqPositions")

    def positions(self):
        return list(self._positions)

    def reqAllOpenOrders(self):
        self.calls.append("reqAllOpenOrders")

    def openTrades(self):
        return list(self._trades)

    def sleep(self, _seconds=0):
        return None

    # mutations -- must never happen on the resolve path
    def placeOrder(self, *_a, **_k):  # pragma: no cover - failure path
        raise AssertionError("resolve must never place an order")

    def cancelOrder(self, *_a, **_k):  # pragma: no cover - failure path
        raise AssertionError("resolve must never cancel an order")


def fake_position(account, con_id, symbol, qty):
    return SimpleNamespace(account=account, contract=SimpleNamespace(conId=con_id, symbol=symbol),
                           position=qty, avgCost=10.0)


def collecting_out():
    captured = {}

    def _out(ok=None, state=None, detail=None, fill=None, **kwargs):
        captured.update(dict(ok=ok, state=state, detail=detail, fill=fill, **kwargs))
        return captured
    return _out, captured


@unittest.skipUnless(HAVE_CANDIDATE, "Set EXEC_REPORTING_TEST_SOURCE to a prepared candidate")
class CandidateIntegrityTests(unittest.TestCase):
    """The tested patch and the shipped candidate must be the same bytes."""

    def test_candidate_matches_the_tested_patch_and_its_manifest(self):
        manifest = json.loads((SOURCE / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["status"], "prepared_only")
        self.assertEqual(sorted(manifest["files"]), sorted(PATCHERS))
        for name, patch in PATCHERS.items():
            with self.subTest(file=name):
                original = (SOURCE / (name + ".original")).read_text(encoding="utf-8-sig")
                rebuilt = patch(original.replace("\r\n", "\n"))
                self.assertEqual(rebuilt, (SOURCE / name).read_text(encoding="utf-8"))

    def test_the_preparer_refuses_a_source_it_was_not_pinned_against(self):
        for name, patch in PATCHERS.items():
            with self.subTest(file=name):
                already = (SOURCE / name).read_text(encoding="utf-8")
                with self.assertRaises(ValueError):
                    patch(already)   # the anchors are consumed; re-applying must refuse


@unittest.skipUnless(HAVE_CANDIDATE, "Set EXEC_REPORTING_TEST_SOURCE to a prepared candidate")
class Item1TestSuiteImportTests(unittest.TestCase):
    """The one broken import that aborted collection of all 18 OneDrive files."""

    @classmethod
    def setUpClass(cls):
        cls.text = (SOURCE / "test_olv_exits.py").read_text(encoding="utf-8")
        cls.original = (SOURCE / "test_olv_exits.py.original").read_text(encoding="utf-8-sig")

    def test_original_bound_the_runner_to_the_dispatcher_shim(self):
        self.assertIn("import olv_exit_moo as runner", self.original)

    def test_candidate_binds_the_runner_to_the_module_that_owns_the_code(self):
        self.assertIn("import olv_exit_pa_legacy as runner", self.text)
        self.assertNotIn("import olv_exit_moo as runner", self.text)
        self.assertIn("import olv_exit_moo as dispatcher", self.text)

    def test_root_names_the_runtime_tree_not_the_test_file_location(self):
        self.assertIn("ROOT = Path(runner.__file__).resolve().parent", self.text)
        self.assertNotIn("ROOT = Path(__file__).resolve().parent", self.text)
        # The OCA assertion must read the runner, not the placement-free shim.
        self.assertNotIn("(ROOT / 'olv_exit_moo.py').read_text", self.text)

    def test_the_dispatcher_itself_keeps_a_guard(self):
        self.assertIn("def test_scheduled_entry_point_dispatches_to_both_runners", self.text)

    def test_the_module_parses_and_collection_no_longer_aborts_on_it(self):
        tree = ast.parse(self.text)
        names = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
        self.assertIn("test_scheduled_entry_point_dispatches_to_both_runners", names)
        # The line that aborted collection was module-level, so it must still be
        # module-level -- and must now reference an attribute that exists.
        self.assertIn("_REAL_LOAD_PLACED = runner.load_placed", self.text)


@unittest.skipUnless(HAVE_CANDIDATE, "Set EXEC_REPORTING_TEST_SOURCE to a prepared candidate")
class Item3NoticeClassificationTests(unittest.TestCase):
    """399 / 2109 / the 2100-2199 warning block are notices, not failures."""

    def namespace(self):
        tree = ast.parse((SOURCE / "execute_order.py").read_text(encoding="utf-8"))
        ns = {"json": json, "print": lambda *_a, **_k: None}
        return load_nodes(tree, {"_ERRORS", "_NOTICES", "_NOTICE_CODES", "_NOTICE_RANGE",
                                 "_BENIGN", "_is_notice", "_on_err", "_out",
                                 "_placed_ids", "_placement_problem", "TERMINAL"}, ns)

    def test_notice_range_boundaries(self):
        ns = self.namespace()
        for code, expected in [(399, True), (398, False), (400, False),
                               (2099, False), (2100, True), (2109, True),
                               (2150, True), (2199, True), (2200, False),
                               (201, False), (202, False), (10052, False),
                               ("2109", True), (None, False), ("nope", False)]:
            with self.subTest(code=code):
                self.assertEqual(ns["_is_notice"](code), expected)

    def test_a_notice_never_lands_in_the_error_list(self):
        ns = self.namespace()
        ns["_on_err"](1, 399, "Order Message: SELL 80 RTX NYSE Warning: repriced")
        ns["_on_err"](1, 2109, "Order Event Warning: Outside RTH ignored")
        self.assertEqual(ns["_ERRORS"], [])
        self.assertEqual(len(ns["_NOTICES"]), 2)

    def test_a_real_reject_still_lands_in_the_error_list(self):
        ns = self.namespace()
        ns["_on_err"](1, 201, "Order rejected - reason: The contract is not available for short sale")
        self.assertEqual(ns["_NOTICES"], [])
        self.assertEqual(len(ns["_ERRORS"]), 1)

    def test_a_benign_code_is_recorded_nowhere(self):
        ns = self.namespace()
        ns["_on_err"](1, 2104, "Market data farm connection is OK")
        self.assertEqual(ns["_ERRORS"], [])
        self.assertEqual(ns["_NOTICES"], [])

    def test_placement_with_only_notices_is_not_a_problem(self):
        ns = self.namespace()
        ns["_on_err"](1, 399, "Order Message: repriced")
        trades = [SimpleNamespace(orderStatus=SimpleNamespace(status="PreSubmitted"))]
        self.assertIsNone(ns["_placement_problem"](trades))

    def test_placement_with_a_real_error_is_still_a_problem(self):
        ns = self.namespace()
        ns["_on_err"](1, 201, "Order rejected")
        trades = [SimpleNamespace(orderStatus=SimpleNamespace(status="PreSubmitted"))]
        self.assertIn("201", ns["_placement_problem"](trades))

    def test_a_notice_is_surfaced_on_success_and_never_on_a_failure(self):
        ns = self.namespace()
        ns["_on_err"](1, 399, "Order Message: repriced")
        emitted = []
        ns["print"] = lambda payload: emitted.append(json.loads(payload))
        ns["_out"](True, "executed", "Bracket submitted")
        ns["_out"](False, "rejected", "live gate: something")
        self.assertIn("IBKR notice", emitted[0]["detail"])
        self.assertIn("399", emitted[0]["detail"])
        self.assertNotIn("IBKR notice", emitted[1]["detail"])

    def test_out_carries_the_structured_keys_and_drops_the_absent_ones(self):
        """`_out` is the only writer of the child's result JSON, so `lock` and
        `snapshot` have to pass through it -- without changing the four keys
        every existing caller emits."""
        ns = self.namespace()
        emitted = []
        ns["print"] = lambda payload: emitted.append(json.loads(payload))
        ns["_out"](False, "rejected", "Nothing sent: blocked", lock={"symbol": "JHX"})
        ns["_out"](True, "executed", "cleared", snapshot={"positions": [], "open_orders": []})
        ns["_out"](False, "rejected", "plain", lock=None, snapshot=None)
        self.assertEqual(emitted[0]["lock"], {"symbol": "JHX"})
        self.assertNotIn("snapshot", emitted[0])
        self.assertEqual(emitted[1]["snapshot"], {"positions": [], "open_orders": []})
        self.assertEqual(sorted(emitted[2]), ["detail", "fill", "ok", "state"])

    def test_the_original_would_have_failed_the_repriced_entry(self):
        tree = ast.parse((SOURCE / "execute_order.py.original").read_text(encoding="utf-8-sig"))
        ns = load_nodes(tree, {"_ERRORS", "_BENIGN", "_on_err", "_placement_problem", "TERMINAL"},
                        {"json": json})
        ns["_on_err"](1, 399, "Order Message: repriced")
        trades = [SimpleNamespace(orderStatus=SimpleNamespace(status="PreSubmitted"))]
        self.assertIn("399", ns["_placement_problem"](trades))


@unittest.skipUnless(HAVE_CANDIDATE, "Set EXEC_REPORTING_TEST_SOURCE to a prepared candidate")
class Item2CommandBoundaryTests(unittest.TestCase):
    """Pre-transmit refusal -> rejected. Post-transmit exception -> unknown."""

    def boundary(self, exception, *, pre_transmit):
        tree = ast.parse((SOURCE / "execute_order.py").read_text(encoding="utf-8"))
        main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
        handler = next(h for n in ast.walk(main) if isinstance(n, ast.Try) for h in n.handlers
                       if any(isinstance(x, ast.Call) and getattr(x.func, "id", "") == "_pre_transmit_refusal"
                              for x in ast.walk(h)))
        captured = {}

        def _out(ok, state, detail, fill=None, **extra):
            captured.update(ok=ok, state=state, detail=detail, **extra)
            return captured
        ns = {"_out": _out, "e": exception, "_pre_transmit_refusal": lambda _exc: pre_transmit,
              "_refusal_lock": lambda exc: getattr(exc, "lock", None)}
        body = ast.parse("def run():\n    pass\n").body[0]
        body.body = copy.deepcopy(handler.body)
        exec(compile(ast.fix_missing_locations(ast.Module(body=[body], type_ignores=[])),
                     "<boundary>", "exec"), ns)
        ns["run"]()
        return captured

    def test_pre_transmit_refusal_reports_rejected_and_says_nothing_was_sent(self):
        out = self.boundary(ValueError("An earlier position action is unresolved"), pre_transmit=True)
        self.assertEqual(out["state"], "rejected")
        self.assertFalse(out["ok"])
        self.assertIn("Nothing sent", out["detail"])
        self.assertNotIn("DO NOT RETRY", out["detail"])

    def test_the_refusal_carries_its_structured_lock_to_the_result(self):
        exc = ValueError("blocked")
        exc.lock = {"symbol": "JHX", "action_type": "close_resize", "action_id": "abc",
                    "created_at": "2026-09-16T10:21:29-04:00", "discrepancy": "exits cover 0 of 625"}
        out = self.boundary(exc, pre_transmit=True)
        self.assertEqual(out["lock"]["action_type"], "close_resize")
        self.assertEqual(out["lock"]["symbol"], "JHX")

    def test_the_refusal_lock_helper_drops_a_half_identity_the_site_cannot_use(self):
        ns = load_nodes(ast.parse((SOURCE / "execute_order.py").read_text(encoding="utf-8")),
                        {"_refusal_lock"}, {"isinstance": isinstance, "all": all, "str": str})
        full = {"symbol": "JHX", "action_type": "close_resize", "action_id": "abc"}
        self.assertEqual(ns["_refusal_lock"](SimpleNamespace(lock=full)), full)
        for partial in ({"symbol": "JHX"}, dict(full, symbol=""), dict(full, action_id=""),
                        "not a dict", None):
            with self.subTest(lock=partial):
                self.assertIsNone(ns["_refusal_lock"](SimpleNamespace(lock=partial)))
        self.assertIsNone(ns["_refusal_lock"](ValueError("no lock attribute")))

    def test_post_transmit_exception_still_reports_unknown(self):
        out = self.boundary(TimeoutError("broker went away mid-placement"), pre_transmit=False)
        self.assertEqual(out["state"], "unknown")
        self.assertFalse(out["ok"])
        self.assertIn("VERIFY IN TWS", out["detail"])

    def test_the_boundary_helper_only_matches_the_refusal_class(self):
        tree = ast.parse((SOURCE / "execute_order.py").read_text(encoding="utf-8"))
        ns = load_nodes(tree, {"_pre_transmit_refusal"}, {})
        module = load_position_actions()
        sys.modules["position_actions"] = module
        try:
            self.assertTrue(ns["_pre_transmit_refusal"](module.PreTransmitRefusal("x")))
            self.assertFalse(ns["_pre_transmit_refusal"](ValueError("x")))
            self.assertFalse(ns["_pre_transmit_refusal"](RuntimeError("x")))
        finally:
            sys.modules.pop("position_actions", None)

    def test_the_original_boundary_had_no_rejected_branch(self):
        original = (SOURCE / "execute_order.py.original").read_text(encoding="utf-8-sig")
        self.assertNotIn("_pre_transmit_refusal", original)


def load_position_actions():
    """Import the CANDIDATE position_actions by path; never the live copy."""
    broker_runtime = str(ROOT / "broker_runtime")
    if broker_runtime not in sys.path:
        sys.path.insert(0, broker_runtime)
    spec = importlib.util.spec_from_file_location(
        "candidate_position_actions", SOURCE / "position_actions.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@unittest.skipUnless(HAVE_CANDIDATE, "Set EXEC_REPORTING_TEST_SOURCE to a prepared candidate")
class PositionActionsCandidateTestCase(unittest.TestCase):
    ACCOUNT = "U8103294"
    CON_ID = 794909199

    @classmethod
    def setUpClass(cls):
        cls.pa = load_position_actions()

    def setUp(self):
        import tempfile
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name) / "position_actions"
        self.root.mkdir(parents=True)
        (self.root / "order_edits").mkdir()
        self.addCleanup(self._tmp.cleanup)

    def ns(self):
        out, captured = collecting_out()
        self._captured = captured
        return {"POSITION_ACTION_STATE_DIR": str(self.root), "_out": out}

    def write_record(self, command_id, *, phase="attention", symbol="JHX", extra=None,
                     edits=False, account_key="pa"):
        record = {"version": 1, "id": command_id, "phase": phase,
                  "account_key": account_key,
                  "payload": {"_broker_account": self.ACCOUNT, "con_id": self.CON_ID,
                              "symbol": symbol, "_command_id": command_id},
                  "identity": [self.ACCOUNT, self.CON_ID, 98, 1554, 1544733398],
                  "session": "2026-09-16",
                  "created": "2026-09-16T10:21:29-04:00",
                  "mutation": "cancel exit",
                  "error": "owning client cannot resolve exactly one matching open order"}
        record.update(extra or {})
        root = self.root / "order_edits" if edits else self.root
        self.pa.save(root, record)
        return record


class Item2StructuredLockTests(PositionActionsCandidateTestCase):
    """The lock is correct; the report of it was not."""

    def test_the_refusal_class_is_a_valueerror_so_existing_handlers_still_catch_it(self):
        self.assertTrue(issubclass(self.pa.PreTransmitRefusal, ValueError))

    def test_the_lock_carries_every_field_the_site_parser_requires(self):
        """`lockRejection()` needs symbol + action_type + action_id together, and
        renders created_at and discrepancy (docs/site_execution_schema.md)."""
        record = self.write_record("890dffb0-2bf9-4b19-8a38-3e9d71da6bf4",
                                   extra={"command_type": "close_resize"})
        lock = self.pa.blocking_lock(record, "position action")
        self.assertEqual(lock["symbol"], "JHX")
        self.assertEqual(lock["action_type"], "close_resize")
        self.assertEqual(lock["action_id"], "890dffb0-2bf9-4b19-8a38-3e9d71da6bf4")
        self.assertEqual(lock["created_at"], "2026-09-16T10:21:29-04:00")
        self.assertIn("resolve exactly one matching open order", lock["discrepancy"])
        self.assertEqual(lock["account"], "pa")
        self.assertEqual(json.loads(json.dumps(lock)), lock)   # must survive the wire

    def test_a_record_written_before_this_change_degrades_to_the_generic_type(self):
        """The locks that exist TODAY carry no command_type; the site must still
        get a resolvable identity rather than a guessed action type."""
        lock = self.pa.blocking_lock(self.write_record("abc-125"), "position action")
        self.assertEqual(lock["action_type"], "position action")
        self.assertTrue(lock["symbol"] and lock["action_id"])

    def test_the_lock_prefers_the_live_broker_observation_for_the_discrepancy(self):
        record = self.write_record("abc-123", extra={
            "observation": {"detail": "current exits cover 0 units but position holds 625"}})
        lock = self.pa.blocking_lock(record, "position action")
        self.assertEqual(lock["discrepancy"], "current exits cover 0 units but position holds 625")
        self.assertIn("exits cover 0 units but position holds 625", self.pa.blocking_summary(lock))

    def test_the_lock_falls_back_to_the_session_date_and_the_contract(self):
        record = self.write_record("abc-124", symbol="", extra={"created": None, "error": None,
                                                                "mutation": None})
        lock = self.pa.blocking_lock(record, "order edit")
        self.assertEqual(lock["symbol"], f"conId {self.CON_ID}")
        self.assertEqual(lock["created_at"], "2026-09-16")
        self.assertIn("phase attention", lock["discrepancy"])

    def test_the_prose_summary_still_names_everything_if_the_structure_is_dropped(self):
        lock = self.pa.blocking_lock(self.write_record("abc-126",
                                                       extra={"command_type": "flatten"}),
                                     "position action")
        summary = self.pa.blocking_summary(lock)
        for fragment in ("JHX", "abc-126", "2026-09-16T10:21:29-04:00", "flatten",
                         "resolve exactly one matching open order"):
            self.assertIn(fragment, summary)

    def test_a_pre_transmit_refusal_is_rejected_even_on_a_record_with_a_mutation_marker(self):
        """The bug: an EARLIER command's mutation marker promoted a clean no-op
        to 'unknown / VERIFY IN TWS'."""
        tree = ast.parse((SOURCE / "position_actions.py").read_text(encoding="utf-8"))
        run = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_run")
        handler = next(h for n in ast.walk(run) if isinstance(n, ast.Try) for h in n.handlers)
        for exc, expected, wording in [
                (self.pa.PreTransmitRefusal("blocked"), "rejected", "Nothing changed"),
                (ValueError("broker went away"), "unknown", "Reconcile in TWS")]:
            with self.subTest(exception=type(exc).__name__):
                record = {"id": "x", "phase": "mutating", "mutation": "cancel exit"}
                ns = {"exc": exc, "record": record, "root": self.root,
                      "PreTransmitRefusal": self.pa.PreTransmitRefusal,
                      "save": lambda *_a, **_k: None, "isinstance": isinstance, "any": any, "dict": dict,
                      "str": str}
                body = ast.parse("def run():\n    pass\n").body[0]
                body.body = copy.deepcopy(handler.body)
                exec(compile(ast.fix_missing_locations(ast.Module(body=[body], type_ignores=[])),
                             "<pa-handler>", "exec"), ns)
                outcome = ns["run"]()
                self.assertEqual(outcome["state"], expected)
                self.assertIn(wording, outcome["detail"])

    def test_the_lock_branches_raise_the_refusal_class_with_the_structured_lock(self):
        text = (SOURCE / "position_actions.py").read_text(encoding="utf-8")
        self.assertIn('raise PreTransmitRefusal(\n                        "An earlier position action is unresolved', text)
        self.assertIn('raise PreTransmitRefusal(\n                        "An earlier order edit is unresolved', text)
        self.assertIn('lock = blocking_lock(previous, "position action")', text)
        self.assertIn('lock = blocking_lock(previous, "order edit")', text)
        self.assertEqual(text.count("+ blocking_summary(lock), lock)"), 2)
        original = (SOURCE / "position_actions.py.original").read_text(encoding="utf-8-sig")
        self.assertIn('raise ValueError("An earlier position action is unresolved', original)

    def test_the_refusal_keeps_its_lock_and_tolerates_not_having_one(self):
        lock = {"symbol": "JHX", "action_type": "flatten", "action_id": "abc"}
        self.assertEqual(self.pa.PreTransmitRefusal("blocked", lock).lock, lock)
        self.assertIsNone(self.pa.PreTransmitRefusal("blocked").lock)
        self.assertIsNone(self.pa.PreTransmitRefusal("blocked", "not a dict").lock)

    def test_new_records_carry_a_created_timestamp_and_the_command_type(self):
        text = (SOURCE / "position_actions.py").read_text(encoding="utf-8")
        self.assertIn('"created": datetime.now(ZoneInfo("America/New_York")).isoformat(timespec="seconds")',
                      text)
        self.assertIn('"command_type": str(ns.get("_COMMAND_TYPE") or "")', text)

    def test_the_command_type_travels_outside_the_payload(self):
        """position_actions compares a reloaded record's payload byte for byte,
        so a new payload key would make every pre-install unresolved action
        unreconcilable -- the very records this fix exists to clear."""
        executor = (SOURCE / "execute_order.py").read_text(encoding="utf-8")
        self.assertIn('globals()["_COMMAND_TYPE"] = str(t or "")', executor)
        self.assertNotIn('p["_command_type"]', executor)
        self.assertNotIn('payload["_command_type"]',
                         (SOURCE / "position_actions.py").read_text(encoding="utf-8"))


class Item4DiagnosticTests(PositionActionsCandidateTestCase):
    """Message-only. The abort is deliberately unchanged -- see the runbook."""

    def test_the_cancel_loop_explains_a_vanished_leg_without_changing_behaviour(self):
        text = (SOURCE / "position_actions.py").read_text(encoding="utf-8")
        self.assertIn("a cancelled OCA sibling goes terminal on its own", text)
        self.assertIn("rung(s) already cancelled", text)
        self.assertIn("OCA {leg.get('oca_group') or 'none'}", text)
        # Still a ValueError, still raised, still aborts the action.
        self.assertIn("raise ValueError(\n                    f\"{exc} [leg ", text)

    def test_the_diagnostic_wraps_only_the_identity_lookup(self):
        tree = ast.parse((SOURCE / "position_actions.py").read_text(encoding="utf-8"))
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "adjust_exits")
        tries = [n for n in ast.walk(fn) if isinstance(n, ast.Try)]
        self.assertEqual(len(tries), 1)
        self.assertEqual(len(tries[0].body), 1)
        self.assertIn("fresh_for_edit", ast.unparse(tries[0].body[0]))


class Item5ResolveHandlerTests(PositionActionsCandidateTestCase):
    """position_action_resolve: read the live book, record it, clear the lock."""

    def broker(self, *, held=625.0, working=True):
        trades = [FakeTrade(account=self.ACCOUNT, con_id=self.CON_ID, symbol="JHX",
                            perm_id=1544733398, order_id=1554, order_ref="EXEC|abc|unified-close",
                            status="PreSubmitted" if working else "Cancelled")]
        return FakeBroker(positions=[fake_position(self.ACCOUNT, self.CON_ID, "JHX", held)],
                          trades=trades)

    def payload(self, **kwargs):
        base = {"action_id": "890dffb0-2bf9-4b19-8a38-3e9d71da6bf4", "symbol": "JHX",
                "operator_note": "flattened by hand in TWS; book is square"}
        base.update(kwargs)
        return base

    def test_it_clears_the_lock_and_records_what_the_live_book_held(self):
        self.write_record("890dffb0-2bf9-4b19-8a38-3e9d71da6bf4")
        broker = self.broker()
        out = self.pa.resolve(self.ns(), broker, self.payload(), "pa")
        self.assertTrue(out["ok"])
        self.assertEqual(out["state"], "executed")
        self.assertIn("cleared by operator", out["detail"])
        self.assertIn("625", out["detail"])
        self.assertIn("flattened by hand in TWS", out["detail"])
        stored = json.loads((self.pa.record_path(
            self.root, "890dffb0-2bf9-4b19-8a38-3e9d71da6bf4")).read_text(encoding="utf-8"))
        self.assertEqual(stored["phase"], "done")
        self.assertEqual(stored["resolution"]["resolved_by"], "operator")
        self.assertEqual(stored["resolution"]["previous_phase"], "attention")
        self.assertEqual(stored["resolution"]["positions"][0]["position"], 625.0)
        self.assertEqual(stored["resolution"]["open_orders"][0]["perm_id"], 1544733398)

    def test_the_response_carries_the_snapshot_the_site_renders(self):
        """Contract: {state, reason, snapshot:{positions, open_orders}} --
        docs/site_execution_schema.md. `fill` stays None: nothing filled."""
        self.write_record("890dffb0-2bf9-4b19-8a38-3e9d71da6bf4")
        out = self.pa.resolve(self.ns(), self.broker(), self.payload(), "pa")
        self.assertIsNone(out["fill"])
        self.assertEqual(sorted(out["snapshot"]), ["open_orders", "positions"])
        self.assertEqual(out["snapshot"]["open_orders"][0]["status"], "PreSubmitted")
        self.assertEqual(out["snapshot"]["positions"][0]["symbol"], "JHX")
        self.assertEqual(json.loads(json.dumps(out["snapshot"])), out["snapshot"])

    def test_a_refusal_carries_no_snapshot(self):
        self.write_record("890dffb0-2bf9-4b19-8a38-3e9d71da6bf4")
        out = self.pa.resolve(self.ns(), self.broker(), self.payload(operator_note=""), "pa")
        self.assertIsNone(out.get("snapshot"))

    def test_it_reads_the_book_and_never_mutates_it(self):
        self.write_record("890dffb0-2bf9-4b19-8a38-3e9d71da6bf4")
        broker = self.broker()
        self.pa.resolve(self.ns(), broker, self.payload(), "pa")
        self.assertIn("reqPositions", broker.calls)
        self.assertIn("reqAllOpenOrders", broker.calls)
        # placeOrder/cancelOrder raise AssertionError if they are ever reached.

    def test_a_cleared_lock_no_longer_blocks_the_next_action(self):
        self.write_record("890dffb0-2bf9-4b19-8a38-3e9d71da6bf4")
        blocking = [r for r in self.pa.records(self.root) if r["phase"] != "done"]
        self.assertEqual(len(blocking), 1)
        self.pa.resolve(self.ns(), self.broker(), self.payload(), "pa")
        self.assertEqual([r for r in self.pa.records(self.root) if r["phase"] != "done"], [])

    def test_it_also_clears_an_order_edit_lock(self):
        self.write_record("edit-1", edits=True)
        out = self.pa.resolve(self.ns(), self.broker(), self.payload(action_id="edit-1"), "pa")
        self.assertEqual(out["state"], "executed")
        self.assertIn("order edit", out["detail"])

    def test_every_refusal_is_rejected_and_says_nothing_was_sent(self):
        self.write_record("890dffb0-2bf9-4b19-8a38-3e9d71da6bf4")
        self.write_record("already-done", phase="done", extra={"id": "already-done"})
        cases = [
            ("unknown id", self.payload(action_id="not-a-real-id"), "pa", "no position action"),
            ("already resolved", self.payload(action_id="already-done"), "pa", "already resolved"),
            ("missing note", self.payload(operator_note=""), "pa", "operator_note"),
            ("short note", self.payload(operator_note="ok"), "pa", "operator_note"),
            ("missing action id", self.payload(action_id=""), "pa", "action_id is required"),
            ("wrong symbol", self.payload(symbol="SPY"), "pa", "is on JHX, not SPY"),
            ("wrong account", self.payload(), "primary", "belongs to account pa"),
            ("unknown account", self.payload(), "paper", "unknown execution account"),
        ]
        for label, payload, account, expected in cases:
            with self.subTest(case=label):
                out = self.pa.resolve(self.ns(), self.broker(), payload, account)
                self.assertFalse(out["ok"])
                self.assertEqual(out["state"], "rejected")
                self.assertIn("Nothing sent", out["detail"])
                self.assertIn(expected, out["detail"])

    def test_a_broker_read_failure_is_still_a_clean_refusal_and_writes_nothing(self):
        self.write_record("890dffb0-2bf9-4b19-8a38-3e9d71da6bf4")

        class Broken(FakeBroker):
            def positions(self):
                raise RuntimeError("socket closed")
        out = self.pa.resolve(self.ns(), Broken(), self.payload(), "pa")
        self.assertEqual(out["state"], "rejected")
        self.assertIn("could not read the live book", out["detail"])
        stored = json.loads((self.pa.record_path(
            self.root, "890dffb0-2bf9-4b19-8a38-3e9d71da6bf4")).read_text(encoding="utf-8"))
        self.assertEqual(stored["phase"], "attention")
        self.assertNotIn("resolution", stored)

    def test_the_result_obeys_the_agent_state_contract(self):
        """exec_agent._execute_live_unlocked rejects anything else as 'unknown'."""
        self.write_record("890dffb0-2bf9-4b19-8a38-3e9d71da6bf4")
        for payload, account in [(self.payload(), "pa"), (self.payload(symbol="SPY"), "pa"),
                                 (self.payload(operator_note=""), "pa")]:
            out = self.pa.resolve(self.ns(), self.broker(), payload, account)
            self.assertIn(out["state"], {"executed", "rejected", "unknown"})
            self.assertEqual(out["state"] == "executed", out["ok"])
            self.assertTrue(isinstance(out["detail"], str) and out["detail"])

    def test_open_orders_are_scoped_to_the_exact_account_and_contract(self):
        broker = FakeBroker(
            positions=[fake_position(self.ACCOUNT, self.CON_ID, "JHX", 625),
                       fake_position("U16584234", self.CON_ID, "JHX", 100),
                       fake_position(self.ACCOUNT, 12259, "SNA", 890)],
            trades=[FakeTrade(account=self.ACCOUNT, con_id=self.CON_ID, perm_id=1),
                    FakeTrade(account="U16584234", con_id=self.CON_ID, perm_id=2),
                    FakeTrade(account=self.ACCOUNT, con_id=12259, perm_id=3)])
        self.assertEqual([r["perm_id"] for r in
                          self.pa.open_orders_for({}, broker, self.ACCOUNT, self.CON_ID)], [1])
        self.assertEqual([r["position"] for r in
                          self.pa.positions_for(broker, self.ACCOUNT, self.CON_ID)], [625])

    def test_open_orders_read_through_the_executors_all_client_reader(self):
        """`ib.openTrades()` returns only THIS client's orders. The snapshot has
        to show what every client left on the contract, so the executor's
        `_fresh_open_trades` is preferred when the namespace carries it."""
        mine = FakeTrade(account=self.ACCOUNT, con_id=self.CON_ID, perm_id=1, client_id=98)
        theirs = FakeTrade(account=self.ACCOUNT, con_id=self.CON_ID, perm_id=2, client_id=11)
        broker = FakeBroker(trades=[mine])
        ns = dict(self.ns(), _fresh_open_trades=lambda _ib: [mine, theirs])
        self.assertEqual([r["perm_id"] for r in
                          self.pa.open_orders_for(ns, broker, self.ACCOUNT, self.CON_ID)], [1, 2])
        self.assertNotIn("reqAllOpenOrders", broker.calls)


@unittest.skipUnless(HAVE_CANDIDATE, "Set EXEC_REPORTING_TEST_SOURCE to a prepared candidate")
class Item6FuturesMappingTests(unittest.TestCase):
    """Brief item 6: CONFIRM, do not change.

    The two `mapped equity-index future is missing its validated exchange or
    expiry` failures (2026-09-04, 2026-09-08, both close_only) were
    `_cluster_symbol` reading the RAW `ib.positions()` contract, which carries a
    blank exchange. The 2026-09-14 18:32 install (the now-retired
    `prepare_execution_repairs`, whose `patch_front` shipped alongside the
    re-pinned `futures_front.py`) qualifies the contract by conId FIRST, which
    is what populates exchange and lastTradeDateOrContractMonth. Asserted on the
    `.original` -- i.e. on LIVE -- because the claim is about what is installed,
    not about anything this candidate adds.
    """

    def test_close_only_qualifies_the_contract_before_the_futures_gate(self):
        original = (SOURCE / "execute_order.py.original").read_text(encoding="utf-8-sig")
        fn = next(n for n in ast.parse(original).body
                  if isinstance(n, ast.FunctionDef) and n.name == "_do_close_only")
        body = ast.unparse(fn)
        self.assertIn("qualify_position(ib, pos)", body)
        self.assertIn("_cluster_symbol(pos.contract)", body)
        self.assertLess(body.index("qualify_position(ib, pos)"),
                        body.index("_cluster_symbol(pos.contract)"))

    def test_this_package_does_not_touch_the_futures_path(self):
        for name in ("execute_order.py", "position_actions.py", "exec_agent.py"):
            with self.subTest(file=name):
                candidate = (SOURCE / name).read_text(encoding="utf-8")
                original = (SOURCE / (name + ".original")).read_text(encoding="utf-8-sig")
                for needle in ("_cluster_symbol", "qualify_position", "select_front_details",
                               "FUTURE_EXCHANGES", "_uncapped_futures"):
                    self.assertEqual(candidate.count(needle), original.count(needle), needle)


@unittest.skipUnless(HAVE_CANDIDATE, "Set EXEC_REPORTING_TEST_SOURCE to a prepared candidate")
class Items2And5AgentReportingTests(unittest.TestCase):
    """`reply` IS the command result the site reads, and it is built field by
    field -- so the structured keys have to be forwarded explicitly."""

    @classmethod
    def setUpClass(cls):
        tree = ast.parse((SOURCE / "exec_agent.py").read_text(encoding="utf-8"))
        handler = next(n for n in ast.walk(tree)
                       if isinstance(n, ast.AsyncFunctionDef) and n.name == "_handle_command")
        wanted = ('res.get(\'lock\')', 'res.get(\'snapshot\')', "position_action_resolve")
        cls.branches = [n for n in ast.walk(handler) if isinstance(n, ast.If)
                        and any(w in ast.unparse(n.test) for w in wanted)]

    def forward(self, res, command_type):
        self.assertEqual(len(self.branches), 3, "expected exactly the three forwarding branches")
        reply = {"type": "result", "state": res["state"], "ok": res["ok"], "detail": res["detail"]}
        ns = {"res": res, "reply": reply, "cmd": {"type": command_type},
              "isinstance": isinstance, "dict": dict}
        body = ast.parse("def run():\n    pass\n").body[0]
        body.body = [copy.deepcopy(b) for b in self.branches]
        exec(compile(ast.fix_missing_locations(ast.Module(body=[body], type_ignores=[])),
                     "<forward>", "exec"), ns)
        ns["run"]()
        return reply

    def test_a_lock_rejection_reaches_the_site_as_a_structured_field(self):
        lock = {"symbol": "JHX", "action_type": "close_resize", "action_id": "abc",
                "created_at": "2026-09-16T10:21:29-04:00", "discrepancy": "exits cover 0 of 625"}
        reply = self.forward({"state": "rejected", "ok": False, "detail": "Nothing sent: blocked",
                              "lock": lock}, "close_resize")
        self.assertEqual(reply["lock"], lock)
        self.assertEqual(reply["state"], "rejected")
        self.assertEqual(reply["reason"], "Nothing sent: blocked")

    def test_resolve_is_renamed_to_the_documented_state_and_keeps_its_snapshot(self):
        snapshot = {"positions": [{"position": 625.0}], "open_orders": []}
        reply = self.forward({"state": "executed", "ok": True, "detail": "cleared by operator",
                              "snapshot": snapshot}, "position_action_resolve")
        self.assertEqual(reply["state"], "resolved")
        self.assertEqual(reply["reason"], "cleared by operator")
        self.assertEqual(reply["snapshot"], snapshot)
        self.assertTrue(reply["ok"])

    def test_a_refused_resolve_stays_rejected(self):
        reply = self.forward({"state": "rejected", "ok": False,
                              "detail": "Nothing sent: operator_note is required"},
                             "position_action_resolve")
        self.assertEqual(reply["state"], "rejected")
        self.assertNotIn("snapshot", reply)

    def test_an_ordinary_execution_is_untouched(self):
        reply = self.forward({"state": "executed", "ok": True, "detail": "Bracket submitted"},
                             "entry_bracket")
        self.assertEqual(reply["state"], "executed")
        self.assertEqual(sorted(reply), ["detail", "ok", "state", "type"])

    def test_the_child_state_vocabulary_gate_is_not_weakened(self):
        """Teaching `_execute_live_unlocked` a fourth word would weaken the
        transmit-trust gate for EVERY command, so the rename happens after it."""
        candidate = ast.parse((SOURCE / "exec_agent.py").read_text(encoding="utf-8"))
        original = ast.parse((SOURCE / "exec_agent.py.original").read_text(encoding="utf-8-sig"))

        def gate(tree):
            fn = next(n for n in ast.walk(tree) if isinstance(n, ast.AsyncFunctionDef)
                      and n.name == "_execute_live_unlocked")
            return ast.unparse(fn)
        self.assertEqual(gate(candidate), gate(original))
        self.assertIn("'executed', 'rejected', 'unknown'", gate(candidate))


@unittest.skipUnless(HAVE_CANDIDATE, "Set EXEC_REPORTING_TEST_SOURCE to a prepared candidate")
class Item5WiringTests(unittest.TestCase):
    """The new type has to reach its handler through all three gates."""

    @classmethod
    def setUpClass(cls):
        cls.executor = ast.parse((SOURCE / "execute_order.py").read_text(encoding="utf-8"))
        cls.agent = ast.parse((SOURCE / "exec_agent.py").read_text(encoding="utf-8"))
        cls.agent_text = (SOURCE / "exec_agent.py").read_text(encoding="utf-8")

    def test_the_executor_supports_and_dispatches_the_new_type(self):
        # SUPPORTED is a literal followed by .add() statements; replay both.
        selected = [node for node in self.executor.body
                    if (isinstance(node, ast.Assign) and any(
                        isinstance(t, ast.Name) and t.id == "SUPPORTED" for t in node.targets))
                    or (isinstance(node, ast.Expr) and "SUPPORTED.add" in ast.unparse(node))]
        ns = {}
        exec(compile(ast.Module(body=selected, type_ignores=[]), "<supported>", "exec"), ns)
        self.assertIn("position_action_resolve", ns["SUPPORTED"])
        text = (SOURCE / "execute_order.py").read_text(encoding="utf-8")
        self.assertIn('if t == "position_action_resolve":', text)
        self.assertIn("_do_position_action_resolve(ib, p, acct)", text)

    def test_the_new_type_is_not_in_the_disabled_set(self):
        ns = load_nodes(self.executor, {"DISABLED_UNSAFE_MUTATIONS"}, {})
        self.assertNotIn("position_action_resolve", ns["DISABLED_UNSAFE_MUTATIONS"])

    def test_the_agent_validates_the_note_and_the_action_id(self):
        fn = next(n for n in self.agent.body if isinstance(n, ast.FunctionDef) and n.name == "_validate")
        branch = next(n for n in ast.walk(fn) if isinstance(n, ast.If)
                      and "position_action_resolve" in ast.unparse(n.test))
        module = load_position_actions()
        sys.modules["position_actions"] = module
        try:
            for payload, account, ok in [
                    ({"action_id": "a", "operator_note": "cleared in TWS"}, "pa", True),
                    ({"action_id": "", "operator_note": "cleared in TWS"}, "pa", False),
                    ({"action_id": "a", "operator_note": "short"}, "pa", False),
                    ({"action_id": "a", "operator_note": "cleared in TWS"}, "paper", False)]:
                with self.subTest(payload=payload, account=account):
                    ns = {"t": "position_action_resolve", "p": payload, "acct": account,
                          "str": str, "len": len, "bool": bool}
                    result = evaluate(branch, ns)
                    self.assertEqual(result[0], ok)
                    self.assertEqual(bool(result[1]), not ok)
        finally:
            sys.modules.pop("position_actions", None)

    def test_the_agent_preview_and_description_say_it_places_nothing(self):
        self.assertIn("places NO order", self.agent_text)
        self.assertIn("Places, cancels and modifies NOTHING", self.agent_text)
        self.assertIn("READ-ONLY", self.agent_text)

    def test_the_agent_never_routes_the_new_type_into_the_position_action_agent(self):
        """`_preview` delegates to position_action_agent.applies() on its first
        line, so the new type must not be claimed there -- its TYPES set is
        close_resize/add_to_position, and the resolve command shares none of
        their payload grammar."""
        sys.path.insert(0, str(ROOT / "broker_runtime"))
        try:
            import position_action_agent
        finally:
            sys.path.pop(0)
        self.assertNotIn("position_action_resolve", position_action_agent.TYPES)
        for account in ("primary", "pa"):
            self.assertFalse(position_action_agent.applies(
                {"account": account, "type": "position_action_resolve"}))
        for name in ("_validate", "_preview"):
            with self.subTest(function=name):
                fn = next(n for n in self.agent.body
                          if isinstance(n, ast.FunctionDef) and n.name == name)
                self.assertTrue([n for n in ast.walk(fn) if isinstance(n, ast.If)
                                 and "position_action_resolve" in ast.unparse(n.test)])


if __name__ == "__main__":
    unittest.main()
