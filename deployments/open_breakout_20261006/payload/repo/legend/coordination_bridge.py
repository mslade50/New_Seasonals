"""Publish only the final actionable decision; gate Legend behind settled breakout.

No pre-open blanket hold. A preemption can consume Legend's existing 20-second
transmit grace and cause a missed entry. No rejected signal automatically
releases the accepted opposite-side latch.
"""
from intraday_coordination import LegendSignal, Scope, from_environment


def coordinate_before_entry(ledger, cfg, st, dec, sched, clock, *, armed, sleeper):
    if ledger is None:
        return True
    if not (armed and not st.forced and st.setup.qualified and sched.strategy == 'Legend_EMA'
            and dec.side in {'BUY', 'SELL_SHORT'} and st.qty > 0):
        return False
    side = 1 if dec.side == 'BUY' else -1
    if (side == 1 and not cfg.allow_longs) or (side == -1 and not cfg.allow_shorts):
        return False
    scope = Scope.for_symbol(clock.now().date().isoformat(), cfg.account, st.symbol)
    signal = LegendSignal(scope, side, st.sig, clock.now(), True, True, True, st.qty)
    ledger.publish(signal, clock.now())
    while clock.now() <= sched.transmit_deadline:
        if ledger.legend_ready(scope, clock.now()):
            return True
        if ledger.read(scope)['operation']['phase'] == 'reconcile':
            return False
        sleeper(.1)
    # Signal suppression policy on failed/rejected Legend needs an owner decision.
    # Latch remains durable; no entry is sent after the original deadline.
    ledger.signal_status(scope, st.sig, 'delivery_unknown')
    return False


def submit_parent(ledger, scope, signal_id, now, send):
    return send() if ledger is None else ledger.submit_legend(scope, signal_id, now, send)


def record_legend_result(ledger, cfg, st, result, clock):
    """Observe failure, never release the already accepted opposite-side latch.

    A returned status is not a substitute for broker reconciliation. Only the
    existing definite rejection statuses are labelled rejected; all other
    non-success results remain uncertain and require an owner decision.
    """
    if ledger is None:
        return
    status = str(result.get('status', ''))
    if status.startswith('SENT') or status == 'TIME_LEG_FAILED_FLATTENED':
        return
    scope = Scope.for_symbol(clock.now().date().isoformat(), cfg.account, st.symbol)
    lifecycle = 'broker_rejected' if status in {'CANCELLED', 'APICANCELLED', 'INACTIVE', 'REJECTED'} else 'delivery_unknown'
    ledger.signal_status(scope, st.sig, lifecycle)
