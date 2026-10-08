"""Recoverable price pause; positive broker responses are evidence, never a guarantee of protection."""
import math
from .strategy import aware

# The existing watchdog snapshot deadline, not a new market-risk tolerance.
EXECUTION_PROOF_SECONDS = 5.


class PriceHealth:
    @property
    def price_pause_enabled(self):
        return self.config.allow_price_only_pause and self.resting_entries

    @property
    def price_entry_blocked(self):
        # Recheck after awaits as well as on the watchdog; short freshness gates remain binding.
        if self.price_pause_enabled:
            reasons = self._price_issues()
            if reasons:
                self._pause_prices(reasons)
        return self.price_paused

    def _observe_price(self, market, kind, timestamp, values):
        """Only an actual valid price callback resets that feed's monotonic age."""
        if not self.price_pause_enabled or market in self.manual_markets:
            return
        try:
            stamp = aware(timestamp)
            age = (aware(self.clock()) - stamp).total_seconds()
            if not -1 <= age <= self.config.stale_seconds or not all(math.isfinite(x) and x > 0 for x in values):
                return
            if kind == 'quote' and values[0] > values[1]:
                return
            key = market + ':' + kind
            old = self.price_observed.get(key)
            if old and stamp <= aware(old['source_timestamp']):
                return  # Duplicate/backdated tick or unchanged cached timestamp.
            # A late arrival must not erase an outage that already reached the boundary.
            self._enforce_price_grace()
            observed = self.monotonic()
            self.price_started.setdefault(market, observed)
            self.price_observed[key] = dict(source_timestamp=stamp.isoformat(),
                                           observed_monotonic=observed)
            return True
        except (TypeError, ValueError):
            return

    def price_quote(self, market, bid, ask, timestamp):
        self._observe_price(market, 'quote', timestamp, (bid, ask))

    def _enforce_price_grace(self):
        expired = {}
        now = self.monotonic()
        for name, state in self.states.items():
            if name in self.manual_markets or state.phase == 'SKIPPED' or not state.opening:
                continue
            for kind in ('trade', 'quote'):
                key = name + ':' + kind
                observation = self.price_observed.get(key)
                # Missing feeds age from the first actual callback, never from another feed's latest tick.
                reference = observation['observed_monotonic'] if observation else self.price_started.get(name)
                if reference is not None and now - reference >= self.config.price_outage_cancel_seconds:
                    expired[key] = dict(age_seconds=now-reference,
                                        source_timestamp=observation['source_timestamp'] if observation else None)
        if expired and not self.halted:
            self.store.event('PRICE_OUTAGE_GRACE_EXPIRED', dict(feeds=expired,
                             threshold_seconds=self.config.price_outage_cancel_seconds))
            self.halt('PRICE_OUTAGE_GRACE_EXPIRED:' + ','.join(sorted(expired)))

    def _pause_prices(self, reasons):
        if not self.price_paused:
            self.price_paused = True
            self.price_pause_at = self.clock().isoformat()
            self.store.event('PRICE_FEED_PAUSED', dict(at=self.price_pause_at, reasons=reasons))
            self.store.set('price_paused', True)
            self.notify('PRICE FEED PAUSED: no new entry decisions; existing entries require fresh broker reconciliation')
        self.price_pause_reasons = reasons

    def _price_issues(self):
        reasons = {}
        now = aware(self.clock())
        for name, s in self.states.items():
            if name in self.manual_markets or s.phase == 'SKIPPED' or not s.opening:
                continue
            ages = {'signal_timestamp': s.last_timestamp}
            ages['monotonic_ages_seconds'] = {kind: self.monotonic()-self.price_observed[name+':'+kind]['observed_monotonic']
                                            for kind in ('trade','quote') if name+':'+kind in self.price_observed}
            signal_age = (now - aware(s.last_timestamp)).total_seconds() if s.last_timestamp else None
            ages['signal_age_seconds'] = signal_age
            try:
                bid, ask, stamp = self.broker.quote(name)
                quote_age = (now - aware(stamp)).total_seconds()
                ages.update(quote_timestamp=aware(stamp).isoformat(), quote_age_seconds=quote_age)
                bad_quote = (not -1 <= quote_age <= self.config.stale_seconds
                             or not all(math.isfinite(x) for x in (bid, ask)) or not 0 < bid <= ask)
            except Exception as exc:
                bad_quote = True
                ages['quote_error'] = str(exc)
            observed_ages = ages['monotonic_ages_seconds']
            callback_stale = any(kind not in observed_ages or observed_ages[kind] > self.config.stale_seconds
                                 for kind in ('trade', 'quote'))
            if callback_stale or signal_age is None or not -1 <= signal_age <= self.config.stale_seconds or bad_quote:
                reasons[name] = ages
        return reasons

    def _check_execution_proof(self, snapshot, own, expected, issues):
        proof = snapshot.get('execution_health') or {}
        try:
            age = (aware(self.clock()) - aware(proof['received_at'])).total_seconds()
            elapsed = proof['roundtrip_seconds']
            valid = ((proof['source'] == 'ibkr_responses'
                      or (proof['source'] == 'offline_simulator' and self.config.mode == 'shadow'))
                     and proof['account'] == self.config.account
                     and proof['client_id'] == self.config.client_id
                     and proof['connected'] is True and proof['healthy'] is True
                     and proof['fresh_orders'] is True and proof['fresh_positions'] is True
                     and proof['fresh_executions'] is True
                     and -1 <= age <= EXECUTION_PROOF_SECONDS
                     and 0 <= elapsed <= EXECUTION_PROOF_SECONDS)
        except (KeyError, TypeError, ValueError):
            valid = False
        if not valid or own is None or issues or any(own.get(cid, 0) != qty for cid, qty in expected.items()):
            raise ValueError('Execution health unverified: fresh responses and matching own book required')
        self.store.set('execution_health', proof)

    def _update_price_health(self, snapshot, own, expected, issues, stable):
        self._check_execution_proof(snapshot, own, expected, issues)
        self._enforce_price_grace()
        reasons = self._price_issues()
        if reasons:
            self._pause_prices(reasons)
        elif self.price_paused and stable and not self.halted:
            self.price_paused = False
            self.price_pause_reasons = {}
            self.store.set('price_paused', False)
            self.store.event('PRICE_FEED_RECOVERED', dict(at=self.clock().isoformat(),
                             paused_at=self.price_pause_at, execution_health=snapshot['execution_health']))
            self.notify('PRICE FEED RECOVERED: fresh prices and broker reconciliation; normal entry gates apply')

    def broker_diagnostic(self, body):
        # Farm warnings are evidence only. Fatal adapter errors still invoke halt().
        key = (body.get('code'), body.get('message'))
        now = self.clock()
        previous = self.broker_diagnostics.get(key)
        if previous is None or (now - previous).total_seconds() >= 30:
            self.broker_diagnostics[key] = now
            self.store.event('BROKER_DIAGNOSTIC', {**body, 'received_at': now.isoformat()})
        # Bound diagnostic memory independently of the number of distinct messages.
        if len(self.broker_diagnostics) > 64:
            del self.broker_diagnostics[next(iter(self.broker_diagnostics))]
