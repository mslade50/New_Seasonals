# Strategy-fit operating baseline

This source-only implementation carries forward the operating check recorded in
`docs/sleeve_operational_check_2026-09-06.md` in review branch
`codebase-review-20260906`, approximately **2026-09-06 10:40 ET**
(`2026-09-06T14:40:00Z`). It did not inspect live state again.

The previous check observed the Primary core book and Event scheduling active.
Monthly Trend was active through the legacy monthly order chain; its replacement
pre-open candidate was unactivated. A cash gate does not retire Trend: its
algorithm remains part of the research comparison baseline while it holds cash.

Legend's SPY/QQQ candidate and the dial hedge protocol candidate were prepared,
but unactivated at that check. Dial-gated SPY was paper. Treasury month-end work
is research only. Those are references, not active comparison strategies.

The exporter reads current source definitions, labels the active sleeves
`PREVIOUSLY_OBSERVED_ACTIVE`, and retains this observation timestamp separately
from catalog creation time. It blocks conclusive family comparisons when the
observation is more than 30 days old. It cannot infer fresh activation from a
source commit, a configured strategy, a candidate's simulated returns, or an
empty current-position inventory.

The registry is a conservative classification for deciding which historical
strategy comparisons to run. It is not a verified execution manifest. New core
or Event definitions without reviewed classifications make the baseline
incomplete. Known Event source definitions include four calendar equity trades
and two SVXY volatility trades. Live fills or current holdings are not inputs.
