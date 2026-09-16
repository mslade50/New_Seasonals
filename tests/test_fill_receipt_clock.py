import pandas as pd
import pytest
from scripts.harvest_fills import validate_source_completeness


@pytest.mark.parametrize('source_age,receipt_age,valid',[
    (6,-.8,True),(6,-5,True),(6,-5.01,False),(6,301,False),
    (301,-.8,False),(-.1,-.8,False),
])
def test_only_relay_receipt_gets_bounded_clock_skew(source_age,receipt_age,valid):
    now=pd.Timestamp('2026-09-10T02:00:00Z')
    account=dict(complete=True,source_at=(now-pd.Timedelta(seconds=source_age)).isoformat(),
                 received_at=(now-pd.Timedelta(seconds=receipt_age)).isoformat())
    payload={'completeness':{'accounts':{'primary':account}}}
    if valid:validate_source_completeness(payload,now=now)
    else:
        with pytest.raises(RuntimeError,match='stale or future'):
            validate_source_completeness(payload,now=now)
