"""Pure qualification validator; import has no I/O."""
from copy import deepcopy
import math

def validate_policy(policy, mode):
    """No production grace is invented. Activation needs measured owner evidence."""
    required = {'schema', 'scope', 'qualification_id', 'source_sha256',
                'soft_ack_seconds', 'hard_ack_seconds', 'cancel_seconds',
                'snapshot_seconds', 'protection_ack_seconds', 'observed_protection_ack_max_seconds', 'observed_ack_max_seconds', 'approved'}
    optional={'paper_evidence_sha256'}
    if not isinstance(policy, dict) or not required<=set(policy) or set(policy)-required-optional or policy['schema'] != 1:
        raise ValueError('Exact versioned order reliability qualification required')
    if policy['scope'] not in {'offline_fixture', 'paper_qualified'} or policy['approved'] is not True:
        raise ValueError('Unapproved reliability qualification')
    if mode == 'live' and policy['scope'] != 'paper_qualified':
        raise ValueError('Offline fixture qualification cannot authorize live orders')
    if policy['scope']=='paper_qualified' and (len(str(policy.get('paper_evidence_sha256','')))!=64
            or any(c not in '0123456789abcdef' for c in policy['paper_evidence_sha256'])):
        raise ValueError('Frozen native paper evidence hash required')
    if not policy['qualification_id'] or len(str(policy['source_sha256'])) != 64:
        raise ValueError('Qualification identity and frozen source hash required')
    for k in ('soft_ack_seconds', 'hard_ack_seconds', 'cancel_seconds', 'snapshot_seconds', 'protection_ack_seconds', 'observed_protection_ack_max_seconds', 'observed_ack_max_seconds'):
        v = policy[k]
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0:
            raise ValueError(f'Invalid measured qualification timer: {k}')
    if not policy['soft_ack_seconds'] < policy['hard_ack_seconds'] <= 60:
        raise ValueError('Ack grace must be bounded and exceed the soft reconciliation threshold')
    if policy['hard_ack_seconds'] <= policy['observed_ack_max_seconds']:
        raise ValueError('Qualified hard grace must exceed measured acknowledgement latency')
    if policy['protection_ack_seconds']<=policy['observed_protection_ack_max_seconds']:
        raise ValueError('Protection deadline must exceed separately measured protective acknowledgment latency')
    if max(policy['cancel_seconds'], policy['snapshot_seconds'], policy['protection_ack_seconds']) > 30:
        raise ValueError('Qualification read/cancel deadline exceeds supported bound')
    return deepcopy(policy)

