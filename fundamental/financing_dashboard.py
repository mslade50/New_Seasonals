"""Conservative research-dashboard filters, independent of the live signal book."""
from dataclasses import asdict, dataclass
from datetime import date
from .financing_opportunity import finite
from .cash_runway import utc


@dataclass(frozen=True)
class DashboardPolicy:
    min_price: float = 5
    min_dollar_volume: float = 10_000_000
    min_median_dollar_volume: float = 5_000_000
    min_return_20: float = .15
    min_return_60: float = .25
    min_relative_60: float = .10
    max_return_20: float = 1
    max_return_60: float = 2
    max_drawdown: float = .25
    max_runway_months: float = 24
    max_balance_age_days: int = 120
    min_history: int = 252

    def to_dict(self):
        return asdict(self)


def price_exclusions(row, policy=DashboardPolicy()):
    reasons=[]
    if row.get('price_status')!='complete': reasons.append('Incomplete prices')
    if not row.get('current_identity_verified'): reasons.append('Listing or identity needs checking')
    if row.get('listing_warning'): reasons.append('Listing warning')
    if row.get('split_recent'): reasons.append('Split within 60 sessions')
    if row.get('history_sessions',0)<policy.min_history: reasons.append('Less than one year of prices')
    checks=[('close',policy.min_price,None,'Price below $5'),
        ('dollar_volume_20',policy.min_dollar_volume,None,'Average turnover below $10m'),
        ('median_dollar_volume_20',policy.min_median_dollar_volume,None,'Median turnover below $5m'),
        ('return_20',policy.min_return_20,policy.max_return_20,'20-session move outside +15% to +100%'),
        ('return_60',policy.min_return_60,policy.max_return_60,'60-session move outside +25% to +200%'),
        ('relative_60',policy.min_relative_60,None,'Insufficient strength versus SPY'),
        ('distance_high_252',-policy.max_drawdown,None,'More than 25% below annual high')]
    for key,low,high,label in checks:
        value=finite(row.get(key))
        if value is None or value<low or (high is not None and value>high): reasons.append(label)
    if not row.get('above_sma50') or not row.get('above_sma200'): reasons.append('Below 50- or 200-session average')
    return reasons


def classify(row,financial,review,*,as_of,price_session,policy=DashboardPolicy()):
    result=dict(row,financial=financial or {},review=review or {},bucket='excluded',exclusions=[],review_items=[])
    reasons=price_exclusions(row,policy)
    if row.get('price_as_of')!=price_session: reasons.append('Price session is stale')
    result['exclusions']=reasons
    if reasons:return result
    result['bucket']='review'
    f=financial or {}
    if f.get('status')!='calculated':
        result['review_items']=['Financial data unavailable'];return result
    age=(utc(as_of).date()-date.fromisoformat(f['balance_date'])).days
    if age>policy.max_balance_age_days:
        result['review_items']=['Balance more than 120 days old'];return result
    if utc(f['filing_accepted_at'])>utc(as_of):
        result['review_items']=['Financial filing after dashboard cutoff'];return result
    burn=finite(f.get('monthly_burn_6m'))
    runway=finite(f.get('runway_6m'))
    if burn is None or burn<=0:
        result['bucket']='excluded';result['exclusions']=['No six-month operating cash burn'];return result
    if runway is None:
        result['review_items']=['Runway unavailable'];return result
    if runway>policy.max_runway_months:
        result['bucket']='excluded';result['exclusions']=['Reported operating runway exceeds 24 months'];return result
    r=review or {}
    bound=(r.get('cik')==row.get('cik') and r.get('balance_date')==f['balance_date'] and r.get('as_of')==as_of)
    if not bound:
        result['review_items']=['Current cash and financing review needed'];return result
    if not r.get('sources'):
        result['review_items'].append('Source evidence missing')
    for field,label in [('liquidity_verified','Cash and current investments unverified'),('newer_releases_checked','Newer earnings releases need checking'),('financing_checked','Recent financing needs checking')]:
        if not r.get(field):result['review_items'].append(label)
    verified=finite(r.get('reported_liquidity'))
    calculated=finite(f.get('reported_liquidity'))
    if verified is None or calculated is None or abs(verified-calculated)>max(1,abs(calculated)*.00001):
        result['review_items'].append('Reviewed liquidity does not tie to filing calculation')
    if r.get('post_balance_financing'):
        result['review_items'].append('Funding changed after the reported balance')
    if r.get('newer_liquidity_disclosure'):
        result['review_items'].append('Newer liquidity disclosure supersedes the balance')
    if r.get('unresolved'):
        result['review_items'].extend(r['unresolved'])
    if not result['review_items']:result['bucket']='watchlist'
    return result
