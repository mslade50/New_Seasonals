// Extend only overlapping, explicitly attested execution query intervals.
// A fresh receipt or the oldest observed fill is not a coverage boundary.
export function extendFillCoverage(previous, current) {
  const anchor=previous?.complete===true?previous:previous?.history_anchor;
  if (!current.complete) return {...current, continuous_from:null,history_anchor:anchor};
  const start=Date.parse(current.query_from || '');
  const end=Date.parse(current.complete_through || '');
  if (!Number.isFinite(start) || !Number.isFinite(end) || start>end) {
    return {...current, complete:false, continuous_from:null, history_anchor:anchor, error:'execution query coverage is unverified'};
  }
  const priorStart=Date.parse(anchor?.continuous_from || '');
  const priorEnd=Date.parse(anchor?.complete_through || '');
  const joins=anchor?.complete===true && anchor.broker_account===current.broker_account
    && Number.isFinite(priorStart) && Number.isFinite(priorEnd)
    && start<=priorEnd && end>=priorEnd;
  const result={...current, continuous_from:new Date(joins?Math.min(start,priorStart):start).toISOString()};
  const scope='OLV_US_STK_NON_OVERNIGHT';
  if(current.olv_coverage?.scope===scope) {
    const cutoff=Date.parse(current.olv_coverage.prior_session_close || '');
    const oldStart=Date.parse(anchor?.olv_continuous_from || '');
    const scopedJoin=anchor?.complete===true && anchor.broker_account===current.broker_account
      && anchor.olv_coverage?.scope===scope && Number.isFinite(oldStart) && oldStart<=priorEnd
      && Number.isFinite(cutoff) && cutoff<start && priorEnd>=cutoff && end>=priorEnd;
    result.olv_continuous_from=new Date(scopedJoin?Math.min(start,oldStart):start).toISOString();
  }
  return result;
}
