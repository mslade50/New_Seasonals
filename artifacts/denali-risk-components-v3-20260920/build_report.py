from pathlib import Path
import csv, json
import report_layout as layout

ROOT = Path(__file__).resolve().parent
components = list(csv.DictReader((ROOT / 'component_forward_returns.csv').open()))
ranges = list(csv.DictReader((ROOT / 'dial_ranges.csv').open()))
manifest = json.loads((ROOT / 'study_manifest.json').read_text())
BASE = manifest['baseline']


def pct(x, scale=1):
    v = float(x) * scale
    return '—' if str(v) == 'nan' else ('0.00%' if abs(v) < .005 else f'{v:+.2f}%')


def comp(name):
    r = {x['horizon']: x for x in components if x['component'] == name}
    n = int(r['5d']['n'])
    return ('table', ['Average forward return', '5 sessions', '21 sessions', '63 sessions'], [
        [f'Signal ON ({n} dates)', *[pct(r[h]['mean_pct']) for h in ['5d', '21d', '63d']]],
        ['All dates (2,075)', *[pct(r[h]['benchmark_mean_pct']) for h in ['5d', '21d', '63d']]]],
        [3.10, 1.40, 1.40, 1.40])


descriptions = {
'Distribution Dominance': 'Counts unusually busy sessions that close down from the open versus those that close up. A large selling imbalance near index highs triggers the signal. We watch it because repeated selling can weaken demand while the index still looks healthy.',
'Defensive Leadership': 'Compares the share of defensive and economically sensitive stocks in uptrends. Defensive stocks are considered more resilient to a slowdown. When they lead on both 50- and 200-session measures while SPY is near its high, the market may be showing less confidence in growth.',
'VIX Range Compression': 'Flags an unusually narrow range in VIX, the options market’s estimate of future SPX swings. VIX must also exceed 13 and its recent average. The concern is that a quiet range can conceal growing unease before a larger market move.',
'Low Absorption Ratio': 'Measures how much sectors move together. An unusually low reading near index highs means separate sector moves are helping keep the index steady. We watch for the risk that a common shock makes those sectors fall together.',
'Seasonal Rank Divergence': 'Flags calendar periods when historical rankings favor defensive stocks over economically sensitive stocks while SPY is near its high. The idea is that the market may be entering a period with less seasonal support for growth-sensitive shares.',
'Dispersion': 'Compares individual-stock swings with swings in the index after a long stretch without a 10% correction. Large moves in opposite directions can leave the index looking calm. The concern is that the calm disappears when those moves become more aligned.',
'Equity P/C Complacency': 'Flags an unusually low ratio of put trading to call trading. Puts can provide downside protection, while calls can express optimism. We watch for enthusiasm accompanied by little apparent demand for protection. It carries weight only on the five-session horizon, which is no longer published, so it currently contributes to no displayed dial. It stays on the board as a monitored signal.',
}


def component_blocks(names):
    blocks = []
    for name in names:
        title = name.replace('Equity P/C', 'Equity Put Call')
        blocks += [('h', title), ('p', descriptions[name]), comp(name)]
    return blocks


labels = {0: 'Below 20 (Robust)', 20: '20–<40 (Calm)', 40: '40–<60 (Neutral)',
          60: '60–<80 (Elevated)', 80: '80+ (Fragile)'}
HS = [5, 10, 21, 42, 63]


def display_table():
    data = [x for x in ranges if x['subset'] == 'all' and x['bins'] == 'display']
    headers = ['Dial range', 'Dates'] + [f'Next {h}' for h in HS] + ['63d losses']
    rows = []
    for r in data:
        n = int(r['days'])
        rows.append([labels[int(float(r['lo']))], str(n)]
                    + [pct(r[f'mean{h}'], 100) if n else '—' for h in HS]
                    + [f"{float(r['negative63']):.0%}" if n else '—'])
    rows.append(['All dates', '2,075'] + [pct(BASE[f'mean{h}'], 100) for h in HS]
                + [f"{BASE['negative63']:.0%}"])
    return ('table', headers, rows, [1.75, .60, .79, .79, .79, .79, .79, 1.00])


fine = []
for r in ranges:
    if r['subset'] == 'all' and r['bins'] == 'ten_point' and int(float(r['lo'])) >= 30 and int(r['days']):
        lo = int(float(r['lo']))
        hi = float(r['hi'])
        name = f'{lo}–<{int(hi)}' if hi != float('inf') and hi <= 100 else f'{lo}+'
        fine.append([name, r['days'], *[pct(r[f'mean{h}'], 100) for h in [5, 21, 63]]])

weights = manifest['weights_63d']
WEIGHT_ORDER = ['Distribution Dominance', 'Defensive Leadership', 'VIX Range Compression',
                'Low Absorption Ratio', 'Seasonal Rank Divergence', 'Dispersion',
                'Equity P/C Complacency']
weight_rows = [[k.replace('Equity P/C', 'Equity Put Call'), f'{weights[k]:.2f}',
                'Weighted average' if weights[k] > 0 else 'No weight at this horizon']
               for k in WEIGHT_ORDER]
weight_rows.append(['NYSE Net New Highs', f"{manifest['nyse_borrowed_weight']:.2f}",
                    'Floor on the finished score, not the weighted average'])

pages = [[
('p', 'This version describes the risk dial as it stands on 20 September 2026 and replaces the 17 September version. The system now publishes **one main dial**, which looks roughly a quarter ahead. The shorter five- and 21-session dials still exist inside the model but are no longer shown anywhere, so this report no longer reports them or their forward returns. One component is new, reading NYSE breadth, and the Pre FOMC Rally component was retired on 17 September 2026 and carries no weight in any dial.'),
('p', 'Each component below has a specific reason for being included, followed by the average returns that occurred after it appeared.'),
('small', 'Study dates: 1 March 2018–2 June 2026; prices through 1 September 2026. Every observation has a complete 63-session outcome. Returns are cumulative SPY returns, including dividends, used here as an S&P 500 proxy. All component and dial tables use the same period and the same price source.'),
*component_blocks(['Distribution Dominance', 'Defensive Leadership', 'VIX Range Compression']),
], [
*component_blocks(['Low Absorption Ratio', 'Seasonal Rank Divergence', 'Dispersion', 'Equity P/C Complacency']),
('small', 'Each Signal ON row includes every flagged date. Consecutive dates can belong to the same episode and share much of their forward-return window, so these counts are not independent tests. Equity Put Call Complacency is shown at all three horizons for comparison, but its only weight sits on the five-session horizon, which is no longer published. Pre FOMC Rally was retired on 17 September 2026 and has been removed from the component set and from the dial.'),
], [
('h', 'NYSE Net New Highs'),
('p', '**What it measures.** The number of NYSE-listed stocks setting new 52-week highs minus the number setting new lows. The trigger is a five-session exponential average of that daily net rather than the single-day figure, so one quiet day inside a negative stretch no longer changes the reading.'),
('p', '**When it arms, and why we watch it.** The smoothed net must be below zero while SPY sits within 3% of its own trailing 252-session closing high. Severity is full within 2% of the high and reduced from 2% through 3%. Narrowing participation while the index is still near its high suggests the advance is being carried by fewer stocks.'),
('p', '**How it enters the dial.** This is the one component that is not added to the weighted average. It builds a separate contribution, using the same weight as Low Absorption Ratio, and the published dial is the greater of the existing score and the expanded one. It can raise the dial and can never lower it.'),
('p', '**Fading, resetting and missing data.** Once armed, the contribution fades over 63 sessions and fades faster as the index falls away from its high. A single reading of the smoothed series at zero or above clears the component and its averaging queues outright; a fresh warning then requires re-entering the near-high zone with negative breadth. If any of the last five sessions has no breadth reading, the smoothed series is blank, which can neither arm a warning nor confirm a recovery, and the existing dial is used instead.'),
('table', ['Average forward return', '5 sessions', '10 sessions', '21 sessions'], [
    ['Smoothed breadth negative near the high', '-0.41%', '-0.63%', '-0.80%'],
    ['Near the high, breadth not negative', '+0.19%', '+0.38%', '+0.76%']],
    [3.10, 1.40, 1.40, 1.40]),
('p', '**Evidence and its limits.** Over 6,440 sessions from December 2000 to September 2026 the smoothed rule flags 244 days in 34 separate episodes, against 345 days in 78 episodes for the single-day version it replaced. The chance that SPY falls 5% at some point within the following 63 sessions is 67% from a smoothed flag against 52% from a single-day flag. Set against that: measured from each rule’s own first alarm, every version tested still shows a positive average return over the following 21 and 63 sessions, the single-day version included. The smoothed rule also fires a median of four sessions later, and against a placebo that simply waits those four sessions and uses no breadth at all, the improvement in drawdown frequency does not reach 1.8 standard errors. The change was a deliberate preference for fewer false alarms and far less flickering, not a demonstrated gain in forecasting power.'),
('p', '**A worked example.** In August 2026 the raw daily net turned non-negative on the 19th, 25th, 26th, 27th and 28th. Under the previous single-day rule each of those cleared the component and both of its averaging queues, so its contribution rebuilt from nothing three separate times in eleven sessions. Under the smoothed rule the 19th and the 25th clear nothing and the state clears once, on the 26th. The smoothed series first armed on 18 August, one session after the single-day version. The warning was still fading on 17 September 2026, when it lifted the published dial to 85.0 from the 81.9 the rest of the composite produced on its own.'),
('small', 'Breadth counts are read each session from the published NYSE latest-close column of the Wall Street Journal Markets Diary. A missing session is left missing and never filled forward.'),
], [
('h', 'Forward returns by main dial range'),
('p', 'Higher readings generally preceded weaker returns in this sample. The relationship was clearest over the next week and month. The table uses the published main dial and the display bands already on the dashboard.'),
display_table(),
('small', 'Next 5 through Next 63 are average percentage returns over that number of trading sessions. 63d losses is the share of 63-session outcomes below zero. Readings on consecutive days share most of their forward-return window, so the date counts are not independent observations, and scores before July 2026 were reconstructed with later model settings.'),
('h', 'Where the main dial became more concerning'),
('p', '**Around 40, returns weakened.** The average three-month return fell from +5.24% below 40 to −0.08% at 40 and above. Negative three-month outcomes rose from 18% to 42%. Most high-reading stretches still finished higher; a few larger declines pulled the average down.'),
('p', '**Above 60 the near-term record was worse:** −0.54% over five sessions and −2.36% over 21 sessions on average, across 181 dates. At 80 and above the 21-session average was −4.73% and 83% of three-month outcomes were negative, on 24 dates.'),
('table', ['Main dial range', 'Dates', 'Next 5', 'Next 21', 'Next 63'], fine, [2.35, .75, 1.40, 1.40, 1.40]),
('p', 'The three-month average turned negative in the 50–60 band, then positive again at 70–80. We would read **40–60 as a caution zone and 60+ as a stronger near-term warning**. The dashboard uses 20/40/60/80 display cutoffs; the data supports broad zones more clearly than exact thresholds.'),
('p', '**Why:** higher scores combine warnings about selling, leadership, stability and participation while the index can still look strong. About 74% of dates at 40 and above occurred within 2% of the yearly high.'),
('p', '**The effect varied by period.** At 40 and above, three-month returns averaged −2.95% in 2018–2021 and +2.52% in 2022–2026, both below their own under-40 comparisons of +6.70% and +3.97%. Spacing observations 63 sessions apart so that none overlap leaves 14 starts at 40 and above and raises their three-month average to +0.62%.'),
('small', 'The 80-and-above group contains only two starts once observations are spaced 63 sessions apart. Reconstructed history and overlapping observations limit the precision of these historical watch zones.'),
], [
('h', 'How the score is built'),
('p', '**Weighting.** Each component carries a weight drawn from how much worse than average returns were after it appeared, together with calibration adjustments. These are relative model inputs, not return forecasts. A larger weight does not by itself mean stronger independent evidence.'),
('p', '**Combining and adjusting.** The weighted share of active and fading warnings is multiplied by 80 and then adjusted for the market backdrop: the past year’s return, distance from the 200-session average, distance below the yearly high, and time since the last meaningful correction. The reasoning is that warnings may matter more after extended strength or prolonged calm. Those adjustments express model assumptions; they do not establish that a decline is due.'),
('p', '**Fading and smoothing.** A warning has full influence while it is on. Once it turns off, its influence fades across the horizon window and fades faster as the index falls. Daily scores are averaged over five sessions, and the published dial is averaged over a further ten. That reduces noise and adds delay. A lower score during a decline can simply reflect older warnings fading as time passes and prices fall; it does not establish a recovery. NYSE Net New Highs sits outside this arithmetic, as a floor on the finished score.'),
('h', 'Current component weights'),
('table', ['Component', '63-session weight', 'How it enters the dial'], weight_rows, [2.60, 1.60, 3.10]),
('small', 'The dial history used here is the production point-in-time record and runs through 18 September 2026. Readings from 2 July 2026 onward were recorded on the day and are never revised; earlier readings were reconstructed with later model settings and should be read as such. From 17 September 2026 the published dial is stored directly, under the definition dated 18 September 2026. The same dial and the same component board are published on the shared site Risk tab.'),
]]

layout.build('Denali_Risk_Dial_and_Forward_Returns_v3',
             'Denali Risk Dial and Forward Returns · Version 3', pages)
