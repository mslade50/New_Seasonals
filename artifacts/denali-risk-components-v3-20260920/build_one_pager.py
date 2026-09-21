from pathlib import Path
import csv
import report_layout as layout

ROOT = Path(__file__).resolve().parent
components = {(x['component'], x['horizon']): x
              for x in csv.DictReader((ROOT / 'component_forward_returns.csv').open())}


def cell(name, horizon, note):
    r = components[(name, horizon)]
    return (f"{horizon}: {float(r['mean_pct']):+.2f}% vs "
            f"{float(r['benchmark_mean_pct']):+.2f}% / {note}")


rows = [
['Distribution\nDominance',
 'Unusually busy selling sessions outnumber buying sessions near market highs. **Why:** selling pressure may weaken support before the index visibly turns.',
 cell('Distribution Dominance', '63d', '92 flagged days, which cluster.')],
['Defensive\nLeadership',
 'Defensive stocks (relatively resilient to a slowdown) lead economically sensitive stocks in the share trending upward. **Why:** cautious leadership near market highs may reveal weakening confidence in growth.',
 cell('Defensive Leadership', '63d', 'Largest gap in the set.')],
['VIX Range\nCompression',
 'Expected market volatility sits in a narrow range but above its recent average. **Why:** apparent stability may conceal rising concern before larger market moves.',
 cell('VIX Range Compression', '63d', 'Evidence is uneven.')],
['Low Absorption\nRatio',
 'Sectors move unusually independently near index highs. **Why:** offsetting sector moves can keep the index calm, and that cushion can disappear if sectors fall together.',
 cell('Low Absorption Ratio', '63d', 'Clearer over one month.')],
['Seasonal Rank\nDivergence',
 'Calendar patterns favor defensive over economically sensitive stocks near market highs. **Why:** seasonal support may be shifting away from the stocks that benefit most from growth.',
 cell('Seasonal Rank Divergence', '21d', 'Mainly a short-term clue.')],
['Dispersion',
 'Individual stocks swing much more than the index after a long period without a correction. **Why:** a steady headline index may hide instability among its members.',
 cell('Dispersion', '63d', 'Only 15 flagged days.')],
['NYSE Net New\nHighs',
 'Fewer NYSE stocks make new 52-week highs than new lows, on a five-session average, while the index is still near its high. **Why:** a narrowing advance is more fragile than a broad one.',
 '21d: -0.80% vs +0.76% on its matched comparison / Enters as a floor, not a weight.'],
['Equity Put Call\nComplacency',
 'Put trading is unusually low relative to call trading. **Why:** enthusiasm with little apparent demand for protection may leave investors exposed to bad news.',
 'No weight on the published dial. Monitored only.'],
]

one = [[
('p', 'Our risk dial combines warning signs associated with **weaker future S&P 500 (SPX) returns**. One dial is published, and it looks roughly one quarter ahead. Stored return studies use SPY as an index proxy. Higher scores mean more warning evidence, not a probability of a market decline.'),
('table', ['Component', 'What it measures and why we watch it', 'Stored forward returns*'], rows, [1.40, 3.95, 1.95]),
('p', '**How they combine:** six components feed a weighted average, which is then adjusted for the market backdrop and smoothed over five and then ten sessions. NYSE net new highs enters separately, as a floor that can raise the finished score and never lower it. Equity put/call carries weight only on a shorter horizon that is no longer published, so it currently contributes to no displayed dial and stays on the board as a monitored signal. Older warnings fade rather than switching off.'),
('small', '*Average return after the signal versus its study comparison average, over the stated horizon, across dates from 1 March 2018 to 2 June 2026. NYSE net new highs is measured on its own longer sample against near-high days whose breadth was not negative. These are recorded historical associations, not independently verified forecasts. Samples differ and signals can overlap, so flagged-day counts are not independent tests. Below-average returns can still be positive. Pre FOMC Rally was retired on 17 September 2026 and carries no weight. The companion report explains each component and the limits of the evidence; the same dial and the same component board are published on the shared site Risk tab.'),
]]

layout.build('Denali_Risk_Dial_One_Page_v3', 'Denali Risk Dial · Version 3', one, compact=True)
