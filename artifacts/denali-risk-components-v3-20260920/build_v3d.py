from pathlib import Path
import csv, json
import report_layout_v3d as layout

ROOT = Path(__file__).resolve().parent
comp = {r['component']: r for r in
        csv.DictReader((ROOT / 'v3d_component_returns.csv').open(encoding='utf-8'))}
bands = list(csv.DictReader((ROOT / 'v3d_main_dial_ranges.csv').open(encoding='utf-8')))
M = json.loads((ROOT / 'v3d_manifest.json').read_text(encoding='utf-8'))
HS = [5, 10, 21]
WINDOWS = ['Next 5', 'Next 10', 'Next 21']
N = f"{M['rows']:,}"
NYSE_N = f"{M['nyse_sample_rows']:,}"
CUT = f"{M['cutoff']:.0f}"


def pct(v):
    v = float(v)
    return '0.00%' if round(v * 100, 2) == 0 else f'{v:+.2%}'


def row(d):
    return [pct(d[f'mean{h}']) for h in HS]


BASELINE = row(M['baseline'])
NYSE_BASELINE = row(M['nyse_baseline'])

NAMES = ['Distribution Dominance', 'Defensive Leadership', 'VIX Range Compression',
         'Low Absorption Ratio', 'Seasonal Rank Divergence', 'Dispersion', 'NYSE Net Highs']

descriptions = {
NAMES[0]: 'Counts distribution and accumulation days over the past 63 trading sessions. Both require SPY volume above its 63-session average and more than 15% above the previous session. Distribution days close below the open; accumulation days close above it. A large selling imbalance near index highs can reveal demand weakening beneath a strong index.',
NAMES[1]: 'Compares the share of defensive and economically sensitive stocks above their 50- and 200-session averages. Defensive businesses tend to hold up better in a slowdown. Their leadership near index highs can signal less confidence in growth and weaker support for further gains.',
NAMES[2]: 'Flags an unusually narrow trading range in VIX, the options market’s estimate of future S&P 500 swings. VIX must also exceed 13 and its recent average. We watch this combination because a tight range alongside rising unease can precede a larger move.',
NAMES[3]: 'Measures how closely sectors move together. A low reading near index highs means separate sector moves can help keep the index steady. A common shock can make those sectors fall together, removing the support that their different paths provided.',
NAMES[4]: 'Compares how defensive and economically sensitive stocks have historically performed at that time of year. When the calendar favors defensive shares while SPY is near its high, the market may have less seasonal support for continued gains.',
NAMES[5]: 'Compares the size of individual-stock moves with moves in the index after a long stretch without a 10% correction. Large gains and losses can cancel out, making the index look calm. We watch for the risk that those stock moves line up on the downside.',
NAMES[6]: 'Subtracts new 52-week lows from new 52-week highs on the New York Stock Exchange (NYSE). A negative total means more listed securities are making lows than highs. When SPY is still near its own 252-session high, that mismatch suggests the index is masking weakness across the broader market. The warning reads a five-session exponential average of the total, so a single quiet day inside a negative stretch does not switch it on or off.',
}
shorts = [
'Busy sessions closing below the open outnumber those closing above it; demand may be weakening.',
'Defensive businesses, more resilient in a slowdown, lead economically sensitive shares; confidence may be fading.',
'VIX (expected S&P 500 volatility) trades in a tight range above 13 and its recent average; unease may be building.',
'Sectors move separately near highs; a common shock could make them fall together.',
'The calendar favors defensive shares; economically sensitive stocks have less seasonal support.',
'Large individual-stock moves cancel out in a calm index; that balance can break.',
'NYSE new lows exceed new highs on a five-session average while SPY is near its high; broad participation is weak.',
]


def component(n):
    d = comp[n]
    baseline = NYSE_BASELINE if n == NAMES[-1] else BASELINE
    total = NYSE_N if n == NAMES[-1] else N
    return [('h', n), ('p', descriptions[n]),
            ('table', ['Average SPY return', *WINDOWS],
             [[f"Warning active ({int(d['n']):,} dates)", *row(d)],
              [f'All dates ({total})', *baseline]], [3.10, 1.4, 1.4, 1.4])]


band_rows = [[b['range'], f"{int(b['n']):,}", *row(b)] for b in bands]
band_rows.append(['All dates', N, *BASELINE])
split_rows = [[f'Below {CUT}', f"{M['below_cutoff']['n']:,}", *row(M['below_cutoff'])],
              [f'{CUT} or above', f"{M['at_or_above_cutoff']['n']:,}",
               *row(M['at_or_above_cutoff'])]]

intro = 'The goal of the risk dial is to proactively identify good and bad trading environments, the “easy dollar” versus “hard penny” markets many seasoned traders recognize. It looks beneath index price action for signs of risk that a steady or rising market can hide.'
note = (f"Study: 8 February 2002 to 18 June 2026; {N} common observation dates, the full span over "
        "which every component can be measured. Returns include SPY dividends, using prices "
        f"through 18 September 2026. All tables use the same sample and show average returns. The "
        f"NYSE Net Highs table covers the {NYSE_N} of those dates whose highs-and-lows record is "
        "complete enough to read the warning, and its comparison row uses the same dates.")
vintage = ('Readings before August 2018 are reconstructed: they show what the current rules '
           'produce when applied to the full price history rather than what was published at the '
           'time. The stored record begins in July 2016, but its own first two years are a '
           'start-up period in which components that need years of trailing history could not yet '
           'register, so the study reads the reconstruction there instead. From August 2018 the '
           'stored record takes over unchanged, and since July 2026 each reading is written on the '
           'day it is produced and never revised afterwards.')

pages = [[
('p', intro),
('p', 'The dial combines the seven components below into one score. Higher readings reflect stronger combined warnings. We focus on what happened over the next **5, 10 and 21 trading days**, roughly one week, two weeks and one month.'),
*component(NAMES[0]), *component(NAMES[1]), *component(NAMES[2]),
], [
*component(NAMES[3]), *component(NAMES[4]), *component(NAMES[5]), *component(NAMES[6]),
], [
('h', 'How to read the dial'),
('p', f"In this study, higher readings were followed by weaker returns. Below 20, the average next-month return was {pct(M['under20_mean21'])}. At {CUT} or above, it was {pct(M['at_or_above_cutoff']['mean21'])}, and the average week ahead was slightly negative. The table shows the full range."),
('table', ['Main dial', 'Dates', *WINDOWS], band_rows, [2.8, .6, 1.3, 1.3, 1.3]),
('h', 'What the ranges mean'),
('p', f"**Below 20, supportive:** the strongest average returns across all three windows. **20 to 40, constructive:** still positive, with smaller gains than the under-20 group. **40 to 55, caution:** gains narrowed again, to between a half and two thirds of the study average. **{CUT} and above, warning:** the deterioration builds with the reading rather than arriving at the line. The {CUT} to 65 band still carried positive averages across all three windows; from 65 to 80 all three turned negative; and at 80 and above, on {bands[-1]['n']} dates, they were weaker still."),
('table', ['Main dial', 'Dates', *WINDOWS], split_rows, [2.8, .6, 1.3, 1.3, 1.3]),
('h', f'Why {CUT} matters'),
('p', f"Fifty-five is our working warning threshold because it marks the top 15% of readings in this study, the 85th percentile. The dial is in warning territory about one day in seven: {M['at_or_above_cutoff']['n']:,} of the {N} dates, arriving in {M['episodes_at_or_above_cutoff']} separate episodes. Below the line, average returns ran at or above the pace of the study as a whole across all three windows. At {CUT} and above, the week-ahead average was slightly negative, the two-week average was flat and the month-ahead average was under a third of the figure below the line. The bands show that the deterioration builds with the reading rather than flipping sign at the line, so {CUT} is a practical dividing line rather than a switch. The score gives us a consistent way to recognize when trading conditions have become less forgiving."),
('p', 'The components describe different kinds of pressure: selling activity, defensive leadership, volatility, sector behavior, seasonality and market breadth. Their weights, how recently they fired and the market backdrop determine the score. Reading the component list alongside the dial helps explain what is driving the warning.'),
('small', note),
('small', vintage + ' Consecutive dates share return windows, so the date counts exceed the number of independent episodes. A favorable reading still carries risk.'),
]]

layout.build('Denali_Risk_Dial_Detailed_v3d', 'Denali Risk Dial', pages)

summary_rows = [[n, reason, *row(comp[n])] for n, reason in zip(NAMES, shorts)]
summary_rows.append(['All dates', 'Comparison for the warning rows', *BASELINE])

layout.build('Denali_Risk_Dial_One_Page_v3d', 'Denali Risk Dial', [[
('p', intro),
('small', 'The dial combines seven components into one score; higher readings reflect stronger combined warnings. We focus on the next 5, 10 and 21 trading days, one week, two weeks and one month. SPY tracks the S&P 500.'),
('table', ['Component', 'Why we watch it', *WINDOWS], summary_rows, [1.55, 3.20, .85, .85, .85]),
('small', 'NYSE net highs are new 52-week highs minus new 52-week lows. The warning reads a five-session exponential average of that total, so a single quiet day does not switch it on or off. It is strongest closest to SPY’s high and clears when the average returns to zero.'),
('table', ['Main dial', 'Dates', *WINDOWS], band_rows, [2.8, .6, 1.3, 1.3, 1.3]),
('p', f'**Reading it:** below 20 has been most supportive; 20 to 40 constructive; 40 to 55 calls for caution; {CUT} and above is the working warning threshold, set at the 85th percentile of readings, with all three windows negative from 65. Taken together the {M["at_or_above_cutoff"]["n"]:,} dates at {CUT} or above averaged {", ".join(row(M["at_or_above_cutoff"]))} against {", ".join(row(M["below_cutoff"]))} on the {M["below_cutoff"]["n"]:,} dates below the line.'),
('small', 'Component rows show returns after each warning was active. ' + note),
('small', vintage),
]], compact=True)
