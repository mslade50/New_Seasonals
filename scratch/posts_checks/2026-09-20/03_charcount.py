"""Char-count the 2026-09-20 drafts before they go into the queue files."""
TEXTS = {
    "1 stat XLU first": (
        "utilities closed at a 52-week closing low on friday with the S&P 2% "
        "off its own high. i went looking for the last time those two printed "
        "together and since 2000 there isn't one. 62 sessions of XLU at a "
        "closing low, and on 59 of them the S&P was more than 15% under water."
    ),
    "2 stat monday vix": (
        "the VIX goes up on mondays. 1,172 of them since 2000, september "
        "excluded: 59.1% higher, mean +1.87%. every other weekday is 42.4% "
        "higher, mean -0.13%. holds in all three decades, weakest in this one. "
        "everyone says weekend decay. i didn't test that part."
    ),
    "3 stat move": (
        "bond vol is up 13% over the last month while the VIX sits 18% below "
        "its 200-day. 49 prior episodes back to 2003: the MOVE goes 15-34 a "
        "month later, -3.2% against +1.2% for a random month. SPY on the same "
        "days is 32-17, +1.0% vs +0.8%. the bond side unwinds, equities shrug."
    ),
    "4 stat kill": (
        "wanted to buy utilities and couldn't get there. worst 21-day return "
        "of its year with the index near a high: 13 episodes since 2000. 8-5 "
        "over five sessions, 5-8 over ten at -0.01% against a +0.37% "
        "baseline. the three since 2018 average negative at every horizon. "
        "no idea tonight."
    ),
    "5 take datamining": (
        "the standing objection to seasonality is that it's data mining, and "
        "it's right. most of what gets published is a sweep with the losers "
        "quietly dropped. the only defense i know is naming the window before "
        "the run and posting the number when it dies. nobody screenshots those."
    ),
    "6 ammo sector low": (
        "in case it comes up: a sector ETF closing at a 52-week low while the "
        "S&P is within 3% of its own high has happened 5 times across the 11 "
        "SPDRs since 2000. four were energy, and ten sessions on those four "
        "went 1-3, -2.9%. the fifth is utilities, friday. i'm not building "
        "anything on four energy trades."
    ),
}
for k, v in TEXTS.items():
    print(f"{len(v):4d}  {'OK ' if len(v) <= 280 else 'LONG'}  {k}")
