from pathlib import Path

TEXTS = {
 1: "long IEF friday MOO, out 9/30 close, no stop. the 10-year ran 15bp+ in two sessions to a 52-week high on wednesday, so i'm two days late. from that second open, 20 runs since 2003: 16-4 three sessions later, +0.51% vs +0.05% normal. 6-0 pre-2018, 10-4 since. half the runs are 2022. grade B.",
 2: "the dollar and the 10-year both closed at 52-week highs today. 7 priors since 2008, four of them in 2022. S&P a month later 6 of 7 up, which sounds like something until you see +0.90% vs +0.87% for any month. a level, not a trade.",
 3: "last september friday tomorrow. USO has closed down on 53 of 89 september fridays since 2006, -0.41% vs +0.11% on every other friday. all three this month were down. before anyone shorts it: 15-35 before 2018, 19-18 since, and from the open it's 42-47. anyway.",
 4: "the dollar long from 9/16 hit its time stop at +2.6R. the drafted-and-not-posted pile is now 4-10, -0.59R a shot. two still open, a small-cap short that ends friday and an energy short. the one time the filter turned up a winner i left it in the drawer.",
 5: "high yield -1.05% on the week. its two-year fit on treasuries and the S&P wanted about -0.35%, a 2-sigma miss, 2nd percentile since 2009. swap small caps in for the S&P and the miss halves. most of it opened monday, when tech ran and nothing else did. duration plus some small-cap beta. a credit-stress read needs more than this.",
 6: "in case someone's calling the bottom in bonds: TLT's worst 5% week of the year printing at a 52-week low, 10 priors declustered a month apart. a month later 2-8, -2.0% vs -0.7% for any month. the first few sessions lean the other way (that's the one i posted). past a week, no.",
}
for k, t in TEXTS.items():
    print(k, len(t))
