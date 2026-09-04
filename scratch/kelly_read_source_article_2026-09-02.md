
## Outline
This article was written by Claude, so it is a bit of an experiment. I provided the chapter structure and did minor editing. I believe it is a helpful to understand Kelly position sizing with no mathematical prerequisites beyond basic school math.
It goes from a single position to a whole portfolio in seven steps. Starting with a single bet, then two independent bets, any two bets, many similar bets, anything in between, and finally any set of bets. So each step adds one idea, and each formula can be checked against the step before it by plugging in simple numbers.

## Rung 0 — Why there is a "right size" at all
A bet can make money on average and still ruin you if you size it too big.
Think of a coin flip. Heads you make +50%, tails you lose −40%, fifty-fifty odds. On average that's +5% per flip — a genuinely good bet. Now actually play it, betting everything each time. One win and one loss: your capital gets multiplied by 1.5, then by 0.6. That's 1.5 × 0.6 = 0.90. You are down 10% after every win-loss pair, and over many flips the pairs pile up and you go broke — on a bet with positive edge.
What happened? When you compound, gains and losses multiply, and multiplication punishes swings. A −40% loss needs a +67% gain just to get back to even. The damage from volatility grows with the square of the swing size, while the benefit from your edge grows only in proportion to it.
There is a rule of thumb that captures this exactly: your compound growth rate ≈ your average return − half the square of your swings. Check it on the coin flip before trusting it. Average return: +5%. Swings: ±45% around that average. Half of 0.45² is about 0.10, so the rule predicts 5% − 10% = −5% per flip — and that matches the −10% per win-loss pair we just computed by hand. The rule works.
Now, instead of betting everything, bet only a fraction f of your capital, and give the bet's two properties names: call its edge μ and its swing size σ (precise definitions in the next rung). At size f, your edge is f × μ and your swings are f × σ, so the rule of thumb becomes a formula:
growth(f) ≈ f μ − ½ f² σ²
Read the two pieces. The first term is a straight line in f: bet twice as much, collect twice the edge. The second term is a parabola: bet twice as much, pay four times the compounding tax. A rising line minus a rising parabola is a hill — an upside-down parabola. At small f the line wins and more size means more growth; at large f the parabola wins and more size means less growth, eventually negative (that's the coin flip, which sits past the peak at f = 1). Somewhere in between is the top of the hill. That top is the "right size," and finding it is everything that follows.

## Rung 1 — One bet: μ / σ²
Rung 0 left us with: growth(f) ≈ f μ − ½ f² σ². Finding the top of a parabola is school math: the peak is where the slope flattens to zero. Differentiate with respect to f and set it to zero:
slope = μ − f σ² = 0  →  f = μ / σ²
So for a single position, the growth-maximizing fraction of your capital is
f = μ / σ²
where the two ingredients are:
μ (mu) — your edge: the return you expect the position to earn per year above cash, after all costs. If you think a stock returns 9% and cash pays 4%, μ is 5%, written as 0.05. This is your honest forecast, not the bull case.
σ (sigma) — the volatility: how much the position swings in a typical year. A stock that routinely moves 30% up or down in a year has σ = 0.30.
Why is sigma squared in the denominator? That's Rung 0 showing up in the formula: the compounding damage grows with the square of the swings, so volatility counts against you twice over. Doubling the vol doesn't halve the right size — it quarters it.
Worked example: μ = 5%, σ = 30%. Then f = 0.05 / 0.09 ≈ 0.55. The formula says put 55% of your capital into this one stock.
That should strike you as insane. It is — deliberately hold that reaction, because it becomes important at the end. The formula gives the size that maximizes long-run growth if your inputs are exactly right, and it is far more aggressive than anything a sane person runs. What matters for now is the structure: size goes up with edge, down with the square of vol.

## Rung 2 — Independent bets: each gets its own μ / σ²
Suppose you have several positions whose returns have nothing to do with each other — genuinely unrelated businesses, unrelated drivers, zero correlation.
Then the answer is almost boring: each position gets its own μ/σ², computed as if the others didn't exist. Independent bets don't interact. Each one's compounding tax is charged only against its own swings, because their wiggles partially cancel rather than pile up.
Notice what this implies for the total. If you find five truly independent positions, each deserving 40%, the formula happily tells you to run 200% gross. Total exposure is not something you decide up front — it is the sum of what your bets individually deserve, and it grows as you find more genuinely different ideas. Gross is an output of the process, not an input to it.
This is the deep reason diversification is valuable in this framework: more independent bets don't just smooth the ride, they justify more total capital at work, which means faster compounding. The catch is the word "independent" — and that's the next rung.

## Rung 3 — Two correlated bets: μ / σ²(1 + ρ)
Now take two positions with the same edge μ and the same vol σ, but whose returns move together with correlation ρ (rho) — a number between −1 and +1. ρ = 0 means unrelated; ρ = 1 means they move in lockstep; ρ = 0.5 means they share about half their movement; negative ρ means one tends to zig when the other zags.
The right size for each of the two becomes:
f = μ / σ²(1 + ρ)
— the solo formula, with the denominator grown by (1 + ρ). Don't take it on faith — check it at the corners, where you already know the answer:
ρ = 0: the denominator is just σ², so each gets its full solo size. That's Rung 2 exactly.. Checks out.
ρ = 1: the denominator doubles, so each gets half its solo size — meaning the two together add up to exactly one solo position. The formula has recognized that two perfectly correlated stocks are one bet with two line items. Two tickets on the same horse don't double your edge; they just split one wager across two pieces of paper.. Checks out.
ρ = 0.5: the denominator is 1.5 σ², so each position shrinks by a third — bigger than half a bet each (they're not clones) but well short of full size (they're not new ideas either).
ρ = −0.5: the denominator is 0.5 σ², so each position gets double its solo size. The formula rewards hedged exposure with more capital, because offsetting swings cancel part of the compounding tax. This is why a hedged long/short pair can responsibly run much bigger gross than the same two positions ever could on the same side — the math agrees with the instinct every pairs trader already has.
One formula, and it reproduces everything you already believed at the corners. That's your license to trust it in between.

## Rung 4 — Many similar bets: the ceiling at μ / σ² × 1 / ρ
Now the case that actually describes a book: not two but many names sharing a theme — same μ, same σ, and the same pairwise correlation ρ between every pair. Ten semis longs. Eight regional banks. Pick your sleeve.
Push the number of names toward infinity and something striking happens. The total capital the whole group deserves does not grow without limit. It rises toward a hard ceiling:
total for the group → μ / σ² × 1 / ρ
and never crosses it, no matter how many names you add.
Read that ceiling in plain terms: the group's maximum justified size is the solo size divided by the correlation. Names that share half their movement (ρ = 0.5): the entire sleeve is never worth more than 2× one normal position — whether it holds five names or fifty. Names that share 70% of their movement: at most 1.4× one position.
Meanwhile each individual name's size shrinks toward zero as you add more — infinitely many tickets on one horse, each ticket worth almost nothing extra.
And you approach the ceiling fast. At ρ = 0.5, a single name already gets you halfway to everything the sleeve can ever justify, and about nine names capture 90% of it. Every look-alike added past that point consumes gross and balance-sheet while adding a rounding error of justified size. This is the concrete answer to "how many similar names are worth holding": a handful, and the more similar they are, the smaller the handful.

## Rung 5 — Any number in between: one curve connects it all
There is a single formula that covers any number of names at any correlation, and it is nothing but the curve connecting the corners you have already verified:
total = N μ / σ²(1 + (N − 1) ρ)
Plug in N = 1 and you get Rung 1. Plug in N = 2 and you get Rung 3. Set ρ = 0 and you get Rung 2. Let N grow huge and it settles on Rung 4's ceiling. Four napkin checks, and the formula stops being something you're asked to trust.
If you want to see it derived rather than checked, the resource is Edward Thorp's paper "The Kelly Criterion in Blackjack, Sports Betting, and the Stock Market" — freely available online, and by some distance the most readable primary source on all of this. The blackjack sections are the gentle on-ramp; the stock-market section is this document with proofs.

## Rung 6 — The real book: Σ⁻¹ μ
A real portfolio has no identical stocks — every name has its own edge, its own vol, and a different correlation to every other name. So μ becomes a list (one expected excess return per name), and Σ (capital Sigma) becomes a table with a row and column per name: down the diagonal, each name's own σ²; in the cell linking name 1 and name 2, their shared movement, σ₁ × σ₂ × ρ₁₂ — vol times vol times correlation. The sizing rule for the whole book is then w = Σ⁻¹ μ — "inverse covariance times expected returns."
In words, what the rule does: it pays each name only for what the rest of the book cannot already replicate — and once unpacked, it turns out to be Rung 1 again, edge over vol squared, applied to that unreplicated piece. The unpacking honestly takes a small dose of linear algebra — genuinely small: what a matrix is, how it multiplies a list of numbers, and what an inverse means. Two focused resources cover exactly that and nothing more. On video: 3Blue1Brown's "Essence of Linear Algebra" series on YouTube — the first four episodes plus the one on inverse matrices, about ninety minutes total, all pictures, no prerequisites, built for exactly this kind of reader. On paper: the first two chapters of Gilbert Strang's "Introduction to Linear Algebra" — the standard engineer's intro, with his matching MIT lectures free online. That's the entire toolkit; you have permission to ignore the rest of both.

## Coda — What everyone does with the answer
Three adjustments stand between the formula and a live book, and they all push the same direction: down.
Bet a fraction — usually half. Remember Rung 1 telling you to put 55% into one stock? Full formula weights maximize growth only if your μ estimates are exactly right, and they never are. Running half the formula's sizes keeps about 75% of the long-run growth rate at half the volatility — and cuts the lifetime odds of ever halving your capital from roughly a coin flip down to about one in eight. That trade is so good that "half-Kelly" is effectively the industry standard among people who use this at all.
Never go over. The penalty for oversizing is not symmetric. Bet below the optimum and you grow a little slower with less risk — a reasonable trade. Bet above it and you grow slower and swing harder: strictly worse on both axes, with bankruptcy at the far end. This is why practitioners shade down and never up, and why the one unforgivable sizing error is too big, not too small.
Distrust your μ more than your ρ. The formula's output is exquisitely sensitive to the expected returns you feed it and only mildly sensitive to the correlations — errors in the edge estimates matter roughly an order of magnitude more. Haircut your forecasts hard before they touch any sizing arithmetic. And when gross is capped below what the formula wants, don't shrink everything proportionally: the look-alike names are the ones burning scarce gross while duplicating exposure you already have, so the clones get cut first and the differentiated positions keep their size longest.
One line: size themes, not tickers; give each idea capital only for what it adds beyond the book; bet half of what the math says; and never, ever bet more.