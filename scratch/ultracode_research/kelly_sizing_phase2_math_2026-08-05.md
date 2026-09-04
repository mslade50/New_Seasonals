# Kelly Sizing — Phase 2 Mathematical Framework (2026-08-05)

Status: research methodology. No allocation result or live change is implied.

## 1. The object being optimized

Let the account's fixed research NAV be

\[
W_0 = \$750{,}000.
\]

For strategy or path \(i\), let \(P_{t,i}\) be its daily flat-basis MTM PnL
under the current sizing framework, including the current base bps, entry and
exit rules, scale-outs, fragility bands, cycle tilts, gap derates, recency
ladder, same-day de-rate, and daily risk cap. Define its daily return stream

\[
r_{t,i}=P_{t,i}/W_0.
\]

The research control is a static multiplier \(m_i\) on the current base size,
not a capital weight. Before cap nonlinearities, the counterfactual portfolio
daily return is

\[
r_t(m)=\sum_i m_i r_{t,i}=m^\top r_t.
\]

Thus \(m_i=1\) is current GRM-1.5 sizing, \(m_i=0.5\) is half the current base
bps for that strategy/path, and \(m_i=2\) is double. OVS P1 and P2 are separate
components. The three pilots — 3x Bear ETF Overbot Fade, 3x Leader Gap Fade,
and Monthly Weak Close — are frozen at \(m_i=1\) in proposal-producing
solutions.

This daily-stream formulation is preferable to assigning capital weights. A
strategy firing 100 times per year contributes more daily mean and covariance
than one firing once per year; overlapping holds and correlated signal days are
present directly in \(r_t\). Nothing has to be corrected by an ad hoc frequency
multiplier.

The linear map is a first-order approximation. The 250 bps per-strategy daily
cap, fixed OVS P2 aggregate cap, OLV ticker-notional cap, integer shares, and
overlap clamp can make \(P_{t,i}(m_i)\neq m_iP_{t,i}(1)\). Any proposal must
therefore be replayed through the engine after the stream-algebra screen.

## 2. Single-strategy empirical Kelly

### 2.1 Empirical R distribution

For a filled signal, let \(R\) be the realized profit divided by the sizing
risk unit. If a fraction \(f\) of wealth is put at risk in that unit, the
one-bet wealth factor is

\[
1+fR.
\]

The empirical Kelly fraction is

\[
f^*=\arg\max_{f\in\mathcal F}
\widehat g(f),\qquad
\widehat g(f)=\frac1N\sum_{j=1}^N\log(1+fR_j).
\]

For long-only sizing of an existing strategy, \(f\ge0\). The solvency domain is

\[
\mathcal F=\left\{f\ge0:1+fR_j>0\ \text{for every }j\right\}.
\]

If the worst observed return is \(R_{\min}<0\), then

\[
f<-1/R_{\min}.
\]

That is a mathematical boundary, not a recommended limit. The sample worst
loss is almost certainly less severe than the unknown population tail, so a
practical solution needs fractional Kelly and drawdown controls well inside
the boundary.

The derivatives are

\[
\widehat g'(f)=\frac1N\sum_j\frac{R_j}{1+fR_j},
\qquad
\widehat g''(f)=-\frac1N\sum_j\frac{R_j^2}{(1+fR_j)^2}\le0.
\]

The objective is concave. If \(\bar R\le0\), the constrained optimum is
\(f^*=0\). If \(\bar R>0\), losses occur with positive probability, and the
first derivative becomes negative before the loss boundary, the root of
\(\widehat g'(f)=0\) is unique.

This is why win-rate/payoff-ratio Kelly is invalid here. It replaces the full
distribution with a two-point gamble and cannot represent gap-through stops,
no-stop strategies, asymmetric targets, scale-outs, or the booked -4.56R tail.

### 2.2 OVS scale-out treatment

OVS near and far tranches are execution pieces of one filled signal, not two
independent bets. For the empirical-log cross-check, rows sharing

`(tier, ticker, signal date, entry date, entry price)`

are collapsed to

\[
R_{\text{signal}}=
\frac{\sum_{k\in\text{tranches}} PnL_k}
     {\sum_{k\in\text{tranches}} Risk_k}.
\]

P1/P2 classification uses the engine rule

\[
\text{P1 if } Open_{T+1}>Close_{signal}+0.25ATR;
\quad \text{P2 otherwise},
\]

after the no-gap skips have already been removed. Daily MTM needs no collapse:
tranche dollars are additive and their different exits are part of the desired
variance smoothing.

### 2.3 Heterogeneous live risk fractions

The current fraction risked on filled signal \(j\) is

\[
q_j=Risk_{j,\,flat}/W_0.
\]

It varies because of overlays and caps. A direct standalone multiplier check
can therefore maximize

\[
\widehat g_i(m_i)=\frac1{N_i}
\sum_j\log(1+m_iq_{ij}R_{ij}),
\]

rather than replacing every \(q_{ij}\) with one base bps number. The result is
a full-Kelly multiplier on the strategy's observed current sizing mix. The
more traditional \(f^*\) from a common R distribution will also be reported,
with \(f^*/\bar q_i\) as an approximate multiplier. Disagreement between the
two is evidence that overlays/cap incidence matters.

This calculation remains a standalone cross-check. It ignores simultaneous
bets and cannot allocate the correlated book.

## 3. Continuous approximation and the frequency mapping

For small \(fR\), expand the logarithm:

\[
E[\log(1+fR)]
\approx f\mu_R-\frac12f^2E[R^2].
\]

Ignoring the small \(\mu_R^2\) part of the second moment gives

\[
f^*\approx\frac{\mu_R}{\sigma_R^2}.
\]

The approximation is best when sizing is small and the distribution is not
strongly skewed. It can be poor for short-volatility-like payoffs: positive
means accumulated through frequent small wins can coexist with a fat left
tail, and the exact logarithm penalizes observations close to the solvency
boundary much more strongly than a variance summary does.

If strategies were independent Poisson bet streams with arrival rates
\(\lambda_i\), non-overlapping bets, and independent R outcomes, their
continuous-time growth rate would be

\[
g(f)=\sum_i\lambda_iE[\log(1+f_iR_i)].
\]

That expression shows why signal frequency matters. It is not the book's
solution because the assumptions fail exactly where risk concentrates:
dip-buy signals arrive together, positions overlap, and outcomes co-move
during corrections. The joint daily stream replaces it with the realized
calendar interaction.

## 4. Correlated multi-strategy Kelly

### 4.1 Exact empirical daily-log problem

Let \(r_t\) be the vector of strategy/path daily returns. With no frozen
background sleeve, the empirical objective is

\[
\widehat g(m)=\frac1T\sum_{t=1}^T\log(1+m^\top r_t),
\]

subject to \(1+m^\top r_t>0\) for every observed day and \(m_i\ge0\).
Negative multipliers would mean reversing a strategy, which is a different
strategy and is not a valid sizing recommendation.

With frozen pilots, write their current return as \(r_{t,F}\) and the seasoned
components as \(r_{t,S}\). The conditional problem becomes

\[
\widehat g(m_S)=\frac1T\sum_t
\log(1+r_{t,F}+m_S^\top r_{t,S}),
\]

with pilot multipliers fixed at one.

Because the logarithm of an affine positive function is concave, this remains
a concave maximization problem over a convex feasible set. Any local optimum
is global for the linear stream model.

### 4.2 Gaussian/small-return approximation

Let

\[
\mu=E[r_t],\qquad \Sigma=Cov(r_t).
\]

The second-order objective is

\[
g(m)\approx m^\top\mu-\frac12m^\top\Sigma m,
\]

and the unconstrained full-Kelly vector is

\[
m_K=\Sigma^{-1}\mu.
\]

More exactly, the quadratic term uses \(E[r_tr_t^\top]=\Sigma+\mu\mu^\top\),
but \(\mu\mu^\top\) is negligible at daily frequencies here. Phase 3 will
solve both the exact empirical-log problem and the quadratic approximation and
report their divergence.

The inverse covariance is the central allocation mechanism. Two dip-buy
strategies with similar positive means do not both receive their standalone
Kelly sizes if their losses occur on the same days. The optimizer rewards only
the part of each mean not replicated by the others.

### 4.3 Constraints and the meaning of a zero

An unconstrained inverse-covariance solution can contain negative weights.
That does not mean the strategy should be shorted; it means its estimated mean
does not pay for its marginal covariance with the rest of the book. The
proposal-producing optimization therefore uses nonnegative multipliers.
A zero is interpreted as “the optimizer cannot justify incremental risk under
these estimates,” not automatically as a recommendation to retire the
strategy. Sample uncertainty, design selection, and operational
diversification still apply.

No arbitrary upper multiplier should be imposed on the analytic solution: it
would hide weak identification. Extreme raw weights are reported as an
instability finding. The equal-risk-budget relative solution and any discrete
engine proposal provide the operational bounds.

## 5. Relative allocation versus absolute Kelly

These are different questions and must not be conflated.

### 5.1 Full-Kelly scale

The raw exact-log or quadratic solution estimates the absolute Kelly vertex in
multiplier space. It answers how far the small-return model would scale the
book if growth alone were the objective.

Because the current vector \(\mathbf 1\) will generally not be parallel to
\(m_K\), there is no unique scalar statement that current sizing is “X Kelly.”
Phase 3 will report three diagnostics rather than manufacture one number:

1. **Risk-budget-equivalent fraction** \(c_B\): the point on the Kelly ray with
   the same average deployed risk budget as the current book.
2. **Variance-equivalent fraction** \(c_\sigma\): the Kelly-ray point with the
   same daily return variance as the current book.
3. **Direction alignment:** the covariance-metric cosine between the current
   vector and \(m_K\). Low alignment means a single fraction is a poor summary.

The GRM replay is the separate current-allocation ray: it scales today's bps
ratios while keeping caps fixed. Comparing the current-allocation and Kelly
rays shows whether changing ratios adds value beyond selecting absolute risk.

### 5.2 Equal-risk-budget relative allocation

For the headline relative answer, define strategy/path \(i\)'s annualized
deployed risk budget

\[
a_i=\frac{252}{T}\sum_j Risk_{ij,flat}/W_0,
\]

where OVS tranche risk is additive and therefore sums back to the filled
signal's risk. This measure embeds frequency and all realized size overlays.
It is preferable to summing quoted bps, which would equate a once-yearly pilot
with OVS.

The current total is \(B_0=\sum_i a_i\). The relative allocation solves the
log or quadratic objective subject to

\[
\sum_i a_im_i=B_0,
\qquad m_i\ge0,
\qquad m_i=1\text{ for frozen pilots}.
\]

This isolates ratios while holding the approximate aggregate amount of filled
signal risk constant. Because the ledger cannot observe unfilled staged risk,
this is an exposure-budget proxy; the final engine replay with staged caps is
the authoritative nonlinear check.

## 6. Fractional Kelly

Along an unconstrained Kelly ray \(cm_K\), the quadratic objective satisfies

\[
g(c)=\left(c-\frac12c^2\right)S,
\qquad
S=\mu^\top\Sigma^{-1}\mu.
\]

Full Kelly is \(c=1\). Relative to its maximum growth \(g(1)=S/2\),

\[
\frac{g(c)}{g(1)}=2c-c^2.
\]

Therefore:

- half-Kelly, \(c=0.5\), retains approximately 75% of full-Kelly growth;
- quarter-Kelly, \(c=0.25\), retains approximately 43.75%;
- return variance on the ray scales as \(c^2\).

Quarter-Kelly is the headline fraction here; half-Kelly is shown for
reference. Fractional Kelly serves two purposes: it reduces the volatility and
drawdown implied by a noisy full-Kelly vertex, and it partially protects
against upward-biased means. It is not a substitute for explicit mean
shrinkage; both are required.

With pilots frozen, the fractional path is

\[
m_F(c)=1,\qquad m_S(c)=c\,m_{K,S}.
\]

The classic 75%/43.75% growth identities become approximate because the fixed
background sleeve contributes its own mean and covariance.

## 7. Drawdown-constrained Kelly

### 7.1 Diffusion intuition

For a single Kelly ray under a Brownian approximation, let full-Kelly's
instantaneous squared Sharpe be \(S\). At fraction \(c\), log wealth has
approximately

\[
d\log W_t=
\left(c-\frac12c^2\right)S\,dt+c\sqrt S\,dB_t.
\]

The probability that wealth ever falls to a fraction \(x<1\) of its starting
value has the hitting-probability form

\[
P\left(\inf_t W_t/W_0\le x\right)
\approx x^{\,2/c-1},\qquad 0<c<2.
\]

For a loss of \(d\) from starting wealth, \(x=1-d\). This is the origin of the
power-law “\(2/c-1\)” Kelly drawdown rule. It is not the probability of a
finite-horizon peak-to-trough drawdown: it assumes a diffusion, constant
parameters, continuous paths, reinvestment, and an infinite horizon. It is
used only as intuition for why reducing the Kelly fraction sharply reduces
deep-loss probability.

### 7.2 Binding operational constraint

The accepted research constraint is

\[
P_{boot}\left(MDD_{252}<-0.20W_0\right)<0.05.
\]

This follows the book's operational convention: a one-year path of flat-basis
dollar PnL is cumulatively added to $750k, and drawdown is the dollar distance
from the running peak divided by starting NAV. It is not the percentage loss
from the contemporaneous compounded peak.

The objective still uses daily log growth, which is the Kelly criterion. At
the book's small daily returns, the difference between additive flat-basis and
compounded daily equity is second order. Phase 3 will show both terminal-log
growth and flat-basis drawdown so that this approximation remains visible.

### 7.3 Stationary block bootstrap

To preserve cross-strategy dependence and volatility clustering, resample the
entire vector \(r_t\) jointly with a circular stationary bootstrap:

- horizon: 252 trading days;
- mean block length: 10 trading days;
- at each next day, continue the current historical block with probability
  0.9 or restart at a uniformly drawn historical day with probability 0.1;
- use the same sampled indices for all strategies/paths;
- compute at least 20,000 paths per candidate fraction with a fixed published
  seed.

For candidate \(m\), calculate path PnL \(W_0m^\top r_t\), cumulative flat
equity, peak-to-trough maxDD, terminal PnL, and compounded log wealth. The
largest fraction satisfying the 20%/5% gate is the drawdown-constrained Kelly
fraction. Quarter-Kelly remains the headline only if it satisfies the gate;
otherwise the constraint wins.

This extends the site's existing Politis-Romano-style mean-10td convention in
`scripts/build_site.py::build_monte_carlo` rather than introducing another
resampling clock.

## 8. Estimation error and empirical-Bayes shrinkage

### 8.1 Why shrink expectancy rather than daily mean blindly

Raw daily means are not exchangeable across strategies: a high-frequency
strategy should have a larger daily mean than an equally good low-frequency
strategy. R-multiple expectancy is closer to a common cross-strategy unit.

For component \(i\), collapse tranches to filled signals and estimate expected
R, \(\theta_i\). Let its observed annualized risk deployment be \(a_i\). The
daily mean implied by a posterior expectancy is

\[
\widetilde\mu_i=\frac{a_i}{252}\widetilde\theta_i.
\]

This preserves frequency and actual sizing while shrinking the edge estimate
in a comparable risk unit.

### 8.2 Episode-adjusted uncertainty

Signals are clustered into strategy/path episodes. The base rule starts a new
episode after more than five trading days without a signal, matching the
existing 3x validation convention. OVS also receives the established
calendar-month cluster estimate because it fires densely. The more
conservative standard error/effective N is used for shrinkage.

For clusters \(c=1,\dots,C_i\), a cluster-robust variance of the trade-weighted
mean is computed from cluster sums. Translate it into an independence-equivalent
sample size

\[
N_{eff,i}=\min\left(N_i,\frac{s_{R,i}^2}{SE_{cluster,i}^2}\right),
\]

with a floor of one. Both the actual cluster count and \(N_{eff}\) are
reported. This prevents 50 same-episode ETF signals or two OVS scale-out rows
from creating 50 or two units of conviction.

### 8.3 Normal-normal empirical Bayes

Use the hierarchy

\[
\widehat\theta_i\mid\theta_i\sim
N(\theta_i,s_i^2),
\qquad
\theta_i\sim N(\theta_0,\tau^2),
\]

where \(s_i\) is the cluster-adjusted standard error. Estimate the common prior
mean \(\theta_0\) and between-strategy variance \(\tau^2\) by marginal maximum
likelihood or an equivalent method-of-moments calculation. The posterior mean
is

\[
\widetilde\theta_i=
\theta_0+\kappa_i(\widehat\theta_i-\theta_0),
\qquad
\kappa_i=\frac{\tau^2}{\tau^2+s_i^2}.
\]

Large, well-dispersed samples such as OVS retain more of their own expectancy.
Small or episode-concentrated samples shrink toward the book prior. A
shrinkage-strength sweep multiplies the estimated prior precision by
0.5/1/2 and reports allocation stability.

This is a principled version of the current 25/30/35/40 nominal-bps conviction
tiers: higher effective N narrows \(s_i\), raises \(\kappa_i\), and allows the
strategy-specific edge to influence size more strongly.

### 8.4 Bayesian-Kelly interpretation

Under uncertain means, maximizing posterior expected quadratic log growth can
be written approximately as

\[
m^\top E[\mu\mid D]
-\frac12m^\top\left(\Sigma+V_{\mu\mid D}\right)m.
\]

Posterior mean uncertainty acts like an additional covariance/ridge penalty.
Using a shrunk posterior mean and reporting fractional Kelly captures the two
main effects without pretending the estimated full-Kelly vertex is known.

## 9. Three mean estimates

Every allocation is computed under three prespecified expectancy vintages:

1. **Full sample:** cluster-shrunk R expectancy using all eligible history.
2. **LOYO-conservative:** for each component, take the minimum mean across all
   leave-one-calendar-year-out samples, then apply the same empirical-Bayes
   shrinkage. This is effectively the drop-best-year expectancy and avoids
   letting one exceptional year set size.
3. **2018+:** recompute exposure, expectancy, shrinkage, and covariance on
   signals/daily returns from 2018 onward.

The recommendation-producing result uses the LOYO-conservative mean. Full
sample and 2018+ are robustness views. A proposed increase must not depend on
the full-sample estimate alone.

## 10. Covariance estimation and crisis dependence

Daily strategy returns are sparse and the inverse covariance can be unstable.
Use a covariance shrinkage estimator toward the diagonal rather than invert the
raw sample matrix. The unshrunk result and a ridge-strength sweep are reported
as diagnostics.

Two covariance regimes are prespecified:

1. **Full sample daily covariance**, estimated from the joint daily MTM matrix.
2. **Crisis covariance**, estimated from the literal brief windows: March
   2020, calendar 2022, August 2024, and April 2025. The same covariance
   shrinkage is applied because the crisis sample is much shorter.

Crisis correlations are also displayed separately from crisis volatility.
This distinguishes “the strategies became more correlated” from “everything
simply became more volatile.” A relative increase must survive both covariance
regimes. The FAMILY4/dip-buy cluster is expected to be the most sensitive.

## 11. Overflow survivorship discipline

The deployed full-book stream includes liquid and overflow tiers because that
is the risk actually taken. But the current-ticker overflow history is
survivorship-flattered and cannot establish an increase.

Phase 3 therefore runs:

- full deployed book;
- liquid tier only;
- full covariance with overflow expectancy set to the corresponding liquid
  strategy/path expectancy where available;
- exclusion of all overflow PnL.

No recommendation to increase size passes unless its sign and broad relative
rank survive a liquid-supported view. A better overflow backtest may reduce
confidence; it can never be used as the sole reason to add risk.

## 12. OVS P1/P2 treatment

P1 and P2 receive separate daily streams, R distributions, effective N,
posterior means, and allocation multipliers. Their covariance with each other
and the rest of the book is retained.

The current effective sizes are 60 bps P1 and 12 bps P2. Under an engine
counterfactual:

- P1 base/path1 bps can change independently;
- P2 bps can change independently;
- the P2 aggregate daily cap remains fixed at 1.125% of NAV, as confirmed;
- cycle tilt, blackout, scale-out, Friday EOD-DD, and precedence remain fixed.

Because the fixed P2 cap makes response nonlinear, the analytic P2 multiplier
is a local direction. The engine replay, not linear extrapolation, determines
the realized P2 risk and growth change.

## 13. Prespecified reporting and decision logic

For every strategy/path, report:

- current nominal and effective bps;
- unique filled signals and ledger rows;
- episode count and effective N;
- full, LOYO-conservative, and 2018+ average R;
- posterior/shrunk average R and shrinkage factor;
- standalone exact-log full-Kelly fraction and multiplier;
- correlated raw full-Kelly multiplier;
- quarter- and half-Kelly multipliers;
- current divided by quarter-Kelly;
- equal-current-risk-budget relative multiplier under full and crisis
  covariance;
- liquid/overflow sensitivity status.

Also report:

- current and Kelly-ray growth/fraction curves for \(c\in[0,1.5]\);
- exact empirical-log and Gaussian approximations;
- one-year stationary-bootstrap median/95th-percentile maxDD and
  \(P(MDD<-20\%)\);
- risk-budget-, variance-, and direction-based location of current GRM-1.5;
- engine replay of at most one discrete proposal, selected without tuning from
  the strongest robust result;
- the explicit null whenever the current allocation lies inside the
  uncertainty/sensitivity band.

The pre-registration draft in phase 4 is written only after these results. No
threshold or estimator is changed in response to an attractive allocation.

## 14. One remaining phase-3 lock choice

The relative optimization needs explicit feasible-set rules. The recommended
lock is:

1. nonnegative multipliers — never reverse a strategy;
2. frozen pilots fixed at 1.0;
3. no arbitrary upper bound in the raw full-Kelly diagnostic;
4. current total annualized filled-risk budget as the equality constraint for
   the headline relative allocation;
5. a zero optimizer weight is evidence against incremental allocation, not an
   automatic strategy-retirement recommendation.

This choice is mathematically and operationally cleaner than imposing a tuned
0.25x-2.0x box. A box can be shown as sensitivity after the unconstrained
instability is visible, but it should not determine the headline result.

