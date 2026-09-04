# study_ovs_liquid results (script output, verbatim)

```
ledger vintage: build_utc=2026-09-04T08:24:57Z source=gha:33852895307 rows=4701
==============================================================================
RECON CHECKS
==============================================================================
OVS positions: liquid 316 (2010+ 255), overflow 956 (2010+ 808)
rows per position: {2: 1215, 1: 144}
path inference: gap-rule vs Size_Mult-rule agreement on liquid 2010+ = 1.0000 (disagreements 0)
ranks recomputed for 100.0% of liquid signals; all four > 85 (strategy filter) on 100.0% of those
liquid signal-date clusters: base 96, recent 55

==============================================================================
PRIMARY: liquid OVS, 2024+ minus 2010-2023, clustered by signal date
==============================================================================
[primary liquid] base N=189 avgR=+0.580 win=76.2% | recent N=66 avgR=-0.028 win=43.9% | diff=-0.608 t_cl=-3.05 (G=151)
monthly block bootstrap (10000 draws, seed 20260904): diff 95% CI [-0.975, -0.197], P(diff>=0)=0.0023
statsmodels cluster check: coef -0.6079 t -3.055

==============================================================================
CUT (i): drop MU DE XLK GLW INTC IBM
==============================================================================
dropped 17 positions (base 5, recent 12)
      era Ticker  n   avgR        pnl
0    base     DE  1  0.429   1929.000
1    base    IBM  1  1.299      4.000
2    base   INTC  2  1.140    971.000
3    base    XLK  1  1.173    585.000
4  recent     DE  1 -2.186  -6403.000
5  recent    GLW  2 -0.299  -2020.000
6  recent    IBM  2 -0.618  -4960.000
7  recent   INTC  1 -1.702  -5743.000
8  recent     MU  3 -1.465 -15927.000
9  recent    XLK  3 -0.803  -7025.000

[cut i] base N=184 avgR=+0.567 win=75.5% | recent N=54 avgR=+0.197 win=50.0% | diff=-0.370 t_cl=-1.83 (G=140)

==============================================================================
CUT (ii): path split P1 / P2 by era
==============================================================================
      era path_gap    n   avgR   win        pnl
0    base       P1  104  0.629 0.760  91209.000
1    base       P2   85  0.519 0.765  22904.000
2  recent       P1   33 -0.161 0.424 -18986.000
3  recent       P2   33  0.104 0.455   3921.000

[cut ii P1] base N=104 avgR=+0.629 win=76.0% | recent N=33 avgR=-0.161 win=42.4% | diff=-0.790 t_cl=-3.14 (G=80)
[cut ii P2] base N=85 avgR=+0.519 win=76.5% | recent N=33 avgR=+0.104 win=45.5% | diff=-0.415 t_cl=-1.59 (G=86)
P1 share of liquid signals: base 0.550 recent 0.500

==============================================================================
CUT (iii): exclude 2026
==============================================================================
[cut iii ex-2026] base N=189 avgR=+0.580 win=76.2% | recent N=37 avgR=+0.188 win=48.6% | diff=-0.391 t_cl=-1.93 (G=127)

==============================================================================
CUT (iv): bottom-extremity share (mean rank_2/5/10/21 < 94.0) and top-cell primary
==============================================================================
bottom share: base 0.339 (N=189), recent 0.197 (N=66)
      era    cell    n   avgR   win    ext
0    base  bottom   64  0.316 0.641 92.048
1    base     top  125  0.715 0.824 96.772
2  recent  bottom   13 -0.377 0.308 92.181
3  recent     top   53  0.057 0.472 96.688

[cut iv top cell (>=94)] base N=125 avgR=+0.715 win=82.4% | recent N=53 avgR=+0.057 win=47.2% | diff=-0.658 t_cl=-3.02 (G=113)
[cut iv bottom cell (<94)] base N=64 avgR=+0.316 win=64.1% | recent N=13 avgR=-0.377 win=30.8% | diff=-0.693 t_cl=-1.97 (G=61)
extremity mean: base 95.17, recent 95.80

==============================================================================
CUT (v): sector / theme concentration of liquid signals by era
==============================================================================
era                     base  recent  n_base  n_recent
sector                                                
Technology             0.053   0.273      10        18
Healthcare             0.048   0.212       9        14
Consumer Defensive     0.101   0.136      19         9
Commodity              0.048   0.106       9         7
Consumer Cyclical      0.095   0.061      18         4
Financial Services     0.196   0.061      37         4
Utilities              0.037   0.061       7         4
Energy                 0.127   0.045      24         3
Industrials            0.143   0.045      27         3
Basic Materials        0.037   0.000       7         0
Communication Services 0.026   0.000       5         0
Index                  0.005   0.000       1         0
Real Estate            0.063   0.000      12         0
UNKNOWN                0.021   0.000       4         0

semis          share: base 0.037 (avgR in-theme +0.719 n=7) | recent 0.121 (avgR in-theme -0.371 n=8, ex-theme +0.019 n=58)
megacap_tech   share: base 0.032 (avgR in-theme +0.970 n=6) | recent 0.045 (avgR in-theme -0.803 n=3, ex-theme +0.009 n=63)
semis|megacap  share: base 0.063 (avgR in-theme +0.850 n=12) | recent 0.167 (avgR in-theme -0.489 n=11, ex-theme +0.064 n=55)
[cut v ex-theme (info only)] base N=177 avgR=+0.561 win=75.1% | recent N=55 avgR=+0.064 win=47.3% | diff=-0.498 t_cl=-2.36 (G=136)
2024+ liquid worst tickers by flat PnL:
        n   avgR        pnl
Ticker                     
MU      3 -1.465 -15927.000
MRK     4 -0.667  -9918.000
XLK     3 -0.803  -7025.000
GLD     4 -1.512  -6709.000
DE      1 -2.186  -6403.000
INTC    1 -1.702  -5743.000
IBM     2 -0.618  -4960.000
UNG     2 -0.720  -3002.000
XLP     1 -0.791  -2316.000
GLW     2 -0.299  -2020.000
CVS     1 -0.392  -1764.000
XBI     2 -1.090  -1471.000

==============================================================================
CUT (vi): signal supply and per-year avgR, liquid 2010-2026
==============================================================================
       n   avgR   win        pnl  p1_share  bottom_share
year                                                    
2010   5  0.581 0.800    240.000     0.200         0.800
2011   4  1.064 1.000  13400.000     0.750         0.500
2012   3  0.269 0.667  -1593.000     0.333         0.667
2013   5 -0.109 0.600   -489.000     0.000         0.200
2016  17  0.283 0.647  15460.000     0.529         0.529
2017   2 -0.326 0.500  -2938.000     1.000         0.500
2018   9 -0.073 0.444    965.000     0.667         0.333
2019  10  0.515 0.900  13952.000     0.400         0.700
2020  51  0.712 0.824   4045.000     0.588         0.235
2021  16  0.410 0.750  20505.000     0.438         0.062
2022  32  0.647 0.750  33229.000     0.406         0.375
2023  35  0.854 0.800  17337.000     0.800         0.286
2024   9 -0.054 0.333   1166.000     0.667         0.333
2025  28  0.266 0.536  11624.000     0.429         0.250
2026  29 -0.305 0.379 -27855.000     0.517         0.103

years (2024-2026) individually below the 2010-2023 mean +0.580: ['2024', '2025', '2026']

==============================================================================
CUT (vii): controls, same era split
==============================================================================
[overflow OVS (upper-bound caveat)] base N=619 avgR=+0.342 win=66.9% | recent N=189 avgR=+0.291 win=63.5% | diff=-0.051 t_cl=-0.45 (G=435)
[3x ETF Overbot Fade] base N=83 avgR=+0.902 win=74.7% | recent N=4 avgR=+0.012 win=50.0% | diff=-0.890 t_cl=-3.24 (G=46)

==============================================================================
CUT (viii): exit-type mix by era
==============================================================================
tranche-row exit mix (share within era):
era        base  recent
Exit Type              
EOD-DD    0.030   0.023
Target    0.439   0.248
Time      0.531   0.729

position-level exit mix (EOD-DD any / Target all / Time all / Mixed):
era       base  recent
exit_pos              
EOD-DD   0.058   0.045
Mixed    0.418   0.212
Target   0.217   0.136
Time     0.307   0.606

position avgR by exit label and era:
      era exit_pos   n   avgR
0    base   EOD-DD  11 -0.627
1    base    Mixed  79  0.969
2    base   Target  41  1.599
3    base     Time  58 -0.442
4  recent   EOD-DD   3 -0.754
5  recent    Mixed  14  0.935
6  recent   Target   9  1.588
7  recent     Time  40 -0.675

[cut viii time-exit positions] base N=58 avgR=-0.442 win=43.1% | recent N=40 avgR=-0.675 win=17.5% | diff=-0.232 t_cl=-1.25 (G=71)
far-tranche time-exit row avgR: base +0.363 (n=137), recent -0.269 (n=54)

==============================================================================
CLOSED DECISION SET INPUTS
==============================================================================
primary t <= -2.0: True (t=-3.05)
cut i t <= -1.5: True (t=-1.83)
cut iv top-cell t <= -1.5: True (t=-3.02); bottom-cell t=-1.97
2024, 2025, 2026 each below base mean: True (['2024', '2025', '2026'])
decision_inputs_all_hold = True; explained_by_extremity = False
```
