"""Symbolic proof that the WITHIN-RUN fill rule is scale-invariant under x -> f*x,
including the open-better-than-limit branch used by backtester/strat_backtester."""
from fractions import Fraction as Fr

# Arbitrary positive prices (rationals to avoid float noise) and factor 0<f<1
C = Fr(3369, 100)    # signal close
A = Fr(7314, 10000)  # atr
mult = Fr(1,4)
f = Fr(990569, 1000000)

def fills_long(close, atr, low, openp):
    limit = close - mult*atr
    # backtester branch: open<limit -> fill@open ; elif low<=limit -> fill@limit
    if openp < limit:
        return True, openp
    if low <= limit:
        return True, limit
    return False, None

# pick a low/open that does NOT fill in RAW
low_raw = Fr(3358, 100); open_raw = Fr(3399, 100)
r_filled, r_px = fills_long(C, A, low_raw, open_raw)
# scale EVERYTHING by f (the adjusted re-run)
a_filled, a_px = fills_long(C*f, A*f, low_raw*f, open_raw*f)
print("RAW filled:", r_filled, " ADJ filled:", a_filled, " (must be equal)")
print("invariance of boolean holds:", r_filled == a_filled)
if r_px is not None and a_px is not None:
    print("price scales by f exactly:", a_px == r_px*f)

# Now try MANY random configs to attempt a counterexample
import random
random.seed(1)
breaks = 0
for _ in range(200000):
    c = Fr(random.randint(500,20000),100)
    a = Fr(random.randint(1,3000),100)
    lo = Fr(random.randint(100,int(c*100)),100)
    op = Fr(random.randint(int(lo*100),int(lo*100)+3000),100)
    ff = Fr(random.randint(800000,999999),1000000)
    rb,_ = fills_long(c,a,lo,op)
    ab,_ = fills_long(c*ff,a*ff,lo*ff,op*ff)
    if rb != ab:
        breaks += 1
        if breaks<=3:
            print("COUNTEREXAMPLE:", c,a,lo,op,ff, rb, ab)
print(f"\nwithin-run invariance counterexamples found: {breaks} / 200000")
