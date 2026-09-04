import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import load_prices

TK = ['KC=F','CT=F','LE=F','ZC=F','ZW=F','ZS=F','SI=F','CL=F','HG=F','HE=F','SB=F','GC=F']
px = load_prices(TK)
for t in TK:
    d = px.get(t)
    if d is None:
        print(t, 'MISSING'); continue
    d = d.tail(7)
    print('===', t)
    prev = None
    for dt, r in d.iterrows():
        o,h,l,c = r['Open'], r['High'], r['Low'], r['Close']
        gap = (o/prev-1)*100 if prev else float('nan')
        sess = (c/prev-1)*100 if prev else float('nan')
        intr = (c/o-1)*100
        bad = []
        if c > h*1.0001: bad.append('CLOSE>HIGH')
        if c < l*0.9999: bad.append('CLOSE<LOW')
        if o > h*1.0001 or o < l*0.9999: bad.append('OPEN_OUT')
        if h < l: bad.append('HIGH<LOW')
        print('  %s O%9.2f H%9.2f L%9.2f C%9.2f | sess%8.2f%% gap%8.2f%% intraday%8.2f%% %s' % (
            str(dt)[:10], o,h,l,c, sess, gap, intr, ' '.join(bad)))
        prev = c
