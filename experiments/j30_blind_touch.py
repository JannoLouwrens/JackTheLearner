"""WORLD_CARD J30 BESIDE (Next-up 4): per-life contact profile, early (<100k) vs late (>=250k) lives, all nine J30 cells. Read-only on /data/jack-logs/j24k1/chains/j24k1_s*.json."""
import json, numpy as np
from scipy.stats import spearmanr
for s in (941,942,943):
    d=json.load(open(f'/data/jack-logs/j24k1/chains/j24k1_s{s}.json'))
    for arm in ('','-blind','-shuffled'):
        L=d['cells'][f'K1-bins16{arm}_s{s}']['lives']
        t=0; rows=[]
        for l in L:
            rows.append((t,l)); t+=l['decisions']
        for lab,sel in (('early<100k',lambda t0:t0<100_000),('late>=250k',lambda t0:t0>=250_000)):
            R=[l for t0,l in rows if sel(t0) and l['cause']!='unfinished' and 'touch' in l]
            if not R: continue
            m=np.array([l['floor_meals'] for l in R],float); n=np.array([l['decisions'] for l in R],float)
            f=np.array([l['touch']['food'] for l in R],float); w=np.array([l['touch']['wall'] for l in R],float)
            o=np.array([l['touch']['obstacle'] for l in R],float)
            rho_w=spearmanr(m,w/n).correlation if m.std()>0 else float('nan')
            rho_n=spearmanr(m,n).correlation if m.std()>0 else float('nan')
            print(f"s{s} {arm or 'sighted':9s} {lab:10s} lives {len(R):3d} meals/life {m.mean():6.2f} dec/life {n.mean():6.0f} wall% {100*w.sum()/n.sum():5.1f} obst% {100*o.sum()/n.sum():4.1f} food% {100*f.sum()/n.sum():5.1f} foodtouch/meal {f.sum()/max(m.sum(),1):5.1f} meals/1kdec {1000*m.sum()/n.sum():5.2f} rho(meals,wall%) {rho_w:+.2f} rho(meals,len) {rho_n:+.2f} zero-meal lives {int((m==0).sum())}")
