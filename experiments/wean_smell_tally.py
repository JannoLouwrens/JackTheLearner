import json,glob,collections
t=collections.defaultdict(dict)
fs=sorted(glob.glob('/data/tmp/wean_smell/*_s9[123].jsonl'))+sorted(glob.glob('/data/tmp/smell_sense/dose2/smell_s9[123].jsonl'))
for f in fs:
    arm,s=f.split('/')[-1][:-6].rsplit('_s',1)
    arm='null(dose2 smell)' if 'dose2' in f else arm
    rows=[json.loads(l) for l in open(f)]
    if not rows: continue
    late=[r for r in rows if r['phase']=='uniform' and r['k']>26000]
    nur=[r for r in rows if r['phase']=='nurse']
    t[int(s)][arm]=dict(weaned_meals=sum(r['ate'] for r in late), weaned_lives=len(late), fed_lives=sum(r['ate']>0 for r in late),
        nurse_meals=sum(r['ate'] for r in nur), nurse_lives=len(nur), r_last=(nur[-1]['r'] if nur else None), k=rows[-1]['k'])
for s in sorted(t):
    for a in sorted(t[s]): print(s, a, t[s][a])
