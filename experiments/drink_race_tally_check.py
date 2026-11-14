"""Synthetic fixtures for experiments/drink_race_tally (known win, each lost clause, world-breaking void, stale exclusion, incomplete). Run: python -m experiments.drink_race_tally_check"""
import json, tempfile, copy
from pathlib import Path
from experiments import drink_race_kcpu as K, drink_race_tally as T
cur={s:K.code_shas(s) for s in K.SPECS}
def ne(g2, hyd, star, drinks=5):
    return {"ok_spread":1,"ok_random_deaths":g2,"ok_forager":1,"forager_drinks":drinks,
            "deaths_dehydration":hyd,"deaths_starvation":star,"deaths_microsleep":0,
            "death_dec_min":300,"death_dec_max":5000,"forager_cause":"","forager_survived_dec":1}
def write(d, arm, g2s, hyd, star, w1=True, drinks=5, stale=False):
    for s in K.SEEDS:
        for spec in K.SPECS:
            m = ne(g2s[s],hyd,star,drinks) if spec=="NE.01" else {"a_pass":1,"c_pass":1}
            rec={"arm":arm,"spec":spec,"seed":s,"metrics":m,"code":cur[spec]}
            if stale and spec=="W.2" and s==0: rec["code"]={"x":1}
            if spec=="W.1": rec["check"]=w1
            (d/f"{arm}_{spec}_s{s}.json").write_text(json.dumps(rec))
def run(**dB):
    d=Path(tempfile.mkdtemp())
    write(d,"shipped",[1,1,1,0,1],5,0)          # base: all dehydration, share 1.0
    write(d,"dB",**dB) if dB else None
    return d, T.tally(d)
_,r=run(g2s=[1,1,1,1,1],hyd=3,star=2);           assert r["verdict"]=="dB SHIPS", r
_,r=run(g2s=[1,1,1,1,1],hyd=5,star=0);           assert r["verdict"]=="none ships" and "max_cause_share not lower" in r["dB"]["lost_on"], r
_,r=run(g2s=[1,1,1,0,0],hyd=3,star=2);           assert r["verdict"]=="none ships" and any("gate 2" in x for x in r["dB"]["lost_on"]), r
_,r=run(g2s=[1,1,1,1,1],hyd=3,star=2,w1=False);  assert r["verdict"]=="none ships", r
_,r=run(g2s=[1,1,1,1,1],hyd=3,star=2,drinks=2);  assert "WORLD-BREAKING" in r["dB"]["void"] and r["verdict"]=="none ships", r
_,r=run(g2s=[1,1,1,1,1],hyd=3,star=2,stale=True);assert r["verdict"] is None and r["dB"]["stale"]==["W.2_s0"], r
_,r=run();                                        assert r["verdict"] is None
print("6/6 + incomplete OK")
