# Rung 13b-imagloss first-read diagnosis (11-25): where the port misses in_rew, compare both f32 decodes to an f64 decode of the same logits.
# Run: nice -n 19 /data/venvs/jack-mlx/bin/python experiments/brain_a/rung13b_decode_probe.py
import sys,os,numpy as np
sys.path.insert(0,'/home/opc/jackthelearner/experiments/brain_a'); sys.argv=['x']
import rung13b as r
import mlx.core as mx, mlx_maths as M
G,R=np.load(r.GOLDEN_A),np.load(r.R13); b=M.bins(255)
for c in (6,8):
    P=lambda h:{k[len(f"c{c}/pre/"):]:mx.array(R[k]) for k in R.files if k.startswith(f"c{c}/pre/{h}/")}
    g=f"c{c}/imag/"
    d=np.concatenate([G[g+"start/deter"][:,None],G[g+"feat/deter"]],1); s=np.concatenate([G[g+"start/stoch"][:,None],G[g+"feat/stoch"]],1)
    x=M.feat2tensor(mx.array(d),mx.array(s))
    hr=np.array(M.twohot_decode(M._mlp_head(P("rew"),"rew",x,"logits"),b)); hc=np.array(mx.sigmoid(M._mlp_head(P("con"),"con",x,"logit")[...,0]))
    gr,gc=G[f"c{c}/imagloss/in_rew"],G[f"c{c}/imagloss/in_con"]
    for nm,a,bb in (("rew",hr,gr),("con",hc,gc)):
        e=np.abs(a-bb)/(np.abs(bb)+1); idx=np.argwhere(e>1e-5)
        print(c,nm,len(idx),"t:",np.unique(idx[:,1]) if len(idx) else None,"rows:",np.unique(idx[:,0])[:10] if len(idx) else None)
        for i,t in idx[:4]: print("   ",i,t,a[i,t],bb[i,t])
    print(c,"golden con unique-ish", np.unique(np.round(gc,3))[:10], "is golden con==1 at t0?", np.allclose(gc[:,0],1), "in_rew t0 sample", gr[:3,0], hr[:3,0])
print("---- f64 decode of the port's logits at missed elements")
bb=np.array(b).astype(np.float64)
for c in (6,8):
    P=lambda h:{k[len(f"c{c}/pre/"):]:mx.array(R[k]) for k in R.files if k.startswith(f"c{c}/pre/{h}/")}
    g=f"c{c}/imag/"
    d=np.concatenate([G[g+"start/deter"][:,None],G[g+"feat/deter"]],1); s=np.concatenate([G[g+"start/stoch"][:,None],G[g+"feat/stoch"]],1)
    x=M.feat2tensor(mx.array(d),mx.array(s))
    L=np.array(M._mlp_head(P("rew"),"rew",x,"logits")).astype(np.float64)
    p=np.exp(L-L.max(-1,keepdims=True)); p/=p.sum(-1,keepdims=True)
    m=127; v64=(p[...,m]*bb[m])+((p[...,m+1:]-p[...,:m][...,::-1])*bb[m+1:]).sum(-1)
    hr=np.array(M.twohot_decode(mx.array(L.astype(np.float32)),b)); gr=G[f"c{c}/imagloss/in_rew"]
    e=np.abs(hr-gr)/(np.abs(gr)+1); idx=np.argwhere(e>1e-5)
    e64p=np.abs(hr-v64)/(np.abs(v64)+1); e64g=np.abs(gr-v64)/(np.abs(v64)+1)
    print(c,"missed",len(idx),"| port vs f64 >1e-5 (all elems):",(e64p>1e-5).sum(),"| golden vs f64 >1e-5:",(e64g>1e-5).sum())
    pf=p.astype(np.float32); top=pf[...,-1]
    print("   top-bin prob at missed elems:",top[tuple(idx[:3].T)], "ulp*bin:", np.spacing(top[tuple(idx[:3].T)])*bb[-1])
    for i,t in idx[:4]: print("   ",i,t,"port",hr[i,t],"golden",gr[i,t],"f64",v64[i,t])
