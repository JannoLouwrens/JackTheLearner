# Timing probe: per-update CPU cost of Ecological RL's Double DQN at the authors' shapes
# (variants/dqn_lifetime/dqn_medium8_mlp_task_partial_variant.py gen_network; batch 64,
# Adam 1e-4, grad clip 5, from experiments/continual/dynamic_static/axe/tool_dqn.py).
import time, torch, torch.nn as nn
torch.set_num_threads(1); torch.manual_seed(0)
A = 8  # minigrid action count order (authors' ToolsEnv; exact n only shifts the last layer)
def mlp(i, hs, o):
    L=[];d=i
    for h in hs: L+= [nn.Linear(d,h), nn.ReLU()]; d=h
    return nn.Sequential(*L, nn.Linear(d,o))
class Net(nn.Module):
    def __init__(s):
        super().__init__(); s.img=mlp(200,[64,64],32); s.inv=mlp(64,[16,16],16); s.fin=mlp(48,[16,16],A)
    def forward(s,x): return s.fin(torch.cat([s.img(x[:,:200]), s.inv(x[:,200:264])],1))
q, qt = Net(), Net(); opt=torch.optim.Adam(q.parameters(),1e-4)
o=torch.rand(64,264); o2=torch.rand(64,264); a=torch.randint(0,A,(64,1)); r=torch.rand(64,1); d=torch.zeros(64,1)
def upd():
    with torch.no_grad():
        best=q(o2).argmax(1,keepdim=True); y=r+(1-d)*0.99*qt(o2).gather(1,best)
    loss=((q(o).gather(1,a)-y)**2).mean(); opt.zero_grad(); loss.backward()
    nn.utils.clip_grad_value_(q.parameters(),5); opt.step()
for _ in range(200): upd()
t=time.process_time(); N=5000
for _ in range(N): upd()
c=(time.process_time()-t)/N
print(f"cpu_s_per_update {c*1e3:.3f} ms; 1M updates (2000 epochs x 500) = {c*1e6/3600:.2f} CPU-h per seed (updates only)")
