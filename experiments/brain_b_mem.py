"""Brain B with a memory (ORDERS J29, PLAN part "His brain", stage 1, step 2 COMBINE).

Published design built from: Ni, Eysenbach & Salakhutdinov 2022, "Recurrent model-free RL can
be a strong baseline for many POMDPs" (arXiv 2110.05038), code `pomdp-baselines@e7c19c32`
(MIT, (c) 2021-22 Tianwei Ni; cloned at /data/vendor_src/pomdp-baselines). CODE ONLY, no
weights. This is a port, not a home-made learner: the network layout below is the authors'
`policies/models/recurrent_actor.py` + `recurrent_critic.py` + `policies/rl/sac.py`, rewritten
in torch 2.15 + gymnasium because their pins (torch 1.12, gym 0.21) live only in
/data/venvs/sb3-17 and their own main.py measured >= 2.3 CPU-s per update on this box
(experiments/brain_b_row7_shim.py, 11-18). Why not a library: under jack-std, sb3_contrib 2.9.2
has only RecurrentPPO and sbx 0.28 has no recurrent SAC (checked 11-27).

Task: Pendulum-V = gymnasium Pendulum-v1 observing only index 2 (angular velocity), 200 steps,
as `envs/pomdp/__init__.py` registers it (partially_obs_dims=[2]).

OURS BESIDE THEIRS (configs/pomdp/pendulum/v/rnn.yml + the model files):
  setting                      theirs                            ours
  seq model / separate         LSTM, separate actor & critic     same
  embeddings obs/act/rew       32 / 8 / 8, ReLU FeatureExtractor same (Linear+ReLU)
  rnn hidden / layers          128 / 1, orthogonal W, zero b     same
  shortcut embedder            critic: (o,a)->48; actor: o->32   same
  heads                        128-128 Q1,Q2 and policy          same
  lr / alpha lr / gamma / tau  3e-4 / 3e-4 / 0.9 / 0.005         same
  entropy                      auto, target -|A|                 same
  batch / seq len              32 whole 200-step episodes        same
  updates per env step         1 (per 200-step rollout: 200)     same
  init / iters                 5 random episodes + 250 x 200     same (51,000 steps)
  eval                         20 deterministic episodes         same
  framework / env lib          torch 1.12, gym 0.21 Pendulum     torch 2.15, gymnasium 1.4 Pendulum-v1  (DIFFERS)
  replay buffer                seq_replay_buffer_vanilla         whole-episode tensors in a list (Pendulum
                                                                 episodes are all exactly 200 -> no masks)
  init                         Mlp fanin/0.1, last +-3e-3, pi +-1e-3  same; embedders = torch default (theirs: plain nn.Linear too)
  CPU threads                  default                           1 (box is shared; declared)
Any other difference found later is a bug in the port, not a choice.
"""
import argparse, json, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import gymnasium as gym

torch.set_num_threads(1)
CFG = dict(obs_emb=32, act_emb=8, rew_emb=8, rnn=128, layers=(128, 128), lr=3e-4, alpha_lr=3e-4,
           gamma=0.9, tau=0.005, batch=32, init_eps=5, iters=250, ep_len=200, eval_eps=20)
LOG_STD_MIN, LOG_STD_MAX = -20, 2


class PendulumV(gym.Wrapper):
    def __init__(self):
        super().__init__(gym.make("Pendulum-v1"))
        self.observation_space = gym.spaces.Box(-8.0, 8.0, (1,), np.float32)

    def reset(self, **k):
        o, i = self.env.reset(**k); return o[2:3].astype(np.float32), i

    def step(self, a):
        o, r, te, tr, i = self.env.step(2.0 * np.asarray(a))   # their NormalizedBoxEnv: [-1,1] -> [-2,2]
        return o[2:3].astype(np.float32), float(r), te, tr, i


def fe(i, o):
    return nn.Sequential(nn.Linear(i, o), nn.ReLU())


def _their_init(lin, last_w=None):
    """torchkit/networks.py Mlp: hidden fanin_init (U +-1/sqrt(fan_in)), bias 0.1; last layer U +-init_w."""
    with torch.no_grad():
        if last_w is None:
            b = 1.0 / np.sqrt(lin.weight.shape[1]); lin.weight.uniform_(-b, b); lin.bias.fill_(0.1)
        else:
            lin.weight.uniform_(-last_w, last_w); lin.bias.uniform_(-last_w, last_w)
    return lin


def mlp(i, hs, o, init_w=3e-3):
    L, d = [], i
    for h in hs:
        L += [_their_init(nn.Linear(d, h)), nn.ReLU()]; d = h
    return nn.Sequential(*L, _their_init(nn.Linear(d, o), init_w) if o else nn.Identity())


class Encoder(nn.Module):
    """(prev action, reward, obs) embeddings -> LSTM, zero initial state (their get_hidden_states)."""
    def __init__(self, od, ad, c):
        super().__init__()
        self.ea, self.er, self.eo = fe(ad, c["act_emb"]), fe(1, c["rew_emb"]), fe(od, c["obs_emb"])
        self.inp = c["act_emb"] + c["rew_emb"] + c["obs_emb"]
        self.rnn = nn.LSTM(self.inp, c["rnn"])
        for n, p in self.rnn.named_parameters():
            nn.init.constant_(p, 0) if "bias" in n else nn.init.orthogonal_(p)

    def forward(self, pa, r, o, h=None):
        x = torch.cat([self.ea(pa), self.er(r), self.eo(o)], -1)
        return self.rnn(x, h)


class Critic(nn.Module):
    def __init__(self, od, ad, c):
        super().__init__()
        self.enc = Encoder(od, ad, c)
        self.sc = fe(od + ad, self.enc.inp)
        self.q1 = mlp(c["rnn"] + self.enc.inp, c["layers"], 1)
        self.q2 = mlp(c["rnn"] + self.enc.inp, c["layers"], 1)

    def forward(self, pa, r, o, a):
        h, _ = self.enc(pa, r, o)
        z = torch.cat([h, self.sc(torch.cat([o, a], -1))], -1)
        return self.q1(z), self.q2(z)


class Actor(nn.Module):
    def __init__(self, od, ad, c):
        super().__init__()
        self.enc = Encoder(od, ad, c)
        self.sc = fe(od, c["obs_emb"])
        self.body = mlp(c["rnn"] + c["obs_emb"], c["layers"], 0)          # hidden 128-128, ReLU
        self.mu = _their_init(nn.Linear(c["layers"][-1], ad), 1e-3)      # TanhGaussianPolicy init_w 1e-3
        self.ls = _their_init(nn.Linear(c["layers"][-1], ad), 1e-3)

    def forward(self, pa, r, o, h=None, deterministic=False):
        z, h = self.enc(pa, r, o, h)
        x = self.body(torch.cat([z, self.sc(o)], -1))
        mu, ls = self.mu(x), self.ls(x).clamp(LOG_STD_MIN, LOG_STD_MAX)
        if deterministic:
            return torch.tanh(mu), None, h
        d = torch.distributions.Normal(mu, ls.exp())
        u = d.rsample(); a = torch.tanh(u)
        lp = (d.log_prob(u) - torch.log(1 - a.pow(2) + 1e-6)).sum(-1, keepdim=True)
        return a, lp, h


class BrainBMem:
    def __init__(self, seed=0, c=CFG):
        torch.manual_seed(seed); self.rng = np.random.default_rng(seed); self.c = c
        od, ad = 1, 1
        self.actor, self.critic, self.critic_t = Actor(od, ad, c), Critic(od, ad, c), Critic(od, ad, c)
        self.critic_t.load_state_dict(self.critic.state_dict())
        self.log_alpha = torch.zeros(1, requires_grad=True); self.target_ent = -float(ad)
        self.oa = torch.optim.Adam(self.actor.parameters(), c["lr"])
        self.oc = torch.optim.Adam(self.critic.parameters(), c["lr"])
        self.oal = torch.optim.Adam([self.log_alpha], c["alpha_lr"])
        self.eps = []   # each: dict of (T+1,1,dim) pa/r/o and (T,1,1) a, done

    def update(self):
        """One SAC update on a batch of whole episodes (their policy_rnn.forward, misalignment kept)."""
        idx = self.rng.integers(len(self.eps), size=self.c["batch"])
        B = {k: torch.cat([self.eps[i][k] for i in idx], 1) for k in ("pa", "r", "o", "a", "d")}
        pa, r, o, a, d = B["pa"], B["r"], B["o"], B["a"], B["d"]       # pa,r,o: T+1 ; a,d: T
        alpha = self.log_alpha.exp().detach()
        with torch.no_grad():
            na, nlp, _ = self.actor(pa, r, o)                            # actions at every o[t], t=0..T
            q1t, q2t = self.critic_t(pa, r, o, na)
            v = torch.min(q1t, q2t) - alpha * nlp
            y = r[1:] + (1 - d) * self.c["gamma"] * v[1:]
        q1, q2 = self.critic(pa, r, o, torch.cat([a, a[-1:]], 0))       # pad last; dropped below
        lc = F.mse_loss(q1[:-1], y) + F.mse_loss(q2[:-1], y)
        self.oc.zero_grad(); lc.backward(); self.oc.step()
        na, nlp, _ = self.actor(pa, r, o)
        q1p, q2p = self.critic(pa, r, o, na)
        la = (alpha * nlp - torch.min(q1p, q2p))[:-1].mean()
        self.oa.zero_grad(); la.backward(); self.oa.step()
        lal = -(self.log_alpha * (nlp[:-1].detach() + self.target_ent)).mean()
        self.oal.zero_grad(); lal.backward(); self.oal.step()
        with torch.no_grad():
            for p, pt in zip(self.critic.parameters(), self.critic_t.parameters()):
                pt.mul_(1 - self.c["tau"]).add_(self.c["tau"] * p)

    def rollout(self, env, random=False, deterministic=False, learn=True):
        o, _ = env.reset(seed=int(self.rng.integers(2**31)))
        pa, rr, oo, aa, dd = [np.zeros(1, np.float32)], [np.zeros(1, np.float32)], [o], [], []
        h, ret = None, 0.0
        for t in range(self.c["ep_len"]):
            if random:
                a = self.rng.uniform(-1, 1, 1).astype(np.float32)
            else:
                with torch.no_grad():
                    x = [torch.as_tensor(v).view(1, 1, -1) for v in (pa[-1], rr[-1], oo[-1])]
                    at, _, h = self.actor(*x, h=h, deterministic=deterministic)
                a = at.view(-1).numpy()
            o, r, te, tr, _ = env.step(a); ret += r
            pa.append(a); rr.append(np.array([r], np.float32)); oo.append(o); aa.append(a)
            dd.append(np.array([float(te)], np.float32))
            if te or tr:
                break
        if learn:
            T = lambda L: torch.as_tensor(np.stack(L)).unsqueeze(1)
            self.eps.append(dict(pa=T(pa), r=T(rr), o=T(oo), a=T(aa), d=T(dd)))
        return ret, len(aa)


def timing(seed, updates):
    """J29 (3): one cell's cost on the box. Not a reading."""
    env, b = PendulumV(), BrainBMem(seed)
    t0 = time.process_time(); w0 = time.time()
    for _ in range(b.c["init_eps"]):
        b.rollout(env, random=True)
    t_env = time.process_time() - t0
    t1 = time.process_time()
    for _ in range(updates):
        b.update()
    per_upd = (time.process_time() - t1) / updates
    t2 = time.process_time(); b.rollout(env); t_pol = time.process_time() - t2
    total = per_upd * 50000 + t_pol * 250 + t_env
    out = dict(seed=seed, updates_timed=updates, cpu_s_per_update=round(per_upd, 4),
               cpu_s_per_policy_episode=round(t_pol, 3), cpu_s_per_1k_steps=round(per_upd * 1000 + t_pol * 5, 1),
               est_cpu_h_51k_one_seed=round(total / 3600, 2), wall_s=round(time.time() - w0, 1),
               threads=torch.get_num_threads(), torch=torch.__version__)
    print(json.dumps(out)); return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["timing"]); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--updates", type=int, default=40)
    a = ap.parse_args()
    timing(a.seed, a.updates)
