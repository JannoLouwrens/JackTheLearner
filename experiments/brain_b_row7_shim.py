"""Run pomdp-baselines' policies/main.py UNMODIFIED. Launcher-only shims (declared):
(1) future_fstrings codec registered (their files declare that coding);
(2) gym.make of LunarLander-* or *Bullet* ids -> CartPole placeholder: envs/pomdp/__init__.py builds
    LunarLander eagerly at import to register LunarLander-P/V; Box2D has no aarch64 wheel;
    the Pendulum-V row never touches it;
(3) an empty `pybullet_envs` module: imported only to register the Bullet suite, unused here."""
import sys, runpy
import future_fstrings; future_fstrings.register()
import types
sys.modules.setdefault("pybullet_envs", types.ModuleType("pybullet_envs"))  # (3) bullet suite unused by Pendulum-V
import gym
_make = gym.make
def make(id, *a, **k):
    if str(id).startswith("LunarLander") or "Bullet" in str(id):
        return _make("CartPole-v1")
    return _make(id, *a, **k)
gym.make = make
sys.argv = ["policies/main.py"] + sys.argv[1:]
runpy.run_path("policies/main.py", run_name="__main__")

# HOW IT WAS RUN (11-18 17:28, JACK Next-up 2, box smoke, NOT a reading):
#   pomdp-baselines @ e7c19c32 cloned to /data/vendor_src/pomdp-baselines (cwd);
#   interpreter /data/venvs/sb3-17 (torch 1.12, gym 0.21 = the authors' gym pin);
#   extra pure deps in /data/scratch_row7/pylib (future-fstrings, ruamel.yaml<0.18,
#   absl-py, psutil, tensorboardX<2.6, protobuf<4, scikit-learn<1.4 + scipy, seaborn<0.13,
#   pycolab; numpy copies deleted so the venv's numpy 1.23.1 is used);
#   PYTHONPATH=$PWD:/data/scratch_row7/pylib nice -n 19 python <this> --cfg rnn.yml(num_iters 3, cuda -1) --seed 0
# Result: imports + env + the 5 random init episodes run (returns -1074..-1348); the first
# training iteration (200 updates on whole 200-step LSTM sequences, batch 32) did NOT log
# within 14.5 min at load ~5.5, 4 torch threads -> > ~4 s/update here (PRIORS said ~0.3 s).
# Untested: torch.set_num_threads(1); Kaggle speed.
# 17:45 re-time: OMP/MKL threads 1, PYTHONUNBUFFERED=1, num_iters 1: no iteration in 8 min (7m52s CPU) -> >= 2.3 CPU-s/update.
