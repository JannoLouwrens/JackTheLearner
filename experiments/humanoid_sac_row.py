"""J10 (2') — the one-kernel walk row: SBX SAC, the RL Zoo's Humanoid entry, Humanoid-v4.

PUBLISHED_PRIORS §16 (J10 question): the cheapest published walk. Humanoid-v4
at 1M steps, 10 seeds (SimBaV2's table): SAC 4,825 [3,784-5,866]. At 2M on v3
(SB3 Zoo, ONE model, 150 episodes): SAC 6,232 +- 280. On Gymnasium Humanoid a
return near 5,000 is STANDING (5 a step x 1,000 steps), so the row also demands
forward metres.

RULE (frozen in this commit, BEFORE the kernel body exists and before seed 0):

  env        Gymnasium `Humanoid-v4`, every default (venv /data/venvs/jack-std:
             gymnasium 1.4.0, mujoco 3.17.0, sbx 0.28.0, jax 0.11.2), one env.
  learner    sbx.SAC with the Zoo's Humanoid entry: n_timesteps 2e6,
             MlpPolicy, learning_starts 10,000; everything else the SB3 SAC
             defaults, SET EXPLICITLY because SBX's defaults are not SB3's:
             lr 3e-4, buffer 1e6, batch 256, tau 0.005, gamma 0.99,
             train_freq 1, gradient_steps 1, ent_coef "auto", net [256, 256].
             VERIFIED 11-18 10:15 (before seed 0): DLR-RM/rl-baselines3-zoo
             @ d802099a (sac.yml last touched b8a11d6e, 2026-10-16), file
             sha256 b05466c5..., `Humanoid-v4: <<: *mujoco-defaults,
             n_timesteps 2e6`; mujoco-defaults = MlpPolicy, learning_starts
             10000; no normalize, no wrapper. Matches the block above. SBX
             0.28 hidden defaults read from source and equal SB3's for the
             non-SDE path: log_std clamp [-20, 2], ent_coef init 1.0,
             target_entropy auto, n_critics 2, net [256, 256], policy_delay 1.
  differs from the published recipe (all listed): SBX (JAX) not SB3 (torch),
             same algorithm; the 2M figure is on v3, ours on v4 (v4 is the
             1M band's env); gymnasium 1.4 / mujoco 3.17 vs the papers'
             versions; Kaggle CPU, not the authors' hardware.
  seeds      0, 1, 2 (one chained run each); extension none.
  eval       at 1M and at 2M: 10 deterministic episodes, env seeds
             10_000..10_009, separate eval env. Per seed: mean return R and
             forward metres per 1,000 steps M = (final x - initial x)
             summed over the 10 episodes / steps taken x 1,000, x = the
             CENTRE OF MASS x (`info["x_position"]` on v4 is mass_center,
             not the torso; corrected 11-18 before seed 0 — the earlier
             "torso" contradicted the named source). Initial x = mass_center
             right after reset, same function.
  PASS       at 2M: R >= 5,950 (6,232 - 280) AND M >= 12.0, on >= 2 of 3
             seeds. Return alone is "STANDS", never "walks".
  1M check   median R over 3 seeds inside [3,784, 5,866]. Outside it the row
             is a NAMED MISS of the published curve (reported beside the 2M
             reading, whatever that is) and the cause is looked for before
             (3) uses the recipe.
  control    late start: same seed 0, learning_starts = 2,000,001 (never
             trains), 50k steps then the same eval; must stay R < 500. If it
             does not, the row is VOID (the eval or the reward is broken).
  resume     0.046 CPU-s a step on the box -> 2M ~ 26 CPU-h: does NOT fit
             one 12 h kernel. Each kernel trains until 10.5 h wall or the
             next 500k mark, then saves model + replay buffer
             (`save_replay_buffer`, ~3.1 GB at 1e6 x 376-dim obs) to its
             output; the next kernel attaches the previous one as a kernel
             source, loads both, and continues with reset_num_timesteps=False.
             The step counter, the ent_coef state and the RNG of the policy
             come from the checkpoint; the env is re-seeded with
             seed*1000 + segment (disclosed: env RNG is not carried across).
             A segment that dies without a checkpoint is re-run from the last
             checkpoint, never skipped. Evals happen only at exactly 1M and 2M.
  budget     Kaggle CPU only (box: nice 19, the 50k control smoke at most).

STATUS: rule + eval + late-start control (`evaluate`, `control_run`; box smoke
11-18) + `train_segment` (resume as the rule says; box resume test
`resume_check`, 11-18) + Kaggle wrapper `kernel` (segment n > 0 mounts
segment n-1's kernel via gpu.py `kernel_sources`; the /kaggle/input/<slug>
mount path is unverified until the first segment 1 runs; `resolve_prev_dir`
(11-19) finds the checkpoint at the rule's path or the ONE deeper mount
carrying the slug, else fails fast listing what is mounted).
"""
from __future__ import annotations

ENV_ID = "Humanoid-v4"
SEEDS = (0, 1, 2)
TOTAL_STEPS = 2_000_000
CHECK_STEPS = 1_000_000
SAC_KW = dict(learning_rate=3e-4, buffer_size=1_000_000, learning_starts=10_000,
              batch_size=256, tau=0.005, gamma=0.99, train_freq=1,
              gradient_steps=1, ent_coef="auto",
              policy_kwargs=dict(net_arch=[256, 256]))
EVAL_SEEDS = tuple(range(10_000, 10_010))
PASS_RETURN = 5_950.0
PASS_METRES_PER_1000 = 12.0
PASS_MIN_SEEDS = 2
CHECK_BAND_1M = (3_784.0, 5_866.0)
CONTROL_LEARNING_STARTS = TOTAL_STEPS + 1
CONTROL_STEPS = 50_000
CONTROL_MAX_RETURN = 500.0
SEGMENT_WALL_H = 10.5
SEGMENT_STEPS = 500_000


def verdict(r2m: dict, m2m: dict, r1m: dict, control_r: float) -> dict:
    """Pure reading of the frozen rule. r2m/m2m/r1m: seed -> value."""
    if control_r >= CONTROL_MAX_RETURN:
        return {"reading": "VOID", "why": f"late-start control R={control_r:.1f} >= {CONTROL_MAX_RETURN}"}
    walks = [s for s in SEEDS if r2m.get(s, -1e9) >= PASS_RETURN and m2m.get(s, -1e9) >= PASS_METRES_PER_1000]
    stands = [s for s in SEEDS if r2m.get(s, -1e9) >= PASS_RETURN]
    vals = sorted(r1m[s] for s in SEEDS if s in r1m)
    med = vals[len(vals) // 2] if len(vals) == 3 else None
    in_band = med is not None and CHECK_BAND_1M[0] <= med <= CHECK_BAND_1M[1]
    reading = "PASS" if len(walks) >= PASS_MIN_SEEDS else (
        "STANDS" if len(stands) >= PASS_MIN_SEEDS else "FAIL")
    return {"reading": reading, "walk_seeds": walks, "stand_seeds": stands,
            "median_1m": med, "check_1m": "IN BAND" if in_band else "NAMED MISS"}


def make_env():
    import gymnasium as gym
    return gym.make(ENV_ID)


def _com_x(env) -> float:
    from gymnasium.envs.mujoco.humanoid_v4 import mass_center
    u = env.unwrapped
    return float(mass_center(u.model, u.data)[0])


def evaluate(policy_fn, seeds=EVAL_SEEDS) -> dict:
    """The rule's eval: one episode per env seed on a SEPARATE env.
    policy_fn(obs) -> action (deterministic). Returns mean R, M and per-episode rows."""
    env = make_env()
    eps, metres, steps = [], 0.0, 0
    for sd in seeds:
        obs, _ = env.reset(seed=int(sd))
        x0, ret, n, done, info = _com_x(env), 0.0, 0, False, {}
        while not done:
            obs, r, term, trunc, info = env.step(policy_fn(obs))
            ret += float(r); n += 1; done = term or trunc
        dx = float(info["x_position"]) - x0
        eps.append({"seed": int(sd), "R": ret, "steps": n, "dx": dx, "fell": bool(term)})
        metres += dx; steps += n
    env.close()
    return {"R": sum(e["R"] for e in eps) / len(eps),
            "M": metres / max(steps, 1) * 1000.0, "episodes": eps}


def control_run(seed: int = 0, steps: int = CONTROL_STEPS) -> dict:
    """Late-start control: the Zoo entry with learning_starts past the end,
    `steps` env steps (never trains), then the rule's eval of its policy."""
    import time
    from sbx import SAC
    kw = dict(SAC_KW, learning_starts=CONTROL_LEARNING_STARTS)
    t0 = time.time()
    model = SAC("MlpPolicy", make_env(), seed=seed, verbose=0, **kw)
    model.learn(total_timesteps=steps)
    t_learn = time.time() - t0
    ev = evaluate(lambda o: model.predict(o, deterministic=True)[0])
    return {"seed": seed, "steps": steps, "learn_s": t_learn, **ev,
            "trained": int(model._n_updates) if hasattr(model, "_n_updates") else None}


def _ckpt_paths(d):
    import os
    return os.path.join(d, "model.zip"), os.path.join(d, "replay.pkl")


class _WallStop:
    """SB3 callback factory: stop learn() once `deadline` (time.time()) passes."""
    def __new__(cls, deadline):
        import time
        from stable_baselines3.common.callbacks import BaseCallback

        class Stop(BaseCallback):
            def _on_step(self) -> bool:
                return time.time() < deadline
        return Stop()


def train_segment(seed: int, out_dir: str, prev_dir: str | None = None, segment: int = 0,
                  wall_h: float = SEGMENT_WALL_H, seg_steps: int = SEGMENT_STEPS,
                  total: int = TOTAL_STEPS, eval_at=(CHECK_STEPS, TOTAL_STEPS),
                  sac_kw: dict | None = None, keep: list | None = None) -> dict:
    """One resume segment of the rule. Fresh model if prev_dir is None, else
    loads model + replay buffer from prev_dir. Trains until the next seg_steps
    mark (capped at `total`) or wall_h, saves model + buffer to out_dir, and
    runs the rule's eval only when num_timesteps lands EXACTLY on an eval_at
    mark. The env is re-seeded seed*1000 + segment (disclosed in the rule)."""
    import json, os, time
    from sbx import SAC
    t0 = time.time()
    os.makedirs(out_dir, exist_ok=True)
    env = make_env()
    env.reset(seed=seed * 1000 + segment)
    if prev_dir is None:
        model = SAC("MlpPolicy", env, seed=seed, verbose=0, **(sac_kw or SAC_KW))
    else:
        mp, rp = _ckpt_paths(prev_dir)
        model = SAC.load(mp, env=env)
        model.load_replay_buffer(rp)
    start = int(model.num_timesteps)
    target = min(total, (start // seg_steps + 1) * seg_steps)
    # SB3 reads total_timesteps as an increment when reset_num_timesteps=False
    model.learn(total_timesteps=target - start, reset_num_timesteps=prev_dir is None,
                callback=_WallStop(t0 + wall_h * 3600.0))
    end = int(model.num_timesteps)
    mp, rp = _ckpt_paths(out_dir)
    model.save(mp)
    model.save_replay_buffer(rp)
    row = {"seed": seed, "segment": segment, "start": start, "end": end, "target": target,
           "hit_target": end == target, "buffer_size": int(model.replay_buffer.size()),
           "n_updates": int(getattr(model, "_n_updates", -1)), "train_s": time.time() - t0}
    if end in tuple(eval_at):
        ev = evaluate(lambda o: model.predict(o, deterministic=True)[0])
        row["eval"] = {"at": end, "R": ev["R"], "M": ev["M"], "episodes": ev["episodes"]}
    with open(os.path.join(out_dir, "segment.json"), "w") as f:
        json.dump(row, f, indent=1)
    if keep is not None:
        keep.append(model)
    return row


def resume_check(work: str, steps: int = 10_000) -> dict:
    """Box test of the resume path at toy size (learning_starts 1,000 so both
    halves train): segment 0 to `steps`, segment 1 resumed to 2*`steps`.
    Checks: step counter continues, buffer carried, actor/critic/ent_coef
    params + actor optimiser state equal in
    memory at the end of segment 0 and after load, updates continue in segment 1.
    Control: a fresh (not loaded) model's params differ from the checkpoint."""
    import os
    import numpy as np, jax
    from sbx import SAC
    kw = dict(SAC_KW, learning_starts=1_000, buffer_size=4 * steps)
    a, b = os.path.join(work, "seg0"), os.path.join(work, "seg1")
    held = []
    r0 = train_segment(0, a, None, 0, seg_steps=steps, total=2 * steps, eval_at=(), sac_kw=kw,
                       keep=held)

    def flat(m):
        leaves = jax.tree_util.tree_leaves((m.policy.actor_state.params, m.policy.qf_state.params,
                                            m.policy.qf_state.target_params, m.ent_coef_state.params,
                                            m.policy.actor_state.opt_state))
        return np.concatenate([np.asarray(x, dtype=np.float64).ravel() for x in leaves])
    # what was saved vs what loads back
    m_saved = SAC.load(_ckpt_paths(a)[0], env=make_env())
    m_fresh = SAC("MlpPolicy", make_env(), seed=0, verbose=0, **kw)
    load_equal = bool(np.array_equal(flat(m_saved), flat(held[0])))
    control_differs = not np.array_equal(flat(m_saved), flat(m_fresh))
    ent0 = float(np.exp(np.asarray(m_saved.ent_coef_state.params["log_ent_coef"])))
    r1 = train_segment(0, b, a, 1, seg_steps=steps, total=2 * steps, eval_at=(), sac_kw=kw)
    ok = (r0["end"] == steps and r1["start"] == steps and r1["end"] == 2 * steps
          and r1["buffer_size"] == 2 * steps and r1["n_updates"] > r0["n_updates"] > 0
          and load_equal and control_differs and abs(ent0 - 1.0) > 1e-3)
    return {"seg0": r0, "seg1": r1, "load_equal": load_equal, "control_differs": control_differs,
            "ent_coef_after_seg0": ent0, "ok": ok}



# ---- Kaggle kernel wrapper (J10 slot 11-18 10:45) ----------------------------
KERNEL_PINS = ["stable-baselines3==2.9.2", "sbx-rl==0.28.0", "gymnasium==1.4.0",
               "mujoco==3.17.0", "jax==0.11.2", "jaxlib==0.11.2"]
KERNEL_REMOTE_TIMEOUT_S = int(11.75 * 3600)
SEGMENT_FETCH = ["segment.json"]  # never the ~6.1 GB model+replay checkpoint

_JOB = r'''
import subprocess as _s, sys as _y
_s.run([_y.executable, "-m", "pip", "install", "-q"] + __PINS__, check=True)
import json, os
os.chdir("/tmp/jack")
from experiments.humanoid_sac_row import train_segment, resolve_prev_dir
out = os.environ["JACK_OUT"]
_p = __PREV__
_p = resolve_prev_dir(_p) if _p else None
print("PREV_DIR " + repr(_p), flush=True)
r = train_segment(__SEED__, out, _p, __SEGMENT__)
print("RESULT " + json.dumps(r), flush=True)
'''


def resolve_prev_dir(expected: str, root: str = "/kaggle/input") -> str:
    """Where the previous segment's checkpoint is actually mounted. The rule's
    path is /kaggle/input/<slug>; Kaggle has also mounted kernel sources one or
    more levels deeper (e.g. .../notebooks/<owner>/<slug>), and that path was
    never verified (J10 11-19). Returns `expected` if it holds model.zip AND
    replay.pkl; else the ONE directory under `root` holding both whose path
    contains the slug; else raises with what IS mounted, so a wrong mount costs
    one pip install, never a fresh-start segment mislabelled as a resume."""
    import os
    need = ("model.zip", "replay.pkl")
    if all(os.path.isfile(os.path.join(expected, n)) for n in need):
        return expected
    slug = os.path.basename(expected.rstrip("/"))
    hits, seen = [], []
    for d, _dirs, files in os.walk(root):
        seen.append(d)
        if all(n in files for n in need) and slug in d.split(os.sep):
            hits.append(d)
    if len(hits) == 1:
        return hits[0]
    raise FileNotFoundError(f"checkpoint for {slug!r}: {len(hits)} candidates {hits} "
                            f"under {root}; dirs seen: {seen[:40]}")


def segment_job_body(seed: int, segment: int, prev_dir: str | None = None) -> str:
    """The job text for one resume segment. Segment 0 trains fresh; a later
    segment loads model + replay from `prev_dir` (the previous segment's kernel
    output, mounted as a kernel source under /kaggle/input)."""
    if seed not in SEEDS:
        raise ValueError(f"seed {seed} is not in the frozen rule's seeds {SEEDS}")
    if (segment == 0) != (prev_dir is None):
        raise ValueError("segment 0 starts fresh (no prev_dir); every later segment needs one")
    return (_JOB.replace("__PINS__", repr(KERNEL_PINS)).replace("__SEED__", repr(int(seed)))
            .replace("__PREV__", repr(prev_dir)).replace("__SEGMENT__", repr(int(segment))))


def kernel(seed: int, segment: int = 0) -> dict:
    """Submit one resume segment as a Kaggle CPU kernel; write segment.json to
    /data/jack-logs/j10/sac_row/s<seed>_seg<segment>.json.

    Segment n > 0 mounts segment n-1's kernel (named in its record's
    "kernel" key) as a kernel source and resumes from /kaggle/input/<slug>;
    REFUSES if that record is missing. Only segment.json is downloaded
    (`fetch_only`), never the ~6.1 GB checkpoint."""
    from pathlib import Path
    import json
    from experiments.gpu import build_job, result_json, submit
    out = Path("/data/jack-logs/j10/sac_row")
    sources, prev_dir = None, None
    if segment > 0:
        prev = out / f"s{seed}_seg{segment - 1}.json"
        src = json.loads(prev.read_text()).get("kernel") if prev.exists() else None
        if not src:
            raise RuntimeError(f"segment {segment} needs {prev} with a 'kernel' key "
                               f"(the previous segment's Kaggle kernel)")
        sources, prev_dir = [src], f"/kaggle/input/{src.split('/', 1)[1]}"
    sub = submit(build_job(segment_job_body(seed, segment, prev_dir)), prefer="kaggle-cpu",
                 est_hours=10.5, timeout_s=KERNEL_REMOTE_TIMEOUT_S, fetch=SEGMENT_FETCH,
                 kernel_sources=sources, fetch_only=True)
    r = result_json(sub, "segment.json")
    r["kernel"] = sub.job_id
    out.mkdir(parents=True, exist_ok=True)
    (out / f"s{seed}_seg{segment}.json").write_text(json.dumps(r, indent=1) + "\n")
    return r

if __name__ == "__main__":
    import json, sys
    if sys.argv[1:2] == ["control"]:
        n = int(sys.argv[2]) if len(sys.argv) > 2 else CONTROL_STEPS
        r = control_run(0, n)
        print(json.dumps({k: v for k, v in r.items() if k != "episodes"}))
        print("VOID" if r["R"] >= CONTROL_MAX_RETURN else "control holds (R < 500)")
    elif sys.argv[1:2] == ["resume_check"]:
        r = resume_check(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 10_000)
        print(json.dumps(r, indent=1))
        print("RESUME OK" if r["ok"] else "RESUME BROKEN")
    elif sys.argv[1:2] == ["kernel"]:
        print(json.dumps(kernel(int(sys.argv[2]), int(sys.argv[3]) if len(sys.argv) > 3 else 0)))
