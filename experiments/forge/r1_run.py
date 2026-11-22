"""Round 1's KERNEL (F3 / ORDERS J18): how one arm of the frozen rule is run.

The rule is `rounds/r1/spec.json` (frozen 594752d0). This file adds no term to
it: every learner setting an arm runs with is READ from the spec
(`arm_settings`), a setting this file does not know how to apply is REFUSED,
and the world is E2c's E-food arm through the same `pilot_b2.pilot_world` call
E2c's kernels made (PILOT_RULE's frames, food_floor_only, native_substep).

One code path for all ten arms (baseline, never-trains twin, K1 K2 K3 K5 K6 N3
C1 C2), `run_arm`: `pilot_b.run_b`'s body with the learner built from the
spec. `experiments/tests/test_forge_r1_run.py` proves the baseline through
here is `pilot_b2.run_pilot_cell("E-food")` life for life and weight for
weight, and that each candidate's change reaches the learner.

What a kernel returns is RAW (`/data/jack-logs/forge/r1/raw/`). Nothing here
writes `results.jsonl`: J18 accepts no result before ORDERS S13 (b)'s
judge-hash tripwire holds, and scoring is the reader's job.

Every kernel of a rung runs ONE commit (`--pin`): the job checks it out after
the launcher's clone, and each record carries the sha it actually ran
(`forge.code_head`), so kernels pushed hours apart cannot run different code.

    python -m experiments.forge.r1_run groups --rung S
    python -m experiments.forge.r1_run chain --rung S --names g0 g2 g4 g6 --pin <sha>
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
SPEC_PATH = Path(__file__).resolve().parent / "rounds" / "r1" / "spec.json"
OUT = Path("/data/jack-logs/forge/r1/raw")
WORLD_ARM = "E-food"          # pilot_b2.E2C_ARMS: arm K + handed 'food', cossin
PER_KERNEL = 4                # E2c's packing: 4 runs on Kaggle's 4 cores
REMOTE_TIMEOUT_S = int(9.5 * 3600)
EST_HOURS = {"S": 4.0, "M": 7.5}
GUARD_RETRIES, GUARD_WAIT_S = 18, 600   # another lane's dirt: wait, never bypass


class Refused(RuntimeError):
    pass


def load_spec(path=None) -> dict:
    return json.loads(Path(path or SPEC_PATH).read_text())


def spec_sha256(path=None) -> str:
    return hashlib.sha256(Path(path or SPEC_PATH).read_bytes()).hexdigest()


def arms(spec: dict) -> list:
    """Baseline, twin(s), then the candidates in the rule's order."""
    return [spec["baseline"], *spec["twins"], *[c["id"] for c in spec["candidates"]]]


# What each rule setting MEANS for the learner. `fixed`: the value is part of
# how every arm runs (run_arm hard-codes it) and the rule must say exactly it.
_FIXED = {"buffer": "the whole run", "norm_obs": True}
_NOTES = ("note", "reset_every")      # prose / the number param_resets derives from
_KNOWN = {"algo", "net_arch", "layer_norm", "lr", "batch_size", "tau", "gamma",
          "learning_starts", "train_freq", "gradient_steps", "ent_coef", "use_sde",
          "n_steps", "top_quantiles_to_drop_per_net", "param_resets",
          "learner_reward_scale"}


def arm_settings(spec: dict, arm: str) -> dict:
    """The rule's settings for one arm: the baseline's, overlaid by the arm's
    own. An unknown arm or an unknown setting is refused, never guessed."""
    st = dict(spec["baseline_settings"])
    if arm == spec["baseline"]:
        over = {}
    elif arm in spec["twins"]:
        over = spec["twin_settings"][arm]
    else:
        cand = {c["id"]: c for c in spec["candidates"]}
        if arm not in cand:
            raise Refused(f"r1: {arm!r} is not an arm of the frozen rule")
        over = cand[arm]["settings"]
    st.update(over)
    for k in _NOTES:
        st.pop(k, None)
    for k, v in _FIXED.items():
        if st.pop(k, v) != v:
            raise Refused(f"r1: arm {arm}: {k} must be {v!r} (run_arm has no other)")
    bad = sorted(set(st) - _KNOWN)
    if bad:
        raise Refused(f"r1: arm {arm}: no knob for setting(s) {bad} — "
                      "a term the kernel cannot meet is an INBOX line")
    return st


def learner_call(st: dict, seed: int, T: int, rung: str | None) -> tuple:
    """(algo name, keyword arguments) for sbx — pure, so a test can read it.
    For the baseline this is `pilot_b.make_b`'s call exactly; a keyword is
    added ONLY when the arm's setting leaves sbx's default."""
    if st["algo"].split()[0] not in ("sbx.SAC", "sbx.TQC"):
        raise Refused(f"r1: unknown algo {st['algo']!r}")
    algo = st["algo"].split()[0].split(".")[1]
    if st["layer_norm"] != "critic":          # sbx: layer_norm=True is critic-only
        raise Refused("r1: layer_norm must be 'critic'")
    pk = dict(layer_norm=True, net_arch=list(st["net_arch"]))
    if "top_quantiles_to_drop_per_net" in st:
        if algo != "TQC":
            raise Refused("r1: quantile truncation needs TQC")
        pk["top_quantiles_to_drop_per_net"] = int(st["top_quantiles_to_drop_per_net"])
    kw = dict(learning_rate=float(st["lr"]), buffer_size=max(int(T), 50_000),
              batch_size=int(st["batch_size"]), tau=float(st["tau"]),
              gamma=float(st["gamma"]), learning_starts=int(st["learning_starts"]),
              train_freq=int(st["train_freq"]), gradient_steps=int(st["gradient_steps"]),
              ent_coef=st["ent_coef"], use_sde=bool(st["use_sde"]), seed=int(seed),
              verbose=0, policy_kwargs=pk)
    if int(st["n_steps"]) != 1:
        kw["n_steps"] = int(st["n_steps"])
    if "param_resets" in st:
        pr = st["param_resets"]
        if isinstance(pr, dict):
            if rung not in pr:
                raise Refused(f"r1: the rule names no reset points for rung {rung!r}")
            pr = pr[rung]
        kw["param_resets"] = [int(x) for x in pr]
    return algo, kw


def _scaled(env, scale: float):
    """K6: the reward AS THE LEARNER SEES IT times `scale`. Sits OUTSIDE
    PilotLives, so the lives, the meals and the metric read the world's own
    reward; only the replay buffer gets the scaled one."""
    import gymnasium as gym

    class LearnerRewardScale(gym.RewardWrapper):
        def reward(self, reward):
            return float(reward) * scale
    return LearnerRewardScale(env)


def run_arm(arm: str, seed: int, T: int, rung: str | None = None, *,
            partial_path=None, spec: dict | None = None, keep: dict | None = None) -> dict:
    """One run of `arm` for T decisions on E-food's world. `rung` names which
    of the rule's rungs this is (None = a smoke/test run, stamped as such).
    `keep` (tests): receives the model and the env."""
    from experiments import pilot_b as pb
    from experiments.pilot_b2 import E2C_ENCODING, PILOT_RULE, arm_cfg, pilot_world
    spec_given = spec is not None
    spec = load_spec() if spec is None else spec
    st = arm_settings(spec, arm)
    algo, kw = learner_call(st, seed, T, rung)
    scale = float(st.get("learner_reward_scale", 1.0))
    pb._single_thread()
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    import sbx
    world = pilot_world(WORLD_ARM, PILOT_RULE["food_floor_only"], PILOT_RULE["eye_frame"],
                        PILOT_RULE["motor_frame"], PILOT_RULE["native_substep"])
    holder = {}

    def _mk():
        w = pb.PilotLives(world(seed), partial_path=partial_path)
        holder["w"] = w
        return w if scale == 1.0 else _scaled(w, scale)

    venv = VecNormalize(DummyVecEnv([_mk]), norm_obs=True, norm_reward=False,
                        clip_obs=10.0)
    venv.seed(seed)
    model = getattr(sbx, algo)("MlpPolicy", venv, **kw)
    if keep is not None:
        keep.update(model=model, env=holder["w"], venv=venv)
        keep.get("hook", lambda m: None)(model)
    tr = []
    t_wall, t_cpu = time.time(), time.process_time()
    model.learn(total_timesteps=int(T), progress_bar=False,
                callback=pb._trace_callback(tr, pb.TRACE_EVERY))
    wall, cpu = time.time() - t_wall, time.process_time() - t_cpu
    w = holder["w"]
    lives = list(w.lives)
    if w.n > 0:
        lives.append(w._life_row({"ate_total": dict(w.env.w.drives.ate_total)},
                                 "unfinished"))
    resets = getattr(model, "reset_idx", 0)
    return {"arm": arm, "seed": int(seed), "T": int(T), "lr": kw["learning_rate"],
            "decisions": int(model.num_timesteps), "wall_s": round(wall, 1),
            "cpu_s": round(cpu, 1),
            "decisions_per_s": round(model.num_timesteps / wall, 2),
            "decisions_per_cpu_s": round(model.num_timesteps / cpu, 2),
            "world": pb._world_stamp(w.env), "trace": tr, "trace_every": pb.TRACE_EVERY,
            "pilot_arm": WORLD_ARM, "handed": arm_cfg(WORLD_ARM)["handed"],
            "handed_encoding": E2C_ENCODING,
            "food_floor_only": bool(PILOT_RULE["food_floor_only"]),
            "eye_frame": PILOT_RULE["eye_frame"], "motor_frame": PILOT_RULE["motor_frame"],
            "forge": {"round": spec["round"], "arm": arm, "rung": rung,
                      "rung_decisions": (spec["rungs"][rung]["decisions"]
                                         if rung in spec["rungs"] else None),
                      "spec_sha256": None if spec_given else spec_sha256(),
                      "algo": algo, "settings": st,
                      "learner_kwargs": {k: v for k, v in kw.items()},
                      "learner_reward_scale": scale, "param_resets_done": int(resets),
                      "code_head": _code_head()},
            "lives": lives}


def _code_head() -> str:
    p = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                       capture_output=True, text=True)
    return p.stdout.strip() if p.returncode == 0 else ""


# ---- packing ---------------------------------------------------------------
def cells(spec: dict, rung: str, climbers: list | None = None) -> list:
    """(arm, seed) for a rung, SEED-major so no arm sits wholly on one slow
    host. S: all ten arms. Above S the entrants are the ladder's to name
    (`climbers` = the arm ids experiments.forge.climb let through, plus the
    baseline, twin and yardstick) — there is no default."""
    if rung == "S":
        if climbers is not None:
            raise Refused("r1: rung S runs every arm of the rule")
        who = arms(spec)
    else:
        if not climbers:
            raise Refused(f"r1: rung {rung} needs the ladder's entrants")
        unknown = [a for a in climbers if a not in arms(spec)]
        if unknown:
            raise Refused(f"r1: not arms of the rule: {unknown}")
        who = list(climbers)
    return [(a, s) for s in spec["rungs"][rung]["seeds"] for a in who]


def groups(spec: dict, rung: str, climbers: list | None = None) -> dict:
    c, n = cells(spec, rung, climbers), PER_KERNEL
    return {f"g{j}": c[n * j:n * (j + 1)] for j in range((len(c) + n - 1) // n)}


def cell_file(arm: str, seed: int) -> str:
    return f"r1_{arm}_s{seed}.json"


def run_parallel(cell_list: list, outdir, rung: str) -> dict:
    """All cells at once as child processes (pilot_b2.run_pilot_parallel's
    shape): own file, partial file every 10 lives, a timed-out cell is `cut`
    with the lives it finished."""
    from experiments.pilot_b import CHILD_TIMEOUT_S, from_partial, partial_file
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    procs = []
    for arm, seed in cell_list:
        f = outdir / cell_file(arm, seed)
        procs.append((arm, seed, f, subprocess.Popen(
            [sys.executable, "-m", "experiments.forge.r1_run", "cell", "--arm", arm,
             "--seed", str(seed), "--rung", rung, "--out", str(f),
             "--partial", str(partial_file(f))],
            cwd=REPO, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)))
    res, t0 = {}, time.time()
    for arm, seed, f, p in procs:
        try:
            _, err = p.communicate(timeout=max(1, CHILD_TIMEOUT_S - (time.time() - t0)))
        except subprocess.TimeoutExpired:
            p.kill()
            _, err = p.communicate()
        rec = (json.loads(f.read_text()) if f.exists()
               else from_partial(arm, seed, partial_file(f)))
        rec["rc"] = p.returncode
        if p.returncode != 0:
            rec["error"] = f"rc={p.returncode}: {err.decode()[-800:]}"
        res[f"{arm}_s{seed}"] = rec
        print(f"{arm} s{seed}: rc {p.returncode}, {len(rec['lives'])} lives", flush=True)
    return res


def cells_from_artifacts(artifacts: dict, cell_list: list) -> dict | None:
    """A kernel with no summary file: assemble from the per-cell files, final
    first, else the partial (marked `cut`), else `not_found` — never invented."""
    from experiments.pilot_b import from_partial, partial_file
    res, found = {}, 0
    for arm, seed in cell_list:
        final = cell_file(arm, seed)
        part = partial_file(Path(final)).name
        if final in artifacts:
            rec = json.loads(Path(artifacts[final]).read_text())
            found += 1
        elif part in artifacts:
            rec = from_partial(arm, seed, Path(artifacts[part]))
            found += 1
        else:
            rec = from_partial(arm, seed, Path("/nonexistent") / part)
        rec["from_cell_file"] = True
        res[f"{arm}_s{seed}"] = rec
    return {"cells": res, "assembled": True} if found else None


_JOB = r"""
import subprocess as _s, sys as _y
_PIN = __PIN__
if _s.run(["git", "-C", "/tmp/jack", "checkout", "-q", _PIN]).returncode != 0:
    _s.run(["git", "-C", "/tmp/jack", "fetch", "-q", "--depth", "1", "origin", _PIN], check=True)
    _s.run(["git", "-C", "/tmp/jack", "checkout", "-q", _PIN], check=True)
_got = _s.run(["git", "-C", "/tmp/jack", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
assert _got == _PIN, ("round pin not checked out", _got, _PIN)
print("FORGE_PIN", _got, flush=True)
_s.run([_y.executable, "-m", "pip", "install", "-q"] + __PINS__, check=True)
import json, os
os.chdir("/tmp/jack")
from experiments.forge.r1_run import run_parallel, spec_sha256
assert spec_sha256() == __SPEC__, "the kernel's spec.json is not the frozen rule"
from experiments.brain_b import versions
r = run_parallel(__CELLS__, os.environ["JACK_OUT"], __RUNG__)
json.dump({"cells": r, "versions": versions(), "round": "r1", "rung": __RUNG__,
           "pin": _PIN, "spec_sha256": __SPEC__},
          open(os.path.join(os.environ["JACK_OUT"], "forge_r1_cells.json"), "w"))
print("DONE", flush=True)
"""


def job_body(cell_list: list, rung: str, pin: str) -> str:
    from experiments.pilot_b import KERNEL_PINS
    if len(pin) != 40 or any(c not in "0123456789abcdef" for c in pin):
        raise Refused("r1: --pin must be a full 40-hex commit sha")
    return (_JOB.replace("__CELLS__", repr([tuple(c) for c in cell_list]))
            .replace("__PINS__", repr(list(KERNEL_PINS)))
            .replace("__RUNG__", repr(str(rung))).replace("__PIN__", repr(pin))
            .replace("__SPEC__", repr(spec_sha256())))


def pin_has_rule(pin: str) -> bool:
    """The pinned commit must hold THIS frozen spec and this kernel file."""
    p = subprocess.run(["git", "-C", str(REPO), "show", f"{pin}:experiments/forge/rounds/r1/spec.json"],
                       capture_output=True)
    q = subprocess.run(["git", "-C", str(REPO), "cat-file", "-e",
                        f"{pin}:experiments/forge/r1_run.py"], capture_output=True)
    return (p.returncode == 0 and q.returncode == 0
            and hashlib.sha256(p.stdout).hexdigest() == spec_sha256())


def kernel(name: str, rung: str, pin: str, climbers: list | None = None) -> dict:
    """Submit one group as a Kaggle CPU kernel -> OUT/<rung>_<name>.json (raw)."""
    from experiments.gpu import build_job, result_json, submit
    spec = load_spec()
    cell_list = groups(spec, rung, climbers)[name]
    if not pin_has_rule(pin):
        raise Refused(f"r1: commit {pin[:10]} does not hold the frozen rule + kernel")
    out = OUT / f"{rung}_{name}.json"
    if out.exists():
        raise Refused(f"r1: {out} is already home — a rung's kernel runs once")
    body = job_body(cell_list, rung, pin)
    sub = submit(build_job(body), prefer="kaggle-cpu", est_hours=EST_HOURS[rung],
                 timeout_s=REMOTE_TIMEOUT_S, fetch=["forge_r1_cells.json"])
    if sub.ok and "forge_r1_cells.json" in sub.artifacts:
        r = result_json(sub, "forge_r1_cells.json")
    else:
        r = cells_from_artifacts(sub.artifacts, cell_list)
        if r is None:
            raise RuntimeError(f"forge r1 kernel {rung}/{name}: {sub.message}")
        print(f"forge r1 kernel {rung}/{name}: no summary ({sub.message or 'missing'}); "
              "assembled from per-cell files", flush=True)
    # what the scoreboard's cost line needs, read off the launcher's receipt
    r["kernel"] = {"job_id": sub.job_id, "duration_s": sub.duration_s,
                   "charge_seconds": sub.charge_seconds, "name": name, "pin": pin}
    OUT.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(r, indent=1) + "\n")
    for k, rec in r["cells"].items():
        print(f"{k}: {len(rec['lives'])} lives" + (" (cut)" if rec.get("cut") else "")
              + (f" ERROR {rec['error']}" if "error" in rec else ""), flush=True)
    return r


def chain(names: list, rung: str, pin: str, climbers: list | None = None,
          sleep=time.sleep) -> dict:
    """Kernels one after another = ONE Kaggle session held. A kernel that fails
    is logged and the chain goes on (its cells stay missing — a missing run is
    the reader's to count, never refilled silently). The launcher's guard
    (another lane's uncommitted code, an unpushed HEAD) is waited out, not
    bypassed."""
    done = {}
    for name in names:
        for attempt in range(GUARD_RETRIES + 1):
            try:
                kernel(name, rung, pin, climbers)
                done[name] = "home"
                break
            except Refused as e:
                done[name] = f"refused: {e}"
                break
            except RuntimeError as e:
                guard = ("Uncommitted changes" in str(e) or "Untracked experiment" in str(e)
                         or "is not on origin" in str(e))
                if guard and attempt < GUARD_RETRIES:
                    print(f"{name}: launcher guard ({str(e).splitlines()[0]}); "
                          f"waiting {GUARD_WAIT_S}s [{attempt + 1}/{GUARD_RETRIES}]", flush=True)
                    sleep(GUARD_WAIT_S)
                    continue
                done[name] = f"failed: {str(e)[:300]}"
                break
        print(f"CHAIN {rung}/{name}: {done[name]}", flush=True)
    print("CHAIN DONE", json.dumps(done), flush=True)
    return done


def main(argv):
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["cell", "groups", "kernel", "chain"])
    p.add_argument("--arm")
    p.add_argument("--seed", type=int)
    p.add_argument("--rung", default="S")
    p.add_argument("--decisions", type=int, default=None,
                   help="cell: a smoke length; stamped rung=None unless it is the rung's")
    p.add_argument("--out")
    p.add_argument("--partial", default=None)
    p.add_argument("--name")
    p.add_argument("--names", nargs="+")
    p.add_argument("--pin")
    p.add_argument("--climbers", nargs="+", default=None)
    a = p.parse_args(argv)
    spec = load_spec()
    if a.cmd == "cell":
        T = spec["rungs"][a.rung]["decisions"]
        rung = a.rung
        if a.decisions is not None and a.decisions != T:
            if any(a.seed in r["seeds"] for r in spec["rungs"].values()):
                raise Refused("r1: a smoke run never uses a rung's seed")
            T, rung = a.decisions, None
        elif a.seed not in spec["rungs"][a.rung]["seeds"]:
            raise Refused(f"r1: seed {a.seed} is not one of rung {a.rung}'s")
        rec = run_arm(a.arm, a.seed, T, rung, partial_path=a.partial)
        Path(a.out).write_text(json.dumps(rec) + "\n")
    elif a.cmd == "groups":
        for k, v in groups(spec, a.rung, a.climbers).items():
            print(k, v)
    elif a.cmd == "kernel":
        kernel(a.name, a.rung, a.pin, a.climbers)
    elif a.cmd == "chain":
        chain(a.names, a.rung, a.pin, a.climbers)


if __name__ == "__main__":
    main(sys.argv[1:])
