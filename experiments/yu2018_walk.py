"""J10 (2) — Yu, Turk & Liu 2018 (arXiv:1801.08093), the published walking recipe,
UNMODIFIED, on its published task (PUBLISHED_PRIORS §4 "Reproduce first").

Source, code only (no weights exist in it; repo has no licence file — read and
run, never redistributed):
  github.com/VincentYu68/SymmetryCurriculumLocomotion @ b50478f8eca673730e3ce1a5441b1948b31a5187
  entry  siggraph_script/run_script_humanoid.sh:
         mpirun -np 8 python3 baselines/baselines/siggraph/humanoid_walk.py --seed 0
  env    DartHumanWalker-v1 (bundled dart-env fork of gym), PPO1 + MlpMirrorPolicy
         (3x64, mirror loss weight 4), staged helping forces (`train_mirror_sig`).
  stack  DART v6.3.0 (= 8090aaec352e6198c4606c98e4990c7942864cb2) from source with
         the repo's external/lcp.cpp; pydart2 @ 113132aceaa40bfd0c2e8c85179fb19b68cf9b8d
         (its head since 2018-05-25, i.e. what the authors' 2019 install pulled) with
         the repo's pydart2_draw.cpp; TF1 -> Python 3.7 (tensorflow 1.15.5, the last
         TF1); MPI.

CORRECTION to the 11-16 17:55 pre-flight (KNOWN_ANSWERS §Row J10-2). The
released script's `num_timesteps=5,000,000` is passed as `max_timesteps` to EACH
curriculum stage's `pposgd_mirror.learn` call — a per-stage cap, not a total.
A stage ends early when mean rollout length > assist_schedule[2][0]/dt AND mean
velocity > 0.5 * final_tv (pposgd_mirror.py:337-350). The helping forces start
as spring GAINS of (2000, 2000) — CORRECTED 11-19, they are not newtons: a
stiffness on the pelvis's sideways position (SPD, N/m, Kd = Kp*dt) and a gain
on its forward-speed error toward the target speed (N per m/s;
human_walker.py:175-212, 262-272) — and shrink x0.75 per stage until < 5
(then 0), then one unaided "last iteration" stage with return threshold 1.2 x 0.7 x stage-0 return: about 22
stages. So the released script's TOTAL is whatever the curriculum takes — the
paper's 10-30 M (500-1,500 iterations of 20k) is the expectation, and
"5 M fits one kernel" was wrong.

OURS BESIDE THEIRS (every difference, before anything runs):
  | setting                | published (script / paper)      | ours                                  |
  |------------------------|---------------------------------|---------------------------------------|
  | learner, env, rewards  | as released                     | unmodified                            |
  | MPI ranks              | 8 (2,500 samples each = 20k/it) | 8, `--oversubscribe` on 4 cores: the  |
  |                        |                                 | 20k batch kept; wall clock ~2x        |
  | hardware               | 16-vCPU EC2, 25-45 s/iteration  | Kaggle CPU kernel, 4 cores, 12 h cap  |
  | OS / compilers         | Ubuntu 16.04/18.04              | Kaggle image (Ubuntu 22.04, gcc 11);  |
  |                        |                                 | apt libs at that image's versions     |
  | Python / TF            | python3 + tensorflow>=1.0       | 3.7 (uv standalone) + TF 1.15.5       |
  | baselines install      | `pip install -e .` (pulls gym,  | `--no-deps` + the deps ppo1 imports;  |
  |                        |  atari, azure, zmq)             | gym is the bundled dart-env either way|
  | across 12 h kernels    | one long process                | resume from the authors' own 10-iter  |
  |                        |                                 | saves via their commented resume hook |
  |                        |                                 | (training_utils.py:54-55) — FULL run  |
  |                        |                                 | only; registered after the smoke      |

RULE (committed before any kernel):
  smoke      ONE Kaggle CPU kernel, build only + a 20-minute timed slice of the
             unmodified script (8 ranks oversubscribed). Records, per step, the
             return code and seconds: apt, python 3.7 env, DART cmake/make/install,
             pydart2 build, baselines + dart-env install, `import pydart2, gym;
             gym.make('DartHumanWalker-v1')`, then PPO iterations completed and
             seconds per iteration. A build that will not compile is recorded
             in KNOWN_ANSWERS as such and is NOT worked around with another
             simulator; a dependency pin (e.g. FCL 0.5.0 from source if 22.04's
             FCL 0.7 breaks DART 6.3's API) is tried once, automatically, and
             listed.
  sizing     from the smoke's measured s/iteration: kernels = ceil(1,500 x s/it
             / 11 h) for the paper's upper total. The full run is registered in
             a follow-up commit with that number, the resume wrapper, and seeds.
  seeds      0, 1, 2 (the script's --seed; MPI rank r uses seed + r as released).
             One seed landing is a PILOT and is labelled so; REPRODUCES needs
             >= 2 of 3.
  accept     PRIMARY: the final policy (policy_params.pkl after the "last
             iteration" stage), assist forces at 0, deterministic actions, 10
             rollouts of the env's full 500 steps: >= 9 of 10 rollouts reach
             step 500 without termination AND mean forward velocity >= 0.75 m/s
             (the script's own stage criterion, 0.5 x final_tv 1.5).
             REPORTED ONLY: the symmetry index, its definition pinned from the
             paper's text before the full run's harvest (draft class: walk SI
             ~0.01, Table 2) — not a pass clause until pinned, said here so it
             cannot be chosen after seeing the number.
  control    the same seed's iteration-0 policy (initial parameters, forces at
             0), same 10 rollouts: must FAIL the primary clause, or the row is
             VOID (the test would not measure learning).
  stage log  (added 11-19, before any full run; reported, not a pass clause.)
             The released loop leaves a stage on EITHER its threshold
             (pposgd_mirror.py:348-350) OR the 5M cap (:211) and prints nothing
             to tell them apart, so all ~22 stages can "complete" having learned
             nothing. `stage_exits` splits rank 0's progress.json where Iter
             restarts and labels each stage THRESHOLD (left below the cap — the
             only other exit), CAP (its last TimestepsSoFar >= 5M; an iteration
             that crossed the cap AND met the threshold is counted CAP, the
             stricter reading, because velocity is not logged), or CUT (the last
             stage of a run that was stopped). It also counts, from stdout, each
             stage's "Revert Iteration" lines and iteration headers: a reverted
             iteration (:237-251) resets TimestepsSoFar, so its 20k samples are
             drawn but never counted toward the cap — the stage's true samples
             are headers x 20k, reported beside the logged count. Every verdict
             line names how many stages left on the cap ("REPRODUCES, k of n
             stages advanced on the cap"); the primary clause is unchanged
             because it tests the final policy's walk itself.
  verdicts   REPRODUCES / MISSES (the curriculum completed but the policy fails
             the primary) / DID-NOT-FINISH (curriculum not done within the
             registered kernels: reported with the stage reached, never PASS) /
             BUILD-FAILED.

Run:  $PY -m experiments.yu2018_walk rule          # print the rule, no compute
      $PY -m experiments.yu2018_walk smoke         # one Kaggle CPU kernel (~1.5 h)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = Path("/data/jack-logs/j10")

YU_REPO = "https://github.com/VincentYu68/SymmetryCurriculumLocomotion"
YU_SHA = "b50478f8eca673730e3ce1a5441b1948b31a5187"
DART_REPO = "https://github.com/dartsim/dart"
DART_SHA = "8090aaec352e6198c4606c98e4990c7942864cb2"   # tag v6.3.0
PYDART2_REPO = "https://github.com/sehoonha/pydart2"
PYDART2_SHA = "113132aceaa40bfd0c2e8c85179fb19b68cf9b8d"
FCL_FALLBACK_TAG = "0.5.0"
PY37_PINS = ["numpy==1.18.5", "tensorflow==1.15.5", "protobuf==3.20.3", "scipy==1.4.1",
             "joblib==0.14.1", "dill", "tqdm", "progressbar2", "cloudpickle==1.2.2",
             "mpi4py==3.0.3", "six", "requests", "pyglet==1.3.2", "matplotlib==3.2.2"]
SMOKE_TRAIN_S = 20 * 60
SMOKE_TIMEOUT_S = int(2.5 * 3600)
MPI_RANKS = 8
ACCEPT = {"rollouts": 10, "min_full": 9, "steps": 500, "min_vel": 0.75}
SEEDS = [0, 1, 2]

# The kernel body: a bash build, stage by stage, every stage's rc + seconds +
# log tail recorded; then a timed slice of the unmodified script.
_BUILD_SH = r"""
set -o pipefail
export DEBIAN_FRONTEND=noninteractive
stage() { local name=$1; shift; local t0=$(date +%s);
  ( "$@" ) > /tmp/yu/log_$name.txt 2>&1; local rc=$?;
  echo "STAGE $name rc=$rc s=$(( $(date +%s) - t0 ))"; tail -n 25 /tmp/yu/log_$name.txt | sed "s/^/  [$name] /";
  return $rc; }
mkdir -p /tmp/yu && cd /tmp/yu
git clone -q __YU_REPO__ repo && git -C repo checkout -q __YU_SHA__ && git -C repo rev-parse HEAD
stage apt bash -c 'apt-get update -qq && apt-get install -y -qq build-essential cmake pkg-config git swig \
  libeigen3-dev libassimp-dev libccd-dev libfcl-dev libboost-regex-dev libboost-system-dev \
  libboost-all-dev libopenscenegraph-dev libbullet-dev liburdfdom-dev libnlopt-dev libxi-dev \
  libxmu-dev freeglut3-dev libtinyxml2-dev libode-dev libopenmpi-dev openmpi-bin' || exit 11
echo "FCL $(pkg-config --modversion fcl)  EIGEN $(pkg-config --modversion eigen3)  GCC $(gcc -dumpversion)"
stage py37 bash -c 'pip install -q uv && uv python install 3.7 && uv venv -q -p 3.7 /tmp/yu/py37 && \
  uv pip install -q --python /tmp/yu/py37/bin/python setuptools==57.5.0 wheel __PY37_PINS__' || exit 12
PY=/tmp/yu/py37/bin/python
dart_build() { rm -rf dart && git clone -q __DART_REPO__ dart && git -C dart checkout -q __DART_SHA__ && \
  cp repo/external/lcp.cpp dart/dart/external/odelcpsolver/lcp.cpp && mkdir dart/build && cd dart/build && \
  cmake .. -DCMAKE_BUILD_TYPE=Release && make -j4 && make install && ldconfig; }
if ! stage dart dart_build; then
  echo "DART failed against the image's FCL; trying the one listed pin: FCL __FCL__ from source"
  stage fcl bash -c 'git clone -q https://github.com/flexible-collision-library/fcl fcl && cd fcl && \
    git checkout -q __FCL__ && mkdir build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release \
    -DFCL_BUILD_TESTS=OFF && make -j4 && make install && ldconfig' || exit 13
  stage dart2 dart_build || exit 14
fi
stage pydart2 bash -c 'git clone -q __PYDART2_REPO__ pydart2 && git -C pydart2 checkout -q __PYDART2_SHA__ && \
  cp repo/external/pydart2_draw.cpp pydart2/pydart2/ && cd pydart2 && \
  '$PY' setup.py build build_ext && '$PY' setup.py develop' || exit 15
stage pkgs bash -c ''$PY' -m pip install -q --no-deps -e repo/baselines && \
  '$PY' -m pip install -q --no-deps -e repo/dart-env' || exit 16
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
stage envmake bash -c 'cd repo && '$PY' -c "import pydart2, gym; e = gym.make(\"DartHumanWalker-v1\"); \
  o = e.reset(); print(\"OBS\", len(o), \"ACT\", e.action_space.shape, \"DT\", e.env.dt); \
  import numpy as np; [e.step(e.action_space.sample()) for _ in range(50)]; print(\"STEP_OK\")"' || exit 17
export OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1 OMP_NUM_THREADS=1
cd repo && t0=$(date +%s)
timeout __TRAIN_S__ mpirun -np __RANKS__ --oversubscribe --allow-run-as-root $PY \
  baselines/baselines/siggraph/humanoid_walk.py --seed 0 > /tmp/yu/log_train.txt 2>&1
echo "STAGE train rc=$? s=$(( $(date +%s) - t0 ))"
grep -c "\*\*\*\*\* Iteration" /tmp/yu/log_train.txt | sed "s/^/ITERATIONS_PRINTED /"
tail -n 40 /tmp/yu/log_train.txt | sed "s/^/  [train] /"
exit 0
"""

_JOB = r'''
import json, os, re, subprocess, time
sh = __SH__
open("/tmp/yu_build.sh", "w").write(sh)
t0 = time.time()
p = subprocess.Popen(["bash", "/tmp/yu_build.sh"], stdout=subprocess.PIPE,
                     stderr=subprocess.STDOUT, text=True)
lines = []
for line in p.stdout:
    print(line, end="", flush=True)
    lines.append(line.rstrip("\n"))
rc = p.wait()
stages = [dict(zip(("stage", "rc", "s"), (m.group(1), int(m.group(2)), int(m.group(3)))))
          for m in (re.match(r"STAGE (\S+) rc=(-?\d+) s=(\d+)", l) for l in lines) if m]
res = {"rc": rc, "wall_s": round(time.time() - t0), "stages": stages,
       "versions": [l for l in lines if l.startswith(("FCL ", "OBS ", "STEP_OK"))]}
prog = []
for root, _, files in os.walk("/tmp/yu/repo/data"):
    for f in files:
        if f == "progress.json":   # baselines logger formats: stdout, log, json
            prog.append(open(os.path.join(root, f)).read())
res["progress_json"] = prog
times = []
for txt in prog:
    for row in txt.strip().splitlines():
        try:
            d = json.loads(row)
        except ValueError:
            continue
        if "TimeElapsed" in d:
            times.append(float(d["TimeElapsed"]))
res["iterations_logged"] = len(times)
if len(times) >= 3:
    res["s_per_iter"] = round((times[-1] - times[0]) / (len(times) - 1), 2)
rows = []
for txt in prog:
    for row in txt.strip().splitlines():
        try:
            rows.append(json.loads(row))
        except ValueError:
            pass
try:
    tl = open("/tmp/yu/log_train.txt").read().splitlines()
except OSError:
    tl = []
train_rc = [s["rc"] for s in stages if s["stage"] == "train"]
res["stages_curriculum"] = stage_exits(rows, tl, finished=bool(train_rc) and train_rc[0] == 0)
res["stage_summary"] = stage_summary(res["stages_curriculum"])
json.dump(res, open(os.path.join(os.environ["JACK_OUT"], "yu2018_smoke.json"), "w"), indent=1)
print("DONE", json.dumps({k: v for k, v in res.items() if k != "progress_json"}), flush=True)
'''


CAP = 5_000_000          # the released script's num_timesteps, per stage
ITER_SAMPLES = 20_000    # 8 ranks x 2,500


def stage_exits(progress_rows, stdout_lines=(), finished=False, cap=CAP, ranks=MPI_RANKS):
    """Per-stage exit reasons for the released curriculum (rule: "stage log").

    progress_rows: rank 0's progress.json rows (dicts with Iter, TimestepsSoFar,
    EpRewMean, EpLenMean). stdout_lines: the training log (all ranks). finished:
    the training process exited by itself (not cut by a timeout or a segment
    end) — only then is the last stage's exit read, else it is CUT.
    """
    stages, prev = [], None
    for r in progress_rows:
        if "Iter" not in r:
            continue
        it = int(r["Iter"])
        if prev is None or it <= prev:
            stages.append([])
        stages[-1].append(r)
        prev = it
    # stdout: a stage ends at its "Current Schedule:" (printed by every rank,
    # same text) or at "Entering Last Iteration!"; count headers and reverts.
    heads, reverts, scheds = [0], [0], []
    seen = set()
    for ln in stdout_lines:
        if "********** Iteration" in ln:
            heads[-1] += 1
        elif "Revert Iteration" in ln:
            reverts[-1] += 1
        elif ln.startswith("Current Schedule:"):
            sch = ln.split(":", 1)[1].strip()
            if sch not in seen:
                seen.add(sch)
                scheds.append(sch)
                heads.append(0)
                reverts.append(0)
    out = []
    for i, rows in enumerate(stages):
        last = rows[-1]
        ts = int(last["TimestepsSoFar"])
        is_last = i == len(stages) - 1
        if is_last and not finished:
            why = "CUT"
        elif ts >= cap:
            why = "CAP"
        else:
            why = "THRESHOLD"
        h = heads[i] if i < len(heads) else 0
        rv = reverts[i] if i < len(reverts) else 0
        out.append({"stage": i, "exit": why, "iters_logged": int(last["Iter"]),
                    "timesteps_logged": ts,
                    "iter_headers": h // ranks, "reverts": rv // ranks,
                    "samples_drawn": (h // ranks) * ITER_SAMPLES,
                    "ep_rew_mean": float(last.get("EpRewMean", float("nan"))),
                    "ep_len_mean": float(last.get("EpLenMean", float("nan"))),
                    "schedule_after": scheds[i] if i < len(scheds) else None})
    return out


def stage_summary(stages) -> str:
    done = [s for s in stages if s["exit"] != "CUT"]
    k = sum(s["exit"] == "CAP" for s in done)
    return f"{k} of {len(done)} stages advanced on the cap"


def build_sh(train_s: int = SMOKE_TRAIN_S) -> str:
    rep = {"__YU_REPO__": YU_REPO, "__YU_SHA__": YU_SHA, "__DART_REPO__": DART_REPO,
           "__DART_SHA__": DART_SHA, "__PYDART2_REPO__": PYDART2_REPO,
           "__PYDART2_SHA__": PYDART2_SHA, "__FCL__": FCL_FALLBACK_TAG,
           "__PY37_PINS__": " ".join(f"'{p}'" for p in PY37_PINS),
           "__TRAIN_S__": str(int(train_s)), "__RANKS__": str(MPI_RANKS)}
    s = _BUILD_SH
    for k, v in rep.items():
        s = s.replace(k, v)
    return s


def job_body(train_s: int = SMOKE_TRAIN_S) -> str:
    import inspect
    helpers = (f"MPI_RANKS = {MPI_RANKS}\nCAP = {CAP}\nITER_SAMPLES = {ITER_SAMPLES}\n"
               + inspect.getsource(stage_exits) + "\n" + inspect.getsource(stage_summary))
    return helpers + _JOB.replace("__SH__", repr(build_sh(train_s)))


def smoke() -> dict:
    from experiments.gpu import build_job, result_json, submit
    sub = submit(build_job(job_body()), prefer="kaggle-cpu", est_hours=1.5,
                 timeout_s=SMOKE_TIMEOUT_S, fetch=["yu2018_smoke.json"])
    if not sub.ok:
        raise RuntimeError(f"yu2018 smoke: {sub.message}")
    r = result_json(sub, "yu2018_smoke.json")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "yu2018_smoke.json").write_text(json.dumps(r, indent=1) + "\n")
    print(json.dumps({k: v for k, v in r.items() if k != "progress_json"}, indent=1))
    return r


def main(argv):
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["rule", "smoke", "script"])
    a = p.parse_args(argv)
    if a.cmd == "rule":
        print(__doc__)
    elif a.cmd == "script":
        print(build_sh())
    else:
        smoke()


if __name__ == "__main__":
    main(sys.argv[1:])
