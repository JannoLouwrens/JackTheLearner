"""LF.02 — a life can be saved and resumed: world, needs, diary, working memory.

T0.03/T0.04 proved the WEIGHTS round-trip. T6.03 proved the companion STORES
survive a genuinely fresh process. ME.1 is the episodic store itself, and ME.8
killed a process mid-thought and got the thought back. This spec is their
composition AT THE LIFE LEVEL: kill a process mid-life in lethal W0 and resume
it such that the continuation is INDISTINGUISHABLE from the uninterrupted run
over the next 1000 decisions — or the resume is a different life wearing the
same weights. `kills:` multi-session lives, hence every life longer than one
Kaggle session or one 50-minute loop iteration — which is also the checkpoint
half of the `lc07-checkpoint-branch` question (every LC.07 full-scale run
measured 1.7-4.8x over the 8.5 h session wall; a resume that cannot be told
from no interruption is the capability that surgery needs to exist).

THE FOUR STORES, and what carries each (the registry's list, made concrete):

  world    MuJoCo's own integration state — `mj_getState(mjSTATE_INTEGRATION)`,
           which is the documented exact-continuation vector (time, qpos, qvel,
           act, ctrl, applied forces, mocap, plugin state, qacc_warmstart) —
           plus every W0 python-side mutable: the decision/sim clocks, the
           drive-gate and audio counters, the life/death bookkeeping, the
           audio band buffer, the contact-onset synth state (events,
           voice_events, _prev_pairs, _last_fired, listener pose), BOTH
           RandomStates (`_rng`, `_spawn_rng` — a resumed life that redraws
           its panel noise or its next respawn is not the same life), the
           legal-spawn cache, and the forager servo's own state.
  needs    every mutable field of `drives.DriveLayer` (e/i/w, the world clock,
           food respawn timers, ate_total, the contact-onset edge detector,
           the submersion clock, the wetness pair, the per-decision
           accumulators, the last_* diagnostics) plus W0's `_prev_drive`.
  diary    the EpisodicMemory event list, row by row.
  wm       ME.8's WorkingMemory: GRU weights, hidden state `h`, `step_idx`.

THE LIFE is LF.01's declared apparatus, reused verbatim: the privileged
forager servo (imported from lf_01, declared in IMPL_DEPS) shuttling between
floor foods in lethal W0 under PS.01's measured calibration, with a
WorkingMemory stepped on [needs, proprio] every decision and diary waypoints
recorded every DIARY_EVERY decisions with DETERMINISTIC timestamps
(t=sim_seconds). Nothing here claims he learned anything; the claim is about
the HARNESS, exactly as LF.01's was.

THE KILL IS REAL (ME.8's discipline). The victim runs in a child process that
checkpoints all four stores ATOMICALLY (tmp + os.replace, T0.05) after every
decision; the parent polls the checkpoint and SIGKILLs the child once it has
passed a drifting mid-life target. The child must die by SIGKILL — rc -9 — or
the run is VOID: a child that finished politely proves nothing.

INDISTINGUISHABLE, operationalised bitwise. Every run records, after every
decision, a feature row: sha256 over the mjSTATE_INTEGRATION vector minus its
time slot ("g"), sim clock t, torso x/y/z, e/i/w, a working-memory digest
("wm": h bytes + step_idx), and a diary digest ("dy": every row's json minus
the wall-clock `t` field). The resumed run's 1000 rows must match the
uninterrupted reference's rows at the same decisions EXACTLY — digest equality
and zero float delta (TRAJ_TOL = 0.0). Bit-exact is deliberate: same machine,
same binary, and MuJoCo's state vector is documented sufficient for exact
continuation, so any drift means a store was missed — which is precisely what
this spec exists to catch. The trajectory gate audits the store list.

  Why the digest EXCLUDES time, and the diary digest excludes Event.t: the
  NULL below starts a fresh life, so its data.time trivially differs from the
  reference's — a digest that includes the clock would hand the null its
  divergence for free and the null gate would be decorative. Time is instead
  carried as the explicit `t` feature and gated exactly on the RESUMED arm
  (restored clock must continue the reference's to the bit). Event.t is
  wall-clock BY DESIGN (EpisodicMemory stamps real time; W0's death rows do
  not pass t=), so it is the one field an honest resume cannot reproduce;
  waypoint rows pass t=sim_seconds explicitly and everything else in every
  row — channel, speaker, text, importance, meta, eid — is gated.

NULL (the registry's: "weights-only resume — T0.04's null, one level up"): a
fresh process that kept only what a naive restart keeps — the code, the
constants, and the WorkingMemory weights (re-derived from the same torch seed;
they never train here, so re-derivation IS the weights file) — and lost world,
needs, diary and hidden state. It replays "the continuation" from a fresh
life. Its rows are compared against the same reference window and must
DIVERGE: null_g_match_frac <= NULL_G_MATCH_MAX and null_max_delta (over
x/y/z/e/i/w, time excluded as above) >= NULL_DIVERGE_MIN. A null that matches
means the instrument cannot tell a real resume from a restart, and the test
measures nothing: FAIL, per law 2.

CONTROL (registry: corrupt each store in turn; T6.03's model): the checkpoint
carries a sha256 per store, verified by the SAME loader the resume child uses,
before anything is applied. Four corruptions — each store's blob truncated to
60% with the recorded sha left in place — and load must RAISE (CorruptStore)
all four times. A loader that shrugs and defaults even once is the
falsified_by list verbatim ("silently defaulted"): FAIL.

VOID LANES (apparatus declined or missed; not refutations):
  - PS.01 calibration refused (borrow_metrics, LF.01's lane verbatim);
  - the kill missed (rc != -9, or the checkpoint decision landed outside
    [K_TARGET, K_TARGET + K_SLACK_MAX]);
  - prefix nondeterminism: the victim's checkpointed feature row at C must
    equal the reference's row at C. Two identical uninterrupted processes
    disagreeing means the apparatus cannot support a bit-exact claim and the
    resume was never tested.

NO rtf gate, stated rather than omitted: T0.32's standing refusal binds long
runs (LF.01's CPU_LONG hour); this is Budget.CPU, ~3,600 decisions (~12 sim-
minutes) per seed, and the whole run fits inside its own cpu<10min class.

WINDOW is 1000 DECISIONS — the strongest available reading of the registry's
"next 1000 steps" (a decision is 40 physics substeps, so this gates 40,000
physics steps of continuation), and the same unit T1.06/LF.01 count in.
"""
from __future__ import annotations

import hashlib
import json
import os
import pickle
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from ..protocol import Ledger, Status, borrow_metrics, run_spec
from ..registry import BY_ID
from ..w0 import SIM_S_PER_DECISION, W0
# After `..w0`, deliberately: importing it puts the repo root on sys.path
# (XL.00's idiom); EpisodicMemory lives there rather than in the package.
from EpisodicMemory import EpisodicMemory  # noqa: E402
# LF.01's forager servo is this spec's declared apparatus too — reused, not
# re-derived, and its file is in IMPL_DEPS so a change there drifts this
# certificate rather than silently rewriting what "the life" means.
from .lf_01_life_to_natural_end import _Forager  # noqa: E402

IMPL_DEPS = ["playground.py", "experiments/w0.py", "experiments/drives.py",
             "EpisodicMemory.py", "WorkingMemory.py", "ContactAudio.py",
             "experiments/tests/lf_01_life_to_natural_end.py"]

REPO = Path(__file__).resolve().parents[2]
PY = "/data/venvs/jackthelearner/bin/python"
_MOD = "experiments.tests.lf_02_life_saved_and_resumed"

# ── the pre-registered numbers, all of them, before the run ────────────────
WINDOW = 1000                  # decisions of continuation under comparison
K_TARGET_BASE, K_TARGET_STEP = 250, 37   # kill target 250/287/324 by seed:
                               # mid-life (50-65 sim-s in), drifting so no
                               # single magic decision is what resume works at
K_SLACK_MAX = 60               # checkpoint may land at most this far past the
                               # target (the child steps on while the parent
                               # polls); further means the kill loop failed
TRAJ_TOL = 0.0                 # resumed floats match the reference exactly
NULL_G_MATCH_MAX = 0.001       # weights-only resume: ~no bit-identical states
NULL_DIVERGE_MIN = 0.5         # and at least half a metre / half a tank of
                               # visible divergence somewhere in the window
DIARY_EVERY = 100              # decisions between deterministic waypoint rows
KILL_POLL_S = 0.01
CHILD_TIMEOUT_S = 420          # any single child stuck longer is an error

WM_OBS_DIM = 18                # needs (6) + proprio (12)
WM_OUT = 4                     # head width; the head's output is unused
WM_HIDDEN = 32                 # ME.8's size
WM_SEED_BASE = 7000            # torch.manual_seed(WM_SEED_BASE + seed)

STORES = ("world", "needs", "diary", "wm")

# Every python-side mutable on W0 (the mjSTATE vector carries the physics).
_W0_FIELDS = ("decisions", "sim_seconds", "drive_gate_open",
              "audio_events_total", "life", "deaths", "died_this_decision",
              "last_death_cause", "_life_started_at", "life_lengths",
              "death_sites", "spawn_sites")
# Every mutable on drives.DriveLayer (state handled separately as a dict).
_DRIVE_FIELDS = ("t", "_respawn_at", "_submerged_since", "_touching_world",
                 "_prev_speed", "_j_max", "_power_dt", "_dt_acc", "_ate",
                 "_rest_dt", "_drown_dt", "last_j", "last_power_w", "last_dt",
                 "last_rest_dt", "n_onsets", "ate_total", "_wet_dt", "_wet_in")

_CACHE: dict = {}              # seed -> tmpdir, shared with _control


class CorruptStore(RuntimeError):
    """A checkpoint store failed its integrity check. Loud by design."""


def _calibration():
    b = borrow_metrics("PS.01", ("j0_ms", "alpha"))
    if not b.ok:
        return None, None, {**b.provenance, "borrow_refusal": b.refusal}
    return b.values["j0_ms"], b.values["alpha"], b.provenance


# ── the life fixture (identical in every child) ────────────────────────────
def _build_life(seed: int, j0: float, alpha: float):
    import torch
    from WorkingMemory import WorkingMemory
    torch.set_num_threads(1)
    diary = EpisodicMemory()
    w = W0(seed=seed, j0=j0, alpha=alpha, lethal=True, diary=diary)
    forager = _Forager(w)
    torch.manual_seed(WM_SEED_BASE + seed)
    wm = WorkingMemory(WM_OBS_DIM, WM_OUT, WM_HIDDEN)
    diary.record("did", "jack", "life began", importance=1.0, t=0.0,
                 meta={"decision": 0})
    return w, diary, forager, wm


def _step_once(w: W0, forager, wm, diary) -> None:
    import mujoco
    import torch
    obs = w.observe()
    feat = torch.from_numpy(
        np.concatenate([obs["needs"], obs["proprio"]]).astype(np.float32))
    with torch.no_grad():
        wm.step(feat)
    w.decide(forager.action("forage"))
    # Normalise the decision boundary: after mj_step, data's kinematics and
    # contacts describe the PRE-integration pose of the last substep — a
    # derived layer the state vector cannot carry, measured as a ~1e-6 m
    # xpos gap on the first restore attempt. This forward pins every derived
    # field to the post-integration state on EVERY side (ref, victim,
    # resume, null run the same line, so cross-arm bit-parity holds), which
    # is what makes "checkpoint = the whole boundary" true rather than
    # approximately true.
    mujoco.mj_forward(w.model, w.data)
    mujoco.mj_rnePostConstraint(w.model, w.data)
    n = w.decisions
    if n % DIARY_EVERY == 0:
        s = w.drives.state
        diary.record(
            "did", "jack",
            f"waypoint: decision {n}, energy {s.e:.4f}, "
            f"eats {int(sum(w.drives.ate_total.values()))}",
            importance=1.0, t=float(w.sim_seconds), meta={"decision": n})


# ── digests and feature rows ───────────────────────────────────────────────
def _mj_state(w: W0) -> np.ndarray:
    import mujoco
    spec = mujoco.mjtState.mjSTATE_INTEGRATION
    buf = np.empty(mujoco.mj_stateSize(w.model, spec), dtype=np.float64)
    mujoco.mj_getState(w.model, w.data, buf, spec)
    return buf


def _g_digest(buf: np.ndarray) -> str:
    # buf[0] is data.time (mjSTATE_TIME is the lowest bit of INTEGRATION);
    # excluded here, gated explicitly via the `t` feature on the resumed arm.
    return hashlib.sha256(buf[1:].tobytes()).hexdigest()


def _wm_digest(wm) -> str:
    h = hashlib.sha256(wm.h.detach().numpy().tobytes())
    h.update(str(int(wm.step_idx)).encode())
    return h.hexdigest()


def _diary_rows(diary) -> list:
    from dataclasses import asdict
    rows = []
    for ev in diary.events:
        d = asdict(ev)
        d.pop("t")             # wall-clock by design; see the docstring
        rows.append(d)
    return rows


def _dy_digest(diary) -> str:
    return hashlib.sha256(json.dumps(_diary_rows(diary), sort_keys=True)
                          .encode()).hexdigest()


def _feature(w: W0, wm, diary) -> dict:
    buf = _mj_state(w)
    s = w.drives.state
    p = w.data.xpos[w.rover_bid]
    return {"g": _g_digest(buf), "t": float(w.sim_seconds),
            "x": float(p[0]), "y": float(p[1]), "z": float(p[2]),
            "e": float(s.e), "i": float(s.i), "w": float(s.w),
            "wm": _wm_digest(wm), "dy": _dy_digest(diary)}


# ── checkpoint: capture / save / load / apply ──────────────────────────────
def _capture(w: W0, forager, wm, diary) -> dict:
    import io

    import torch
    world = {
        "mjstate": _mj_state(w).tobytes(),
        "warmstart": np.array(w.data.qacc_warmstart, dtype=np.float64),
        "audio": w._audio.tobytes(),
        "rng": w._rng.get_state(), "spawn_rng": w._spawn_rng.get_state(),
        "legal": None if w._legal is None else np.asarray(w._legal),
        "synth": {"listener_pos": np.asarray(w.synth.listener_pos),
                  "listener_yaw": float(w.synth.listener_yaw),
                  "events": list(w.synth.events),
                  "voice_events": list(w.synth.voice_events),
                  "prev_pairs": set(w.synth._prev_pairs),
                  "last_fired": dict(w.synth._last_fired)},
        "forager": {"target": forager.target,
                    "prev_xy": np.asarray(forager.prev_xy),
                    "prev_ate": dict(forager.prev_ate)},
    }
    for k in _W0_FIELDS:
        world[k] = getattr(w, k)
    dl = w.drives
    needs = {k: getattr(dl, k, None) for k in _DRIVE_FIELDS}
    needs["state"] = vars(dl.state)
    needs["prev_drive"] = vars(w._prev_drive)
    buf = io.BytesIO()
    torch.save({"sd": wm.state_dict(), "h": wm.h.detach().clone(),
                "step": int(wm.step_idx)}, buf)
    stores = {"world": pickle.dumps(world),
              "needs": pickle.dumps(needs),
              "diary": pickle.dumps([vars(ev) for ev in diary.events]),
              "wm": buf.getvalue()}
    return {"decision": int(w.decisions),
            "feature": _feature(w, wm, diary),
            "stores": stores,
            "sha": {k: hashlib.sha256(v).hexdigest()
                    for k, v in stores.items()}}


def _save_atomic(ckpt: dict, path: str) -> None:
    tmp = f"{path}.tmp"
    with open(tmp, "wb") as f:
        f.write(pickle.dumps(ckpt))
    os.replace(tmp, path)      # T0.05: the atomic write is what makes a
    # SIGKILL survivable — the file on disk is always a complete checkpoint.


def _load_checkpoint(path: str) -> dict:
    """Verify EVERY store's sha before anything is applied. This is the loud
    loader the control corrupts against: a mismatch RAISES, never defaults."""
    with open(path, "rb") as f:
        d = pickle.loads(f.read())
    for name in STORES:
        blob = d["stores"][name]
        got = hashlib.sha256(blob).hexdigest()
        if got != d["sha"][name]:
            raise CorruptStore(
                f"checkpoint store {name!r} failed integrity: sha {got[:12]} "
                f"!= recorded {d['sha'][name][:12]} — refusing to load a "
                f"corrupted life")
    return d


def _apply(ckpt: dict, seed: int, j0: float, alpha: float):
    """Rebuild the life from scratch and pour the four stores back in."""
    import io

    import mujoco
    import torch

    from ..drives import DriveState
    from EpisodicMemory import Event, _tokens
    w, diary, forager, wm = _build_life(seed, j0, alpha)

    world = pickle.loads(ckpt["stores"]["world"])
    buf = np.frombuffer(world["mjstate"], dtype=np.float64)
    spec = mujoco.mjtState.mjSTATE_INTEGRATION
    want = mujoco.mj_stateSize(w.model, spec)
    if buf.shape[0] != want:
        raise CorruptStore(f"mj state is {buf.shape[0]} floats, model wants "
                           f"{want}: not the same world")
    mujoco.mj_setState(w.model, w.data, buf.copy(), spec)
    mujoco.mj_forward(w.model, w.data)
    mujoco.mj_rnePostConstraint(w.model, w.data)
    # The forward just overwrote qacc_warmstart with a fresh solve at the
    # restored state; the reference's next substep starts its solver from the
    # SAVED warmstart, so put it back or the first substep diverges in its
    # final bits and chaos does the rest.
    w.data.qacc_warmstart[:] = world["warmstart"]
    for k in _W0_FIELDS:
        setattr(w, k, world[k])
    w._audio = np.frombuffer(world["audio"], dtype=np.float32).copy()
    w._rng.set_state(world["rng"])
    w._spawn_rng.set_state(world["spawn_rng"])
    w._legal = world["legal"]
    sy = world["synth"]
    w.synth.listener_pos = np.asarray(sy["listener_pos"]).copy()
    w.synth.listener_yaw = float(sy["listener_yaw"])
    w.synth.events = list(sy["events"])
    w.synth.voice_events = list(sy["voice_events"])
    w.synth._prev_pairs = set(sy["prev_pairs"])
    w.synth._last_fired = dict(sy["last_fired"])
    fg = world["forager"]
    forager.target = fg["target"]
    forager.prev_xy = np.asarray(fg["prev_xy"]).copy()
    forager.prev_ate = dict(fg["prev_ate"])

    needs = pickle.loads(ckpt["stores"]["needs"])
    for k in _DRIVE_FIELDS:
        setattr(w.drives, k, needs[k])
    w.drives.state = DriveState(**needs["state"])
    w._prev_drive = DriveState(**needs["prev_drive"])

    diary.events = []
    diary._tok = []
    for row in pickle.loads(ckpt["stores"]["diary"]):
        diary.events.append(Event(**row))
        diary._tok.append(_tokens(row["text"]))

    snap = torch.load(io.BytesIO(ckpt["stores"]["wm"]), weights_only=False)
    wm.load_state_dict(snap["sd"])
    wm.h = snap["h"]
    wm.step_idx = int(snap["step"])
    return w, diary, forager, wm


# ── child roles ─────────────────────────────────────────────────────────────
def _child_ref(tmp: str, seed: int, j0: float, alpha: float, n: int) -> None:
    w, diary, forager, wm = _build_life(seed, j0, alpha)
    rows = []
    for _ in range(n):
        _step_once(w, forager, wm, diary)
        rows.append(_feature(w, wm, diary))
    _dump_rows(f"{tmp}/ref.json", rows)


def _child_victim(tmp: str, seed: int, j0: float, alpha: float,
                  cap: int) -> None:
    w, diary, forager, wm = _build_life(seed, j0, alpha)
    for _ in range(cap):
        _step_once(w, forager, wm, diary)
        _save_atomic(_capture(w, forager, wm, diary), f"{tmp}/ckpt.pkl")
    sys.exit(3)                # reaching here means the kill missed


def _child_resume(tmp: str, seed: int, j0: float, alpha: float) -> None:
    ckpt = _load_checkpoint(f"{tmp}/ckpt.pkl")
    w, diary, forager, wm = _apply(ckpt, seed, j0, alpha)
    back = _feature(w, wm, diary)
    roundtrip = int(back == ckpt["feature"])
    rows = []
    for _ in range(WINDOW):
        _step_once(w, forager, wm, diary)
        rows.append(_feature(w, wm, diary))
    _dump_rows(f"{tmp}/resume.json", rows,
               extra={"roundtrip_ok": roundtrip,
                      "loaded_decision": ckpt["decision"]})


def _child_null(tmp: str, seed: int, j0: float, alpha: float) -> None:
    # Weights-only: _build_life re-derives the WorkingMemory weights from the
    # same torch seed (they never train here, so re-derivation IS the weights
    # file); world, needs, diary and hidden state are what a naive restart
    # loses, and this arm loses them.
    w, diary, forager, wm = _build_life(seed, j0, alpha)
    rows = []
    for _ in range(WINDOW):
        _step_once(w, forager, wm, diary)
        rows.append(_feature(w, wm, diary))
    _dump_rows(f"{tmp}/null.json", rows)


def _dump_rows(path: str, rows: list, extra: dict | None = None) -> None:
    out = {"rows": rows}
    out.update(extra or {})
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        json.dump(out, f)
    os.replace(tmp, path)


def _run_child(role: str, tmp: str, seed: int, j0: float, alpha: float,
               *args: str, expect_kill: bool = False):
    cmd = [PY, "-m", _MOD, role, tmp, str(seed), repr(j0), repr(alpha),
           *map(str, args)]
    proc = subprocess.Popen(cmd, cwd=REPO, stdout=subprocess.DEVNULL,
                            stderr=subprocess.PIPE)
    if not expect_kill:
        _, err = proc.communicate(timeout=CHILD_TIMEOUT_S)
        if proc.returncode != 0:
            raise RuntimeError(f"{role} child rc={proc.returncode}: "
                               f"{err.decode()[-800:]}")
        return proc.returncode
    return proc


def _kill_at(proc, ckpt_path: str, target: int) -> int:
    """Poll the checkpoint; SIGKILL the child once it has passed `target`."""
    deadline = time.time() + CHILD_TIMEOUT_S
    while time.time() < deadline:
        try:
            with open(ckpt_path, "rb") as f:
                if pickle.loads(f.read())["decision"] >= target:
                    break
        except (FileNotFoundError, EOFError, pickle.UnpicklingError):
            pass
        if proc.poll() is not None:
            break              # died on its own: rc will say so
        time.sleep(KILL_POLL_S)
    if proc.poll() is None:
        os.kill(proc.pid, signal.SIGKILL)
    proc.wait()
    return proc.returncode


# ── comparison ──────────────────────────────────────────────────────────────
_FLOATS_RESUME = ("t", "x", "y", "z", "e", "i", "w")
_FLOATS_NULL = ("x", "y", "z", "e", "i", "w")   # t excluded: a fresh clock
# trivially differs from a continued one, and the null must not be handed its
# divergence for free (same reason the g digest excludes time).


def _compare(ref_rows: list, rows: list, keys: tuple) -> tuple:
    g = sum(a["g"] == b["g"] for a, b in zip(ref_rows, rows)) / len(rows)
    wmf = sum(a["wm"] == b["wm"] for a, b in zip(ref_rows, rows)) / len(rows)
    dyf = sum(a["dy"] == b["dy"] for a, b in zip(ref_rows, rows)) / len(rows)
    delta = max(abs(a[k] - b[k])
                for a, b in zip(ref_rows, rows) for k in keys)
    return g, wmf, dyf, delta


def _experiment(seed: int) -> dict:
    j0, alpha, prov = _calibration()
    m: dict = {"calibrated": float(j0 is not None), **prov}
    if j0 is None:
        return m
    tmp = tempfile.mkdtemp(prefix=f"lf02_s{seed}_", dir="/data")
    _CACHE[seed] = tmp
    k_target = K_TARGET_BASE + K_TARGET_STEP * seed
    t0 = time.perf_counter()

    # 1. The victim: checkpoint every decision, die by SIGKILL mid-life.
    victim = _run_child("victim", tmp, seed, j0, alpha,
                        k_target + K_SLACK_MAX + 5, expect_kill=True)
    rc = _kill_at(victim, f"{tmp}/ckpt.pkl", k_target)
    ckpt = _load_checkpoint(f"{tmp}/ckpt.pkl")
    c = int(ckpt["decision"])
    m["killed_by_sigkill"] = float(rc == -signal.SIGKILL)
    m["ckpt_decision"] = float(c)
    m["ckpt_in_window"] = float(k_target <= c <= k_target + K_SLACK_MAX)

    # 2. The uninterrupted reference, run past the same window.
    _run_child("ref", tmp, seed, j0, alpha, c + WINDOW)
    ref = json.load(open(f"{tmp}/ref.json"))["rows"]

    # 3. Prefix determinism: the checkpointed row at C must equal the
    #    reference's row at C, or the apparatus cannot host a bitwise claim.
    m["prefix_ok"] = float(ckpt["feature"] == ref[c - 1])

    # 4. Resume in a fresh process; 5. the weights-only null.
    _run_child("resume", tmp, seed, j0, alpha)
    _run_child("null", tmp, seed, j0, alpha)
    res = json.load(open(f"{tmp}/resume.json"))
    nul = json.load(open(f"{tmp}/null.json"))["rows"]

    window_ref = ref[c:c + WINDOW]
    g, wmf, dyf, delta = _compare(window_ref, res["rows"], _FLOATS_RESUME)
    m["restore_roundtrip_ok"] = float(res["roundtrip_ok"])
    m["resume_g_match_frac"] = g
    m["resume_wm_match_frac"] = wmf
    m["resume_dy_match_frac"] = dyf
    m["resume_max_delta"] = delta
    ng, _, _, ndelta = _compare(window_ref, nul, _FLOATS_NULL)
    m["null_g_match_frac"] = ng
    m["null_max_delta"] = ndelta
    m["wall_s"] = round(time.perf_counter() - t0, 1)
    return m


def _control(seed: int) -> dict:
    """Truncate each store in turn: the loader must RAISE all four times."""
    import shutil
    tmp = _CACHE[seed]
    raised, errors = 0, []
    try:
        with open(f"{tmp}/ckpt.pkl", "rb") as f:
            good = pickle.loads(f.read())
        for name in STORES:
            bad = {"decision": good["decision"], "feature": good["feature"],
                   "stores": dict(good["stores"]), "sha": dict(good["sha"])}
            blob = bad["stores"][name]
            bad["stores"][name] = blob[: int(len(blob) * 0.6)]
            path = f"{tmp}/corrupt_{name}.pkl"
            with open(path, "wb") as f:
                f.write(pickle.dumps(bad))
            try:
                _load_checkpoint(path)
                errors.append(f"{name}:SILENT")
            except Exception as e:
                raised += 1
                errors.append(f"{name}:{type(e).__name__}")
        return {"corrupt_raised_frac": raised / len(STORES),
                "corrupt_outcomes": ";".join(errors)}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        _CACHE.pop(seed, None)


def _check(m: dict, c: dict):
    if m.get("calibrated") != 1.0 or c.get("calibrated", 1.0) != 1.0:
        return Status.VOID     # PS.01 refused; the refusal is in the metrics.
    if m.get("killed_by_sigkill") != 1.0 or m.get("ckpt_in_window") != 1.0:
        return Status.VOID     # the kill missed: nothing was interrupted, so
        # nothing was resumed. Apparatus, not refutation.
    if m.get("prefix_ok") != 1.0:
        return Status.VOID     # two identical uninterrupted processes
        # disagreed at C — the apparatus cannot host a bitwise claim and the
        # resume was never tested.
    return bool(
        m.get("restore_roundtrip_ok") == 1.0
        and m.get("resume_g_match_frac") == 1.0
        and m.get("resume_wm_match_frac") == 1.0
        and m.get("resume_dy_match_frac") == 1.0
        and m.get("resume_max_delta", 1.0) <= TRAJ_TOL
        and m.get("null_g_match_frac", 1.0) <= NULL_G_MATCH_MAX
        and m.get("null_max_delta", 0.0) >= NULL_DIVERGE_MIN
        and c.get("corrupt_raised_frac") == 1.0)


def run(ledger: Ledger | None = None):
    return run_spec(BY_ID["LF.02"], _experiment, _check,
                    control_fn=_control, ledger=ledger)


def _main() -> None:
    role, tmp, seed = sys.argv[1], sys.argv[2], int(sys.argv[3])
    j0, alpha = float(sys.argv[4]), float(sys.argv[5])
    if role == "ref":
        _child_ref(tmp, seed, j0, alpha, int(sys.argv[6]))
    elif role == "victim":
        _child_victim(tmp, seed, j0, alpha, int(sys.argv[6]))
    elif role == "resume":
        _child_resume(tmp, seed, j0, alpha)
    elif role == "null":
        _child_null(tmp, seed, j0, alpha)
    else:
        raise SystemExit(f"unknown role {role!r}")


if __name__ == "__main__":
    _main()
