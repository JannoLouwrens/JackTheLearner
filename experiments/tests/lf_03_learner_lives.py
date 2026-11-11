"""LF.03 — one whole life with the learner in the loop.

LF.01/LF.02 composed world + needs + diary + working memory around a SCRIPTED
forager. This spec puts the seated learning core (`wm-latent`, cores.py; seat
in docs/CHAMPIONS.md) where the script was, and asks the integration question
only: does the creature-as-built survive being run end to end?

    lethal W0, every W0 sense live (vision, audio, touch, proprio, needs;
    language present-and-dropped per W0-4), the learner choosing every action
    and updating its weights online, across >= N_DEATHS deaths, the diary
    written by the world and carried across every death, ONE save-and-resume
    mid-life in a fresh process, no crash, no non-finite state, weights
    measurably changing.

AN INTEGRATION CLAIM, NOT A PERFORMANCE CLAIM (ORDERS J3). Nothing here gates
on how long he lives or whether he gets better. life spans, eats and
optimiser steps are REPORTED so the next spec can size a performance claim.

THE LOOP is `survival.run_survival`'s core path, step for step — same arm
construction (torch seed 9000+seed, .eval()), same exploration schedule
(EXPLORE_STD decayed over a fixed horizon), same homeo-dr reward with D_DEATH
termination, same _TargetRing / _gae / _updates_due / lc_update, same
train_ratio 0.25 LC.02 certified. It is re-expressed as a stepping object
here ONLY because run_survival is one closure and a closure cannot be
checkpointed mid-life; editing survival.py would stale every LC certificate.
Every piece it uses is imported, none re-derived. Two declared differences:
(1) LF.02's boundary normalisation (mj_forward + mj_rnePostConstraint after
every decision) — required for a bit-exact resume, applied identically to
every arm; (2) the diary is EpisodicMemory and W0 writes the death rows into
it (W0-3), plus one "life began" row at birth.

SAVE-AND-RESUME (LF.02's mechanism, carrying the LEARNER this time). Process A
lives from birth to the first decision >= K_SAVE at which the current life has
lasted >= SAVE_MIN_INTO_LIFE decisions, checkpoints FIVE stores atomically
(world, needs, diary — LF.02's — plus `learner`: core weights, Adam state,
the replay ring, the open GAE segment, the exploration RNG, the torch batch
generator and global RNG, the decision counter, the optimiser-step counter),
each with its own sha256, then lives RESUME_CHECK more decisions recording a
feature row per decision as the UNINTERRUPTED reference. Process B — fresh,
nothing shared but the file — verifies every sha (CorruptStore, loud), rebuilds
the life and pours the stores back, lives the same RESUME_CHECK decisions, and
its rows must equal A's BIT FOR BIT (world digest, clock, pose, needs, a
learner digest over every weight + step count, diary digest). Then B lives on
to >= N_DEATHS deaths. A resume that is not bit-exact is a different life
wearing the same weights: FAIL. (LF.02 already certified the SIGKILL half; a
clean process exit here is declared, not hidden.)

CONTROLS THAT MUST FAIL, EACH ACTUALLY RUN (registry):
  frozen twin   the same life with the optimiser never stepped. The SAME
                detector that must read "weights changed" on the real life
                (relative L2 change >= WEIGHT_CHANGE_MIN AND optimiser steps
                > 0) must read "not changed" on it. A detector that passes the
                frozen twin cannot certify learning-in-the-loop.
  wiped diary   the same life with the diary emptied right after every death.
                The SAME detector that must read "everything crossed" on the
                real life (at every respawn, the diary holds every earlier
                life's death row) must read "lost" on it.
  Each control lives to CONTROL_DEATHS deaths (>= 2, so it crosses death
  twice) — smaller than the real life on purpose: the control tests the
  DETECTOR, and a detector that fires at 2 crossings fires at 5.

VOID LANES (apparatus declined; not refutations):
  - PS.01 calibration refused (LF.01's lane);
  - the life hit DECISION_CAP before N_DEATHS deaths (he outlived the
    apparatus — nothing about integration was refuted);
  - the save never happened (no mid-life decision found before the cap).

PILOT (2026-11-11, before this file): one natural life of wm-latent in lethal
W0 via run_survival, seed 0, e0=1.0 — 503.0 sim-s (~2,515 decisions), 140.8 s
wall, 860 optimiser steps, physics finite. Five deaths ~12.6k decisions
~12 min/seed; with controls ~20 min/seed; cpu<2h at 3 seeds, run detached.
"""
from __future__ import annotations

import hashlib
import json
import os
import pickle
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from ..protocol import Ledger, Status, borrow_metrics, run_spec
from ..registry import BY_ID
from ..w0 import W0
from EpisodicMemory import EpisodicMemory  # noqa: E402  (repo root on path via ..w0)
from .lf_02_life_saved_and_resumed import (  # noqa: E402
    _DRIVE_FIELDS, _W0_FIELDS, CorruptStore, _dy_digest, _g_digest,
    _mj_state, _save_atomic)

IMPL_DEPS = ["playground.py", "experiments/w0.py", "experiments/drives.py",
             "experiments/cores.py", "experiments/survival.py",
             "EpisodicMemory.py", "ContactAudio.py",
             "experiments/tests/lf_02_life_saved_and_resumed.py"]

REPO = Path(__file__).resolve().parents[2]
PY = "/data/venvs/jackthelearner/bin/python"
_MOD = "experiments.tests.lf_03_learner_lives"

# ── the pre-registered numbers, all of them, before the run ────────────────
ARM = "wm-latent"              # the Learning-core seat holder (CHAMPIONS.md)
N_DEATHS = 5                   # the order's ">= 5 deaths"
DECISION_CAP = 30_000          # ~2.4x the piloted 5-death length; past it VOID
HORIZON = 15_000               # exploration std decays over this many decisions
TRAIN_RATIO = 0.25             # LC.02's certified cadence (run_survival default)
K_SAVE_BASE, K_SAVE_STEP = 6_000, 97   # save in life ~3, drifting by seed
SAVE_MIN_INTO_LIFE = 100       # the save is MID-life, not at a respawn
RESUME_CHECK = 200             # decisions of bit-exact continuation compared
WEIGHT_CHANGE_MIN = 1e-3       # relative L2 change of all weights, end vs birth
CONTROL_DEATHS = 2             # each control crosses death twice
SENSES = ("vision", "audio", "touch", "proprio", "needs")
CHILD_TIMEOUT_S = 3_600

STORES = ("world", "needs", "diary", "learner")


def _calibration():
    b = borrow_metrics("PS.01", ("j0_ms", "alpha"))
    if not b.ok:
        return None, None, {**b.provenance, "borrow_refusal": b.refusal}
    return b.values["j0_ms"], b.values["alpha"], b.provenance


def _flat(core):
    import torch
    return torch.cat([p.detach().reshape(-1) for p in core.parameters()])


class _Life:
    """run_survival's core path as a stepping object (see the docstring)."""

    def __init__(self, seed: int, j0: float, alpha: float, *,
                 train: bool = True, wipe_diary: bool = False):
        import torch

        from ..cores import build_arm, make_optimizer
        from ..survival import _TargetRing
        torch.set_num_threads(1)
        self.seed, self.train, self.wipe_diary = seed, train, wipe_diary
        self.diary = EpisodicMemory()
        self.w = W0(seed=seed, j0=j0, alpha=alpha, lethal=True,
                    diary=self.diary)
        self.rng = np.random.RandomState(seed * 6553 + 11)
        self.gen = torch.Generator().manual_seed(seed * 7919 + 3)
        torch.manual_seed(9_000 + seed)
        self.core = build_arm(ARM)
        self.core.eval()                                   # F1
        self.opt = make_optimizer(self.core) if train else None
        self.ring = _TargetRing(self.w.observe())
        self.theta0 = _flat(self.core).clone()
        self.seg = {"obs": [], "act": [], "rew": [], "val": []}
        self.k = 0
        self.opt_steps = 0
        self.nonfinite_loss = 0
        self.sense_n = 0
        self.sense_s = {s: 0.0 for s in SENSES}
        self.sense_ss = {s: 0.0 for s in SENSES}
        self.sense_nonfinite = 0
        self.carried: list = []           # per respawn: all prior deaths kept?
        self.diary.record("did", "jack", "life began", importance=1.0,
                          t=0.0, meta={"life": 0})

    # -- one decision -----------------------------------------------------
    def step(self) -> None:
        import mujoco
        import torch

        from ..cores import ACTION_DIM, LC_BATCH, lc_update
        from ..survival import D_DEATH, EXPLORE_STD, _updates_due
        w = self.w
        obs = w.observe()
        for s in SENSES:
            v = obs[s].astype(np.float64)
            if not np.all(np.isfinite(v)):
                self.sense_nonfinite += 1
            self.sense_s[s] += float(v.sum())
            self.sense_ss[s] += float((v * v).sum())
        self.sense_n += 1
        obs_t = {kk: torch.from_numpy(v).unsqueeze(0) for kk, v in obs.items()}
        with torch.no_grad():
            s = self.core.shared_state(obs_t, dropped=W0.DROPPED)
            mean = self.core.act(obs_t, s).squeeze(0).numpy()
            v = float(self.core.critic(s))
        frac = min(1.0, self.k / max(1, HORIZON - 1))
        std = EXPLORE_STD[0] + (EXPLORE_STD[1] - EXPLORE_STD[0]) * frac
        a = np.clip(mean + std * self.rng.randn(ACTION_DIM), -1.0, 1.0)

        d_before = w.drives.state.d()
        w.decide(a)
        # LF.02's boundary normalisation — see its _step_once.
        mujoco.mj_forward(w.model, w.data)
        mujoco.mj_rnePostConstraint(w.model, w.data)
        died = w.died_this_decision
        d_after = D_DEATH if died else w.drives.state.d()
        r = d_before - d_after

        self.seg["obs"].append({kk: torch.from_numpy(x.copy())
                                for kk, x in obs.items()})
        self.seg["act"].append(torch.from_numpy(a.astype(np.float32)))
        self.seg["rew"].append(r)
        self.seg["val"].append(v)
        from ..survival import SEG
        if died or len(self.seg["rew"]) >= SEG:
            self._close_segment(boot=v, terminal=died)
        if self.train and self.ring.n >= LC_BATCH:
            for _ in range(_updates_due(w.decisions, TRAIN_RATIO)):
                batch, targets = self.ring.sample(self.gen)
                loss = lc_update(self.core, batch, targets, self.opt,
                                 dropped=W0.DROPPED)
                if not np.isfinite(loss):
                    self.nonfinite_loss += 1
                self.opt_steps += 1
        if died:
            if self.wipe_diary:
                self.diary.events = []
                self.diary._tok = []
            prior = sum(1 for ev in self.diary.events
                        if str(ev.text).startswith("life ended")
                        and (ev.meta or {}).get("life", 1e9) < w.life)
            self.carried.append(int(prior >= w.life))
        self.k += 1

    def _close_segment(self, boot: float, terminal: bool) -> None:
        from ..survival import _gae
        sg = self.seg
        if not sg["rew"]:
            return
        adv, tgt = _gae(sg["rew"], sg["val"], boot, terminal)
        for i in range(len(sg["rew"])):
            self.ring.push(sg["obs"][i], sg["act"][i], float(tgt[i]),
                           float(adv[i]), value_lp=0.0)
        self.seg = {"obs": [], "act": [], "rew": [], "val": []}

    # -- read-outs ----------------------------------------------------------
    def learner_digest(self) -> str:
        h = hashlib.sha256(_flat(self.core).numpy().tobytes())
        h.update(f"{self.opt_steps}:{self.ring.n}:{self.k}".encode())
        return h.hexdigest()

    def feature(self) -> dict:
        w = self.w
        st = w.drives.state
        p = w.data.xpos[w.rover_bid]
        return {"g": _g_digest(_mj_state(w)), "t": float(w.sim_seconds),
                "x": float(p[0]), "y": float(p[1]), "z": float(p[2]),
                "e": float(st.e), "i": float(st.i), "w": float(st.w),
                "ln": self.learner_digest(), "dy": _dy_digest(self.diary)}

    def facts(self) -> dict:
        import torch
        w = self.w
        theta = _flat(self.core)
        rel = float(torch.linalg.vector_norm(theta - self.theta0)
                    / torch.linalg.vector_norm(self.theta0))
        stds = {}
        for s in SENSES:
            m1 = self.sense_s[s]
            m2 = self.sense_ss[s]
            # pooled over every channel of the sense and every decision
            n = max(1, self.sense_n) * self._dims[s]
            var = max(0.0, m2 / n - (m1 / n) ** 2)
            stds[s] = var ** 0.5
        params_finite = bool(torch.all(torch.isfinite(theta)))
        return {
            "deaths": int(w.deaths), "decisions": int(w.decisions),
            "sim_seconds": float(w.sim_seconds),
            "life_spans": [round(float(x), 2) for x in w.life_lengths],
            "optimiser_steps": int(self.opt_steps),
            "weight_change_rel": rel,
            "nonfinite_loss": int(self.nonfinite_loss),
            "sense_std": stds,
            "sense_nonfinite": int(self.sense_nonfinite),
            "physics_finite": bool(np.all(np.isfinite(w.data.qpos))
                                   and np.all(np.isfinite(w.data.qvel))),
            "params_finite": params_finite,
            "carried": list(self.carried),
            "diary_rows": len(self.diary.events),
            "ate_total": int(sum(w.drives.ate_total.values())),
        }

    @property
    def _dims(self) -> dict:
        from ..cores import MODALITIES
        return {s: MODALITIES[s] for s in SENSES}


# ── the five-store checkpoint ──────────────────────────────────────────────
def _capture(L: _Life) -> dict:
    import io

    import torch
    w = L.w
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
    }
    for k in _W0_FIELDS:
        world[k] = getattr(w, k)
    dl = w.drives
    needs = {k: getattr(dl, k, None) for k in _DRIVE_FIELDS}
    needs["state"] = vars(dl.state)
    needs["prev_drive"] = vars(w._prev_drive)
    buf = io.BytesIO()
    ring = L.ring
    torch.save({
        "core": L.core.state_dict(),
        "opt": None if L.opt is None else L.opt.state_dict(),
        "ring": {"obs": ring.obs, "action": ring.action, "value": ring.value,
                 "value_lp": ring.value_lp, "advantage": ring.advantage,
                 "n": ring.n},
        "seg": L.seg, "theta0": L.theta0,
        "np_rng": L.rng.get_state(), "gen": L.gen.get_state(),
        "torch_rng": torch.get_rng_state(),
        "k": L.k, "opt_steps": L.opt_steps,
        "nonfinite_loss": L.nonfinite_loss,
        "sense": (L.sense_n, L.sense_s, L.sense_ss, L.sense_nonfinite),
        "carried": L.carried,
    }, buf)
    stores = {"world": pickle.dumps(world), "needs": pickle.dumps(needs),
              "diary": pickle.dumps([vars(ev) for ev in L.diary.events]),
              "learner": buf.getvalue()}
    return {"decision": int(w.decisions), "feature": L.feature(),
            "stores": stores,
            "sha": {k: hashlib.sha256(v).hexdigest()
                    for k, v in stores.items()}}


def _load_checkpoint(path: str) -> dict:
    with open(path, "rb") as f:
        d = pickle.loads(f.read())
    for name in STORES:
        got = hashlib.sha256(d["stores"][name]).hexdigest()
        if got != d["sha"][name]:
            raise CorruptStore(f"checkpoint store {name!r} failed integrity: "
                               f"refusing to load a corrupted life")
    return d


def _apply(ckpt: dict, seed: int, j0: float, alpha: float) -> _Life:
    import io

    import mujoco
    import torch

    from ..drives import DriveState
    from EpisodicMemory import Event, _tokens
    L = _Life(seed, j0, alpha)
    w = L.w
    world = pickle.loads(ckpt["stores"]["world"])
    buf = np.frombuffer(world["mjstate"], dtype=np.float64)
    spec = mujoco.mjtState.mjSTATE_INTEGRATION
    if buf.shape[0] != mujoco.mj_stateSize(w.model, spec):
        raise CorruptStore("mj state size differs: not the same world")
    mujoco.mj_setState(w.model, w.data, buf.copy(), spec)
    mujoco.mj_forward(w.model, w.data)
    mujoco.mj_rnePostConstraint(w.model, w.data)
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

    needs = pickle.loads(ckpt["stores"]["needs"])
    for k in _DRIVE_FIELDS:
        setattr(w.drives, k, needs[k])
    w.drives.state = DriveState(**needs["state"])
    w._prev_drive = DriveState(**needs["prev_drive"])

    L.diary.events, L.diary._tok = [], []
    for row in pickle.loads(ckpt["stores"]["diary"]):
        L.diary.events.append(Event(**row))
        L.diary._tok.append(_tokens(row["text"]))

    sn = torch.load(io.BytesIO(ckpt["stores"]["learner"]), weights_only=False)
    L.core.load_state_dict(sn["core"])
    L.opt.load_state_dict(sn["opt"])
    for k in ("obs", "action", "value", "value_lp", "advantage", "n"):
        setattr(L.ring, k, sn["ring"][k])
    L.seg, L.theta0 = sn["seg"], sn["theta0"]
    L.rng.set_state(sn["np_rng"])
    L.gen.set_state(sn["gen"])
    torch.set_rng_state(sn["torch_rng"])
    L.k, L.opt_steps = sn["k"], sn["opt_steps"]
    L.nonfinite_loss = sn["nonfinite_loss"]
    L.sense_n, L.sense_s, L.sense_ss, L.sense_nonfinite = sn["sense"]
    L.carried = sn["carried"]
    return L


# ── child roles ─────────────────────────────────────────────────────────────
def _dump(path: str, obj: dict) -> None:
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f)
    os.replace(tmp, path)


def _child_first(tmp: str, seed: int, j0: float, alpha: float) -> None:
    L = _Life(seed, j0, alpha)
    k_save = K_SAVE_BASE + K_SAVE_STEP * seed
    saved = None
    while L.k < DECISION_CAP:
        L.step()
        if (L.k >= k_save and not L.w.died_this_decision
                and (L.w.sim_seconds - L.w._life_started_at)
                >= SAVE_MIN_INTO_LIFE * (L.w.sim_seconds / L.w.decisions)):
            _save_atomic(_capture(L), f"{tmp}/ckpt.pkl")
            saved = {"decision": L.k, "life": int(L.w.life),
                     "into_life_s": float(L.w.sim_seconds
                                          - L.w._life_started_at)}
            break
    rows = []
    if saved is not None:
        for _ in range(RESUME_CHECK):
            L.step()
            rows.append(L.feature())
    _dump(f"{tmp}/first.json", {"saved": saved, "rows": rows})


def _child_second(tmp: str, seed: int, j0: float, alpha: float) -> None:
    ckpt = _load_checkpoint(f"{tmp}/ckpt.pkl")
    L = _apply(ckpt, seed, j0, alpha)
    roundtrip = int(L.feature() == ckpt["feature"])
    rows = []
    for _ in range(RESUME_CHECK):
        L.step()
        rows.append(L.feature())
    while L.w.deaths < N_DEATHS and L.k < DECISION_CAP:
        L.step()
    _dump(f"{tmp}/second.json", {"roundtrip_ok": roundtrip, "rows": rows,
                                 "facts": L.facts()})


def _run_child(role: str, tmp: str, seed: int, j0: float, alpha: float):
    cmd = [PY, "-m", _MOD, role, tmp, str(seed), repr(j0), repr(alpha)]
    p = subprocess.run(cmd, cwd=REPO, stdout=subprocess.DEVNULL,
                       stderr=subprocess.PIPE, timeout=CHILD_TIMEOUT_S)
    if p.returncode != 0:
        raise RuntimeError(f"{role} child rc={p.returncode}: "
                           f"{p.stderr.decode()[-800:]}")


# ── the shared detectors (experiment and controls read the SAME ones) ──────
def _weights_changed(f: dict) -> bool:
    return f["optimiser_steps"] > 0 and f["weight_change_rel"] >= WEIGHT_CHANGE_MIN


def _diary_carried(f: dict) -> bool:
    return len(f["carried"]) >= 1 and all(f["carried"])


def _experiment(seed: int) -> dict:
    import shutil
    j0, alpha, prov = _calibration()
    m: dict = {"calibrated": float(j0 is not None), **prov}
    if j0 is None:
        return m
    tmp = tempfile.mkdtemp(prefix=f"lf03_s{seed}_", dir="/data/tmp")
    t0 = time.perf_counter()
    try:
        _run_child("first", tmp, seed, j0, alpha)
        first = json.load(open(f"{tmp}/first.json"))
        m["saved"] = float(first["saved"] is not None)
        if first["saved"] is None:
            return m
        m["save_decision"] = float(first["saved"]["decision"])
        m["save_into_life_s"] = first["saved"]["into_life_s"]
        _run_child("second", tmp, seed, j0, alpha)
        sec = json.load(open(f"{tmp}/second.json"))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    f = sec["facts"]
    ref, res = first["rows"], sec["rows"]
    m["resume_roundtrip_ok"] = float(sec["roundtrip_ok"])
    m["resume_exact_frac"] = (sum(a == b for a, b in zip(ref, res))
                              / max(1, len(ref)))
    m["deaths"] = float(f["deaths"])
    m["reached_deaths"] = float(f["deaths"] >= N_DEATHS)
    m["decisions"] = float(f["decisions"])
    m["life_spans"] = json.dumps(f["life_spans"])
    m["optimiser_steps"] = float(f["optimiser_steps"])
    m["weight_change_rel"] = f["weight_change_rel"]
    m["weights_changed"] = float(_weights_changed(f))
    m["diary_carried"] = float(_diary_carried(f))
    m["diary_rows"] = float(f["diary_rows"])
    m["ate_total"] = float(f["ate_total"])
    for s, v in f["sense_std"].items():
        m[f"sense_std_{s}"] = v
    m["senses_live"] = float(all(v > 0 for v in f["sense_std"].values()))
    m["finite"] = float(f["physics_finite"] and f["params_finite"]
                        and f["nonfinite_loss"] == 0
                        and f["sense_nonfinite"] == 0)
    m["wall_s"] = round(time.perf_counter() - t0, 1)
    return m


def _live_to(L: _Life, deaths: int) -> dict:
    while L.w.deaths < deaths and L.k < DECISION_CAP:
        L.step()
    return L.facts()


def _control(seed: int) -> dict:
    j0, alpha, _ = _calibration()
    if j0 is None:
        return {"calibrated": 0.0}
    fz = _live_to(_Life(seed, j0, alpha, train=False), CONTROL_DEATHS)
    wd = _live_to(_Life(seed, j0, alpha, wipe_diary=True), CONTROL_DEATHS)
    return {
        "frozen_deaths": float(fz["deaths"]),
        "frozen_weight_change_rel": fz["weight_change_rel"],
        "frozen_flagged": float(not _weights_changed(fz)),
        "wiped_deaths": float(wd["deaths"]),
        "wiped_carried": json.dumps(wd["carried"]),
        "wiped_flagged": float(not _diary_carried(wd)),
        # the wiped twin still learns: its weights detector must NOT fire,
        # or the two detectors are not independent
        "wiped_weights_changed": float(_weights_changed(wd)),
    }


def _check(m: dict, c: dict):
    if m.get("calibrated") != 1.0 or c.get("calibrated", 1.0) != 1.0:
        return Status.VOID
    if m.get("saved") != 1.0:
        return Status.VOID     # no mid-life save point before the cap
    if m.get("reached_deaths") != 1.0:
        return Status.VOID     # he outlived the apparatus; not a refutation
    if (c.get("frozen_deaths", 0) < CONTROL_DEATHS
            or c.get("wiped_deaths", 0) < CONTROL_DEATHS):
        return Status.VOID     # a control that never crossed death tested nothing
    # every per-seed flag is 0/1 and aggregated as a mean: == 1.0 is EVERY seed
    return bool(
        m.get("resume_roundtrip_ok") == 1.0
        and m.get("resume_exact_frac") == 1.0
        and m.get("weights_changed") == 1.0
        and m.get("diary_carried") == 1.0
        and m.get("senses_live") == 1.0
        and m.get("finite") == 1.0
        and c.get("frozen_flagged") == 1.0
        and c.get("wiped_flagged") == 1.0
        and c.get("wiped_weights_changed") == 1.0)


def run(ledger: Ledger | None = None):
    return run_spec(BY_ID["LF.03"], _experiment, _check,
                    control_fn=_control, ledger=ledger)


def _main() -> None:
    role, tmp, seed = sys.argv[1], sys.argv[2], int(sys.argv[3])
    j0, alpha = float(sys.argv[4]), float(sys.argv[5])
    if role == "first":
        _child_first(tmp, seed, j0, alpha)
    elif role == "second":
        _child_second(tmp, seed, j0, alpha)
    else:
        raise SystemExit(f"unknown role {role!r}")


if __name__ == "__main__":
    _main()
