"""LF.06 — the hand-over: a Jack trained here still learns on another machine.

LEGS REVERSED (ruling 7, docs/DECISIONS_RESOLVED.md 2026-11-14 #7; ORDERS J8).
"Here" = a Kaggle CPU kernel: N_HERE learning lives, the hand-over file, the
exam taken there (`_here`, one kernel per claim seed, the checkpoint comes
back through gpu.submit's fetch-out). "There" = THIS box (other OS image,
other CPU): the frozen exam, M_THERE further learning lives, the frozen
hand-over twin, and both controls, as niced child processes, <= 2 at once.
Claim, seeds, bars and both controls unchanged; which machine is which does
not enter the claim. No input channel is built. Only `run()` writes the
ledger, through run_spec.

Pieces, each reusing the creature's existing code paths, never re-deriving:
  checkpoint   LF.03's five-store `_capture` / `_load_checkpoint` / `_apply`
               (world, needs, diary, learner = weights + optimiser + replay
               ring + RNGs). The hand-over file IS that checkpoint.
  strip        CONTROL 1: same file with the diary emptied, the optimiser
               reset to a fresh one and the replay ring emptied — weights only.
  fresh        CONTROL 2: same architecture, NEW init (torch seed salted),
               nothing carried.
  exam         K_EXAM frozen lives: optimiser never stepped, exploration std
               pinned at its schedule's END value (L.k = HORIZON), exam world
               seeds 9000 + 10*seed + i, ONE fixed noise salt; exam = mean
               sim-s. A censored exam life (LIFE_CAP) -> the exam is VOID.
  tol_from_pilot  TOL_S = max(20, 2 * sd_exam * sqrt(2/5)), sd_exam the pooled
               single-life sd of the seed-90 pilot exam over 2 noise salts
               (LF.04's margin rule, in the exam's own units).

All numbers are copied from the registered spec's notes; if this file and the
spec disagree, the spec wins and this file is the bug.
"""
from __future__ import annotations

import json
import os
import pickle
import platform
import subprocess
import sys

import numpy as np

from .lf_03_learner_lives import (HORIZON, _apply, _calibration, _capture,
                                  _load_checkpoint)
from .lf_04_living_teaches import LIFE_CAP, _Life4, live_lives

N_HERE = 10
M_THERE = 10
K_EXAM = 5
CLAIM_SEEDS = (110, 111, 112)
PILOT_SEED = 90
TOL_FLOOR_S = 20.0
MARGIN_S = 272.812                    # LF.04's declared margin, same venue
EXAM_SALT = 2_000_029                 # the ONE fixed exam noise salt
PILOT_SALTS = (EXAM_SALT, 3_000_017)  # the pilot's 2 noise salts
FRESH_TORCH_SALT = 77_777
PILOT_FILE = "/data/tmp/lf06/exam_pilot_s90.jsonl"


def exam_seed(seed: int, i: int) -> int:
    return 9000 + 10 * seed + i


# ── the hand-over file and its two controls ────────────────────────────────
def handover(L) -> dict:
    """The checkpoint that travels: LF.03's five stores, unchanged."""
    return _capture(L)


def strip(ckpt: dict, seed: int, j0: float, alpha: float):
    """CONTROL 1: weights only. Load the full file, then empty the diary,
    replace the optimiser with a fresh one and empty the replay ring."""
    from ..cores import make_optimizer
    L = _apply(ckpt, seed, j0, alpha)
    L.diary.events, L.diary._tok = [], []
    L.opt = make_optimizer(L.core)
    L.ring.n = 0
    L.seg = {"obs": [], "act": [], "rew": [], "val": []}
    return L


def fresh(seed: int, j0: float, alpha: float, train: bool = False):
    """CONTROL 2: same architecture, a NEW init; nothing carried."""
    import torch
    L = _Life4(seed, j0, alpha, train=train)
    torch.manual_seed(9_000 + seed + FRESH_TORCH_SALT)
    from ..cores import build_arm, make_optimizer
    from .lf_03_learner_lives import ARM
    L.core = build_arm(ARM)
    L.core.eval()
    L.opt = make_optimizer(L.core) if train else None
    return L


# ── the frozen exam ────────────────────────────────────────────────────────
def exam(core_state: dict, seed: int, j0: float, alpha: float,
         salt: int = EXAM_SALT, out=None, tag: str = "exam") -> dict:
    """K_EXAM frozen lives, one per exam world seed, the given weights.
    Returns {"exam_s": mean sim-s | None, "spans": [...], "void": reason}."""
    spans, void = [], ""
    for i in range(K_EXAM):
        L = _Life4(exam_seed(seed, i), j0, alpha, train=False,
                   explore_salt=salt)
        L.core.load_state_dict(core_state)
        L.k = HORIZON                  # std pinned at the schedule's END
        rows = live_lives(L, 1, out, tag=f"{tag}:s{seed}:e{i}")
        if rows[0]["censored"]:
            void = f"exam life {i} censored at LIFE_CAP={LIFE_CAP}"
            break
        spans.append(rows[0]["sim_s"])
    return {"exam_s": None if void else float(np.mean(spans)),
            "spans": spans, "void": void}


def tol_from_pilot(path: str = PILOT_FILE) -> dict:
    """The declared TOL_S rule, from the seed-90 pilot exam rows only."""
    rows = [json.loads(x) for x in open(path)]
    ex = [r for r in rows if r["tag"].startswith("exam")]
    if len(ex) < 2 * K_EXAM or any(r["censored"] for r in ex):
        return {"tol_s": None, "why": f"pilot incomplete or censored "
                                      f"({len(ex)} rows)"}
    sd = float(np.std([r["sim_s"] for r in ex], ddof=1))
    return {"tol_s": max(TOL_FLOOR_S, 2.0 * sd * np.sqrt(2.0 / K_EXAM)),
            "sd_exam": sd, "n": len(ex)}


def _pilot(out_path: str = PILOT_FILE) -> None:
    """Seed 90 (not a claim seed): N_HERE learning lives, then the frozen
    exam of the end weights under both pilot salts. Streams JSON lines."""
    import os
    j0, alpha, prov = _calibration()
    if j0 is None:
        raise SystemExit(f"PS.01 calibration refused: {prov}")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    L = _Life4(PILOT_SEED, j0, alpha, train=True)
    with open(out_path, "a") as out:
        live_lives(L, N_HERE, out, tag=f"here:s{PILOT_SEED}")
        state = {k: v.clone() for k, v in L.core.state_dict().items()}
        for salt in PILOT_SALTS:
            exam(state, PILOT_SEED, j0, alpha, salt=salt, out=out,
                 tag=f"exam{salt}")


def _selftest() -> dict:
    """No lives: build a creature, take a few decisions, hand over, and check
    that the full file carries every store, that strip() drops exactly the
    stores it declares, and that fresh() differs in weights."""
    import torch

    from .lf_03_learner_lives import _flat, _Life
    j0, alpha, prov = _calibration()
    if j0 is None:
        return {"ok": False, "why": f"calibration refused: {prov}"}
    L = _Life(PILOT_SEED, j0, alpha)
    while L.opt_steps < 2:             # until the ring + optimiser are live
        L.step()
    ck = handover(L)
    full = _apply(ck, PILOT_SEED, j0, alpha)
    st = strip(ck, PILOT_SEED, j0, alpha)
    fr = fresh(PILOT_SEED, j0, alpha)
    w_full = torch.equal(_flat(full.core), _flat(L.core))
    w_strip = torch.equal(_flat(st.core), _flat(L.core))
    w_fresh = torch.equal(_flat(fr.core), _flat(L.core))
    res = {
        "full_weights_equal": w_full,
        "full_diary_n": len(full.diary.events),
        "full_ring_n": int(full.ring.n),
        "full_opt_state_n": len(full.opt.state_dict()["state"]),
        "strip_weights_equal": w_strip,
        "strip_diary_n": len(st.diary.events),
        "strip_ring_n": int(st.ring.n),
        "strip_opt_state_n": len(st.opt.state_dict()["state"]),
        "fresh_weights_equal": w_fresh,
    }
    res["ok"] = bool(w_full and w_strip and not w_fresh
                     and res["full_diary_n"] > 0 and res["full_ring_n"] > 0
                     and res["full_opt_state_n"] > 0 and res["strip_diary_n"] == 0
                     and res["strip_ring_n"] == 0
                     and res["strip_opt_state_n"] == 0)
    return res


# ── the claim run: legs reversed (ruling 7) ────────────────────────────────
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(
    __file__))))
PY = "/data/venvs/jackthelearner/bin/python"
_MOD = "experiments.tests.lf_06_handover"
LAST = 5                              # "his last five lives there"
TOL_S = 20.0                          # tol_from_pilot() on the s90 pilot:
                                      # sd_exam 1.31 (n 10) -> the floor, 20.0
CLAIM_DIR = "/data/tmp/lf06"
REMOTE_TIMEOUT_S = 4 * 3600           # 10 lives + 5 exam lives, one process
CHILD_TIMEOUT_S = 4 * 3600


def hardware() -> str:
    """The machine string the VOID clause compares: arch | CPU | OS image."""
    cpu = ""
    try:
        for ln in open("/proc/cpuinfo"):
            k = ln.split(":")[0].strip().lower()
            if k in ("model name", "cpu part", "hardware") and not cpu:
                cpu = ln.split(":", 1)[1].strip()
    except OSError:
        pass
    return f"{platform.machine()} | {cpu} | {platform.platform()}"


def _ckpt_bytes(ck: dict) -> bytes:
    return pickle.dumps(ck)            # what _load_checkpoint reads back


def _here(seed: int, out_dir: str) -> dict:
    """THE KERNEL'S LEG: N_HERE learning lives on a claim seed, the hand-over
    file written beside the exam taken here. Returns the JSON it also writes."""
    j0, alpha, prov = _calibration()
    if j0 is None:
        r = {"calibrated": 0.0, "prov": str(prov)}
    else:
        L = _Life4(seed, j0, alpha, train=True)
        rows = live_lives(L, N_HERE, None, tag=f"here:s{seed}")
        ck = handover(L)
        with open(os.path.join(out_dir, f"lf06_ckpt_s{seed}.pkl"), "wb") as f:
            f.write(_ckpt_bytes(ck))
        state = {k: v.clone() for k, v in L.core.state_dict().items()}
        ex = exam(state, seed, j0, alpha, tag="exam_here")
        r = {"calibrated": 1.0, "seed": seed, "here_rows": rows,
             "exam_here": ex, "hw_here": hardware(), "sha": ck["sha"],
             "decision": ck["decision"]}
    with open(os.path.join(out_dir, f"lf06_here_s{seed}.json"), "w") as f:
        json.dump(r, f)
    return r


_JOB = r"""
import subprocess as _s, sys as _y
_s.run([_y.executable, "-m", "pip", "install", "-q", "mujoco"], check=True)
import os
from experiments.tests.lf_06_handover import _here
_here(__SEED__, os.environ["JACK_OUT"])
print("DONE", flush=True)
"""


def _claim_dir(seed: int) -> str:
    d = f"{CLAIM_DIR}/claim_s{seed}"
    os.makedirs(d, exist_ok=True)
    return d


def _fetch_here(seed: int) -> dict:
    """The hand-over arrives: one Kaggle CPU kernel, or its cached result if
    THIS seed's file is already here with a matching checkpoint sha."""
    import hashlib
    d = _claim_dir(seed)
    jp, cp = f"{d}/here.json", f"{d}/ckpt.pkl"
    if os.path.exists(jp) and os.path.exists(cp):
        r = json.load(open(jp))
        ck = _load_checkpoint(cp)
        if r.get("sha") == ck["sha"]:
            return r
    from ..gpu import build_job, result_json, submit
    names = [f"lf06_here_s{seed}.json", f"lf06_ckpt_s{seed}.pkl"]
    res = submit(build_job(_JOB.replace("__SEED__", str(int(seed)))),
                 prefer="kaggle-cpu", est_hours=1.0,
                 timeout_s=REMOTE_TIMEOUT_S, fetch=names)
    if not res.ok:
        raise RuntimeError(f"LF.06 kaggle-cpu here s{seed}: {res.message}")
    r = result_json(res, names[0])
    if r.get("calibrated") != 1.0:
        return r
    blob = open(res.artifacts[names[1]], "rb").read()
    ck = pickle.loads(blob)
    for k in ck["sha"]:                # the file that crossed must be intact
        if hashlib.sha256(ck["stores"][k]).hexdigest() != r["sha"][k]:
            raise RuntimeError(f"LF.06 s{seed}: store {k!r} changed in transit")
    with open(cp, "wb") as f:
        f.write(blob)
    with open(jp, "w") as f:
        json.dump(r, f)
    return r


def _there_child(role: str, seed: int, ckpt_path: str, out_path: str) -> None:
    """ONE box-side job on a claim seed, from the hand-over file.
    learner  : the full file, M_THERE learning lives
    twin     : the full file, frozen (never updated), same world + noise
    strip    : CONTROL 1, weights only, M_THERE learning lives
    exam     : the frozen exam of the file's weights
    exam_strip: CONTROL 1's exam, of the stripped file's weights
    exam_fresh: CONTROL 2, the frozen exam of a NEW init."""
    import torch
    torch.set_num_threads(1)
    j0, alpha, prov = _calibration()
    if j0 is None:
        raise SystemExit(f"PS.01 calibration refused: {prov}")
    ck = _load_checkpoint(ckpt_path)
    with open(out_path, "w") as out:
        if role in ("learner", "twin"):
            L = _apply(ck, seed, j0, alpha)
            if role == "twin":
                L.train, L.opt = False, None
            live_lives(L, M_THERE, out, tag=f"{role}:s{seed}")
        elif role == "strip":
            live_lives(strip(ck, seed, j0, alpha), M_THERE, out,
                       tag=f"strip:s{seed}")
        elif role in ("exam", "exam_strip", "exam_fresh"):
            if role == "exam":
                core = _apply(ck, seed, j0, alpha).core
            elif role == "exam_strip":
                core = strip(ck, seed, j0, alpha).core
            else:
                core = fresh(seed, j0, alpha).core
            state = {k: v.clone() for k, v in core.state_dict().items()}
            r = exam(state, seed, j0, alpha, out=None, tag=role)
            out.write(json.dumps({"tag": role, "exam": r,
                                  "hw": hardware()}) + "\n")
        else:
            raise SystemExit(f"unknown there role {role!r}")


def _there(roles, seed: int) -> dict:
    """The box's jobs, <= 2 niced children at once. {role: rows}."""
    d = _claim_dir(seed)
    res, todo = {}, list(roles)
    while todo:
        batch, todo = todo[:2], todo[2:]
        from ..cpu_budget import admit_detached
        adm = admit_detached(f"LF.06:there:s{seed}")
        if not adm.admitted:           # a kcpu spec's box legs still answer
            raise RuntimeError(f"LF.06 box legs refused: {adm.reason}")
        procs = {}
        for role in batch:             # billed to the CPU day by wall clock
            out = f"{d}/there_{role}.jsonl"
            procs[role] = (subprocess.Popen(
                [PY, "-m", "experiments.cpu_budget", "wrap",
                 f"LF.06:there:{role}", "nice", "-n", "19", PY, "-m", _MOD,
                 "there", role, str(seed), f"{d}/ckpt.pkl", out], cwd=REPO,
                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE), out)
        for role, (p, out) in procs.items():
            _, err = p.communicate(timeout=CHILD_TIMEOUT_S)
            if p.returncode != 0:
                raise RuntimeError(f"LF.06 there {role} s{seed} rc="
                                   f"{p.returncode}: {err.decode()[-800:]}")
            res[role] = [json.loads(x) for x in open(out)]
    return res


def _last(rows) -> tuple:
    """(mean sim-s of the last LAST of M_THERE lives, any censored). A
    censored life ends the arm; its span fills the lives it did not live."""
    sp = [r["sim_s"] for r in rows]
    cens = any(r["censored"] for r in rows)
    if rows and rows[-1]["censored"]:
        sp += [sp[-1]] * (M_THERE - len(sp))
    return float(np.mean(sp[-LAST:])), cens


def _experiment(i: int) -> dict:
    seed = CLAIM_SEEDS[i]
    h = _fetch_here(seed)
    if h.get("calibrated") != 1.0:
        return {"calibrated": 0.0}
    t = _there(["exam", "learner", "twin"], seed)
    ex_here = h["exam_here"]
    ex_there = t["exam"][0]["exam"]
    hw_there = t["exam"][0]["hw"]
    m = {"calibrated": 1.0,
         "exam_void": float(bool(ex_here["void"] or ex_there["void"])),
         "hw_same": float(h["hw_here"] == hw_there),
         "hw_here": h["hw_here"], "hw_there": hw_there,
         "here_censored": float(any(r["censored"] for r in h["here_rows"])),
         "tol_s": TOL_S, "margin_s": MARGIN_S}
    if m["exam_void"]:
        return m
    lL, cL = _last(t["learner"])
    lT, cT = _last(t["twin"])
    d_exam = abs(ex_there["exam_s"] - ex_here["exam_s"])
    m.update({
        "exam_here_s": ex_here["exam_s"], "exam_there_s": ex_there["exam_s"],
        "exam_gap_s": d_exam, "a_match": float(d_exam <= TOL_S),
        "last5_learner_s": lL, "last5_twin_s": lT, "gain_vs_twin_s": lL - lT,
        "b_gain": float(lL - lT >= MARGIN_S),
        "learner_censored": float(cL), "twin_censored": float(cT),
        "spans_learner": json.dumps([r["sim_s"] for r in t["learner"]]),
        "spans_twin": json.dumps([r["sim_s"] for r in t["twin"]]),
        "spans_here": json.dumps([r["sim_s"] for r in h["here_rows"]]),
    })
    return m


def _control(i: int) -> dict:
    seed = CLAIM_SEEDS[i]
    h = _fetch_here(seed)              # cached by _experiment
    if h.get("calibrated") != 1.0:
        return {"calibrated": 0.0}
    d = _claim_dir(seed)
    twin = [json.loads(x) for x in open(f"{d}/there_twin.jsonl")]
    t = _there(["strip", "exam_strip", "exam_fresh"], seed)
    lT, _ = _last(twin)
    lS, cS = _last(t["strip"])
    ex_here = h["exam_here"]["exam_s"]
    exf = t["exam_fresh"][0]["exam"]
    exs = t["exam_strip"][0]["exam"]   # both controls RUN through gate (a)
    strip_a = (not exs["void"]) and abs(exs["exam_s"] - ex_here) <= TOL_S
    strip_b = lS - lT >= MARGIN_S      # the hand-over twin is the null for
                                       # (b), as it is for the experiment
    c = {"strip_censored": float(cS), "last5_strip_s": lS,
         "strip_gain_vs_twin_s": lS - lT,
         "strip_exam_s": exs["exam_s"], "strip_passed_a": float(strip_a),
         "strip_passed_b": float(strip_b),
         "strip_passed_both": float(bool(strip_a and strip_b)),
         "fresh_exam_void": float(bool(exf["void"] or exs["void"]))}
    if not exf["void"]:
        c["fresh_exam_s"] = exf["exam_s"]
        c["fresh_gap_s"] = abs(exf["exam_s"] - ex_here)
        c["fresh_passed_a"] = float(c["fresh_gap_s"] <= TOL_S)
    return c


def _check(m: dict, c: dict):
    from ..protocol import Status
    if m.get("calibrated") != 1.0 or c.get("calibrated", 1.0) != 1.0:
        return Status.VOID
    # per-seed 0/1 flags aggregate as means: 1.0 = every seed, 0.0 = none
    if m.get("exam_void", 1.0) != 0.0 or c.get("fresh_exam_void", 1.0) != 0.0:
        return Status.VOID     # a censored exam life (declared)
    if m.get("twin_censored", 1.0) != 0.0:
        return Status.VOID     # a frozen twin life outlived the cap (declared)
    if m.get("hw_same", 1.0) != 0.0:
        return Status.VOID     # the hand-over did not cross machines
    return bool(m.get("a_match") == 1.0 and m.get("b_gain") == 1.0
                and c.get("strip_passed_both") == 0.0
                and c.get("fresh_passed_a") == 0.0)


def run(ledger=None):
    from ..protocol import run_spec
    from ..registry import BY_ID
    return run_spec(BY_ID["LF.06"], _experiment, _check,
                    control_fn=_control, ledger=ledger)


if __name__ == "__main__":
    if sys.argv[1:] == ["selftest"]:
        print(json.dumps(_selftest(), indent=1))
    elif sys.argv[1:] == ["pilot"]:
        _pilot()
    elif sys.argv[1:2] == ["here"]:          # by hand: one kernel leg locally
        print(json.dumps(_here(int(sys.argv[2]), sys.argv[3]))[:2000])
    elif sys.argv[1:2] == ["there"]:
        _there_child(sys.argv[2], int(sys.argv[3]), sys.argv[4], sys.argv[5])
    else:
        raise SystemExit("usage: python -m experiments.tests.lf_06_handover "
                         "selftest|pilot|here SEED DIR|there ROLE SEED CKPT OUT")
