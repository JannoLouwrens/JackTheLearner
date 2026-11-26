"""J24 (5): the kernel side of J24's three self-advancing chains (ORDERS J24 (5)).

"One seed's three cells a kernel, carry through `save_life`." This module is
the job a Kaggle CPU kernel runs for ONE seed: the three J24 arms as three
child processes (`j24_chain.run_cell_carried`), each saving its carry into the
kernel's output before the segment's wall and, in segment n > 0, resuming from
the ONE whole carry segment n-1 left (mounted under /kaggle/input). Shapes are
the forge's (`r1_run.run_parallel` / `job_body` / `find_carry`), reused where
they are not tied to a forge round.

The freeze guard: a job is built only for a pinned commit whose
`docs/research/prereg/eye_bins16.md` says `STATUS: FROZEN` — a learner seed
(941–943) never runs on a draft (ORDERS J24 (3): "Freeze BEFORE any learner
seed"). Smoke seeds never get a job either.

Still to build (next slot): `kernel` / `chain` — submit, fetch, segment files,
assemble — which `r1_run.kernel` / `chain` do for a round.
"""
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from experiments import j24_chain as jc
from experiments.forge import r1_run as rr

REPO = Path(__file__).resolve().parent.parent
PREREG = "docs/research/prereg/eye_bins16.md"
SEEDS = (941, 942, 943)
SUMMARY = "j24_summary.json"
OUT = Path("/data/jack-logs/j24/chains")


def cells(seed: int) -> list:
    if int(seed) not in SEEDS:
        raise rr.Refused(f"j24_launch: seed {seed} is not one of the prereg's {SEEDS}")
    return [(a, int(seed)) for a in jc.J24_ONLY]


def cell_file(arm: str, seed: int) -> str:
    return f"j24_{arm}_s{seed}.json"


def prereg_at(pin: str) -> bytes | None:
    p = subprocess.run(["git", "-C", str(REPO), "show", f"{pin}:{PREREG}"],
                       capture_output=True)
    return p.stdout if p.returncode == 0 else None


def frozen(text: bytes | None) -> bool:
    """The prereg's STATUS line (first one) begins `**STATUS: FROZEN`."""
    if not text:
        return False
    for line in text.decode().splitlines():
        if line.startswith("**STATUS:"):
            return line.startswith("**STATUS: FROZEN")
    return False


def find_carry(root, arm: str, seed: int) -> Path:
    """The ONE whole J24 carry for (arm, seed) under `root`; none or two -> Refused."""
    name = jc.carry_name(arm, seed)
    hits = sorted(Path(d) for d, _dirs, _f in os.walk(root)
                  if Path(d).name == name and rr.carry_whole(d))
    if len(hits) != 1:
        raise rr.Refused(f"j24_launch: {len(hits)} whole carries named {name} under "
                         f"{root}: {[str(h) for h in hits]}")
    return hits[0]


def run_parallel(cell_list: list, outdir, T: int, carry_root, resume_root=None,
                 deadline: float | None = None) -> dict:
    """The seed's cells at once as child processes; one missing carry in a
    continuation refuses the whole kernel before any cell runs."""
    from experiments.pilot_b import CHILD_TIMEOUT_S, from_partial, partial_file
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    src = ({c: find_carry(resume_root, *c) for c in map(tuple, cell_list)}
           if resume_root is not None else {})
    procs = []
    for arm, seed in cell_list:
        f = outdir / cell_file(arm, seed)
        argv = [sys.executable, "-m", "experiments.j24_launch", "cell", "--arm", arm,
                "--seed", str(seed), "--T", str(int(T)), "--out", str(f),
                "--partial", str(partial_file(f)),
                "--carry-dir", str(Path(carry_root) / jc.carry_name(arm, seed))]
        if deadline is not None:
            argv += ["--deadline", repr(float(deadline))]
        if (arm, seed) in src:
            argv += ["--resume-from", str(src[(arm, seed)])]
        procs.append((arm, seed, f, subprocess.Popen(
            argv, cwd=REPO, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)))
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


_JOB = r"""
import subprocess as _s, sys as _y, time as _t
_T0 = _t.time()
_PIN = __PIN__
if _s.run(["git", "-C", "/tmp/jack", "checkout", "-q", _PIN]).returncode != 0:
    _s.run(["git", "-C", "/tmp/jack", "fetch", "-q", "--depth", "1", "origin", _PIN], check=True)
    _s.run(["git", "-C", "/tmp/jack", "checkout", "-q", _PIN], check=True)
_got = _s.run(["git", "-C", "/tmp/jack", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
assert _got == _PIN, ("j24 pin not checked out", _got, _PIN)
print("J24_PIN", _got, flush=True)
_s.run([_y.executable, "-m", "pip", "install", "-q"] + __PINS__, check=True)
import hashlib, json, os
os.chdir("/tmp/jack")
assert hashlib.sha256(open(__PREREG__, "rb").read()).hexdigest() == __PREREG_SHA__, "prereg moved"
from experiments.forge.r1_run import SEG_WALL_S
from experiments.j24_launch import run_parallel
from experiments.brain_b import versions
r = run_parallel(__CELLS__, os.environ["JACK_OUT"], __T__,
                 carry_root=os.path.join(os.environ["JACK_OUT"], "carry"),
                 resume_root=__RESUME__, deadline=_T0 + SEG_WALL_S)
json.dump({"cells": r, "versions": versions(), "round": "j24", "rung": "J24",
           "pin": _PIN, "spec_sha256": __PREREG_SHA__, "segment": __SEGMENT__},
          open(os.path.join(os.environ["JACK_OUT"], __SUMMARY__), "w"))
print("DONE", flush=True)
"""


def job_body(seed: int, pin: str, segment: int = 0, T: int | None = None) -> str:
    """One seed's kernel, segment `segment` (0 = fresh, n > 0 = resume from the
    mounted segment n-1). Refuses a short pin, a segment out of range, a seed
    not the prereg's, and a pin whose prereg is not FROZEN."""
    from experiments.pilot_b import KERNEL_PINS
    from experiments.pilot_b2 import J24_T
    if len(pin) != 40 or any(c not in "0123456789abcdef" for c in pin):
        raise rr.Refused("j24_launch: --pin must be a full 40-hex commit sha")
    if not 0 <= int(segment) < rr.MAX_SEGMENTS:
        raise rr.Refused(f"j24_launch: segment {segment}; at most {rr.MAX_SEGMENTS} kernels a run")
    cl = cells(seed)
    text = prereg_at(pin)
    if not frozen(text):
        raise rr.Refused(f"j24_launch: {PREREG} at {pin[:10]} is not FROZEN — "
                         "no learner seed runs on a draft")
    return (_JOB.replace("__RESUME__", repr("/kaggle/input" if segment else None))
            .replace("__SEGMENT__", repr(int(segment)))
            .replace("__CELLS__", repr([tuple(c) for c in cl]))
            .replace("__T__", repr(int(T or J24_T)))
            .replace("__PINS__", repr(list(KERNEL_PINS)))
            .replace("__PREREG__", repr(PREREG))
            .replace("__PREREG_SHA__", repr(hashlib.sha256(text).hexdigest()))
            .replace("__SUMMARY__", repr(SUMMARY))
            .replace("__PIN__", repr(pin)))


def main(argv):
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["cell", "job"])
    p.add_argument("--arm")
    p.add_argument("--seed", type=int)
    p.add_argument("--T", type=int)
    p.add_argument("--out")
    p.add_argument("--partial", default=None)
    p.add_argument("--carry-dir", default=None)
    p.add_argument("--resume-from", default=None)
    p.add_argument("--deadline", type=float, default=None)
    p.add_argument("--pin")
    p.add_argument("--segment", type=int, default=0)
    a = p.parse_args(argv)
    if a.cmd == "cell":
        if a.carry_dir is None:
            raise rr.Refused("j24_launch: a J24 cell always carries: --carry-dir")
        rec = jc.run_cell_carried(a.arm, a.seed, a.T, carry_dir=a.carry_dir,
                                  resume_from=a.resume_from, deadline=a.deadline,
                                  partial_path=a.partial)
        Path(a.out).write_text(json.dumps(rec) + "\n")
    elif a.cmd == "job":
        print(job_body(a.seed, a.pin, a.segment))


if __name__ == "__main__":
    main(sys.argv[1:])
