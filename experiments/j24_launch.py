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

`kernel(seed, pin, segment)` submits one segment (Kaggle CPU), writes its
payload to OUT/segments, and assembles OUT/j24_s<seed>.json with the forge's
`r1_run.assemble` once nothing is left to carry or the last segment is home.
`chain(seeds, pin)` runs each seed's segments back to back in one session.
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


EST_HOURS = 7.5
SEGS = OUT / "segments"


def seg_file(seed: int, segment: int) -> Path:
    return SEGS / f"j24_s{seed}_seg{segment}.json"


def raw_file(seed: int) -> Path:
    return OUT / f"j24_s{seed}.json"


def kernel(seed: int, pin: str, segment: int = 0) -> dict:
    """One segment of one seed's chain -> SEGS (and OUT/j24_s<seed>.json when done)."""
    from experiments.gpu import build_job, result_json, submit
    from experiments.pilot_b import partial_file
    cl = cells(seed)
    if raw_file(seed).exists():
        raise rr.Refused(f"j24_launch: {raw_file(seed)} is already home — a seed runs once")
    if seg_file(seed, segment).exists():
        raise rr.Refused(f"j24_launch: {seg_file(seed, segment)} is already home")
    prior = [json.loads(seg_file(seed, i).read_text()) for i in range(segment)
             if seg_file(seed, i).exists()]
    if len(prior) != segment:
        raise rr.Refused(f"j24_launch: segment {segment} of s{seed} needs 0..{segment - 1} home")
    sources = None
    if segment:
        if prior[-1].get("pin") != pin:
            raise rr.Refused("j24_launch: a continuation runs the pin its run started on")
        cl = rr.resumable(prior[-1])
        if not cl:
            raise rr.Refused(f"j24_launch: segment {segment - 1} of s{seed} carried nothing")
        job = (prior[-1].get("kernel") or {}).get("job_id")
        if not job or "/" not in str(job):
            raise rr.Refused(f"j24_launch: segment {segment - 1} names no kernel to mount")
        sources = [job]
    body = job_body(seed, pin, segment)
    fetch = [SUMMARY] + [n for a, sd in cl
                         for n in (cell_file(a, sd), partial_file(Path(cell_file(a, sd))).name)]
    sub = submit(build_job(body), prefer="kaggle-cpu", est_hours=EST_HOURS,
                 timeout_s=rr.REMOTE_TIMEOUT_S, fetch=fetch, kernel_sources=sources,
                 fetch_only=True)
    if sub.ok and SUMMARY in sub.artifacts:
        r = result_json(sub, SUMMARY)
    else:
        r = _from_artifacts(sub.artifacts, cl)
        if r is None:
            raise RuntimeError(f"j24 kernel s{seed} seg{segment}: {sub.message}")
        r.update(round="j24", rung="J24", pin=pin, segment=int(segment),
                 spec_sha256=hashlib.sha256(prereg_at(pin)).hexdigest())
    r["kernel"] = {"job_id": sub.job_id, "duration_s": sub.duration_s,
                   "charge_seconds": sub.charge_seconds, "name": f"j24_s{seed}", "pin": pin}
    for k, rec in r["cells"].items():
        print(f"{k}: {len(rec['lives'])} lives" + (" (cut)" if rec.get("cut") else "")
              + (" (carried on)" if rec.get("resumable") else "")
              + (f" ERROR {rec['error']}" if "error" in rec else ""), flush=True)
    SEGS.mkdir(parents=True, exist_ok=True)
    seg_file(seed, segment).write_text(json.dumps(r, indent=1) + "\n")
    if rr.resumable(r) and segment + 1 < rr.MAX_SEGMENTS:
        r["next_segment"] = segment + 1
        return r
    raw = rr.assemble(prior + [r])
    raw_file(seed).write_text(json.dumps(raw, indent=1) + "\n")
    return raw


def _from_artifacts(artifacts: dict, cl: list) -> dict | None:
    """No summary: per-cell files, final else partial (cut) else not_found — never invented."""
    from experiments.pilot_b import from_partial, partial_file
    res, found = {}, 0
    for arm, seed in cl:
        final = cell_file(arm, seed)
        part = partial_file(Path(final)).name
        if final in artifacts:
            rec, found = json.loads(Path(artifacts[final]).read_text()), found + 1
        elif part in artifacts:
            rec, found = from_partial(arm, seed, Path(artifacts[part])), found + 1
        else:
            rec = from_partial(arm, seed, Path("/nonexistent") / part)
        rec["from_cell_file"] = True
        res[f"{arm}_s{seed}"] = rec
    return {"cells": res, "assembled": True} if found else None


def chain(seeds: list, pin: str, sleep=time.sleep, kernel_fn=None) -> dict:
    """Each seed's segments back to back (one Kaggle session held). A refused or
    failed seed is logged and the chain goes on; the launcher's guard (dirt,
    unpushed HEAD) is waited out, never bypassed."""
    kernel_fn = kernel_fn or kernel
    done = {}
    for seed in seeds:
        seg, waits = 0, 0
        while True:
            try:
                r = kernel_fn(seed, pin, seg)
                if isinstance(r, dict) and r.get("next_segment"):
                    seg = r["next_segment"]
                    print(f"s{seed}: segment {seg - 1} home, carrying on", flush=True)
                    continue
                done[seed] = "home"
            except rr.Refused as e:
                done[seed] = f"refused: {e}"
            except RuntimeError as e:
                guard = ("Uncommitted changes" in str(e) or "Untracked experiment" in str(e)
                         or "is not on origin" in str(e))
                if guard and waits < rr.GUARD_RETRIES:
                    waits += 1
                    print(f"s{seed}: launcher guard; waiting {rr.GUARD_WAIT_S}s "
                          f"[{waits}/{rr.GUARD_RETRIES}]", flush=True)
                    sleep(rr.GUARD_WAIT_S)
                    continue
                done[seed] = f"failed: {str(e)[:300]}"
            break
        print(f"CHAIN j24 s{seed}: {done[seed]}", flush=True)
    print("CHAIN DONE", json.dumps({str(k): v for k, v in done.items()}), flush=True)
    return done


def main(argv):
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["cell", "job", "kernel", "chain"])
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
    p.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
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
    elif a.cmd == "kernel":
        kernel(a.seed, a.pin, a.segment)
    elif a.cmd == "chain":
        chain(a.seeds, a.pin)


if __name__ == "__main__":
    main(sys.argv[1:])
