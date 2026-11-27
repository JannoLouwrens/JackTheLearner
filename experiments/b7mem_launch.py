"""J29 (5): the Kaggle CPU side of row B-7's memory arm `B7mem` (ORDERS J29).

One seed a chain of segments, one Kaggle CPU kernel a segment. A segment runs
`python -m experiments.brain_b_mem run --seed S --threads 4 --carry ... --deadline
T0 + SEG_WALL_S`: the run cuts itself at the first iteration boundary past the
deadline into its carry (nets, targets, alpha, optimisers, replay, RNG, counters,
curve — bit-equal on resume, tests.test_brain_b_mem), and segment n > 0 first
copies the ONE carry segment n-1 left (mounted under /kaggle/input via
`kernel_sources`). Where the cut falls does not move the run. Shapes follow
`experiments/j24_launch.py`; nothing here touches the learner.

The freeze guard: a job is built only for a pinned commit whose
`docs/research/KNOWN_ANSWERS.md` carries row B-7's FROZEN RULE (c848d888) — no seed
of the arm runs before its bar is sealed. Only the frozen seeds 0–3 get a job.

Cost, measured on the box (J29 (3)): 1.62 CPU-s an update, 52,000 updates a seed
(1,000 init + 51,000) ≈ 22.5 CPU-h; 2 threads halved the wall. A Kaggle CPU kernel
has 4 cores, so a seed takes 1–3 segments of 7.5 h. MAX_SEGMENTS = 4 (30 h of wall)
covers even no thread speed-up. A segment that ends without the run having moved
stops the chain (no blind re-buys). Per the frozen row a seed that does not reach
51,000 steps is LOST and the row VOID — never replaced, never re-run under another
seed.

Declared settings that differ from the box timing (not from the recipe): threads 4
(the module's default is 1 because the box is shared); Kaggle's own torch build
(recorded in every segment's `versions`) — every segment of one seed runs on the
same image, which the bit-equal carry needs.

    python -m experiments.b7mem_launch job --seed 0 --pin <sha>       # print the body
    python -m experiments.b7mem_launch chain --seeds 0 --pin <sha>    # one seed's chain
"""
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
ROW_FILE = "docs/research/KNOWN_ANSWERS.md"
ROW_HEAD = "## Row B-7 "
FROZEN_MARK = "- **FROZEN RULE (11-27 09:40, J29 (4)"
SEEDS = (0, 1, 2, 3)
STEPS = 51000                       # 5 random episodes + 250 x 200 (frozen row)
THREADS = 4
PIP = ["gymnasium==1.4.0"]          # jack-std's; torch is the image's (recorded)
SEG_WALL_S = int(7.5 * 3600)        # the run cuts itself by here (Kaggle's wall is 12 h)
REMOTE_TIMEOUT_S = int(9.5 * 3600)
MAX_SEGMENTS = 4
EST_HOURS = 7.5
GUARD_RETRIES, GUARD_WAIT_S = 18, 600
OUT = Path("/data/jack-logs/j29/b7mem")
SEGS = OUT / "segments"
SUMMARY = "b7mem_summary.json"


class Refused(Exception):
    pass


def out_name(seed: int) -> str:
    return f"b7mem_s{int(seed)}.json"


def carry_name(seed: int) -> str:
    return f"b7mem_s{int(seed)}.carry"


def seg_file(seed: int, segment: int) -> Path:
    return SEGS / f"b7mem_s{int(seed)}_seg{int(segment)}.json"


def raw_file(seed: int) -> Path:
    return OUT / out_name(seed)


def row_at(pin: str) -> str | None:
    """Row B-7's section of KNOWN_ANSWERS.md at `pin` (heading to the next `## `)."""
    p = subprocess.run(["git", "-C", str(REPO), "show", f"{pin}:{ROW_FILE}"],
                       capture_output=True)
    if p.returncode:
        return None
    lines, out = p.stdout.decode().splitlines(), None
    for ln in lines:
        if out is None:
            if ln.startswith(ROW_HEAD):
                out = [ln]
        elif ln.startswith("## "):
            break
        else:
            out.append(ln)
    return "\n".join(out) + "\n" if out else None


def frozen(row: str | None) -> bool:
    return bool(row) and any(ln.startswith(FROZEN_MARK) for ln in row.splitlines())


def _check_pin(pin: str) -> None:
    if len(pin) != 40 or any(c not in "0123456789abcdef" for c in pin):
        raise Refused("b7mem_launch: --pin must be a full 40-hex commit sha")


_JOB = r"""
import subprocess as _s, sys as _y, time as _t
_T0 = _t.time()
_PIN = __PIN__
if _s.run(["git", "-C", "/tmp/jack", "checkout", "-q", _PIN]).returncode != 0:
    _s.run(["git", "-C", "/tmp/jack", "fetch", "-q", "--depth", "1", "origin", _PIN], check=True)
    _s.run(["git", "-C", "/tmp/jack", "checkout", "-q", _PIN], check=True)
_got = _s.run(["git", "-C", "/tmp/jack", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
assert _got == _PIN, ("b7mem pin not checked out", _got, _PIN)
print("B7MEM_PIN", _got, flush=True)
_s.run([_y.executable, "-m", "pip", "install", "-q"] + __PINS__, check=True)
import json, os, shutil
os.chdir("/tmp/jack")
_y.path.insert(0, "/tmp/jack")
from experiments.b7mem_launch import row_at, frozen
import hashlib
_row = row_at(_PIN)
assert frozen(_row) and hashlib.sha256(_row.encode()).hexdigest() == __ROW_SHA__, "B-7 row moved"
_O = os.environ["JACK_OUT"]
_carry = os.path.join(_O, __CARRY__)
_out = os.path.join(_O, __OUTF__)
if __SEGMENT__:
    _hits = [os.path.join(d, __CARRY__) for d, _x, f in os.walk("/kaggle/input") if __CARRY__ in f]
    assert len(_hits) == 1, ("b7mem: need exactly one prior carry", _hits)
    shutil.copy(_hits[0], _carry)
_T1 = _t.time()
_p = _s.run([_y.executable, "-m", "experiments.brain_b_mem", "run", "--seed", str(__SEED__),
             "--threads", str(__THREADS__), "--out", _out, "--carry", _carry,
             "--deadline", repr(_T0 + __WALL__)], capture_output=True, text=True)
_rec = json.load(open(_out)) if os.path.exists(_out) else None
import torch, numpy, gymnasium
json.dump({"seed": __SEED__, "segment": __SEGMENT__, "pin": _PIN, "row_sha256": __ROW_SHA__,
           "rc": _p.returncode, "stderr": _p.stderr[-1500:], "record": _rec,
           "setup_s": _T1 - _T0, "run_wall_s": _t.time() - _T1, "nproc": os.cpu_count(),
           "versions": {"torch": torch.__version__, "numpy": numpy.__version__,
                        "gymnasium": gymnasium.__version__, "python": _y.version.split()[0]}},
          open(os.path.join(_O, __SUMMARY__), "w"), indent=1)
print("DONE rc", _p.returncode, flush=True)
"""


def job_body(seed: int, pin: str, segment: int = 0, wall_s: int = SEG_WALL_S) -> str:
    """One seed's kernel, segment `segment` (0 fresh, n > 0 resumes the mounted carry).
    Refuses a short pin, a seed not 0–3, a segment out of range, and a pin whose B-7
    row is not FROZEN."""
    _check_pin(pin)
    if int(seed) not in SEEDS:
        raise Refused(f"b7mem_launch: seed {seed} is not one of B-7's frozen {SEEDS}")
    if not 0 <= int(segment) < MAX_SEGMENTS:
        raise Refused(f"b7mem_launch: segment {segment}; at most {MAX_SEGMENTS} kernels a seed")
    row = row_at(pin)
    if not frozen(row):
        raise Refused(f"b7mem_launch: row B-7 at {pin[:10]} is not FROZEN — no seed runs "
                      "before its bar is sealed")
    return (_JOB.replace("__PINS__", repr(list(PIP)))
            .replace("__ROW_SHA__", repr(hashlib.sha256(row.encode()).hexdigest()))
            .replace("__CARRY__", repr(carry_name(seed)))
            .replace("__OUTF__", repr(out_name(seed)))
            .replace("__SEGMENT__", repr(int(segment)))
            .replace("__SEED__", repr(int(seed)))
            .replace("__THREADS__", repr(THREADS))
            .replace("__WALL__", repr(int(wall_s)))
            .replace("__SUMMARY__", repr(SUMMARY))
            .replace("__PIN__", repr(pin)))


def last_step(rec: dict | None) -> int:
    ev = (rec or {}).get("evals") or []
    return int(ev[-1]["step"]) if ev else 0


def kernel(seed: int, pin: str, segment: int = 0, submit_fn=None) -> dict:
    """One segment -> SEGS/b7mem_s<seed>_seg<n>.json; when the run is complete (or the
    seed is lost) -> OUT/b7mem_s<seed>.json, the record the B-7 read takes."""
    _check_pin(pin)
    if raw_file(seed).exists():
        raise Refused(f"b7mem_launch: {raw_file(seed)} is already home — a seed runs once")
    if seg_file(seed, segment).exists():
        raise Refused(f"b7mem_launch: {seg_file(seed, segment)} is already home")
    prior = [json.loads(seg_file(seed, i).read_text()) for i in range(segment)
             if seg_file(seed, i).exists()]
    if len(prior) != segment:
        raise Refused(f"b7mem_launch: segment {segment} of s{seed} needs 0..{segment - 1} home")
    sources = None
    if segment:
        if prior[-1].get("pin") != pin:
            raise Refused("b7mem_launch: a continuation runs the pin its seed started on")
        job = (prior[-1].get("kernel") or {}).get("job_id")
        if not job or "/" not in str(job):
            raise Refused(f"b7mem_launch: segment {segment - 1} names no kernel to mount")
        sources = [job]
    body = job_body(seed, pin, segment)
    if submit_fn is None:
        from experiments.gpu import build_job, submit
        submit_fn = lambda b, **k: submit(build_job(b), **k)  # noqa: E731
    sub = submit_fn(body, prefer="kaggle-cpu", est_hours=EST_HOURS, timeout_s=REMOTE_TIMEOUT_S,
                    fetch=[SUMMARY, out_name(seed)], kernel_sources=sources, fetch_only=True)
    arts = getattr(sub, "artifacts", {}) or {}
    if SUMMARY in arts:
        r = json.loads(Path(arts[SUMMARY]).read_text())
    elif out_name(seed) in arts:          # summary lost: the curve file, never invented
        r = {"seed": int(seed), "segment": int(segment), "pin": pin, "rc": None,
             "record": json.loads(Path(arts[out_name(seed)]).read_text()), "from_out_file": True}
    else:
        raise RuntimeError(f"b7mem kernel s{seed} seg{segment}: {getattr(sub, 'message', sub)}")
    r["kernel"] = {"job_id": sub.job_id, "duration_s": sub.duration_s,
                   "charge_seconds": getattr(sub, "charge_seconds", None), "pin": pin}
    rec = r.get("record")
    step, before = last_step(rec), (last_step(prior[-1].get("record")) if prior else 0)
    complete = bool(rec and rec.get("complete"))
    print(f"b7mem s{seed} seg{segment}: rc {r.get('rc')}, step {before} -> {step}"
          + (" COMPLETE" if complete else ""), flush=True)
    SEGS.mkdir(parents=True, exist_ok=True)
    seg_file(seed, segment).write_text(json.dumps(r, indent=1) + "\n")
    moved = step > before and r.get("rc") in (0, None)   # a crashed run is not carried on
    if not complete and moved and segment + 1 < MAX_SEGMENTS:
        r["next_segment"] = segment + 1
        return r
    raw = dict(rec or {"arm": "B7mem", "seed": int(seed), "complete": False, "evals": []})
    raw.update(pin=pin, row_sha256=r.get("row_sha256"), segments=segment + 1,
               versions=r.get("versions"),
               kernel={"job_id": [(p.get("kernel") or {}).get("job_id") for p in prior + [r]],
                       "duration_s": [(p.get("kernel") or {}).get("duration_s")
                                      for p in prior + [r]]})
    if not complete:
        raw["lost"] = ("stalled or crashed: a segment ended without the run moving cleanly" if not moved
                       else f"not complete after {MAX_SEGMENTS} segments")
    OUT.mkdir(parents=True, exist_ok=True)
    raw_file(seed).write_text(json.dumps(raw, indent=1) + "\n")
    return raw


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
                done[seed] = "lost: " + r["lost"] if r.get("lost") else "home"
            except Refused as e:
                done[seed] = f"refused: {e}"
            except RuntimeError as e:
                guard = ("Uncommitted changes" in str(e) or "Untracked experiment" in str(e)
                         or "is not on origin" in str(e))
                if guard and waits < GUARD_RETRIES:
                    waits += 1
                    print(f"s{seed}: launcher guard; waiting {GUARD_WAIT_S}s "
                          f"[{waits}/{GUARD_RETRIES}]", flush=True)
                    sleep(GUARD_WAIT_S)
                    continue
                done[seed] = f"failed: {str(e)[:300]}"
            break
        print(f"CHAIN b7mem s{seed}: {done[seed]}", flush=True)
    print("CHAIN DONE", json.dumps({str(k): v for k, v in done.items()}), flush=True)
    return done


def main(argv):
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["job", "kernel", "chain"])
    p.add_argument("--seed", type=int)
    p.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    p.add_argument("--pin", required=True)
    p.add_argument("--segment", type=int, default=0)
    a = p.parse_args(argv)
    if a.cmd == "job":
        print(job_body(a.seed, a.pin, a.segment))
    elif a.cmd == "kernel":
        print(json.dumps({k: v for k, v in kernel(a.seed, a.pin, a.segment).items()
                          if k != "record"}, indent=1))
    else:
        chain(a.seeds, a.pin)


if __name__ == "__main__":
    main(sys.argv[1:])
