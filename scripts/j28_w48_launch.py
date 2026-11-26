"""J28: W48's spare GPU on three cheap cells, armed so no worker need be awake. Modelled on
`scripts/kws_w48_launch.py` (same worktree guard, anchored waits, dry-run with a faked clock).

A POLLER, not a computing job (JACK_WATCHER=gpu). It fires at the FIRST of:
  (a) the ear's twin is home: /data/jack-logs/kws_row2/kws2_twin.json exists (the order: "cells queue
      behind the ear on W48"), and the week is open; or
  (b) FALLBACK: Sun 11-29 00:01 UTC, whatever the ear did.
Then its OWN clean worktree at origin/main runs ONE chain, one Kaggle GPU session each, in order:
  1. experiments.body.mjx_timing kernel          (MJX HumanoidStand, 10 min)
  2. experiments.brain_a.fp16_timing kernel       (A size S, precision off/on)
  3. experiments.brain_a.ratio64 kernel 20 --control, then 21, then 22   (prereg a2_ratio64.md)
A non-zero rc (a refusal, a cut) STOPS the chain (ruling 124: nothing retries) — except cell 1: a
failure there is itself the reading (order: "cells 2-3 still run"), so the chain goes on after it.
Results: /data/jack-logs/body/mjx_timing.json, /data/jack-logs/brain_a/rows/{fp16_timing,a1r64_s*}.json.
Receipts: copy the worktree's `experiments/gpu_submissions.jsonl` rows into main's on harvest.

    python scripts/j28_w48_launch.py                                    # the real thing (detached)
    python scripts/j28_w48_launch.py --dry-run --now 2026-11-29T00:05:00Z [--twin FILE]

Log: /data/jack-logs/j28/w48_launch.log (launch_detached's LOG).
"""
from __future__ import annotations

import datetime as dt
import os
import subprocess
import sys
import time

MAIN = "/home/opc/jackthelearner"
os.chdir(MAIN)                       # pinned (LESSONS: detached scripts die at import otherwise)
sys.path.insert(0, MAIN)

WT = "/data/wt/jack-j28-w48"
PY = "/data/venvs/jackthelearner/bin/python"
OPENS = dt.datetime(2026, 11, 28, 0, 1, tzinfo=dt.timezone.utc)
FALLBACK = dt.datetime(2026, 11, 29, 0, 1, tzinfo=dt.timezone.utc)
TWIN = "/data/jack-logs/kws_row2/kws2_twin.json"
CELLS = (("mjx", ["experiments.body.mjx_timing", "kernel"], False),          # (name, module+args, stops?)
         ("fp16", ["experiments.brain_a.fp16_timing", "kernel"], True),
         ("r64_s20", ["experiments.brain_a.ratio64", "kernel", "20", "--control"], True),
         ("r64_s21", ["experiments.brain_a.ratio64", "kernel", "21"], True),
         ("r64_s22", ["experiments.brain_a.ratio64", "kernel", "22"], True))


def say(*a):
    print(dt.datetime.now(dt.timezone.utc).strftime("%FT%TZ"), *a, flush=True)


def fire_reason(now: dt.datetime, twin_home: bool):
    """None (keep sleeping) or the reason to fire now."""
    if now < OPENS:
        return None
    if twin_home:
        return "the ear's twin is home"
    if now >= FALLBACK:
        return f"fallback {FALLBACK:%FT%TZ} reached (ear's twin NOT home)"
    return None


def ff_worktree(dry: bool) -> str:
    if not os.path.isdir(WT):
        if dry:
            say(f"DRY: would run git worktree add --detach {WT} origin/main")
            return "(not created in a dry run)"
        subprocess.run(["git", "-C", MAIN, "fetch", "-q", "origin"], check=True)
        subprocess.run(["git", "-C", MAIN, "worktree", "add", "-q", "--detach", WT, "origin/main"], check=True)
    dirty = subprocess.run(["git", "-C", WT, "status", "--porcelain", "--untracked-files=no"],
                           capture_output=True, text=True).stdout.strip()
    if dirty:
        raise SystemExit(f"REFUSED: {WT} has tracked changes:\n{dirty}")
    if not dry:
        subprocess.run(["git", "-C", WT, "fetch", "-q", "origin"], check=True)
        subprocess.run(["git", "-C", WT, "merge", "-q", "--ff-only", "origin/main"], check=True)
        subprocess.run(["git", "-C", WT, "diff", "--quiet", "HEAD", "origin/main"], check=True)
    return subprocess.run(["git", "-C", WT, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()


def run_chain(run, dry: bool) -> int:
    """run(cmd) -> rc. Returns the rc that stopped the chain, or 0."""
    for name, mod, stops in CELLS:
        c = ["nice", "-n", "19", PY, "-m", *mod]
        say("LAUNCH", name, " ".join(c), f"cwd={WT}")
        rc = 0 if dry else run(c)
        say(f"{name} rc {rc}")
        if rc != 0 and stops:
            say(f"STOP: {name} rc {rc} (ruling 124: nothing retries)")
            return rc
    return 0


def main(argv) -> int:
    dry = "--dry-run" in argv
    arg = lambda k: next((argv[i + 1] for i, a in enumerate(argv) if a == k), None)  # noqa: E731
    now_arg, twin = arg("--now"), arg("--twin") or TWIN
    clock = ((lambda: dt.datetime.fromisoformat(now_arg.replace("Z", "+00:00"))) if now_arg
             else (lambda: dt.datetime.now(dt.timezone.utc)))
    say(f"j28_w48_launch {'DRY-RUN ' if dry else ''}now={clock():%FT%TZ} opens={OPENS:%FT%TZ} "
        f"fallback={FALLBACK:%FT%TZ} twin={twin}")
    while True:
        why = fire_reason(clock(), os.path.exists(twin))
        if why:
            break
        if dry:
            say("DRY: would keep sleeping (no trigger yet)")
            return 0
        time.sleep(300 if clock() >= OPENS else max(1.0, min(1800.0, (OPENS - clock()).total_seconds())))
    say("FIRE:", why)
    head = ff_worktree(dry)
    say(f"worktree {WT} at {head}" + (" (dry: not fetched)" if dry else ""))
    env = dict(os.environ, JACK_REPO=WT, JACK_WATCHER="gpu")
    rc = run_chain(lambda c: subprocess.run(c, cwd=WT, env=env).returncode, dry)
    say("DRY-RUN DONE" if dry else f"J28 chain done rc {rc}; a worker harvests and writes §29 / a2_ratio64 score")
    return rc


if __name__ == "__main__":
    sys.exit(main(sys.argv))
