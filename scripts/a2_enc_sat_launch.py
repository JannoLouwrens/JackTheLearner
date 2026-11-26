"""J14 (a): the Saturday launcher for A2-v1S-enc seeds 10 -> 11 -> 12 (prereg a2_size_s_enc.md).

A POLLER, not a computing job (JACK_WATCHER=gpu): it sleeps until Sat 2026-11-28 00:01 UTC (Kaggle's
W48 opens 00:00), makes (or fast-forwards) its OWN clean worktree at origin/main, then runs ONE self-advancing chain,
one seed a Kaggle GPU session: `rows enc-kernel 10 --control`, then 11, then 12. A seed whose kernel
is refused or fails stops the chain (the next seed is not started blind; the log says why). No worker
needs to be awake.

    python scripts/a2_enc_sat_launch.py                         # the real thing (detached)
    python scripts/a2_enc_sat_launch.py --dry-run --now 2026-11-28T00:02:00Z   # prints the launch lines

Log: /data/jack-logs/a2_enc/sat_launch.log (launch_detached's LOG).
"""
from __future__ import annotations

import datetime as dt
import json
import os
import subprocess
import sys
import time

MAIN = "/home/opc/jackthelearner"
os.chdir(MAIN)                       # pinned (LESSONS: detached scripts die at import otherwise)
sys.path.insert(0, MAIN)

WT = "/data/wt/jack-a2enc"   # its own worktree: jack-clean carries the forge's receipt dirt
PY = "/data/venvs/jackthelearner/bin/python"
OPENS = dt.datetime(2026, 11, 28, 0, 1, tzinfo=dt.timezone.utc)
CHAIN = ((10, True), (11, False), (12, False))   # the twin (seed 10) rides with seed 10
OUT = "/data/jack-logs/brain_a/rows"


def say(*a):
    print(dt.datetime.now(dt.timezone.utc).strftime("%FT%TZ"), *a, flush=True)


def cmd_for(seed: int, control: bool) -> list:
    return [PY, "-m", "experiments.brain_a.rows", "enc-kernel", str(seed)] + (["--control"] if control else [])


def fetched_ok(seed: int, control: bool) -> tuple:
    name = "v1s_enc_s%d%s.json" % (seed, "_c" if control else "")
    p = os.path.join(OUT, name)
    if not os.path.exists(p):
        return False, f"{p} missing"
    r = json.load(open(p))
    if not isinstance(r, dict) or r.get("stopped") or not r.get("runs"):
        return False, f"{p}: stopped={r.get('stopped') if isinstance(r, dict) else r!r}"
    return True, p


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


def main(argv) -> int:
    dry = "--dry-run" in argv
    now_arg = next((argv[i + 1] for i, a in enumerate(argv) if a == "--now"), None)
    now = (dt.datetime.fromisoformat(now_arg.replace("Z", "+00:00")) if now_arg
           else dt.datetime.now(dt.timezone.utc))
    say(f"a2_enc_sat_launch {'DRY-RUN ' if dry else ''}now={now:%FT%TZ} opens={OPENS:%FT%TZ}")
    if not dry:
        while dt.datetime.now(dt.timezone.utc) < OPENS:
            left = (OPENS - dt.datetime.now(dt.timezone.utc)).total_seconds()
            time.sleep(max(1.0, min(1800.0, left)))
        say("W48 open")
    elif now < OPENS:
        say(f"DRY: would sleep {(OPENS - now).total_seconds():.0f} s")
    head = ff_worktree(dry)
    say(f"worktree {WT} at {head}" + (" (dry: not fetched)" if dry else ""))
    env = dict(os.environ, JACK_REPO=WT, JACK_WATCHER="gpu")
    for seed, control in CHAIN:
        c = cmd_for(seed, control)
        say("LAUNCH", " ".join(c), f"cwd={WT}")
        if dry:
            continue
        rc = subprocess.run(c, cwd=WT, env=env).returncode
        ok, why = fetched_ok(seed, control)
        say(f"seed {seed}: rc {rc}, fetched {'OK' if ok else 'NOT OK'}: {why}")
        if rc != 0 or not ok:
            say(f"CHAIN STOPPED at seed {seed}; later seeds not started")
            return 1
    say("CHAIN DONE" if not dry else "DRY-RUN DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
