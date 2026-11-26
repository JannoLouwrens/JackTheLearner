"""J14 (b): the ear's last two runs (N7-kws-2 s2 + the shuffled-label twin) on Kaggle W48, armed so no
worker need be awake (ruling 109). The order: "s2 + the twin run on W48 after A's chain is launched".

A POLLER, not a computing job (JACK_WATCHER=gpu). It fires at the FIRST of:
  (a) A's Saturday launcher (`scripts/a2_enc_sat_launch.py`) has logged its seed-10 LAUNCH line after
      "W48 open", plus GRACE (15 min) so A's kernel takes its session first; or
  (b) FALLBACK: OPENS + 2 h, whatever A did (the ear does not depend on A; a dead A launcher must not
      cost the ear its week).
Then it makes (or fast-forwards) its OWN clean worktree at origin/main and runs ONE chain:
`kws_row kaggle2 s2 twin` (s2, then the twin; a run home and complete is skipped; a refused or cut run
stops the chain — `colab_chain`'s own rule). Results: /data/jack-logs/kws_row2/kws2_{s2,twin}.json.
Scoring (`score2` into KNOWN_ANSWERS N7-kws-2) is left to a worker. Receipts: copy the worktree's
`experiments/gpu_submissions.jsonl` rows into main's on harvest.

    python scripts/kws_w48_launch.py                                    # the real thing (detached)
    python scripts/kws_w48_launch.py --dry-run --now 2026-11-28T00:20:00Z [--alog FILE]

Log: /data/jack-logs/kws_row2/w48_launch.log (launch_detached's LOG).
"""
from __future__ import annotations

import datetime as dt
import os
import re
import subprocess
import sys
import time

MAIN = "/home/opc/jackthelearner"
os.chdir(MAIN)                       # pinned (LESSONS: detached scripts die at import otherwise)
sys.path.insert(0, MAIN)

WT = "/data/wt/jack-kws-w48"
PY = "/data/venvs/jackthelearner/bin/python"
OPENS = dt.datetime(2026, 11, 28, 0, 1, tzinfo=dt.timezone.utc)
GRACE = dt.timedelta(minutes=15)
FALLBACK = OPENS + dt.timedelta(hours=2)
A_LOG = "/data/jack-logs/a2_enc/sat_launch.log"
RUNS = ("s2", "twin")

# Anchored: a say() line starts with a timestamp. launch_detached's header ("LAUNCH <ts> cwd=... cmd:
# ...a2_enc_sat_launch.py") starts with LAUNCH and must NOT match (memory: unanchored grep scar).
_TS = r"^(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ) "
OPEN_RE = re.compile(_TS + r"W48 open\s*$")
A10_RE = re.compile(_TS + r"LAUNCH \S+ -m experiments\.brain_a\.rows enc-kernel 10\b")


def say(*a):
    print(dt.datetime.now(dt.timezone.utc).strftime("%FT%TZ"), *a, flush=True)


def a_launched_at(text: str):
    """Time of A's seed-10 LAUNCH line, only if it follows a 'W48 open' line; else None."""
    opened = False
    for line in text.splitlines():
        if OPEN_RE.match(line):
            opened = True
        elif opened:
            m = A10_RE.match(line)
            if m:
                return dt.datetime.strptime(m.group(1), "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=dt.timezone.utc)
    return None


def fire_reason(now: dt.datetime, a_text: str):
    """None (keep sleeping) or the reason to fire now."""
    if now < OPENS:
        return None
    t = a_launched_at(a_text)
    if t is not None and now >= t + GRACE:
        return f"A's seed 10 launched {t:%FT%TZ}, grace {GRACE} passed"
    if now >= FALLBACK:
        return f"fallback {FALLBACK:%FT%TZ} reached (A seed-10 launch {'seen' if t else 'NOT seen'})"
    return None


def read(p: str) -> str:
    try:
        return open(p, errors="replace").read()
    except OSError:
        return ""


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
    arg = lambda k: next((argv[i + 1] for i, a in enumerate(argv) if a == k), None)  # noqa: E731
    now_arg, a_log = arg("--now"), arg("--alog") or A_LOG
    clock = ((lambda: dt.datetime.fromisoformat(now_arg.replace("Z", "+00:00"))) if now_arg
             else (lambda: dt.datetime.now(dt.timezone.utc)))
    say(f"kws_w48_launch {'DRY-RUN ' if dry else ''}now={clock():%FT%TZ} opens={OPENS:%FT%TZ} "
        f"fallback={FALLBACK:%FT%TZ} a_log={a_log}")
    while True:
        why = fire_reason(clock(), read(a_log))
        if why:
            break
        if dry:
            say("DRY: would keep sleeping (no trigger yet)")
            return 0
        time.sleep(300 if clock() >= OPENS else max(1.0, min(1800.0, (OPENS - clock()).total_seconds())))
    say("FIRE:", why)
    head = ff_worktree(dry)
    say(f"worktree {WT} at {head}" + (" (dry: not fetched)" if dry else ""))
    c = ["nice", "-n", "19", PY, "-m", "experiments.kws_row", "kaggle2", *RUNS]
    say("LAUNCH", " ".join(c), f"cwd={WT}")
    if dry:
        say("DRY-RUN DONE")
        return 0
    env = dict(os.environ, JACK_REPO=WT, JACK_WATCHER="gpu")
    rc = subprocess.run(c, cwd=WT, env=env).returncode
    say(f"kws chain rc {rc} — results in /data/jack-logs/kws_row2/kws2_{{s2,twin}}.json; worker scores with score2")
    return rc


if __name__ == "__main__":
    sys.exit(main(sys.argv))
