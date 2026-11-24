"""S16 / ruling 97 — a jack slot with nothing to do costs nothing.

    idle_skip.py check   exit 10 = SKIP (prints the three readings), 0 = RUN (prints why)
    idle_skip.py mark start|end   record a real slot's live detached jobs + now
    idle_skip.py test    the planted tests

A slot is skipped only when ALL three readings say idle: every Kaggle CPU
session (5, ruling 85) holds one of our kernels and Kaggle itself says each in-
flight kernel is RUNNING; no detached box job recorded at the last real slot has
since died; `docs/ORDERS.md` is clean and was not committed after that slot
ended. Any reading that cannot be taken RUNS the slot (lesson 11). At most
MAX_SKIPS in a row, then the slot runs regardless. GPU openness is not read:
the skip cap bounds what that can cost to 3 slots.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
STATE = Path(os.environ.get("JACK_IDLE_STATE", "/data/jack-logs/idle_skip.json"))
DECL = Path(os.environ.get("JACK_PROC_DECL", "/data/jack-logs/declared_pids"))
SUBS = REPO / "experiments/gpu_submissions.jsonl"
KAGGLE = "/data/venvs/kaggle/bin/kaggle"
SESSIONS = 5          # Kaggle CPU sessions in all (ruling 85)
MAX_SKIPS = 3
SLUG = {"kaggle-cpu": "cpu", "kaggle": "ladder", "kaggle-tpu": "tpu"}


class Unreadable(Exception):
    pass


def _alive(key: str) -> bool:
    """`pid:starttime` as lib_procwatch.sh's proc_key writes it."""
    pid, _, start = key.partition(":")
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    except OSError:
        return False
    return fields[19] == start


def detached_keys() -> set[str]:
    """Live declared launch_detached/dispatch jobs (not regate's run_spec rows)."""
    try:
        lines = DECL.read_text().splitlines()
    except FileNotFoundError:
        return set()
    out = set()
    for ln in lines:
        key, _, rest = ln.partition("\t")
        label = rest.partition("\t")[2]
        if label.startswith(("launch_detached", "dispatch")) and _alive(key):
            out.add(key)
    return out


def kaggle_in_flight() -> list[tuple[str, str]]:
    """[(backend, slug)] for kaggle attempts with no result row, last 13 h."""
    try:
        rows = [json.loads(l) for l in SUBS.read_text().splitlines() if l.strip()]
    except (OSError, ValueError) as e:
        raise Unreadable(f"gpu_submissions unreadable: {e}")
    done = {r.get("attempt_id") for r in rows if r.get("phase") != "attempt"}
    now = time.time()
    return [(r["backend"], f"jack-{SLUG[r['backend']]}-{int(r['ts'])}")
            for r in rows
            if r.get("phase") == "attempt" and r.get("backend") in SLUG
            and r.get("attempt_id") not in done and now - r.get("ts", 0) < 13 * 3600]


def kaggle_status(slug: str) -> str:
    try:
        p = subprocess.run([KAGGLE, "kernels", "status", slug],
                           capture_output=True, text=True, timeout=90)
    except Exception as e:
        raise Unreadable(f"kaggle status {slug}: {e}")
    if p.returncode or "KernelWorkerStatus." not in p.stdout:
        raise Unreadable(f"kaggle status {slug}: rc {p.returncode} "
                         + " ".join((p.stderr or p.stdout).split())[:120])
    return p.stdout.split("KernelWorkerStatus.", 1)[1].split('"')[0].strip()


def orders_changed_after(ts: float) -> str | None:
    def git(*a):
        return subprocess.run(["git", "-C", str(REPO), *a], capture_output=True, text=True)
    if git("diff", "--quiet", "HEAD", "--", "docs/ORDERS.md").returncode:
        return "ORDERS.md has uncommitted edits"
    p = git("log", "-1", "--format=%ct %h", "--", "docs/ORDERS.md")
    if p.returncode or not p.stdout.strip():
        raise Unreadable("git log of ORDERS.md unreadable")
    ct, h = p.stdout.split()
    return f"ORDERS.md changed after the last slot ({h})" if int(ct) > ts else None


def load_state() -> dict:
    try:
        return json.loads(STATE.read_text())
    except (OSError, ValueError):
        return {}


def save_state(s: dict) -> None:
    tmp = STATE.with_suffix(".tmp")
    tmp.write_text(json.dumps(s))
    tmp.replace(STATE)


def decide(state: dict, live: set[str], in_flight, status, orders) -> tuple[bool, str]:
    """(skip?, readings). `status(slug)` and `orders(ts)` may raise Unreadable."""
    if "ts" not in state or "keys" not in state:
        return False, "no record of the last real slot"
    skips = state.get("skips", 0)
    if skips >= MAX_SKIPS:
        return False, f"{skips} skips in a row — this slot runs regardless"
    try:
        flight = in_flight()
        cpu = [s for b, s in flight if b == "kaggle-cpu"]
        if len(cpu) < SESSIONS:
            return False, f"kaggle cpu {len(cpu)}/{SESSIONS} sessions in flight — one is free"
        for _, slug in flight:
            st = status(slug)
            if st != "RUNNING":
                return False, f"kaggle {slug} is {st}"
        gone = set(state["keys"]) - live
        if gone:
            return False, f"{len(gone)} detached job(s) finished since the last slot"
        why = orders(state["ts"])
        if why:
            return False, why
    except Unreadable as e:
        return False, f"unreadable: {e}"
    return True, (f"kaggle {len(flight)} in flight all RUNNING ({len(cpu)}/{SESSIONS} cpu); "
                  f"detached {len(live)} live, 0 finished; orders unchanged; "
                  f"skip {skips + 1}/{MAX_SKIPS}")


def check() -> int:
    state = load_state()
    skip, why = decide(state, detached_keys(), kaggle_in_flight, kaggle_status,
                       orders_changed_after)
    print(why)
    if skip:
        state["skips"] = state.get("skips", 0) + 1
        save_state(state)
        return 10
    return 0


def mark(phase: str) -> int:
    """A real slot's `start` records the live jobs fresh; its `end` adds the
    ones it launched, so a job that dies during or after the slot is seen."""
    keys = detached_keys()
    if phase == "end":
        keys |= set(load_state().get("keys", []))
    save_state({"ts": time.time(), "keys": sorted(keys), "skips": 0})
    return 0


def test() -> int:
    busy = lambda: [("kaggle-cpu", f"jack-cpu-{i}") for i in range(5)]
    running = lambda slug: "RUNNING"
    same = lambda ts: None
    st = {"ts": 1.0, "keys": ["1:2"], "skips": 0}
    def unread(slug):
        raise Unreadable("planted outage")
    cases = [
        ("all busy skips", st, busy, running, same, True),
        ("one finished kernel runs", st, busy,
         lambda s: "COMPLETE" if s.endswith("3") else "RUNNING", same, False),
        ("unreadable kaggle runs", st, busy, unread, same, False),
        ("4th skip runs", dict(st, skips=3), busy, running, same, False),
        ("free session runs", st, lambda: busy()[:4], running, same, False),
        ("finished box job runs", dict(st, keys=["1:2", "9:9"]), busy, running, same, False),
        ("director edit runs", st, busy, running, lambda ts: "ORDERS.md changed", False),
        ("no record runs", {}, busy, running, same, False),
    ]
    bad = 0
    for name, s, fl, stt, od, want in cases:
        got, why = decide(s, {"1:2"}, fl, stt, od)
        ok = got == want
        bad += not ok
        print(f"{'ok  ' if ok else 'FAIL'} {name}: skip={got} — {why}")
    print(f"{len(cases) - bad}/{len(cases)} planted cases")
    return 1 if bad else 0


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "check"
    if cmd == "mark":
        sys.exit(mark(sys.argv[2] if len(sys.argv) > 2 else "start"))
    sys.exit({"check": check, "test": test}[cmd]())
