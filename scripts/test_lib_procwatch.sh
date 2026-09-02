#!/bin/bash
# Self-test for scripts/lib_procwatch.sh — the guard that answers "did this
# iteration leave compute running on a box with paying tenants?"
#
# WHY THIS FILE EXISTS. Law 1 applies to conduct code: the capability claimed
# here is "the system can now see an orphaned project process", and it is only
# claimed by cases that could have failed. Every assertion below runs against
# REAL processes in /proc, not a stub — the whole defect class this file guards
# lives in how /proc is read (argv[0] vs the whole command line, pid vs
# pid:starttime, ancestry through a fork), and a mocked /proc would test the
# mock. The processes are `sleep`-shaped, nice 19, and reaped on exit.
#
# The case that matters most is PROSE: the auditor's own instrument
# (`pgrep -f '/data/venvs/jackthelearner'`) matches any process merely QUOTING
# the venv path — including the builder's own `claude`, whose prompt contains
# it. If that case ever goes green here, the detector has become the bug.
#
# Run:  bash scripts/test_lib_procwatch.sh     (exit 0 = all green)
set -uo pipefail

REAL_REPO=/home/opc/jackthelearner
VENV_PY=/data/venvs/jackthelearner/bin/python
FAIL=0
LOGLINE=""
say() { LOGLINE="$LOGLINE$*
"; }

ok()  { printf '  ok    %s\n' "$1"; }
bad() { printf '  FAIL  %s\n     %s\n' "$1" "$2"; FAIL=$((FAIL + 1)); }
chk() { [ "$2" = "$3" ] && ok "$1" || bad "$1" "expected [$3], got [$2]"; }

TMP=$(mktemp -d) || exit 1
KIDS=""
cleanup() {
  # Leave no process running — this file, of all files, must obey it.
  for p in $KIDS; do kill -9 "$p" 2>/dev/null; done
  rm -rf "$TMP"
}
trap cleanup EXIT

export JACK_PROC_DECL="$TMP/declared_pids"
. "$REAL_REPO/scripts/lib_procwatch.sh"

# spawn CMD... -> echoes the pid, records it for cleanup, waits for /proc.
spawn() {
  setsid nice -n 19 "$@" >/dev/null 2>&1 &
  local p=$!
  KIDS="$KIDS $p"
  local i=0
  while [ $i -lt 50 ] && [ ! -d "/proc/$p" ]; do sleep 0.05; i=$((i + 1)); done
  echo "$p"
}

echo "== predicate: what counts as this project's compute =="

VENV_PID=$(spawn "$VENV_PY" -c 'import time; time.sleep(45)')
chk "a venv python is ours (argv[0] under the venv)" \
    "$(_proc_is_ours "$VENV_PID" && echo yes || echo no)" yes

SLEEP_PID=$(spawn /bin/sleep 45)
chk "a plain /bin/sleep is not ours" \
    "$(_proc_is_ours "$SLEEP_PID" && echo yes || echo no)" no

# THE PROSE CASE. argv is `sh -c 'sleep 45' /data/venvs/jackthelearner/bin/python`
# — the venv path is IN the command line as $0 of the script, exactly as it is
# in the builder's claude prompt. `pgrep -f` matches this; we must not.
# The trailing `:` is load-bearing: sh EXECS a lone simple command, replacing
# itself with `sleep` and dropping the argv this case exists to carry.
PROSE_PID=$(spawn /bin/sh -c 'sleep 45; :' "$VENV_PY")
chk "a process merely QUOTING the venv path is not ours" \
    "$(_proc_is_ours "$PROSE_PID" && echo yes || echo no)" no
chk "  ...and pgrep -f DOES match it (the instrument this replaces)" \
    "$(pgrep -u "$(id -u)" -f /data/venvs/jackthelearner | grep -cxF "$PROSE_PID")" 1

# THE SCAR ITSELF: a bare `python -c` whose cwd is the repo.
BARE_PID=$( cd "$REAL_REPO" && spawn /usr/bin/python3.9 -c 'import time; time.sleep(45)' )
chk "a bare system python with cwd=repo is ours (the 3749514 shape)" \
    "$(_proc_is_ours "$BARE_PID" && echo yes || echo no)" yes

OUT_PID=$( cd /tmp && spawn /usr/bin/python3.9 -c 'import time; time.sleep(45)' )
chk "the same python outside the repo is not ours" \
    "$(_proc_is_ours "$OUT_PID" && echo yes || echo no)" no

echo "== keys: a declaration must survive pid reuse =="

KEY=$(proc_key "$VENV_PID")
chk "proc_key is pid:starttime" "$(echo "$KEY" | grep -cE "^$VENV_PID:[0-9]+$")" 1
chk "proc_key on a dead pid is empty" "$(proc_key 999999 2>/dev/null; echo "rc=$?")" "rc=1"
chk "proc_starttime is stable across reads" \
    "$([ "$(proc_starttime "$VENV_PID")" = "$(proc_starttime "$VENV_PID")" ] && echo yes)" yes

echo "== snapshot and leak detection =="

BEFORE=$(proc_snapshot)
chk "the venv python is in the snapshot" \
    "$(printf '%s\n' "$BEFORE" | grep -cxF "$KEY")" 1
chk "the prose process is not in the snapshot" \
    "$(printf '%s\n' "$BEFORE" | grep -cE "^$PROSE_PID:")" 0

# NOT inside $( ): proc_leaks sets PROC_LEAK_N and appends to LOGLINE, and a
# subshell would swallow both — which is how the first draft of this file read
# "0 leaks" while the detector was working correctly.
LOGLINE=""; PROC_LEAK_N=-1
proc_leaks "$BEFORE" say && R=clean || R=leak
chk "a process present BEFORE is not a leak" "$R" clean
chk "  ...and PROC_LEAK_N is 0" "$PROC_LEAK_N" 0

NEW_PID=$(spawn "$VENV_PY" -c 'import time; time.sleep(45)')
LOGLINE=""; PROC_LEAK_N=-1
proc_leaks "$BEFORE" say && R=clean || R=leak
chk "a NEW undeclared project process is a leak" "$R" leak
chk "  ...counted once" "$PROC_LEAK_N" 1
chk "  ...named with its pid" "$(printf '%s' "$LOGLINE" | grep -c "LEFTOVER PROCESS $NEW_PID:")" 1
chk "  ...and reported with its command line" \
    "$(printf '%s' "$LOGLINE" | grep -c 'cmd: .*time.sleep')" 1

echo "== declaration: a detached run is legitimate and must survive =="

proc_declare "$NEW_PID" "test-detached-run"
LOGLINE=""
chk "a DECLARED new process is not a leak" \
    "$(proc_leaks "$BEFORE" say && echo clean || echo leak)" clean

# Forged declaration: right pid, wrong incarnation. This is what a stale
# declaration would look like after pid reuse, and it must not launder.
: > "$JACK_PROC_DECL"
printf '%s\t%s\t%s\n' "$NEW_PID:1" "$(date -Iseconds)" "stale-reused-pid" >> "$JACK_PROC_DECL"
LOGLINE=""
chk "a declaration with a stale starttime does NOT attribute the pid" \
    "$(proc_leaks "$BEFORE" say && echo clean || echo leak)" leak

echo "== the Python writer: run_spec's self-declaration must read back =="

# 61st audit B3: experiments/protocol.py now declares the runner's own pid so
# an inline spec run (the third LEFTOVER=1, T3.09's runner) stops reading as a
# leak. Two independent implementations of "pid:starttime" — Python's rsplit
# and the shell's ${s##*") "} — must agree or the declaration silently fails
# open, so the case runs the REAL writer and the REAL reader against each
# other, not either one against a fixture.
RS_PID=$( cd "$REAL_REPO" && spawn "$VENV_PY" -c '
import sys; sys.path.insert(0, "/home/opc/jackthelearner")
from experiments.protocol import _declare_to_procwatch
_declare_to_procwatch("test-run-spec-self-declaration")
import time; time.sleep(45)' )
i=0
while [ $i -lt 100 ] && ! grep -q "^$RS_PID:" "$JACK_PROC_DECL" 2>/dev/null; do
  sleep 0.1; i=$((i + 1))
done
chk "the python-written line is pid:starttime, tab-separated" \
    "$(grep -cE "^$RS_PID:[0-9]+	" "$JACK_PROC_DECL")" 1
chk "the shell reader attributes the self-declared runner" \
    "$(_proc_attributed "$RS_PID" && echo yes || echo no)" yes

echo "== ancestry: the work is a fork of the declared watcher =="

: > "$JACK_PROC_DECL"
PARENT_PID=$(spawn "$VENV_PY" -c '
import subprocess, sys, time
subprocess.Popen([sys.executable, "-c", "import time; time.sleep(45)"])
time.sleep(45)')
sleep 1
CHILD_PID=$(pgrep -P "$PARENT_PID" | head -1)
KIDS="$KIDS $CHILD_PID"
chk "the fork exists" "$([ -n "$CHILD_PID" ] && echo yes || echo no)" yes
proc_declare "$PARENT_PID" "test-parent"
chk "a child of a declared process is attributed" \
    "$(_proc_attributed "$CHILD_PID" && echo yes || echo no)" yes
chk "an unrelated process is not attributed by that declaration" \
    "$(_proc_attributed "$NEW_PID" && echo yes || echo no)" no

echo "== housekeeping =="

: > "$JACK_PROC_DECL"
proc_declare "$VENV_PID" "alive"
printf '%s\t%s\t%s\n' "999999:1" "$(date -Iseconds)" "long-dead" >> "$JACK_PROC_DECL"
proc_prune_declarations
chk "pruning drops the dead declaration" "$(grep -c '^999999:' "$JACK_PROC_DECL")" 0
chk "pruning keeps the live one" "$(cut -f1 "$JACK_PROC_DECL" | grep -cxF "$(proc_key "$VENV_PID")")" 1
chk "declaring a dead pid fails loudly" \
    "$(proc_declare 999999 x 2>/dev/null; echo "rc=$?")" "rc=1"

rm -f "$JACK_PROC_DECL"
LOGLINE=""
chk "a missing declaration file is not a crash" \
    "$(proc_leaks "$(proc_snapshot)" say && echo clean || echo leak)" clean

kill -9 "$VENV_PID" 2>/dev/null
sleep 0.3
chk "a leak that has EXITED is no longer reported" \
    "$(BEF=$(printf '%s\n' "$BEFORE" | grep -vxF "$KEY"); proc_leaks "$BEF" say; printf '%s' "$LOGLINE" | grep -c "PROCESS $VENV_PID:")" 0

echo
if [ "$FAIL" -eq 0 ]; then echo "ALL GREEN"; else echo "$FAIL FAILURE(S)"; fi
exit $((FAIL > 0))
