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
export JACK_MEM_RECEIPTS="$TMP/mem_receipts"
. "$REAL_REPO/scripts/lib_procwatch.sh"

# spawn CMD... -> echoes the pid, records it for cleanup, waits for the EXEC
# CHAIN, not just for /proc: the pid's directory exists while argv[0] still
# reads `setsid` or `nice`, and a predicate case that samples it in that
# window measures the trampoline, not the target (flaked live 2026-09-03 —
# "a venv python is ours" read `no` once in four runs). argv[0] of the final
# image is exactly $1 at every call site in this file, including the sh -c
# prose case, whose trailing `:` keeps sh from exec-replacing itself.
spawn() {
  setsid nice -n 19 "$@" >/dev/null 2>&1 &
  local p=$!
  KIDS="$KIDS $p"
  local i=0
  while [ $i -lt 50 ] && \
        [ "$(tr '\0' '\n' < "/proc/$p/cmdline" 2>/dev/null | head -1)" != "$1" ]; do
    sleep 0.05; i=$((i + 1))
  done
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

echo "== memory: a peak over the ceiling is NAMED, never killed (63rd B2) =="

# `b"j" * n`, not `bytearray(n)`: multiplication WRITES every page, so the
# rss actually rises — a zero-filled allocation the kernel never commits
# would test the mock, not the high-water mark. VmHWM is the point: the test
# would still see this peak even after the memory were freed.
FAT_PID=$( cd "$REAL_REPO" && spawn "$VENV_PY" -c \
  'x = b"j" * (300 * 1024 * 1024); import time; time.sleep(45)' )
i=0
while [ $i -lt 100 ] && [ "$(proc_peak_rss_mb "$FAT_PID" 2>/dev/null || echo 0)" -lt 250 ]; do
  sleep 0.1; i=$((i + 1))
done
chk "proc_peak_rss_mb sees the 300 MB peak" \
    "$([ "$(proc_peak_rss_mb "$FAT_PID")" -ge 250 ] && echo yes || echo no)" yes

JACK_MEM_CEILING_MB=200
LOGLINE=""; PROC_MEM_N=-1
proc_memory_report say && R=clean || R=over
chk "a project python peaking over the ceiling is reported" "$R" over
chk "  ...named with its pid" "$(printf '%s' "$LOGLINE" | grep -c "MEMORY $FAT_PID:")" 1
chk "  ...with the peak as a number in MB" \
    "$(printf '%s' "$LOGLINE" | grep -cE "MEMORY $FAT_PID:[0-9]+ — peak rss [0-9]+ MB")" 1
chk "  ...while a lean project python is NOT named" \
    "$(printf '%s' "$LOGLINE" | grep -c "MEMORY $VENV_PID:")" 0

# A declaration attributes a pid to a purpose; it does not waive the RAM
# constraint. The T2.00 shape — a legitimate, declared spec run at 5x the
# ceiling — must still be named or the guard has a laundering path.
proc_declare "$FAT_PID" "test-declared-but-fat"
LOGLINE=""
proc_memory_report say || true
chk "a DECLARED process over the ceiling is still named" \
    "$(printf '%s' "$LOGLINE" | grep -c "MEMORY $FAT_PID:")" 1

JACK_MEM_CEILING_MB=100000
LOGLINE=""; PROC_MEM_N=-1
proc_memory_report say && R=clean || R=over
chk "under the ceiling the report is clean" "$R" clean
chk "  ...and PROC_MEM_N is 0" "$PROC_MEM_N" 0
JACK_MEM_CEILING_MB=1536

echo "== memory receipts: an EXITED breach is harvested, not lost (142nd FTB 3) =="

# The defect this arm exists for: T6.01's child peaked at 2129 MB inside the
# 00:07 slot, EXITED, and the live scan at slot end had nothing to read —
# VmHWM dies with the pid. The receipt is the peak carried out by the process
# itself; these cases run the REAL python writer against the REAL shell
# harvester, same discipline as the self-declaration section above.
rm -f "$JACK_MEM_RECEIPTS"
chk "mark on a missing receipt file reads 0" "$(mem_receipts_mark)" 0

# A receipt from BEFORE the slot's mark is prior work, not this slot's breach.
printf '%s\t%s\t%s\t%s\n' "999998:1" "$(date -Iseconds)" 3000 "T6.PRIOR" >> "$JACK_MEM_RECEIPTS"
MARK=$(mem_receipts_mark)

# The real writer: plants the 2129 MB shape under its own (about to exit)
# pid:starttime, so the python and shell key formats meet over a real death.
( cd "$REAL_REPO" && "$VENV_PY" -c '
import sys; sys.path.insert(0, "/home/opc/jackthelearner")
from experiments.protocol import _report_mem_receipt
_report_mem_receipt("T6.FAKE", 2129.0)' )
LOGLINE=""; PROC_MEM_N=-1
proc_memory_report say "$MARK" && R=clean || R=over
chk "an exited over-ceiling receipt is reported" "$R" over
chk "  ...as an EXITED line carrying the self-reported peak" \
    "$(printf '%s' "$LOGLINE" | grep -c "(EXITED) — peak rss 2129 MB")" 1
chk "  ...labelled with the spec that recorded it" \
    "$(printf '%s' "$LOGLINE" | grep -c "run_spec T6.FAKE")" 1
chk "  ...and the pre-mark receipt is NOT re-reported" \
    "$(printf '%s' "$LOGLINE" | grep -c "T6.PRIOR")" 0
chk "  ...PROC_MEM_N counts the exited breach" "$PROC_MEM_N" 1

# A --gate sweep writes one receipt per spec from ONE process; the high-water
# mark is monotone, so the harvest must collapse them to one line at the max.
MARK2=$(mem_receipts_mark)
printf '%s\t%s\t%s\t%s\n' "999997:3" "$(date -Iseconds)" 900  "SWEEP.A" >> "$JACK_MEM_RECEIPTS"
printf '%s\t%s\t%s\t%s\n' "999997:3" "$(date -Iseconds)" 2048 "SWEEP.B" >> "$JACK_MEM_RECEIPTS"
printf '%s\t%s\t%s\t%s\n' "999996:3" "$(date -Iseconds)" 400  "LEAN.C"  >> "$JACK_MEM_RECEIPTS"
LOGLINE=""; PROC_MEM_N=-1
proc_memory_report say "$MARK2" || true
chk "a sweep's receipts collapse to ONE line at the process max" \
    "$(printf '%s' "$LOGLINE" | grep -c "999997:3 (EXITED) — peak rss 2048 MB")" 1
chk "  ...an under-ceiling exited process stays silent" \
    "$(printf '%s' "$LOGLINE" | grep -c "999996:3")" 0

# A receipt whose process is STILL ALIVE is the live scan's to name from the
# kernel's own counter — the harvest reporting it too would double-count.
ALIVE_PID=$(spawn "$VENV_PY" -c 'import time; time.sleep(45)')
ALIVE_KEY=$(proc_key "$ALIVE_PID")
MARK3=$(mem_receipts_mark)
printf '%s\t%s\t%s\t%s\n' "$ALIVE_KEY" "$(date -Iseconds)" 7777 "STILL.ALIVE" >> "$JACK_MEM_RECEIPTS"
LOGLINE=""
proc_memory_report say "$MARK3" || true
chk "a receipt whose process is still alive is left to the live scan" \
    "$(printf '%s' "$LOGLINE" | grep -c "STILL.ALIVE")" 0

# No mark -> the harvest arm stays off: the pre-mark/over-ceiling receipts
# above must NOT leak into a caller that only asked for the live scan.
LOGLINE=""
proc_memory_report say || true
chk "without a mark the harvest arm stays off (live scan only)" \
    "$(printf '%s' "$LOGLINE" | grep -c "EXITED")" 0

# The trim is bounded and only ever runs BEFORE a mark is taken.
seq 1 4100 | awk -v OFS='\t' '{print "1:" $1, "t", 10, "x"}' > "$JACK_MEM_RECEIPTS"
mem_receipts_trim
chk "an oversized receipt log trims to its tail" "$(wc -l < "$JACK_MEM_RECEIPTS")" 2000
seq 1 100 | awk -v OFS='\t' '{print "1:" $1, "t", 10, "x"}' > "$JACK_MEM_RECEIPTS"
mem_receipts_trim
chk "a small receipt log is left alone" "$(wc -l < "$JACK_MEM_RECEIPTS")" 100
rm -f "$JACK_MEM_RECEIPTS"
chk "a missing receipt file is not a crash" \
    "$(proc_memory_report say "0" && echo clean || echo over)" clean

# The call-site pin, grep-level on purpose (the notice-before-prune lesson):
# the trim MOVES byte offsets, so it must run before the mark is taken, and
# the slot-end report must actually receive the mark.
LOOP="$REAL_REPO/scripts/ladder_loop.sh"
T_LINE=$(grep -n '^mem_receipts_trim' "$LOOP" | head -1 | cut -d: -f1)
M_LINE=$(grep -n '^MEM_MARK_BEFORE=\$(mem_receipts_mark)' "$LOOP" | head -1 | cut -d: -f1)
chk "ladder_loop.sh trims BEFORE taking the mark" \
    "$([ -n "$T_LINE" ] && [ -n "$M_LINE" ] && [ "$T_LINE" -lt "$M_LINE" ] && echo yes || echo no)" yes
chk "ladder_loop.sh passes the mark to the slot-end report" \
    "$(grep -c 'proc_memory_report say "\${MEM_MARK_BEFORE:-}"' "$LOOP")" 1

echo "== housekeeping =="

: > "$JACK_PROC_DECL"
proc_declare "$VENV_PID" "alive"
printf '%s\t%s\t%s\n' "999999:1" "$(date -Iseconds)" "long-dead" >> "$JACK_PROC_DECL"
proc_prune_declarations
# Stamp-then-drop (99th audit B3): the FIRST prune after a death stamps the
# row EXITED so a paced-out reader can tell a finished run from a live one;
# the SECOND drops it. One-step deletion erased exactly that distinction.
chk "first prune STAMPS the dead declaration EXITED, not drops it" \
    "$(grep -c $'^999999:1\t.*\tEXITED ' "$JACK_PROC_DECL")" 1
chk "pruning keeps the live one" "$(cut -f1 "$JACK_PROC_DECL" | grep -cxF "$(proc_key "$VENV_PID")")" 1
proc_prune_declarations
chk "second prune drops the stamped dead declaration" "$(grep -c '^999999:' "$JACK_PROC_DECL")" 0
chk "  ...and still keeps the live one unstamped" \
    "$(grep -c $'\tEXITED ' "$JACK_PROC_DECL")" 0
chk "declaring a dead pid fails loudly" \
    "$(proc_declare 999999 x 2>/dev/null; echo "rc=$?")" "rc=1"

echo "== notice-before-prune: the slot-start ORDERING is load-bearing (2026-09-19) =="

# The two-slot lifecycle as ladder_loop.sh actually runs it. A run_spec child
# dies with its session; the slot-END prune stamps it EXITED. What the NEXT
# slot's start does with that stamp is pure call ordering:
#   notice -> prune   announces the death once, then sweeps the stamp  (correct)
#   prune -> notice   sweeps the stamp first, the notice reads a clean file,
#                     and the death is never said — the 09-19 defect: the
#                     09:5x and 10:1x PS.06 losses, both declared and both
#                     stamped, produced no notice at the 10:07/11:07 starts.
: > "$JACK_PROC_DECL"
printf '%s\t%s\t%s\n' "888888:1" "$(date -Iseconds)" "run_spec FAKE.01" >> "$JACK_PROC_DECL"
proc_prune_declarations           # the slot-END prune: stamps the dead run
chk "slot-end prune stamped the dead run_spec" \
    "$(grep -c $'^888888:1\t.*\tEXITED ' "$JACK_PROC_DECL")" 1

# The repaired live path: notice first, prune second.
LOGLINE=""
notice_exited_dispatches LIVE
proc_prune_declarations
chk "notice-then-prune ANNOUNCES the prior-slot death" \
    "$(printf '%s' "$LOGLINE" | grep -c "LIVE NOTICE: declared dispatch 'run_spec FAKE.01'")" 1
chk "  ...and the stamp is swept after being said" \
    "$(grep -c '^888888:' "$JACK_PROC_DECL")" 0

# THE CONTROL, and it must come out silent: rebuild the stamped state and run
# the PRE-repair order. If this ever announces, the prune has stopped dropping
# and the two-step lifecycle is broken in the other direction — either way a
# human looks.
printf '%s\t%s\t%s\tEXITED %s\n' "888888:1" "$(date -Iseconds)" \
    "run_spec FAKE.01" "$(date -Iseconds)" >> "$JACK_PROC_DECL"
LOGLINE=""
proc_prune_declarations
notice_exited_dispatches LIVE
chk "prune-then-notice is provably BLIND (the defect, kept as the control)" \
    "$(printf '%s' "$LOGLINE" | grep -c "NOTICE")" 0

# B6's original silence cases, made durable: a non-dispatch exit stays quiet.
# (The label must not CONTAIN dispatch/run_spec/detached — the filter is a
# substring match, and a first draft of this fixture labelled the row
# "not-a-dispatch" and correctly got announced.)
: > "$JACK_PROC_DECL"
printf '%s\t%s\t%s\tEXITED %s\n' "777777:1" "$(date -Iseconds)" \
    "editor-probe" "$(date -Iseconds)" >> "$JACK_PROC_DECL"
LOGLINE=""
notice_exited_dispatches LIVE
chk "an EXITED non-dispatch row is not announced" \
    "$(printf '%s' "$LOGLINE" | grep -c NOTICE)" 0

# The call-site pin, grep-level on purpose: the defect was one line of call
# ordering in ladder_loop.sh that no fixture of the function alone could see.
# The first LIVE notice must sit after the pace gate and before the first
# slot-start prune.
LOOP="$REAL_REPO/scripts/ladder_loop.sh"
G_LINE=$(grep -n 'pace_gate say' "$LOOP" | head -1 | cut -d: -f1)
N_LINE=$(grep -n '^notice_exited_dispatches LIVE' "$LOOP" | head -1 | cut -d: -f1)
P_LINE=$(grep -n '^proc_prune_declarations' "$LOOP" | awk -F: -v g="$G_LINE" '$1 > g {print $1; exit}')
chk "ladder_loop.sh live path calls the notice BEFORE the slot-start prune" \
    "$([ -n "$G_LINE" ] && [ -n "$N_LINE" ] && [ -n "$P_LINE" ] && \
       [ "$N_LINE" -gt "$G_LINE" ] && [ "$N_LINE" -lt "$P_LINE" ] && echo yes || echo no)" yes

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
