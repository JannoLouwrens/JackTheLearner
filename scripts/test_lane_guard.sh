#!/bin/bash
# Self-test for the LANE GUARD in experiments/run.py — the refusal that ends
# the dies-with-parent class (nine occurrences; LESSONS.md "dispatch watcher"
# entry and its corollaries 4-9; 103rd audit item 2; wrapped-lane visibility
# per the 105th audit item 1).
#
# WHY THIS FILE EXISTS. Law 1 applies to conduct code: the capability claimed
# is "a registered run launched into a PROVABLY ABANDONED lane is REFUSED at
# launch, and the lane the runner cannot distinguish is LOUDLY MARKED", and it
# is only claimed by cases that could have failed. Every case below runs the
# REAL runner against REAL launch lanes — a mocked verdict would test the mock.
#
# THE MEASUREMENT THIS FILE ENCODES (2026-09-19, and the first version of the
# guard got it wrong the same day): stdin=/dev/null is NOT proof of a
# background launch — this harness gives /dev/null stdin to some ordinary
# foreground calls (sandboxing), and a run_in_background task is byte-identical
# to those in everything observable at launch (tree, sid, fds, env). The v1
# guard refused on stdin and blocked its own T0.36 re-buy within minutes of
# shipping. What IS provable abandonment: reparenting to pid 1 (`&`-style,
# the launcher exited without waiting) and setsid session leadership (the lane
# D20 closed). So: orphan/setsid REFUSE, /dev/null WARNS.
#
# THE ORDERING LESSON IS PINNED HERE TOO. Occurrences six and seven proved
# that a fixture verifying the FUNCTION but not the CALL SITE verifies
# nothing. The spend-path cases below go through `python -m experiments.run
# <argv>` itself, using an argv that CANNOT spend (an unregistered id refuses
# at the argv gate) — if the guard is ever moved below argv validation or
# dropped, the "call site" case fails by message, not by accident.
#
# THE CONTROL THAT MUST FAIL: the waived invocation IS the pre-repair order —
# with the guard stood down, the abandoned launch sails straight through to
# the spend machinery and nothing else at that boundary catches the lane.
#
# Run:  bash scripts/test_lane_guard.sh     (exit 0 = all green)
set -uo pipefail

REAL_REPO=/home/opc/jackthelearner
VENV_PY=/data/venvs/jackthelearner/bin/python
FAIL=0

ok()  { printf '  ok    %s\n' "$1"; }
bad() { printf '  FAIL  %s\n     %s\n' "$1" "$2"; FAIL=$((FAIL + 1)); }
chk() { [ "$2" = "$3" ] && ok "$1" || bad "$1" "expected [$3], got [$2]"; }

cd "$REAL_REPO" || exit 1
TMP=$(mktemp -d) || exit 1
trap 'rm -rf "$TMP"' EXIT

# Every invocation strips an inherited waiver AND an inherited lane marker:
# this fixture may itself be run from a dispatch.sh or launch_detached.sh
# descendant one day, and an inherited waiver would turn every refusal case
# green-by-accident just as an inherited marker would put the DECLARED notice
# in every case's output.
RUN() { env -u JACK_LANE_WAIVER -u JACK_DETACHED_LANE "$VENV_PY" -m experiments.run "$@"; }

# Poll a detached case's output file for its EXIT receipt (the runner's last
# stdout line is always `EXIT <rc>`, T0.23 P8) — a fixed sleep would flake.
exit_receipt() {
  local f="$1" i=0
  while [ $i -lt 60 ]; do
    grep -m1 '^EXIT ' "$f" 2>/dev/null && return 0
    sleep 0.2; i=$((i + 1))
  done
  echo "EXIT none"
}

echo "== the verdict, via the read-only probe (run lane) =="

echo x | RUN lane >"$TMP/fg" 2>&1
chk "a pipe-held stdin is launchable (rc)" "$?" 0
chk "  ...with no warning" "$(grep -c 'LANE WARNING' "$TMP/fg")" 0

RUN lane </dev/null >"$TMP/null" 2>&1
chk "stdin=/dev/null alone is LAUNCHABLE — not proof of background (rc)" "$?" 0
chk "  ...but carries the loud mark" \
    "$(grep -c 'LANE WARNING' "$TMP/null")" 1

echo x | setsid env -u JACK_LANE_WAIVER "$VENV_PY" -m experiments.run lane \
    >"$TMP/setsid" 2>&1
chk "a setsid session leader is refused (rc) — the lane D20 closed" "$?" 3
chk "  ...and the reason names setsid" \
    "$(grep -c 'session leader' "$TMP/setsid")" 1

# The launcher must PROVABLY exit: bash optimizes `( cmd & )` inside a script
# so the job's parent (this script) stays alive and nothing orphans. A
# throwaway `sh -c '... &'` exits the instant the job is started, which is
# exactly the `&`-lane being guarded.
/bin/sh -c 'env -u JACK_LANE_WAIVER "$1" -m experiments.run lane >"$2" 2>&1 </dev/null &' \
    _ "$VENV_PY" "$TMP/orph"
chk "an \`&\` orphan is refused — the launcher exited without waiting" \
    "$(exit_receipt "$TMP/orph")" "EXIT 3"
chk "  ...and the reason names the orphaning" \
    "$(grep -c 'orphaned' "$TMP/orph")" 1

echo "== the CALL SITE: the spend path itself, with an argv that cannot spend =="

echo x | setsid env -u JACK_LANE_WAIVER "$VENV_PY" -m experiments.run ZZ.99 \
    >"$TMP/site" 2>&1
chk "an abandoned spec launch is refused at the spend boundary (rc)" "$?" 3
chk "  ...by the LANE guard, not the argv gate" \
    "$(grep -c 'ABANDONED' "$TMP/site")" 1
chk "  ...before argv validation ever ran" \
    "$(grep -c 'unrecognised' "$TMP/site")" 0

RUN ZZ.99 </dev/null >"$TMP/fgspend" 2>&1
chk "the indistinguishable lane reaches the argv gate (rc)" "$?" 2
chk "  ...carrying the loud mark on the way through" \
    "$(grep -c 'LANE WARNING' "$TMP/fgspend")" 1

echo x | RUN ZZ.99 >"$TMP/clean" 2>&1
chk "a clean foreground reaches the argv gate (rc)" "$?" 2
chk "  ...with no lane text at all" \
    "$(grep -cE 'LANE WARNING|LANE NOTICE|ABANDONED' "$TMP/clean")" 0

echo "== the lane that DECLARES itself: scripts/launch_detached.sh, the REAL launcher =="

# The 105th audit's defect, pinned so it cannot regrow: every setsid case
# above hand-rolls its own launcher, and this repository does not launch that
# way — launch_detached.sh interposes `cpu_budget wrap` between setsid and
# the spend, so the session-leader refusal NEVER fires on the spending
# process, and a fixture of hand-rolled setsids certifies a shape the system
# does not use (LESSONS.md, foot). This case invokes the launcher ITSELF.
# Cost: ~1 s of the CPU day meter (admit + a sub-second read-only probe) and
# the launcher's 15 s liveness sleep. On an exhausted day the admit gate
# refuses the launch and this case fails LOUDLY with the admit message —
# that is a real refusal doing its job, not a fixture fault.
DLOG="$TMP/detached.log"
env -u JACK_LANE_WAIVER -u JACK_DETACHED_LANE \
    scripts/launch_detached.sh "$DLOG" "$VENV_PY" -m experiments.run lane \
    >"$TMP/dlaunch" 2>&1
# The launcher's own rc is 1 here BY DESIGN — a payload that finishes under
# 15 s is reported DEAD by its documented contract. The verdict is in $DLOG.
chk "the wrapped lane is VISIBLE — the payload names its declared lane" \
    "$(grep -c 'DETACHED LANE, DECLARED — launch_detached.sh' "$DLOG")" 1
chk "  ...and refuse/permit stays where D32 found it: PERMITTED" \
    "$(grep -c 'lane: launchable' "$DLOG")" 1
chk "  ...not refused" "$(grep -c 'ABANDONED' "$DLOG")" 0
# Prove this case exercised the WRAPPED shape, not a direct setsid: the
# probe prints its own sid/pid, and through the wrapper they must differ
# (the wrapper is the session leader, the spend is its child).
psid=$(grep -m1 '^stdin=' "$DLOG" | sed 's/.*sid=\([0-9]*\).*/\1/')
ppid_=$(grep -m1 '^stdin=' "$DLOG" | sed 's/.*pid=\([0-9]*\)$/\1/')
chk "  ...through the wrapper (spend is NOT the session leader)" \
    "$([ -n "$psid" ] && [ -n "$ppid_" ] && [ "$psid" != "$ppid_" ] && echo distinct || echo same)" \
    "distinct"

echo "== the CONTROL THAT MUST FAIL: the pre-repair order, via the waiver =="

echo x | setsid env -u JACK_LANE_WAIVER \
    JACK_LANE_WAIVER="lane-guard fixture control" \
    "$VENV_PY" -m experiments.run ZZ.99 </dev/null >"$TMP/ctrl" 2>&1
chk "with the guard stood down, the abandoned launch sails through (rc)" \
    "$?" 2
chk "  ...to the spend machinery (argv gate is what finally stops it)" \
    "$(grep -c 'unrecognised' "$TMP/ctrl")" 1
chk "  ...and the waiver is a LOUD MARK, not a silence" \
    "$(grep -c 'LANE WAIVER' "$TMP/ctrl")" 1

echo
if [ "$FAIL" -eq 0 ]; then
  echo "ALL GREEN"
else
  echo "$FAIL FAILURE(S)"
fi
exit "$FAIL"
