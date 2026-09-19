#!/bin/bash
# Self-test for the LANE GUARD in experiments/run.py — the refusal that ends
# the dies-with-parent class (seven occurrences; LESSONS.md "dispatch watcher"
# entry and its corollaries 4-7; 103rd audit item 2).
#
# WHY THIS FILE EXISTS. Law 1 applies to conduct code: the capability claimed
# is "a registered run launched outside a session foreground is REFUSED at
# launch", and it is only claimed by cases that could have failed. Every case
# below runs the REAL runner against REAL launch lanes (stdin=/dev/null,
# setsid, a pipe) — the discriminator was measured on the live harness
# (foreground Bash holds stdin on a socket; run_in_background and `&` both get
# /dev/null), and a mocked verdict would test the mock.
#
# THE ORDERING LESSON IS PINNED HERE TOO. Occurrences six and seven proved
# that a fixture verifying the FUNCTION but not the CALL SITE verifies
# nothing: the notice was wired one line after the janitor that blinded it.
# So the spend-path cases below go through `python -m experiments.run <argv>`
# itself, using an argv that CANNOT spend (an unregistered id refuses at the
# argv gate) — if the guard is ever moved below argv validation or dropped,
# the "call site" case fails by message, not by accident.
#
# THE CONTROL THAT MUST FAIL: the waived invocation IS the pre-repair order —
# with the guard stood down, the backgrounded launch sails straight through
# to the spend machinery and nothing else at that boundary catches the lane.
# If that case ever starts being caught, something new is guarding and this
# file is stale; if the waiver's banner ever goes quiet, the loud mark died.
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

# Every invocation strips an inherited waiver: this fixture may itself be run
# from a dispatch.sh descendant one day, and an inherited waiver would turn
# every refusal case green-by-accident.
RUN() { env -u JACK_LANE_WAIVER "$VENV_PY" -m experiments.run "$@"; }

echo "== the verdict, via the read-only probe (run lane) =="

echo x | RUN lane >"$TMP/fg" 2>&1
chk "a pipe-held stdin is a foreground (rc)" "$?" 0

RUN lane </dev/null >"$TMP/null" 2>&1
chk "stdin=/dev/null is refused (rc)" "$?" 3
chk "  ...and the reason names stdin" \
    "$(grep -c 'stdin is /dev/null' "$TMP/null")" 1

echo x | setsid env -u JACK_LANE_WAIVER "$VENV_PY" -m experiments.run lane \
    >"$TMP/setsid" 2>&1
chk "a setsid session leader is refused (rc) — the lane D20 closed" "$?" 3
chk "  ...and the reason names setsid" \
    "$(grep -c 'session leader' "$TMP/setsid")" 1

echo "== the CALL SITE: the spend path itself, with an argv that cannot spend =="

RUN ZZ.99 </dev/null >"$TMP/site" 2>&1
chk "a backgrounded spec launch is refused at the spend boundary (rc)" "$?" 3
chk "  ...by the LANE guard, not the argv gate" \
    "$(grep -c 'not in a session foreground' "$TMP/site")" 1
chk "  ...before argv validation ever ran" \
    "$(grep -c 'unrecognised' "$TMP/site")" 0

echo x | RUN ZZ.99 >"$TMP/fgspend" 2>&1
chk "the same argv in a foreground reaches the argv gate (rc)" "$?" 2
chk "  ...with no lane text" \
    "$(grep -c 'session foreground' "$TMP/fgspend")" 0

echo "== the CONTROL THAT MUST FAIL: the pre-repair order, via the waiver =="

env -u JACK_LANE_WAIVER JACK_LANE_WAIVER="lane-guard fixture control" \
    "$VENV_PY" -m experiments.run ZZ.99 </dev/null >"$TMP/ctrl" 2>&1
chk "with the guard stood down, the backgrounded launch sails through (rc)" \
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
