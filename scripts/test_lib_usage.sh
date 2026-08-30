#!/bin/bash
# Self-test for scripts/lib_usage.sh — the gates that decide whether the builder
# runs at all.
#
# WHY THIS FILE EXISTS. Three functions decide whether any organ on this box
# executes: `usage_gate` (the owner's 90% stop), `pace_gate` (the line), and now
# `model_gate`/`model_chain` (D14's model floor). Until 2026-08-30 not one of
# them had a test, and their failures are the most expensive this project has
# had: `lib_credits.sh`'s third-wording scar cost two uncounted dead slots, and
# the 44th audit fitted a REGRESSION to `pace_gate`'s allowance line — a closed
# arithmetic form — and derived "243 hours to clear" 40 minutes before the gap
# closed to 1. An organ nobody can test is an organ everybody guesses about.
#
# The governing rule applies to conduct code too: a capability may only be
# claimed by a test that could have failed. Each case below fails if the gate
# stops doing what its comments say.
#
# Run:  bash scripts/test_lib_usage.sh     (exit 0 = all green)
set -uo pipefail

REAL_REPO=/home/opc/jackthelearner
FAIL=0
LOGLINE=""
say() { LOGLINE="$LOGLINE$*
"; }

ok()  { printf '  ok    %s\n' "$1"; }
bad() { printf '  FAIL  %s\n     %s\n' "$1" "$2"; FAIL=$((FAIL + 1)); }
chk() { [ "$2" = "$3" ] && ok "$1" || bad "$1" "expected [$3], got [$2]"; }

# ---------------------------------------------------------------- the stub ---
# lib_usage.sh shells out to "$REPO/scripts/claude_usage.py" for every reading,
# so a fake REPO is the whole seam. JACK_TEST_MAP drives it:
#   all=<pct>            -> `--pct` with no --model
#   <model>=<pct>        -> `--model <M> --pct`
#   <model>=             -> that model has NO separate weekly line (exit 2),
#                           which is the real behaviour of Opus and Sonnet
#   elapsed=<pct>        -> `--week-elapsed`
# A key that is absent exits 2 with no output, same as the real tool.
TMP=$(mktemp -d) || exit 1
trap 'rm -rf "$TMP"' EXIT
mkdir -p "$TMP/scripts"
cat > "$TMP/scripts/claude_usage.py" <<'PY'
import os, sys
args = sys.argv[1:]
key = "all"
if "--model" in args:
    key = args[args.index("--model") + 1].lower()
if "--week-elapsed" in args:
    key = "elapsed"
table = dict(
    kv.split("=", 1) for kv in os.environ.get("JACK_TEST_MAP", "").split(",") if kv
)
val = table.get(key)
if not val:
    sys.exit(2)
print(val)
PY
REPO="$TMP"
# shellcheck source=/dev/null
. "$REAL_REPO/scripts/lib_usage.sh"

run_case() { LOGLINE=""; JACK_TEST_MAP="$1"; export JACK_TEST_MAP; }

echo "model_gate — D14's model floor"

# The case that fired on 2026-08-30: Fable pinned at 100 all day.
run_case "fable=100"
model_gate fable say; chk "a model at 100% is refused" "$?" "1"
case "$LOGLINE" in *"REFUSING fable"*"100%"*) ok "the refusal names the model and the reading";;
  *) bad "the refusal names the model and the reading" "log was [$LOGLINE]";; esac

# The boundary is INCLUSIVE. D14 says "at a 95% floor"; 95 must refuse.
run_case "fable=95"
model_gate fable say; chk "the 95% boundary refuses (>=, not >)" "$?" "1"
run_case "fable=94"
model_gate fable say; chk "94% is attempted" "$?" "0"
chk "an attempted model logs nothing" "$LOGLINE" ""

# FAILING OPEN on an unreadable line is the deliberate departure from
# usage_gate's UNKNOWN-IS-NOT-ZERO rule, and it is what keeps Opus runnable.
# If this ever flips, the loop aborts every slot forever on the fallback model.
run_case "fable=100"
model_gate opus say; chk "a model with NO separate line is attempted" "$?" "0"

# A floor above 100 is the no-op escape hatch: exactly today's behaviour.
run_case "fable=100"
MODEL_FLOOR=101; model_gate fable say; chk "floor 101 disables the gate" "$?" "0"
MODEL_FLOOR=95

echo
echo "model_chain — which models the slot may attempt"

# THE MEASURED CASE OF 2026-08-30, and the whole reason for the (b-effective)
# reading: Fable exhausted, Opus with no line. The literal reading aborted the
# slot here; 19 iterations and 4 registered verdicts rode on it not doing that.
run_case "fable=100,opus=,sonnet="
chk "fable capped -> the chain is the fallbacks, NOT empty" \
    "$(model_chain say fable opus sonnet | tr '\n' ' ')" "opus sonnet "

run_case "fable=10,opus=,sonnet="
chk "nothing capped -> the primary stays first" \
    "$(model_chain say fable opus sonnet | tr '\n' ' ')" "fable opus sonnet "

# The abort branch. Unreachable with the stock chain (only Fable has a line),
# which is exactly why it needs a test rather than a run to prove it works.
run_case "fable=100,opus=99,sonnet=100"
chk "every model capped -> EMPTY chain (the slot is refused)" \
    "$(model_chain say fable opus sonnet)" ""

# FALLBACK_MODELS may name the primary; the old inline walk skipped it and the
# rewrite must not attempt it twice at ~50 minutes a go.
run_case "opus="
chk "the primary is not attempted twice" \
    "$(model_chain say opus opus sonnet | tr '\n' ' ')" "opus sonnet "

echo
echo "chain_reading — D14's two readings, and what each costs"

# THE FORK, reduced to one assertion each. Same inputs — the exact state of
# 2026-08-30, Fable at 100% with a live Opus fallback — and opposite outcomes.
run_case "fable=100,opus=,sonnet="
LIVE=$(model_chain say fable opus sonnet)
chk "effective: the primary is capped, the slot still runs on opus" \
    "$(chain_reading effective fable "$LIVE" | tr '\n' ' ')" "opus sonnet "
chk "literal: the primary is capped, so the SLOT is refused" \
    "$(chain_reading literal fable "$LIVE")" ""

run_case "fable=10,opus=,sonnet="
LIVE=$(model_chain say fable opus sonnet)
chk "literal: an uncapped primary runs normally" \
    "$(chain_reading literal fable "$LIVE" | tr '\n' ' ')" "fable opus sonnet "

chk "an already-empty chain stays empty, and prints no blank line" \
    "$(chain_reading effective fable "" | wc -c)" "0"

# An unknown value must not silently become `literal` — that would delete
# iterations on a typo.
run_case "fable=100,opus=,sonnet="
LIVE=$(model_chain say fable opus sonnet)
chk "an unrecognised reading behaves as effective, never as literal" \
    "$(chain_reading LiTeRaL-ish fable "$LIVE" | tr '\n' ' ')" "opus sonnet "

echo
echo "pace_gate — the line is arithmetic, not a race"

# The 44th audit regressed a slope out of a closed form. The line is
# PACE_FLOOR + ceil((PACE_CAP-PACE_FLOOR)*elapsed/100), and these cases pin it.
unset JACK_NO_PACE
run_case "all=85,elapsed=95,fable=100"
pace_gate say; chk "85% at 95% of the week proceeds (line 87)" "$?" "0"
run_case "all=87,elapsed=95,fable=100"
pace_gate say; chk "87% at 95% of the week SKIPS (line 87, >=)" "$?" "1"
case "$LOGLINE" in *"week:all models"*"not the gate"*)
    ok "the skip line prints both meters and names the gate";;
  *) bad "the skip line prints both meters and names the gate" "log was [$LOGLINE]";; esac
run_case "all=30,elapsed=0,fable=1"
pace_gate say; chk "the floor is 25 at the reset (30% skips)" "$?" "1"
run_case "all=89,elapsed=100,fable=1"
pace_gate say; chk "the line converges ON 90, not beside it (89 proceeds)" "$?" "0"

# FAILS OPEN on an unreadable week position — deliberate: usage_gate already
# refused on an unreadable meter one function above.
run_case "all=99"
pace_gate say; chk "unreadable week position fails open" "$?" "0"

echo
echo "usage_gate — the owner's 90% stop"

run_case "all=89"
usage_gate say; chk "89% proceeds" "$?" "0"
run_case "all=90"
usage_gate say; chk "90% stops (>=, not >)" "$?" "1"
case "$LOGLINE" in *"paused until the owner resumes"*) ok "the stop names the resume";;
  *) bad "the stop names the resume" "log was [$LOGLINE]";; esac

# UNKNOWN IS NOT ZERO. This is the one place unreadable must REFUSE, and it is
# the opposite of model_gate's rule — if these two ever converge, one of them
# has been changed by someone who did not read why.
run_case ""
usage_gate say; chk "an unreadable meter refuses to run" "$?" "1"
case "$LOGLINE" in *"ABORT: usage unreadable"*) ok "the abort says the meter was unreadable";;
  *) bad "the abort says the meter was unreadable" "log was [$LOGLINE]";; esac

# An override lifts a KNOWN ceiling, never a blind one.
printf 'ceiling=95\nuntil=%s\n' "$(( $(date +%s) + 3600 ))" > "$TMP/.usage-resumed"
run_case "all=92"
usage_gate say; chk "an owner resume lifts 92% under a 95% ceiling" "$?" "0"
run_case ""
usage_gate say; chk "an owner resume does NOT lift an unreadable meter" "$?" "1"
run_case "all=96"
usage_gate say; chk "past the resume ceiling still stops" "$?" "1"
printf 'ceiling=95\nuntil=1\n' > "$TMP/.usage-resumed"
run_case "all=92"
usage_gate say; chk "an EXPIRED resume stops" "$?" "1"
[ -f "$TMP/.usage-resumed" ] && bad "an expired resume is deleted" "the file survived" \
                             || ok "an expired resume is deleted"

echo
if [ "$FAIL" = 0 ]; then echo "ALL GREEN"; exit 0; fi
echo "${FAIL} FAILURE(S)"; exit 1
