#!/bin/bash
# The Jack validation ladder loop.
#
# Runs headless on this box every hour. The box is the right host and not merely
# a convenient one: it holds the torch venv, the ledger, and BOTH GPU credentials
# (Colab ADC at ~/.config/gcloud, Kaggle at ~/.kaggle). A cloud runner has none of
# those, so it could never execute the training tiers — the part most worth
# automating.
#
# Guardrails, in order of how badly their absence would hurt:
#
#  1. TENANT SAFETY. This box serves paying customers. The loop runs at nice 19
#     with 2 OMP threads and refuses to start if load or free memory say the box
#     is already busy. It must never be the reason a tenant is slow.
#  2. DISK. Refuses to start below 3 GB free on / — a filled root once destroyed
#     Playwright's browser install and nearly took the ingress with it.
#  3. NO OVERLAP. A flock, because a single spec can run for 12 minutes and two
#     concurrent torch processes on 4 cores would thrash.
#  4. BOUNDED. --max-turns caps a runaway iteration; the hourly cron restarts it
#     cleanly rather than letting one session grind.
#  5. EXPLICIT MODEL. --model is passed rather than inherited: the interactive
#     /model command rewrites the shared settings file, so an unattended loop
#     that trusts ambient config can silently change model mid-week and nobody
#     would know from the log. Override with JACK_LOOP_MODEL=fable (or any
#     alias) in the crontab line.
#
# Install:  crontab -e  ->  7 * * * * /home/opc/jackthelearner/scripts/ladder_loop.sh
# Watch:    tail -f /data/jack-logs/ladder.log
# Stop:     touch /home/opc/jackthelearner/.loop-paused
set -uo pipefail

REPO=/home/opc/jackthelearner
LOGDIR=/data/jack-logs          # /data, not /var/log — root is the tight volume
# The loop's OWN lock, guarding only against overlapping loop iterations.
# NOT /tmp/jack-ladder.lock: experiments/run.py now takes that one itself
# (non-blocking, skip-on-held), so a loop holding it for the whole session
# would make every in-loop runner invocation silently skip — an hour of
# Claude reasoning against a ladder it was locked out of.
LOCK=/tmp/jack-loop.lock
# The loop's guards below check CPU load, not the ladder lock — a GPU spec
# polling Kaggle holds only /tmp/jack-ladder-gpu.lock now, so the builder can
# still do CPU work beside it (2026-08-09: the box sat at 4% for hours
# because those two were the same lock).
PAUSE="$REPO/.loop-paused"
FALLBACK_MODELS="${JACK_LOOP_FALLBACK:-opus sonnet}"  # tried in order when the primary is out of credits
MIN_FREE_GB=3
MAX_LOAD=6.0

mkdir -p "$LOGDIR"
LOG="$LOGDIR/ladder.log"
say() { echo "$(date -Iseconds) $*" >> "$LOG"; }

if [ -f "$PAUSE" ]; then
  # Credit-caused pauses SELF-EXPIRE: credits refresh on their own schedule,
  # so a machine that strands itself permanently over a transient exhaustion
  # would wait days for a human who was told he could leave. Retry after 4h.
  # Manual pauses (any other content) never self-expire — a human's stop
  # means stop.
  if grep -q "^credits" "$PAUSE" 2>/dev/null; then
    AGE=$(( $(date +%s) - $(stat -c %Y "$PAUSE") ))
    if [ "$AGE" -gt 14400 ]; then
      say "credit pause aged out after ${AGE}s — self-resuming"
      rm -f "$PAUSE"
    else
      say "paused (credits; self-retry in $((14400-AGE))s)"; exit 0
    fi
  else
    say "paused (remove $PAUSE to resume)"; exit 0
  fi
fi

exec 9>"$LOCK"
flock -n 9 || { say "previous iteration still running — skipping"; exit 0; }

FREE_GB=$(df -BG --output=avail / | tail -1 | tr -dc '0-9')
if [ "${FREE_GB:-0}" -lt "$MIN_FREE_GB" ]; then
  say "ABORT: only ${FREE_GB}GB free on / (need ${MIN_FREE_GB}GB)"
  exit 0
fi

LOAD=$(awk '{print $1}' /proc/loadavg)
if awk -v l="$LOAD" -v m="$MAX_LOAD" 'BEGIN{exit !(l>m)}'; then
  say "ABORT: load ${LOAD} above ${MAX_LOAD} — leaving the box to the tenants"
  exit 0
fi

# STOP AT 90%. Owner's rule, 2026-08-09: pause ALL agents at 90% weekly usage
# until the owner resumes them. Real number from `claude -p /usage`, not a
# proxy. Nothing else is throttled — this is the only limit.
# UNKNOWN IS NOT ZERO: if usage cannot be read, do NOT run. A meter that fails
# open is not a limit.
PCT=$(/data/venvs/jackthelearner/bin/python "$REPO/scripts/claude_usage.py" --pct 2>/dev/null)
case "$PCT" in ''|*[!0-9]*) say "ABORT: usage unreadable — refusing to run"; exit 0;; esac
if [ "$PCT" -ge 90 ]; then
  say "STOPPED at ${PCT}% weekly usage — all agents paused until the owner resumes"
  exit 0
fi
cd "$REPO" || exit 0
BEFORE=$(/data/venvs/jackthelearner/bin/python -c \
  "import json;d=json.load(open('experiments/ledger.json'))['results'];print(sum(1 for v in d.values() if v['status']=='PASS'))" 2>/dev/null || echo 0)
# Live count, not a constant: the ladder GROWS (57 -> 105 already) and a
# hardcoded total both lies in the log and mis-times the self-pause below.
TOTAL=$(/data/venvs/jackthelearner/bin/python -c \
  "from experiments.registry import LADDER; print(len(LADDER))" 2>/dev/null || echo 105)

say "iteration start — ${BEFORE}/${TOTAL} demonstrated, model ${JACK_LOOP_MODEL:-opus}, load ${LOAD}, ${FREE_GB}GB free"

PROMPT=$(cat "$REPO/scripts/ladder_prompt.md")

run_claude() {
  nice -n 19 ionice -c3 env TMPDIR=/data/tmp OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
    PLAYWRIGHT_BROWSERS_PATH=/data/caches/ms-playwright \
    HF_HOME=/data/caches/huggingface \
    timeout 50m claude -p "$PROMPT" \
      --model "$1" \
      --dangerously-skip-permissions \
      --max-turns 120 \
      >> "$LOG" 2>&1
}

MODEL="${JACK_LOOP_MODEL:-opus}"
run_claude "$MODEL"
RC=$?

# CREDIT EXHAUSTION IS NOT A CRASH, and it does not look like one: the CLI
# prints "out of usage credits" and exits in ~3 seconds, so an hourly loop
# burns every remaining slot doing nothing. It cost 8 dead iterations on
# 2026-08-09 before anyone looked. Walk a fallback chain instead of idling.
for FB in $FALLBACK_MODELS; do
  tail -5 "$LOG" | grep -qi "out of usage credits" || break
  [ "$FB" = "$MODEL" ] && continue
  say "OUT OF CREDITS on ${MODEL} — falling back to ${FB}"
  MODEL="$FB"
  run_claude "$FB"
  RC=$?
done
if tail -5 "$LOG" | grep -qi "out of usage credits"; then
  say "OUT OF CREDITS on every model — credit-pausing (self-resumes in 4h)"
  echo "credits $(date -Iseconds)" > "$PAUSE"
fi

AFTER=$(/data/venvs/jackthelearner/bin/python -c \
  "import json;d=json.load(open('experiments/ledger.json'))['results'];print(sum(1 for v in d.values() if v['status']=='PASS'))" 2>/dev/null || echo 0)

say "iteration end rc=${RC} — ${BEFORE} -> ${AFTER} demonstrated"

if [ "$AFTER" -ge "$TOTAL" ]; then
  say "LADDER COMPLETE — all ${TOTAL} specs demonstrated. Pausing the loop."
  touch "$PAUSE"
fi
exit 0
