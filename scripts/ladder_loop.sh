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
# Iterations lost to credit/session limits, one line each, appended when a
# run dies limited and cleared by the next SUCCESSFUL iteration — which
# announces the count it inherited, so lost capacity is a number in the log
# rather than a pattern someone has to notice (14th audit, B4).
LOST="$LOGDIR/lost_iterations.log"
say() { echo "$(date -Iseconds) $*" >> "$LOG"; }
. "$REPO/scripts/lib_credits.sh"
. "$REPO/scripts/lib_usage.sh"
. "$REPO/scripts/lib_pause.sh"

pause_gate say || exit 0

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

# Both filesystems: the repo and logs sit on /, but the venvs, artifacts and
# /data/jack-logs live on /data — which is where a 45 GB WAL actually filled a
# disk once, and where this guard was NOT pointed until the 28th audit (B6).
FREE_GB=$(df -BG --output=avail / | tail -1 | tr -dc '0-9')
FREE_DATA_GB=$(df -BG --output=avail /data | tail -1 | tr -dc '0-9')
if [ "${FREE_GB:-0}" -lt "$MIN_FREE_GB" ]; then
  say "ABORT: only ${FREE_GB}GB free on / (need ${MIN_FREE_GB}GB)"
  exit 0
fi
if [ "${FREE_DATA_GB:-0}" -lt "$MIN_FREE_GB" ]; then
  say "ABORT: only ${FREE_DATA_GB}GB free on /data (need ${MIN_FREE_GB}GB)"
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
usage_gate say || exit 0
# A PACE SKIP DEFERS CLAUDE SPEND — IT MUST NOT DEFER WORK THAT COSTS NONE.
# A detached run (dispatch.sh / launch_detached.sh) can finish during a gated
# hour and write its row into experiments/ledger.json; committing that row
# spends no meter, and leaving it uncommitted is exactly what stalled DP.05's
# finished FAIL overnight on 2026-08-24 (27th audit, B3) — and what an
# owner-side `git add -A` swept into an unrelated commit the same day. Stage
# ONLY the three RUNNER_OUTPUTS a harvest legitimately writes — the ledger row
# plus its two GPU receipts (29th audit B4: the row was committed at 05:07
# while gpu_budget.json and gpu_submissions.jsonl, the receipts accounting for
# that exact row, sat uncommitted for hours). Every other path in this tree
# may be the owner's or a live session's (the add -A ban, one level down). The
# full harvest write-up (docstring FAIL RECORD, journal, amend) still belongs
# to the next unskipped iteration — this commits the evidence, not the
# interpretation.
HARVEST_PATHS="experiments/ledger.json experiments/gpu_budget.json experiments/gpu_submissions.jsonl"
harvest_bookkeeping() {
  cd "$REPO" || return 0
  # Against HEAD, not the index: an iteration killed between `git add` and
  # `git commit` leaves the row STAGED, where a worktree-vs-index diff reads
  # clean and the harvest skips it (28th audit B2).
  # shellcheck disable=SC2086
  git diff --quiet HEAD -- $HARVEST_PATHS 2>/dev/null && return 0
  # Refuse a torn file: the detached runner may be mid-write at this instant.
  # Any DIRTY harvest path must parse (the .jsonl per line) or the whole
  # harvest waits — the row and its receipts travel in one commit or not at all.
  /data/venvs/jackthelearner/bin/python - <<'PY' 2>/dev/null || {
import json, subprocess
for p in ("experiments/ledger.json", "experiments/gpu_budget.json"):
    if subprocess.run(["git", "diff", "--quiet", "HEAD", "--", p]).returncode:
        json.load(open(p))
p = "experiments/gpu_submissions.jsonl"
if subprocess.run(["git", "diff", "--quiet", "HEAD", "--", p]).returncode:
    for line in open(p):
        if line.strip():
            json.loads(line)
PY
    say "bookkeeping: a dirty harvest file is unparseable (runner mid-write?) — left for the next iteration"
    return 0; }
  ROWS=$(/data/venvs/jackthelearner/bin/python - <<'PY' 2>/dev/null
import json, subprocess
old = subprocess.run(["git", "show", "HEAD:experiments/ledger.json"],
                     capture_output=True, text=True).stdout
old = (json.loads(old) if old else {}).get("results", {})
new = json.load(open("experiments/ledger.json")).get("results", {})
changed = {k for k in new if new.get(k) != old.get(k)} | {k for k in old if k not in new}
print(" ".join(sorted(changed)) or "unknown")
PY
)
# shellcheck disable=SC2086
  git add $HARVEST_PATHS 2>/dev/null || return 0
  # The pathspec is load-bearing (28th audit B2): without it this commits the
  # WHOLE index, so anything a killed iteration left pre-staged rides along
  # under a message asserting one file — the add -A sweep through a new door,
  # in the one path that runs unattended with no agent watching.
  # shellcheck disable=SC2086
  if git commit -q -m "pace-skip bookkeeping: detached-run ledger row(s) [${ROWS:-unknown}] + GPU receipts committed while the builder was pace-gated (27th audit B3, 29th audit B4). Mechanical commit from ladder_loop.sh; the next unskipped iteration owes the harvest write-up. Only the three harvest RUNNER_OUTPUTS staged." -- $HARVEST_PATHS 2>/dev/null; then
    say "bookkeeping: committed detached ledger row(s) [${ROWS:-unknown}] during pace skip"
    git push -q 2>/dev/null && say "bookkeeping: pushed" || say "bookkeeping: push failed — next iteration retries"
  else
    say "bookkeeping: commit failed — left for the next iteration"
  fi
}

# ...and spread what is left across the week, so the loop is still awake on the
# Sunday that Kaggle's free quota expires. Builder ONLY: it is ~82% of all organ
# runs (168/wk against the overseer's 28, review's 7, field watch's 1), so
# pacing it captures nearly all the benefit while the oversight organs — the
# machinery that catches drift — keep the plain 90% gate at full strength.
pace_gate say || { harvest_bookkeeping; exit 0; }
cd "$REPO" || exit 0
BEFORE=$(/data/venvs/jackthelearner/bin/python -c \
  "import json;d=json.load(open('experiments/ledger.json'))['results'];print(sum(1 for v in d.values() if v['status']=='PASS'))" 2>/dev/null || echo 0)
# Live count, not a constant: the ladder GROWS (57 -> 105 already) and a
# hardcoded total both lies in the log and mis-times the self-pause below.
TOTAL=$(/data/venvs/jackthelearner/bin/python -c \
  "from experiments.registry import LADDER; print(len(LADDER))" 2>/dev/null || echo 105)

say "iteration start — ${BEFORE}/${TOTAL} demonstrated, model ${JACK_LOOP_MODEL:-opus}, load ${LOAD}, ${FREE_GB}GB free"
if [ -s "$LOST" ]; then
  say "inheriting $(wc -l < "$LOST") iteration(s) lost to limits since the last success (see $LOST)"
fi

# SILENCE MUST NEVER READ AS SUCCESS. Two iterations on 2026-08-10 (17:07 and
# 22:07 the day before) did their work, committed, and emitted no `iteration
# end` line: the 50m `timeout` or an OOM kill took the shell before it got
# there, so the Review counted them as neither success nor failure and had to
# infer from the log body what the instrument should have said. The overseer
# asked for this trap on 2026-08-09 18:48; it is the same principle the
# organ-liveness check enforces one level up — an organ that stops reporting
# must report that it stopped.
ITER_ENDED=0
on_exit() {
  [ "$ITER_ENDED" = 1 ] && return 0
  say "iteration end rc=KILLED — the shell died before recording an end (timeout, signal or OOM). Work may still have been committed; the log body is the only record."
}
trap on_exit EXIT

PROMPT=$(cat "$REPO/scripts/ladder_prompt.md")

run_claude() {
  mark_log            # bound the credit check to THIS run's output
  # JACK_ITER_DEADLINE: epoch seconds after which this iteration is dead (the
  # `timeout 50m` below, minus 60 s of margin). gpu.submit() refuses to start a
  # Colab job that cannot return before it — a Colab result dies with its
  # watcher, and the 2026-08-13 T2.03 pilot was lost to exactly that.
  nice -n 19 ionice -c3 env TMPDIR=/data/tmp OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
    PLAYWRIGHT_BROWSERS_PATH=/data/caches/ms-playwright \
    HF_HOME=/data/caches/huggingface \
    JACK_ITER_DEADLINE=$(( $(date +%s) + 2940 )) \
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
# A SESSION LIMIT gets the same walk (it cost 3 iterations on 2026-08-13
# before anyone looked): a limit on the primary may not bind a fallback, and
# a failed fallback attempt costs ~3 s.
for FB in $FALLBACK_MODELS; do
  limit_hit || break
  [ "$FB" = "$MODEL" ] && continue
  say "LIMITED on ${MODEL} (credits or session) — falling back to ${FB}"
  MODEL="$FB"
  run_claude "$FB"
  RC=$?
done
if credits_out; then
  say "OUT OF CREDITS on every model — credit-pausing (self-resumes in 4h)"
  echo "credits $(date -Iseconds)" > "$PAUSE"
  echo "$(date -Iseconds) credits model=${MODEL}" >> "$LOST"
elif session_limited; then
  # No pause: the message names its own reset time, and an hourly ~3 s retry
  # is cheaper than stranding the loop for 4 h past the reset. The marker is
  # the point — a dead iteration must be a number, not a silence.
  say "SESSION LIMIT on every model — marking the lost iteration"
  echo "$(date -Iseconds) session-limit model=${MODEL}" >> "$LOST"
fi

AFTER=$(/data/venvs/jackthelearner/bin/python -c \
  "import json;d=json.load(open('experiments/ledger.json'))['results'];print(sum(1 for v in d.values() if v['status']=='PASS'))" 2>/dev/null || echo 0)

ITER_ENDED=1
say "iteration end rc=${RC} — ${BEFORE} -> ${AFTER} demonstrated"

if [ "$RC" = 0 ] && [ -s "$LOST" ] && ! limit_hit; then
  say "recovered — clearing $(wc -l < "$LOST") lost-iteration marker(s)"
  : > "$LOST"
fi

if [ "$AFTER" -ge "$TOTAL" ]; then
  say "LADDER COMPLETE — all ${TOTAL} specs demonstrated. Pausing the loop."
  touch "$PAUSE"
fi
exit 0
