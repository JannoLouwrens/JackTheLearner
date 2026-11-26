#!/bin/bash
# answer.sh — the Director answers workers' questions BETWEEN sittings.
#
# Owner, 2026-11-26: "workers must be able to ask director questions." On that
# day seven worker slots ended "nothing to do, waiting on the Director": two
# blockers were raised at 10:14-11:40, after the 06:37 sitting, and would have
# waited ~20 hours for the next one.
#
# How a worker asks: ONE line under `## INBOX` in docs/ORDERS.md that starts
#     - QUESTION (lane, order): <the decision needed> | options: <A / B> | default: <what I do if unanswered>
# then it carries on with other work. This poll finds such lines; if there are
# none it exits WITHOUT starting a model. If there are, it holds a short
# answering session (manager model, 12 minutes) that rules on each question
# and rewrites its line to start `- ANSWERED`.
#
# Install:  52 * * * * /home/opc/jackthelearner/scripts/answer.sh
# Stop:     touch /home/opc/jackthelearner/.answer-paused
# Bounds:   at most JACK_ANSWER_MAX (default 6) answering sessions a day; never
#           while a full sitting runs; the same pause/disk/load/usage gates as
#           every other desk. JACK_ANSWER_DRY=1 prints the decision and stops.
set -uo pipefail
REPO="${JACK_REPO:-/home/opc/jackthelearner}"
LOG="${JACK_ANSWER_LOG:-/data/jack-logs/answer.log}"
COUNT_DIR="${JACK_ANSWER_STATE:-/data/jack-logs/answer_state}"
PAUSE="$REPO/.answer-paused"
MAX="${JACK_ANSWER_MAX:-6}"
DRY="${JACK_ANSWER_DRY:-}"
say() { echo "$(date -Iseconds) $*" >> "$LOG"; }
decide() { [ -n "$DRY" ] && echo "$1"; [ -n "$DRY" ] || [ "$1" = "none" ] || say "$1"; }

# Open questions = INBOX lines starting "- QUESTION". No model is started to find this out.
open_questions() {
  awk '/^## INBOX/{f=1;next} f&&/^## /{exit} f&&/^- QUESTION/{n++} END{print n+0}' "$REPO/docs/ORDERS.md" 2>/dev/null
}
N=$(open_questions)
[ "${N:-0}" -eq 0 ] && { decide "none"; exit 0; }

mkdir -p "$COUNT_DIR"
TODAY=$(date -u +%F); CF="$COUNT_DIR/$TODAY.count"
HELD=$(cat "$CF" 2>/dev/null || echo 0)
[ "$HELD" -ge "$MAX" ] && { decide "SKIP: $N question(s) open but today's $MAX answering sessions are spent — they wait for the next sitting"; exit 0; }
[ -z "${JACK_ANSWER_TEST:-}" ] && pgrep -f '^/bin/bash .*scripts/review\.sh' >/dev/null && { decide "SKIP: a full sitting is running — it empties the INBOX itself"; exit 0; }
[ -n "$DRY" ] && { echo "WOULD ANSWER: $N question(s), session $((HELD+1)) of $MAX today"; exit 0; }

exec 9>"$COUNT_DIR/lock"; flock -n 9 || exit 0
. "$REPO/scripts/models.env"
. "$REPO/scripts/lib_credits.sh"
. "$REPO/scripts/lib_usage.sh"
. "$REPO/scripts/lib_pause.sh"
pause_gate say "$PAUSE" || exit 0
FREE_GB=$(df -BG --output=avail / | tail -1 | tr -dc '0-9')
[ "${FREE_GB:-0}" -lt 3 ] && { say "ABORT: ${FREE_GB}GB free on /"; exit 0; }
LOAD=$(awk '{print $1}' /proc/loadavg)
awk -v l="$LOAD" 'BEGIN{exit !(l>6.0)}' && { say "ABORT: load ${LOAD} — tenants first"; exit 0; }
usage_gate say answer || { say "answering DEFERRED — the usage gate refused it; $N question(s) stay open"; exit 0; }
. "$REPO/scripts/lib_card.sh"
MODEL="${JACK_ANSWER_MODEL:-$MANAGER_MODEL}"
model_gate "$MODEL" say || MODEL="$MANAGER_FALLBACK"

echo $((HELD+1)) > "$CF"
say "answer start — $N question(s), session $((HELD+1)) of $MAX, model ${MODEL}"
usage_ledger answer start "$MODEL"
cd "$REPO" || exit 0
nice -n 19 env TMPDIR=/data/tmp timeout 12m claude -p "$(philosophy_card; cat "$REPO/scripts/answer_prompt.md")" \
  --model "$MODEL" --dangerously-skip-permissions --max-turns 40 >> "$LOG" 2>&1
RC=$?
usage_ledger answer end "$MODEL"
say "answer end rc=$RC — $(open_questions) question(s) still open"
