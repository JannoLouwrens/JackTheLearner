#!/bin/bash
# The CLERICAL lane: re-buy cheap stale certificates on cron, with no model call.
#
# See scripts/regate.py for the full argument. In one line: a stale `cpu<10min`
# certificate costs 40 seconds to run and was costing an hour of an hourly Claude
# iteration to decide, out of a weekly pool this project measured its own share of
# at 9%. Clerical work should not cost a model call.
#
# Deliberately NOT gated on usage_gate/pace_gate: this process never calls a
# model, so rationing it against the model meter would starve a free lane to
# protect a resource it does not consume. It DOES honour `.paused`.
#
# Install:  43 */2 * * *  /home/opc/jackthelearner/scripts/regate.sh
#   :43 on even hours — clear of the builder (:07) and the audit desks (:37).
set -uo pipefail

REPO=/home/opc/jackthelearner
LOG=/data/jack-logs/regate.log
LOCK=/tmp/jack-regate.lock          # its own lock; it YIELDS the runner's
PY=/data/venvs/jackthelearner/bin/python

mkdir -p "$(dirname "$LOG")"
say() { echo "$(date -Iseconds) $*" >> "$LOG"; }

# One sweep at a time. Non-blocking: a sweep still running after two hours is a
# problem to see in the log, not to queue behind.
exec 9>"$LOCK" || exit 0
flock -n 9 || { say "previous sweep still running — skipping"; exit 0; }

. "$REPO/scripts/lib_pause.sh"
pause_gate say || exit 0

cd "$REPO" || exit 0

# Tenant safety, same floors the builder uses: this box serves paying customers
# and a clerical sweep is never worth making one of them slow.
LOAD=$(awk '{print $1}' /proc/loadavg)
awk -v l="$LOAD" 'BEGIN{exit !(l>6.0)}' && { say "load $LOAD too high — skipping"; exit 0; }
FREE_GB=$(df -BG --output=avail / | tail -1 | tr -dc '0-9')
[ "${FREE_GB:-0}" -lt 3 ] && { say "only ${FREE_GB}GB free on / — skipping"; exit 0; }

OUT=$(nice -n 19 timeout 3000 "$PY" scripts/regate.py 2>&1)
printf '%s\n' "$OUT" | while IFS= read -r l; do say "$l"; done

# Only the runner writes the ledger; this just commits what the runner wrote.
# Explicit paths, because another writer shares this tree (LESSONS.md, the
# 2026-08-24 double sweep) — `git add -A` is banned here for that reason.
if ! git diff --quiet experiments/ledger.json 2>/dev/null; then
  MOVED=$(printf '%s\n' "$OUT" | grep -o 'STATUS MOVED on .*' || true)
  git add experiments/ledger.json
  git commit -q -m "regate sweep: cheap stale certificates re-bought mechanically

$(printf '%s\n' "$OUT" | tail -12)

No model call, no judgment, no threshold touched — the runner produced every
verdict here and this lane only chose which cheap stale rows to ask for.
${MOVED:+A status MOVED, which is the lane working: it keeps the scoreboard TRUE, not green.}" \
    && git push -q origin HEAD 2>/dev/null \
    && say "committed and pushed" || say "commit/push failed — rows are on the ledger, tree left for the loop"
else
  say "no ledger change"
fi
