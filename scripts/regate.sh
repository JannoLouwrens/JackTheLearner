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

# THE DEAD-MAN CHECK. Every other liveness check in this project lives inside an
# organ that needs a model, a readable meter and a working login — so the
# failures that stop ALL of them (an expired login, an unreadable meter, the
# disk floor, a launcher that will not start) were reported nowhere the owner
# looks, by organs that were themselves down. This clerk needs none of those. If
# no worker has finished a run for 8 hours and nothing is paused, it stamps the
# owner's page with the reason — the worker log's own last line — and pushes.
# It only ever ADDS a banner (lib_seal.sh's stale_output, which refuses a dirty
# file and never stamps twice); the Director's next sitting clears it.
. "$REPO/scripts/lib_seal.sh"
LAST_OK=$(grep -a 'iteration end rc=0' /data/jack-logs/ladder.log 2>/dev/null | tail -1 | cut -d' ' -f1)
if [ -n "$LAST_OK" ]; then
  LAST_S=$(date -d "$LAST_OK" +%s 2>/dev/null || echo 0)
  AGE_H=$(( ( $(date +%s) - LAST_S ) / 3600 ))
  if [ "$LAST_S" -gt 0 ] && [ "$AGE_H" -ge 8 ]; then
    WHY=$(grep -aE '^[0-9]{4}-[0-9]{2}-[0-9]{2}T' /data/jack-logs/ladder.log | tail -1 | cut -c1-220)
    say "DEAD-MAN: no worker run has finished for ${AGE_H} h — last log line: ${WHY}"
    stale_output docs/PROGRESS.md "machine" \
      "NO WORKER HAS FINISHED A RUN FOR ${AGE_H} HOURS (checked $(date -u '+%F %H:%M') UTC by the clerk, which needs no model). The worker log's last line: ${WHY}" \
      say && git push -q origin HEAD 2>/dev/null
  fi
fi

# Tenant safety, same floors the builder uses: this box serves paying customers
# and a clerical sweep is never worth making one of them slow.
LOAD=$(awk '{print $1}' /proc/loadavg)
awk -v l="$LOAD" 'BEGIN{exit !(l>6.0)}' && { say "load $LOAD too high — skipping"; exit 0; }
FREE_GB=$(df -BG --output=avail / | tail -1 | tr -dc '0-9')
[ "${FREE_GB:-0}" -lt 3 ] && { say "only ${FREE_GB}GB free on / — skipping"; exit 0; }

# STREAM, DO NOT BUFFER. The first version did `OUT=$(...)` and wrote the log
# only after the sweep returned, so a sweep killed partway logged NOTHING while
# having really re-bought five certificates (2026-09-28 10:25: ME.11.B, ME.11.C,
# PS.08, T0.27, T0.28 all ran and landed, and the log's last line was the pause
# test from twenty minutes earlier). Their rows were only committed because the
# builder's own pace-skip bookkeeping found them orphaned.
#
# That is a lesson this project has already paid for once, on a different lane:
# the LC.03 registered run printed nothing for 15 hours and taught that liveness
# is worker CPU time, not log bytes. A log that appears only on success cannot
# report a failure, which is the one thing a log is for. `-u` keeps python from
# holding lines back; the read loop stamps and appends each as it arrives, and
# TMP keeps a copy for the commit message because the loop runs in a subshell.
TMP=$(mktemp) || exit 0
trap 'rm -f "$TMP"' EXIT
say "sweep start — $("$PY" scripts/regate.py --list 2>/dev/null | tail -1 | tr -d '\n')"
nice -n 19 timeout 3000 "$PY" -u scripts/regate.py 2>&1 | while IFS= read -r l; do
  say "$l"
  printf '%s\n' "$l" >> "$TMP"
done
OUT=$(cat "$TMP")

# Only the runner writes these files; this just commits what the runner wrote.
# The staged set is protocol.RUNNER_OUTPUTS — DERIVED, not hand-copied, because
# the hand-copied version ("ledger.json alone") orphaned cpu_budget.json and
# SO.10's bakeoff record in DECISIONS_RESOLVED.md on 2026-09-28: the sweep
# pushed a clean-looking commit and left a dirty tree behind it, which is the
# "evidence log that invalidates the evidence" scar (protocol.py documents four
# occurrences) minted a fifth way. Still explicit paths, never `git add -A` —
# another writer shares this tree (LESSONS.md, the 2026-08-24 double sweep);
# RUNNER_OUTPUTS is exactly the set no human hand writes mid-run. Diffed
# against HEAD, not the index, so a sweep killed between add and commit is
# still harvested next pass (ladder_loop's 28th-audit-B2 lesson).
STAGE=$("$PY" -c 'import os
from experiments.protocol import RUNNER_OUTPUTS
print(" ".join(p for p in RUNNER_OUTPUTS if not p.endswith(".tmp") and os.path.exists(p)))')
# shellcheck disable=SC2086
# A manager that is sitting right now may be half-way through a ruling in
# docs/DECISIONS_RESOLVED.md. That file is also a runner output, so this clerk
# used to stage it — and would have pushed half a ruling under "regate sweep".
# While a Director or auditor is alive, the file is theirs to commit.
if ps -eo args= | awk '/scripts\/(review|overseer)\.sh/ && !/awk/' | grep -q .; then
  STAGE=$(printf '%s\n' $STAGE | grep -vx 'docs/DECISIONS_RESOLVED.md' | tr '\n' ' ')
fi
if [ -n "$STAGE" ] && ! git diff --quiet HEAD -- $STAGE 2>/dev/null; then
  MOVED=$(printf '%s\n' "$OUT" | grep -o 'STATUS MOVED on .*' || true)
  # shellcheck disable=SC2086
  git add -- $STAGE
  git commit -q -m "regate sweep: cheap certificates re-bought mechanically (stale bills first, then the standing-PASS rotation)

$(printf '%s\n' "$OUT" | tail -12)

No model call, no judgment, no threshold touched — the runner produced every
verdict here and this lane only chose which cheap stale rows to ask for.
Staged: the changed protocol.RUNNER_OUTPUTS (ledger row + every receipt the
runner wrote beside it), nothing else.
${MOVED:+A status MOVED, which is the lane working: it keeps the scoreboard TRUE, not green.}" \
    -- $STAGE \
    && git push -q origin HEAD 2>/dev/null \
    && say "committed and pushed" || say "commit/push failed — rows are on the ledger, tree left for the loop"
else
  say "no runner-output change"
fi
