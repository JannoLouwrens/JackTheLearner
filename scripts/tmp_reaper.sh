#!/bin/bash
# Keep root from filling with dead scratch — WITHOUT touching live sessions.
#
# Scar 1 (why it exists): / reached 3.6 GB against the loop's 3 GB floor, ~0.6 GB
# from the builder silently refusing to run overnight. Cause: 9.6 GB of finished
# research-agent scratch (ASR model downloads, checkpoints) on the small disk.
#
# Scar 2 (why it looks like this): v1 selected session dirs by the DIRECTORY's
# own mtime and reaped the scratchpad of a LIVE session — the directory was
# created Aug 1, so an 8-day-old mtime hid a session that was actively running.
# A directory's mtime says when its entries last changed, NOT when its contents
# were last touched. Now: recurse for the newest file inside, and refuse to
# touch any session whose id appears in a running process. Two independent
# liveness checks, because one was demonstrably not enough.
set -uo pipefail
LOG=/data/jack-logs/tmp_reaper.log
IDLE_DAYS=2
# bfs (this box's find) rejects relative -newermt strings; hand it an absolute one.
IDLE_TS=$(date -Iseconds -d "${IDLE_DAYS} days ago")
before=$(df -BG --output=avail / | tail -1 | tr -dc '0-9')
killed=0

# Liveness signal 1: session ids named by a RUNNING CLAUDE PROCESS. Scoped to
# claude processes on purpose — scanning every process's argv also matches the
# grep/ls/rm that merely MENTIONS an id, which spares dead scratch forever and
# makes the reaper a no-op. (Caught by the control on 2026-08-09: the test's own
# command line kept its fixture alive.)
live_ids=$(ps -eo comm=,args= 2>/dev/null | awk '$1 ~ /claude/' \
  | grep -oE '[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}' | sort -u)

for d in /tmp/claude-1000/*/*/ /data/tmp/*/; do
  [ -d "$d" ] || continue
  id=$(basename "$d")
  # 1. never touch a session named by a running claude process
  if printf '%s\n' "$live_ids" | grep -qx "$id" 2>/dev/null; then continue; fi
  # 2. never touch a tree containing a file modified inside the idle window.
  #    Recurses on purpose: a session dir's OWN mtime is when its entries last
  #    changed, which can be days stale while the session writes busily one
  #    level down. That confusion is what deleted a live scratchpad.
  if find "$d" -type f -newermt "$IDLE_TS" -print -quit 2>/dev/null | grep -q .; then
    continue
  fi
  rm -rf "$d" 2>/dev/null && killed=$((killed+1))
done

# gpu.py job dirs and loose /tmp files: no liveness concern, plain age.
find /data -maxdepth 1 -type d -name "tmp??????*" -mtime +2 -exec rm -rf {} + 2>/dev/null
find /tmp -maxdepth 1 -type f -mtime +3 \( -name "*.log" -o -name "*.json" \) -delete 2>/dev/null

after=$(df -BG --output=avail / | tail -1 | tr -dc '0-9')
echo "$(date -Iseconds) reaped ${killed} idle scratch dirs: / ${before}G -> ${after}G free" >> "$LOG"
