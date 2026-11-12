#!/bin/bash
# gap_fill.sh — close the idle gap between hourly worker slots.
#
# Owner, 2026-11-12: "of course close the gap we are work horses over here".
# The hourly worker launches at :07 with up to 50 minutes; when it finishes
# early (07:33 on the day this was written) the box idled until the next :07.
# Cron calls this at :22 and :37. It launches ONE extra jack-lane worker sized
# to be dead by :04, three minutes before the next hourly launch — so a fill
# can never hold the lock into a system or lab slot and make it skip.
#
# Everything else is ladder_loop.sh's own: the lock (a worker still running =
# skip), the pause file, the 95% usage stop, the load and disk guards.
set -u
MIN_NOW=$((10#$(date +%M)))
LEFT=$(( (64 - MIN_NOW) % 60 ))          # minutes until the next :04
[ "$LEFT" -ge 22 ] || exit 0              # too short to do honest work
[ "$LEFT" -le 50 ] || LEFT=50
# Cheap pre-check so a busy hour leaves no "skipping" line in the worker log.
exec 8>>/tmp/jack-loop.lock
flock -n 8 || exit 0
flock -u 8; exec 8>&-
JACK_SLOT_MIN="$LEFT" exec /home/opc/jackthelearner/scripts/ladder_loop.sh jack
