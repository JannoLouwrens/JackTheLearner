#!/bin/bash
# lib_liveness.sh — SCHEDULE-side liveness for an organ that publishes a report.
#
# THE SCAR, 2026-08-30 (52nd audit, FOR THE BUILDER B1). `review.sh` started at
# 06:37:03 in FULL mode — the only mode that does Part 2, the test
# re-examination, and the run that owed the `w0-too-shallow` world design. It
# hit `Error: Reached max turns (60)` at 06:48:03, ELEVEN minutes into a
# forty-minute budget, and exited having written nothing: no `PROGRESS.md`, no
# `PROGRESS_LOG.md` row, no commit.
#
# `seal_output` behaved correctly and did nothing, because it stamps a report
# that is DIRTY — a run that dies before writing leaves the file clean. So one
# scar's repair assumed the other scar's opposite for the second time in the
# same file: the 08-24 repair covered death-before-writing by NOT publishing,
# the 08-30 repair covered death-after-writing by sealing, and death-before-
# writing-while-the-page-still-claims-to-be-current fell between them.
#
# The consequence stood for two days with nothing red anywhere: `docs/PROGRESS.md`
# opened *"2026-08-29 … Ladder: 84/187 (44.9%). Fifth consecutive day on which
# not one figure in this table has moved"* while the ladder was at 92/200 and
# the builder had just run the most productive 48 hours in the project's
# history. A current-state page that silently describes a two-day-old world is
# the same disease as an uncommitted draft, in the other direction.
#
# WHAT THE ARTIFACT-SIDE CHECK PROVABLY CANNOT SEE, and why this file is keyed
# to the SCHEDULE instead. `seal_output` is called BY the organ, so it can only
# fire when the organ ran. It is blind to every way of producing nothing that
# does not involve running: the usage gate refusing (08-16 at 95%, 08-23 at
# 94%), the pause file left on, cron edited, the box down. Measured over this
# organ's whole life, that blindness is total — see below.
#
# THE FINDING THIS FILE WAS WRITTEN AGAINST, and it is larger than the audit's:
# **the Review has never once completed a FULL run.** All 11 rows in
# `docs/PROGRESS_LOG.md` are `DAILY`. Exactly one FULL run has ever started, and
# these are the only three Sundays the organ has existed for:
#
#     2026-08-16 06:37  STOPPED at 95% weekly usage        (never started)
#     2026-08-23 06:37  STOPPED at 94% weekly usage        (never started)
#     2026-08-30 06:37  started, died at max turns in 11m   (wrote nothing)
#
# Three consecutive weeks, three different failure modes, and Part 2 of this
# project's review has therefore NEVER HAPPENED. Two of the three were invisible
# to any artifact-side instrument by construction. That is the whole argument
# for keying liveness to the schedule.
#
# THE RULE, generalised (27th audit's corollary, written then and never built):
# an organ that is the destination of routed work must have its liveness watched
# by something OTHER THAN ITSELF. This file is the something else; the overseer
# is where it is called from, because the overseer runs 4x/day, takes no lock,
# and only reads.
#
# Run:  bash scripts/test_lib_liveness.sh     (exit 0 = all green)

# ------------------------------------------------------------------ parsing ---
# Rows of a markdown table whose first column is a date: "YYYY-MM-DD MODE".
# Deliberately NOT anchored on "the last line" — rows are appended by hand-ish
# agents and an out-of-order append must not read as freshness. Every caller
# below takes a MAX over the dates, so a stale organ cannot look alive by
# appending an old row.
_md_table_rows() {
  [ -f "$1" ] || return 0
  awk -F'|' '
    /^\| *[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9] *\|/ {
      d = $2; m = $3
      # The mode field is HUMAN PROSE, not a token: the first-ever FULL row was
      # written "**FULL**" as reasonable emphasis, and an exact compare read it
      # as "FULL has never completed" — a stuck alarm that stamped a truthful
      # 684-line report STALE (56th audit). Strip emphasis before comparing, or
      # the authors of the document are silently maintaining this parser.
      gsub(/[ \t]/, "", d); gsub(/[ \t*_]/, "", m)
      print d, m
    }' "$1"
}

# Newest date in the table, or empty if the file has no data rows at all.
history_newest_date() { _md_table_rows "$1" | awk '{print $1}' | sort | tail -1; }

# Newest date carrying a given mode, or empty if that mode has NEVER appeared.
# Empty is the answer that found the finding above — do not collapse it to
# "very old", because "never" and "stale" want different sentences.
history_newest_mode_date() {
  _md_table_rows "$1" | awk -v m="$2" '$2 == m {print $1}' | sort | tail -1
}

# Calendar days from <YYYY-MM-DD> to today. Prints 99999 for empty/unparseable,
# which is the conservative direction: an unreadable history reads as dead, not
# as healthy. UNKNOWN IS NOT ZERO — the same rule `usage_gate` states one
# surface over.
_days_since() {
  local d="$1" then now
  [ -n "$d" ] || { echo 99999; return; }
  then=$(date -u -d "$d 00:00:00" +%s 2>/dev/null) || { echo 99999; return; }
  [ -n "$then" ] || { echo 99999; return; }
  now=$(date -u -d "$(date -u +%F) 00:00:00" +%s)
  echo $(( (now - then) / 86400 ))
}

# ---------------------------------------------------------- the assertions ---
# table_liveness <history-file> <max-row-age-days> <mode> <max-mode-age-days>
#
# Prints a one-line reason and returns 1 when the schedule has been missed;
# returns 0 silently when the organ is keeping to it.
#
# WHY DAYS AND NOT HOURS. The audit asked for "a row within the last 25 h", but
# the rows carry a DATE, not a time, so an hour-precision test would be false
# precision: a row written today at 06:37 parses as today 00:00 and would read
# 25 h old by 01:07 tomorrow — a false alarm every night. Days is what the data
# supports. Cost of the coarser grain: a daily organ that dies at 06:37 is
# caught at 00:00 the next day rather than 25 h later. That is the same day's
# builder either way.
#
# WHY max-mode-age IS 7 FOR A WEEKLY MODE, NOT 8. Consecutive Sundays are
# exactly 7 days apart, so 7 is the largest age a HEALTHY weekly organ can ever
# show (Sunday morning, before that day's run). 8 means a Sunday came and went
# and produced nothing. Setting it to 8 to be "safe" costs a full day of
# blindness and buys nothing, because 7 is not a noisy estimate — it is the
# cadence, read off the crontab.
table_liveness() {
  local file="$1" max_row="$2" mode="$3" max_mode="$4"
  local newest newest_mode age age_mode

  if [ ! -f "$file" ]; then
    echo "$file does not exist — the organ has never published a history row"
    return 1
  fi

  newest=$(history_newest_date "$file")
  age=$(_days_since "$newest")
  if [ "$age" -gt "$max_row" ]; then
    if [ -z "$newest" ]; then
      echo "$file has no data rows at all"
    else
      echo "newest row in $file is $newest (${age}d old; the schedule allows ${max_row}d)"
    fi
    return 1
  fi

  [ -n "$mode" ] || return 0
  newest_mode=$(history_newest_mode_date "$file" "$mode")
  if [ -z "$newest_mode" ]; then
    echo "no $mode row has EVER been written to $file — that mode has never completed"
    return 1
  fi
  age_mode=$(_days_since "$newest_mode")
  if [ "$age_mode" -gt "$max_mode" ]; then
    echo "newest $mode row in $file is $newest_mode (${age_mode}d old; the cadence allows ${max_mode}d)"
    return 1
  fi
  return 0
}

# review_liveness [sayfn]
#
# The Review's concrete schedule assertion, and the only caller wired up today.
# Adding an organ means adding a function like this one — the cadences come off
# the crontab, not off a guess:
#
#     7 * * * *   ladder_loop.sh    hourly
#     37 */6 * * *  overseer.sh     6-hourly
#     37 6 * * *    review.sh       daily, FULL on Sundays
#     37 5 * * 1    field_watch.sh  weekly (Mondays)
#
# Returns 0 when the Review is keeping to its schedule, 1 when it is not. When
# it is not, `docs/PROGRESS.md` is stamped STALE and committed, because the log
# line alone is what failed for two days: the exit code lives in
# /data/jack-logs and the confident page lives in docs/, and until this call
# nothing joined them.
review_liveness() {
  local sayfn="${1:-:}" reason rc=0
  # A PAUSED organ is not a DEAD organ. The owner's pause file is a decision;
  # shouting about it would train the reader to ignore the banner.
  if [ -f "${REPO:-.}/.review-paused" ]; then
    "$sayfn" "review liveness: SKIPPED — .review-paused is set (owner decision, not a fault)"
    return 0
  fi
  reason=$(table_liveness docs/PROGRESS_LOG.md 1 FULL 7) || rc=1
  if [ "$rc" -eq 0 ]; then
    "$sayfn" "review liveness: OK — $(history_newest_date docs/PROGRESS_LOG.md) daily, $(history_newest_mode_date docs/PROGRESS_LOG.md FULL) FULL"
    return 0
  fi
  "$sayfn" "REVIEW LIVENESS FAILED — $reason"
  stale_output docs/PROGRESS.md review \
    "the Review has missed its schedule: $reason" "$sayfn"
  return 1
}
