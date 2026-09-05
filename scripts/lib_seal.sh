#!/bin/bash
# seal_output — an organ's report must not outlive the run that wrote it.
#
# THE SCAR, 2026-08-30 06:50. The 49th audit hit `Reached max turns (60)` and
# exited rc=1. `overseer.sh` handled that correctly IN THE LOG:
#
#     audit end rc=1 — verdict: UNKNOWN (audit did not complete)
#
# and then exited without touching `docs/OVERSIGHT.md`. But the dying run had
# ALREADY written that file, all the way to its closing footer — so the file on
# disk opened `VERDICT: ON TRACK`, the first non-DRIFTING verdict in four
# audits, and the run that reached it never finished its own checklist. It was
# also never committed, so the next builder found a confident audit sitting
# uncommitted in a shared tree with nothing to say it was a draft.
#
# THE ASSUMPTION THAT FAILED is written in overseer.sh's own comment: *"A run
# that died (rc!=0) did not write it."* That was the repair for the 08-24 scar,
# where a 2-second death republished the PREVIOUS run's verdict, and for THAT
# failure it is true. It is false for a run that dies LATE — max turns, a
# timeout, an API error after the report is on disk — which is the more likely
# death for an organ whose last act is to write a long file. One scar's repair
# assumed the other scar's opposite.
#
# So the log and the file disagreed, and nothing joined them: the exit code
# lives in `/data/jack-logs/*.log` and the verdict lives in `docs/`, and every
# human and agent who opens `docs/` sees only the confident half. Same shape as
# the background-liveness scar one layer up — an organ reporting health for
# work that had already died, because the organ that would notice is the organ
# that stopped.
#
# THE RULE: if a run exits non-zero and its output file is dirty, that run
# wrote a DRAFT. Stamp it as one, in the file, above the verdict, and commit it
# so it is neither lost nor mistaken for a finding.
#
#     seal_output <rc> <repo-relative-output-file> <organ> [say-fn] [max-clean-age-h]
#
# AND THE THIRD CASE, added 2026-08-31 (52nd audit, B1b). The two branches above
# cover a run that died EARLY (leave the committed file alone) and a run that
# died LATE (stamp its draft). Neither covers a run that died BEFORE WRITING on
# a schedule it OWED: the file is clean, so the DRAFT branch skips it, and the
# page goes on presenting an old report as current state with nothing on it
# saying so. That is what happened to `docs/PROGRESS.md` on 2026-08-30 and it
# stood for two days. `stale_output` below is that branch, and it is shared with
# `scripts/lib_liveness.sh`, which reaches the same conclusion from the schedule
# side when the organ never ran at all.
#
# The clean branch takes an AGE, and refusing to stamp a young file is the whole
# point of it. The overseer publishes 4x/day: if its 12:37 run dies before
# writing, the 06:37 report is six hours old and perfectly current, and stamping
# it STALE would be noise that teaches the reader to skip banners. So a clean
# file is stale only once it is older than its organ's own cadence — 7 h for the
# 6-hourly overseer, 25 h for the daily Review, 169 h for the weekly field
# watch. Those numbers are read off the crontab; none of them is an estimate.
#
# Committing is part of the repair, not a convenience: the 49th audit's file
# had to be recovered by hand by the next builder, and an uncommitted report in
# a shared tree is one `git clean` from gone (SM.03 lost five days to exactly
# that). The commit is PATH-SCOPED — `git commit -- <file>` — because
# `git commit` otherwise writes the whole index and these organs share a tree
# with a builder that may have staged anything.

# Hours since the commit that last touched <file>, or 99999 if git cannot say.
# mtime is the wrong clock: a checkout rewrites it and would make a
# months-stale report look minutes fresh. UNKNOWN IS NOT ZERO — an unanswerable
# age reads as old, and `stale_output` then refuses on its own dirty check if
# the file is merely untracked.
_seal_file_age_hours() {
  local ct
  ct=$(git log -1 --format=%ct -- "$1" 2>/dev/null)
  [ -n "$ct" ] || { echo 99999; return; }
  echo $(( ( $(date +%s) - ct ) / 3600 ))
}

# stale_output <file> <organ> <reason> [say-fn]
#
# The file is not this run's draft — it is the PREVIOUS run's report, still
# accurate as a record and no longer accurate as current state. Say so at the
# top of the page and commit it, because the log line alone is what failed.
stale_output() {
  local file="$1" organ="$2" reason="$3" sayfn="${4:-:}"
  [ -f "$file" ] || return 0
  # NEVER touch a file someone else is editing. This is the `git add -A` lesson
  # applied to a stamper: a writer on a shared tree must bound itself to its own
  # edits, and a dirty output file here means either a live organ session or the
  # owner. Refusing is the honest failure — say it and stop.
  if [ -n "$(git status --porcelain -- "$file" 2>/dev/null)" ]; then
    "$sayfn" "NOT stamping $file stale — it is dirty in the shared tree; someone is writing it"
    return 1
  fi
  if head -8 "$file" | grep -q "STALE — "; then
    "$sayfn" "$file already carries a stale banner — leaving it"
    return 0
  fi
  local stamp
  stamp="$(date -Iseconds)"
  {
    printf '> **STALE — THE RUN THAT OWED THIS PAGE AN UPDATE PRODUCED NOTHING.**\n'
    printf '> %s\n' "$reason"
    printf '> So everything below is the PREVIOUS run of the %s and is a RECORD,\n' "$organ"
    printf '> not current state: its counts, its "current state" framing and any\n'
    printf '> claim about what has or has not moved describe an older world.\n'
    printf '> Stamped %s by scripts/lib_seal.sh. It disappears the next time the\n' "$stamp"
    printf '> %s completes a run and rewrites this file.\n\n' "$organ"
    cat "$file"
  } > "$file.stale" && mv "$file.stale" "$file"
  "$sayfn" "stamped $file STALE — $reason"
  git add -- "$file" 2>/dev/null
  git commit -q -m "$organ: schedule missed — $file stamped STALE, not current state

$reason

Committed by scripts/lib_seal.sh, not by the organ's agent. The content is the
previous run's real work and is kept; only its claim to describe TODAY is
withdrawn. The banner clears itself the next time the organ completes." -- "$file" 2>/dev/null \
    && "$sayfn" "committed the stale banner" \
    || "$sayfn" "WARNING: could not commit the stale banner — it is dirty in the tree"
  return 0
}

# seal_output <rc> <repo-relative-output-file> <organ> [say-fn] [max-clean-age-h] [run-start-epoch]
#
# THE SIXTH ARGUMENT, added 2026-09-05 (74th audit B1). The signature above
# takes ONE path because the 49th-audit scar was about one page — and a run's
# product is rarely its page. On 2026-09-05 the daily Review died at max turns
# with FIVE files dirty: the report got the banner, and the other four (a live
# owner decision, a shrink-only ratchet move, the week's only queue disposal,
# the builder's 122-line priority block) went out six hours later as ordinary
# work. The seal protected the receipt and let the transactions through.
#
# So on rc!=0 the seal now also sweeps the run's OTHER dirty paths: commits
# them in one path-scoped commit naming the rc and the organ, and lists them
# inside the sealed report, so neither a reader of the page nor a reader of
# `git log` needs a hand-check to learn their provenance.
#
# THE BOUND, and it is the `git add -A` lesson applied to the sweeper: this is
# a SHARED tree, and a writer on it must bound itself to its own edits. The
# wrapper passes the epoch at which its run started; only dirty paths whose
# mtime is at or after that moment are swept. Anything older (the owner's
# uncommitted draft, another organ's staged work) is LEFT dirty for its author
# — but still NAMED in the report, so it is visible without being seized.
# Deleted paths have no mtime and are never swept, only named. With no epoch
# given, nothing is swept and everything dirty is named: an unbounded sweep
# would be the ddbe6b7 scar with a banner on it.
seal_output() {
  local rc="$1" file="$2" organ="$3" sayfn="${4:-:}" max_clean_age="${5:-25}" run_start="${6:-}"
  [ "$rc" -eq 0 ] && return 0
  [ -f "$file" ] || return 0
  # Dirty means THIS dying run wrote it (or an earlier one did and nobody
  # sealed it) -> the DRAFT branch below. A clean file is the previous run's
  # committed report; it is not this run's draft, but if the run that owed it an
  # update died without writing, it is no longer current state either.
  if [ -z "$(git status --porcelain -- "$file" 2>/dev/null)" ]; then
    local age_h
    age_h=$(_seal_file_age_hours "$file")
    if [ "$age_h" -le "$max_clean_age" ]; then
      "$sayfn" "$file untouched by this rc=$rc run and only ${age_h}h old (cadence allows ${max_clean_age}h) — still current, not stamping"
      return 0
    fi
    stale_output "$file" "$organ" \
      "The $organ run that would have rewritten it exited rc=$rc without writing a word, and the file is now ${age_h}h old against a ${max_clean_age}h cadence." \
      "$sayfn"
    return 0
  fi
  # The run's whole dirty set, partitioned BEFORE the banner is written so the
  # banner can name both halves (74th audit B1). `swept` = this run's own acts
  # (mtime >= run start); `left` = everything else dirty, named but untouched.
  local -a swept=()
  local swept_names="" left_names="" _line _p
  while IFS= read -r _line; do
    [ -n "$_line" ] || continue
    _p="${_line:3}"; _p="${_p##* -> }"
    [ "$_p" = "$file" ] && continue
    if [ -n "$run_start" ] && [ -e "$_p" ] \
       && [ "$(stat -c %Y "$_p" 2>/dev/null || echo 0)" -ge "$run_start" ]; then
      swept+=("$_p"); swept_names="${swept_names:+$swept_names, }$_p"
    else
      left_names="${left_names:+$left_names, }$_p"
    fi
  done < <(git status --porcelain 2>/dev/null)
  # Never stamp twice — but still commit. A second dying run that appended to
  # an already-sealed draft leaves the same uncommitted file the seal exists to
  # prevent, and one banner is enough to say the same thing.
  if head -3 "$file" | grep -q "INCOMPLETE RUN"; then
    "$sayfn" "$file already carries a draft banner — committing as it stands"
    if [ -n "$swept_names$left_names" ] && ! grep -q "also left dirty" "$file"; then
      {
        printf '\n> Files this run also left dirty'
        [ -n "$swept_names" ] && printf ', committed unbannered by the seal: %s' "$swept_names"
        [ -n "$left_names" ] && printf '; left dirty and NOT committed (predate this run, or no run-start known): %s' "$left_names"
        printf '.\n'
      } >> "$file"
    fi
  else
    local stamp
    stamp="$(date -Iseconds)"
    {
      printf '> **INCOMPLETE RUN — THIS IS A DRAFT, NOT A FINDING.**\n'
      printf '> The %s run that wrote this file exited rc=%s and did not\n' "$organ" "$rc"
      printf '> complete its own checklist (%s). Everything below was\n' "$stamp"
      printf '> written before the run stopped: any verdict, any section claiming\n'
      printf '> "no findings", and any instrument table in it are UNVERIFIED.\n'
      printf '> Sealed automatically by scripts/lib_seal.sh; the exit code is in\n'
      printf '> the log, and this banner is what joins the two.\n'
      [ -n "$swept_names" ] && \
        printf '> Files this run also left dirty, committed unbannered by the seal: %s.\n' "$swept_names"
      [ -n "$left_names" ] && \
        printf '> Left dirty and NOT committed (predate this run, or no run-start known): %s.\n' "$left_names"
      printf '\n'
      cat "$file"
    } > "$file.sealed" && mv "$file.sealed" "$file"
    "$sayfn" "sealed $file as an INCOMPLETE RUN draft (rc=$rc)"
  fi
  git add -- "$file" 2>/dev/null
  git commit -q -m "$organ: run exited rc=$rc mid-report — $file sealed as a draft

Committed by scripts/lib_seal.sh, not by the organ's agent. The run wrote this
file and then died before finishing its checklist, so its verdict is unearned.
Preserved rather than discarded: the content is real work; only its status is
in doubt." -- "$file" 2>/dev/null \
    && "$sayfn" "committed the sealed draft" \
    || "$sayfn" "WARNING: could not commit the sealed draft — it is dirty in the tree"
  # The run's ACTS — dispositions, decisions, steering — committed in one
  # path-scoped commit that names the rc, the organ and the sealed report, so
  # `git log` joins them the way the banner joins the report to the log.
  if [ "${#swept[@]}" -gt 0 ]; then
    git add -- "${swept[@]}" 2>/dev/null
    git commit -q -m "$organ: rc=$rc run's other dirty files, committed unbannered — see the sealed $file

These paths were left dirty by the same $organ run whose report was sealed as
an INCOMPLETE RUN draft (rc=$rc). They are that run's acts, kept rather than
discarded — an uncommitted disposition in a shared tree is one git clean from
gone — but their author never finished its own checklist, and the sealed
report names them (74th audit B1). Swept only because their mtime postdates
the run's start; committed by scripts/lib_seal.sh, not by the organ's agent." \
      -- "${swept[@]}" 2>/dev/null \
      && "$sayfn" "committed ${#swept[@]} other dirty file(s) from the dying run: $swept_names" \
      || "$sayfn" "WARNING: could not commit the run's other dirty files: $swept_names"
  fi
  [ -n "$left_names" ] && "$sayfn" "left dirty for their author (predate run start or no run-start known): $left_names"
  return 0
}
