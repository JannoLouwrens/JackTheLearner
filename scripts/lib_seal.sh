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
#     seal_output <rc> <repo-relative-output-file> <organ> [say-fn]
#
# Committing is part of the repair, not a convenience: the 49th audit's file
# had to be recovered by hand by the next builder, and an uncommitted report in
# a shared tree is one `git clean` from gone (SM.03 lost five days to exactly
# that). The commit is PATH-SCOPED — `git commit -- <file>` — because
# `git commit` otherwise writes the whole index and these organs share a tree
# with a builder that may have staged anything.

seal_output() {
  local rc="$1" file="$2" organ="$3" sayfn="${4:-:}"
  [ "$rc" -eq 0 ] && return 0
  [ -f "$file" ] || return 0
  # Dirty means THIS dying run wrote it (or an earlier one did and nobody
  # sealed it). A clean file is the previous run's committed report and is not
  # this run's draft — leave it alone.
  [ -n "$(git status --porcelain -- "$file" 2>/dev/null)" ] || return 0
  # Never stamp twice — but still commit. A second dying run that appended to
  # an already-sealed draft leaves the same uncommitted file the seal exists to
  # prevent, and one banner is enough to say the same thing.
  if head -3 "$file" | grep -q "INCOMPLETE RUN"; then
    "$sayfn" "$file already carries a draft banner — committing as it stands"
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
      printf '> the log, and this banner is what joins the two.\n\n'
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
  return 0
}
