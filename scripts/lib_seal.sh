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

# THE FOURTH CASE, added 2026-09-14 (D25's armed default, option (iii) FIX THE
# SEAL, BUY NOTHING — the owner did not rule by 2026-09-13, so the
# pre-registered default fired).
#
# THE SCAR: 2026-09-06. The Sunday FULL wrote its whole page, committed its
# dispositions as it made them, and appended its own `PROGRESS_LOG` row at
# 07:12 — the LAST item on its checklist. `timeout(1)` killed it at the
# 40-minute wall five minutes later, and this file, reading only `rc != 0`,
# gave that page the identical banner it correctly gives a run that died having
# written nothing: *"THIS IS A DRAFT, NOT A FINDING ... any verdict ... any
# instrument table in it are UNVERIFIED."* The builder then spent twenty-four
# hours executing seven of that page's nine `FOR THE BUILDER` items off a
# document formally marked unverified — and was right to. One verdict for two
# opposite events is how a seal teaches its readers to skip banners, which is
# the one failure mode a seal cannot survive.
#
# THE DISCRIMINATOR is the organ's own TAIL RECEIPT: the last item on its
# checklist, written by the agent, in a file the seal can read. The caller
# passes it (`receipt_file` + `receipt_pat`); if a matching line is there when
# the seal runs, the run reached the end of its own list and died on the tail,
# and the banner says THAT instead of calling the page a draft. A line that
# declares itself INCOMPLETE is never a completion receipt — `review.sh` writes
# exactly such a row for a run that died before its own append (76th audit B4),
# and a receipt an organ's own dead-run fallback can satisfy is not a receipt.
#
# ONE DEVIATION FROM THE DEFAULT'S LETTER, recorded because it is a deviation.
# `D25`'s default reads *"if this run committed `docs/PROGRESS.md` AND appended
# its `PROGRESS_LOG` row"*. Git says the first conjunct is false of the very run
# the decision cites: on 2026-09-06 the agent never committed the page — it left
# it dirty and `lib_seal.sh` itself committed it at 07:17:11 (`cf18320`). Taken
# literally the new branch would never fire on its own scar. So the gate is the
# RECEIPT, and the page's custody is REPORTED rather than required: in this
# branch the page is this run's complete product and is committed in the same
# breath, by the run or by the seal. Strictly monotone either way — it can only
# replace a false banner with a truer one, it moves no threshold, refuses no
# run, fails no spec and stales no certificate. A run with no receipt keeps
# today's wording BYTE-FOR-BYTE, which is what the default requires.

# _seal_stamp_emissions <path> <organ> <rc> <stamp> [tail_complete]
#
# THE SCAR (76th audit 9.3, 2026-09-06): `D23` — an armed owner decision with a
# default that fires by silence — was written by the 09-05 Review run that died
# at max turns. The seal bannered THE PAGE, the sweep committed the run's other
# dirty files with provenance in the COMMIT MESSAGE — and the entry itself
# reached the owner's desk reading like any other, because `decisions --check`
# and every human reads the FILE, not `git log`. A dead run's routed artefacts
# outlived the banner that was supposed to qualify them.
#
# So: when the sweep commits a registry the dead run wrote into, each entry the
# run ADDED (a `##`/`###` header new in the diff) gets one provenance line
# directly under its header — a disclosure, never a reversal: nothing is
# re-dated, weakened or un-armed. Scoped to the three files where an entry can
# sit on a clock or hold a seat; other swept files keep commit-message
# provenance only, EXCEPT `experiments/ledger.json`, which gets row-level
# provenance via `_seal_stamp_ledger` below (79th audit 1.1). Idempotent: a
# header already followed by a PROVENANCE line is left alone.
_seal_stamp_emissions() {
  local p="$1" organ="$2" rc="$3" stamp="$4" tail_complete="${5:-}" hdr note
  case "$p" in
    docs/DECISIONS_NEEDED.md|docs/REVIEW_QUEUE.md|docs/CHAMPIONS.md) ;;
    *) return 0;;
  esac
  if [ -n "$tail_complete" ]; then
    # D25: the run finished its checklist and was killed on the tail. The entry
    # is still a dead run's artefact and still unaudited — but saying its report
    # is "sealed as an INCOMPLETE RUN draft" would be the same falsehood this
    # decision fired to remove, one file over.
    note="> PROVENANCE (scripts/lib_seal.sh, ${stamp}): this entry was written by a ${organ} run that reached the last item on its own checklist and was then killed at its wall clock (rc=${rc}); its report is sealed as COMPLETE, not as a draft. The entry's facts are that run's, unverified until an audit re-measures them; nothing here is re-dated, weakened or un-armed by this line."
  else
    note="> PROVENANCE (scripts/lib_seal.sh, ${stamp}): this entry was written by a ${organ} run that exited rc=${rc} without completing its own checklist — its report is sealed as an INCOMPLETE RUN draft. The entry's facts are that dead run's, unverified until an audit re-measures them; nothing here is re-dated, weakened or un-armed by this line."
  fi
  # Added headers only: the diff is unstaged working-tree vs HEAD, taken before
  # the sweep's `git add`, so `+## ...` lines are exactly what this run created.
  while IFS= read -r hdr; do
    [ -n "$hdr" ] || continue
    grep -qxF "$hdr" "$p" || continue
    awk -v h="$hdr" -v n="$note" '
      { lines[NR] = $0 }
      END {
        for (i = 1; i <= NR; i++) {
          print lines[i]
          if (lines[i] == h && lines[i+1] !~ /^> PROVENANCE/) print n
        }
      }' "$p" > "$p.provstamp" && mv "$p.provstamp" "$p"
  done < <(git diff -- "$p" 2>/dev/null | grep -E '^\+#{2,3} ' | sed 's/^\+//')
  return 0
}

# _seal_stamp_ledger <path> <organ> <rc> <stamp>
#
# THE SCAR (79th audit 1.1, 2026-09-06): the Sunday FULL hit its 40-minute
# wall and the sweep committed `experiments/ledger.json` carrying `ME.1`'s
# FAIL row — correctly, the row was a true measurement and a clean-tree re-run
# superseded it — but the only place saying a dying organ committed it was
# `review.log`. `PROGRESS.md` got a banner; the ledger got nothing, and every
# reader of the ledger reads the FILE, not the organ's log. A row committed by
# a timed-out organ should be as visible in the ledger as `+dirty` makes a
# dirty tree.
#
# So: before the sweep commits the ledger, every spec row whose content
# differs from HEAD gets an additive `seal_provenance` string — a DISCLOSURE,
# never a change. Status, metrics, commit, history are untouched; the runner's
# own lock (`ledger.json.lock`) and tmp+fsync+replace discipline are reused so
# a live run cannot be raced; and the key vanishes the next time a real run
# records that spec, exactly as a clean re-run supersedes a `+dirty` stamp.
# NOT routed through `amended`: an amendment records a CHANGE (field/from/to)
# and feeds the staleness lane; this key records an unchanged row's chain of
# custody. Any failure inside returns 0 — the seal must never make the
# dying-run path worse than an unstamped commit.
_seal_stamp_ledger() {
  local p="$1" organ="$2" rc="$3" stamp="$4" tail_complete="${5:-}"
  [ "$p" = "experiments/ledger.json" ] || return 0
  SEAL_ORGAN="$organ" SEAL_RC="$rc" SEAL_STAMP="$stamp" \
  SEAL_TAIL="${tail_complete:+1}" \
  /data/venvs/jackthelearner/bin/python - "$p" <<'PYEOF' || return 0
import fcntl, json, os, subprocess, sys, tempfile
path = sys.argv[1]
organ = os.environ["SEAL_ORGAN"]; rc = os.environ["SEAL_RC"]
stamp = os.environ["SEAL_STAMP"]
tail = os.environ.get("SEAL_TAIL", "")
died = (f"run that reached the last item on its own checklist and was then "
        f"killed at its wall clock (rc={rc})" if tail else
        f"run that exited rc={rc} without completing its own checklist")
note = (f"committed by scripts/lib_seal.sh at {stamp}, swept from a {organ} "
        f"{died}. The "
        "row is as the runner wrote it, unmodified apart from this key — "
        "provenance, not an amendment. The next real run of this spec "
        "supersedes it.")
try:
    head = json.loads(subprocess.run(
        ["git", "show", f"HEAD:{path}"], capture_output=True, text=True,
        check=True).stdout).get("results", {})
except Exception:
    head = {}
lock = open(path + ".lock", "a+")
fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
try:
    with open(path) as f:
        payload = json.load(f)
    stamped = []
    for sid, row in payload.get("results", {}).items():
        cur = dict(row); cur.pop("seal_provenance", None)
        prev = dict(head.get(sid) or {}); prev.pop("seal_provenance", None)
        if cur != prev:
            row["seal_provenance"] = note
            stamped.append(sid)
    if stamped:
        fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path) or ".",
                                   suffix=".tmp")
        with os.fdopen(fd, "w") as f:
            f.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
            f.flush(); os.fsync(f.fileno())
        os.replace(tmp, path)
        print("seal_provenance stamped on ledger row(s): "
              + ", ".join(sorted(stamped)))
finally:
    fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
PYEOF
}

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
  local rc="$1" file="$2" organ="$3" sayfn="${4:-:}" max_clean_age="${5:-25}" run_start="${6:-}" \
        receipt_file="${7:-}" receipt_pat="${8:-}"
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
  # D25's TAIL RECEIPT (see the fourth case above): did this run reach the last
  # item on its own checklist? Read BEFORE the banner is chosen, and read from
  # the tree rather than from git, because an agent that appended its row and
  # died before committing still finished the list — the sweep below commits it.
  # A row that declares itself INCOMPLETE is the organ's own dead-run fallback
  # and is never a receipt.
  local tail_complete="" receipt_line=""
  if [ -n "$receipt_file" ] && [ -n "$receipt_pat" ] && [ -f "$receipt_file" ]; then
    receipt_line=$(grep -E "$receipt_pat" "$receipt_file" 2>/dev/null \
                   | grep -v 'INCOMPLETE' | tail -1)
    if [ -n "$receipt_line" ]; then
      tail_complete=1
      "$sayfn" "tail receipt found in $receipt_file — this run finished its checklist and was killed on the tail"
    else
      "$sayfn" "no tail receipt in $receipt_file — sealing as an INCOMPLETE RUN draft"
    fi
  fi
  # Never stamp twice — but still commit. A second dying run that appended to
  # an already-sealed draft leaves the same uncommitted file the seal exists to
  # prevent, and one banner is enough to say the same thing.
  if head -3 "$file" | grep -qE "INCOMPLETE RUN|CHECKLIST COMPLETE"; then
    "$sayfn" "$file already carries a seal banner — committing as it stands"
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
      if [ -n "$tail_complete" ]; then
        # D25 (iii). The page is a finished product whose run lost its exit,
        # not its work. Two things this banner is careful NOT to say: that the
        # page has been verified by anyone else, and that anything AFTER the
        # receipt in the organ's checklist ran.
        printf '> **CHECKLIST COMPLETE — THE RUN WAS KILLED ON THE TAIL, NOT MID-REPORT.**\n'
        printf '> The %s run that wrote this file exited rc=%s (%s), and\n' "$organ" "$rc" "$stamp"
        printf '> before it died it reached the LAST item on its own checklist: %s\n' "$receipt_file"
        printf '> carries this run'"'"'s own row —\n'
        printf '>     %s\n' "$receipt_line"
        printf '> So this page is that run'"'"'s finished product and is committed as\n'
        printf '> such: what the run lost was its exit, not its work. It is still\n'
        printf '> UNAUDITED — no other organ has checked it — and any checklist item\n'
        printf '> that comes AFTER the row above did not run. Sealed by\n'
        printf '> scripts/lib_seal.sh (D25, armed default fired 2026-09-14); the exit\n'
        printf '> code is in the log, and this banner is what joins the two.\n'
      else
      printf '> **INCOMPLETE RUN — THIS IS A DRAFT, NOT A FINDING.**\n'
      printf '> The %s run that wrote this file exited rc=%s and did not\n' "$organ" "$rc"
      printf '> complete its own checklist (%s). Everything below was\n' "$stamp"
      printf '> written before the run stopped: any verdict, any section claiming\n'
      printf '> "no findings", and any instrument table in it are UNVERIFIED.\n'
      printf '> Sealed automatically by scripts/lib_seal.sh; the exit code is in\n'
      printf '> the log, and this banner is what joins the two.\n'
      fi
      [ -n "$swept_names" ] && \
        printf '> Files this run also left dirty, committed unbannered by the seal: %s.\n' "$swept_names"
      [ -n "$left_names" ] && \
        printf '> Left dirty and NOT committed (predate this run, or no run-start known): %s.\n' "$left_names"
      printf '\n'
      cat "$file"
    } > "$file.sealed" && mv "$file.sealed" "$file"
    if [ -n "$tail_complete" ]; then
      "$sayfn" "sealed $file as COMPLETE — killed on the tail (rc=$rc)"
    else
      "$sayfn" "sealed $file as an INCOMPLETE RUN draft (rc=$rc)"
    fi
  fi
  git add -- "$file" 2>/dev/null
  # The subject line is what `git log` shows, so it carries the same
  # distinction the banner does (D25): a page whose run finished its checklist
  # is not "sealed as a draft", and a reader of the log should not have to open
  # the file to learn which of the two happened.
  local _msg
  if [ -n "$tail_complete" ]; then
    _msg="$organ: run killed at its wall clock on the tail (rc=$rc) — $file sealed as COMPLETE

Committed by scripts/lib_seal.sh, not by the organ's agent. The run reached the
last item on its own checklist ($receipt_file carries its row) and was then
killed, so what it lost was its exit, not its work. The page is preserved as a
finished — and still unaudited — product, which is what D25's armed default
fired to make this instrument able to say."
  else
    _msg="$organ: run exited rc=$rc mid-report — $file sealed as a draft

Committed by scripts/lib_seal.sh, not by the organ's agent. The run wrote this
file and then died before finishing its checklist, so its verdict is unearned.
Preserved rather than discarded: the content is real work; only its status is
in doubt."
  fi
  git commit -q -m "$_msg" -- "$file" 2>/dev/null \
    && "$sayfn" "committed the sealed report" \
    || "$sayfn" "WARNING: could not commit the sealed report — it is dirty in the tree"
  # The run's ACTS — dispositions, decisions, steering — committed in one
  # path-scoped commit that names the rc, the organ and the sealed report, so
  # `git log` joins them the way the banner joins the report to the log.
  if [ "${#swept[@]}" -gt 0 ]; then
    # A routed artefact gets the provenance its page gets (76th audit B3):
    # stamp entries the dying run ADDED to the decision/queue/champions
    # registries before they are committed, so the disclosure travels in the
    # FILE the next reader opens, not only in a commit message nobody greps.
    local _sp _stamp
    _stamp="$(date -Iseconds)"
    for _sp in "${swept[@]}"; do
      _seal_stamp_emissions "$_sp" "$organ" "$rc" "$_stamp" "$tail_complete"
      _seal_stamp_ledger "$_sp" "$organ" "$rc" "$_stamp" "$tail_complete"
    done
    git add -- "${swept[@]}" 2>/dev/null
    local _how
    if [ -n "$tail_complete" ]; then
      _how="whose report was sealed as COMPLETE (rc=$rc) — it reached the last
item on its own checklist and was then killed at its wall clock"
    else
      _how="whose report was sealed as an INCOMPLETE RUN draft (rc=$rc) — its
author never finished its own checklist"
    fi
    git commit -q -m "$organ: rc=$rc run's other dirty files, committed unbannered — see the sealed $file

These paths were left dirty by the same $organ run $_how. They are that run's
acts, kept rather than discarded — an uncommitted disposition in a shared tree
is one git clean from gone — and the sealed report names them (74th audit B1).
Swept only because their mtime postdates the run's start; committed by
scripts/lib_seal.sh, not by the organ's agent." \
      -- "${swept[@]}" 2>/dev/null \
      && "$sayfn" "committed ${#swept[@]} other dirty file(s) from the dying run: $swept_names" \
      || "$sayfn" "WARNING: could not commit the run's other dirty files: $swept_names"
  fi
  [ -n "$left_names" ] && "$sayfn" "left dirty for their author (predate run start or no run-start known): $left_names"
  return 0
}
