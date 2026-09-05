#!/bin/bash
# Self-test for scripts/lib_liveness.sh and the STALE branch of lib_seal.sh —
# the pair that decides whether a dead organ is visible.
#
# WHY THIS FILE EXISTS. The governing rule applies to conduct code too: a
# capability may only be claimed by a test that could have failed. The thing
# being claimed here is "the system can now see a Review that did not run", and
# the failure this guards against is exactly the one it repairs — an instrument
# that reports health because the organ that would notice is the organ that
# stopped. Every case below fails if the checker stops doing what its comments
# say.
#
# The seam is a REAL git repo in a tmpdir, not a stub: `stale_output` refuses to
# touch a dirty file and commits path-scoped, and both behaviours are the point
# of it, so mocking git would test nothing that matters.
#
# Run:  bash scripts/test_lib_liveness.sh     (exit 0 = all green)
set -uo pipefail

REAL_REPO=/home/opc/jackthelearner
FAIL=0
LOGLINE=""
say() { LOGLINE="$LOGLINE$*
"; }

ok()  { printf '  ok    %s\n' "$1"; }
bad() { printf '  FAIL  %s\n     %s\n' "$1" "$2"; FAIL=$((FAIL + 1)); }
chk() { [ "$2" = "$3" ] && ok "$1" || bad "$1" "expected [$3], got [$2]"; }

TMP=$(mktemp -d) || exit 1
trap 'rm -rf "$TMP"' EXIT

. "$REAL_REPO/scripts/lib_seal.sh"
. "$REAL_REPO/scripts/lib_liveness.sh"

# A markdown history table with rows at given <days-ago> <mode>.
mklog() {
  local f="$1"; shift
  { echo "| date | mode | one line |"; echo "|---|---|---|"; } > "$f"
  while [ $# -gt 0 ]; do
    printf '| %s | %s | text |\n' "$(date -u -d "$1 days ago" +%F)" "$2" >> "$f"
    shift 2
  done
}

printf '\n--- table_liveness: the daily row ---\n'

mklog "$TMP/h1.md" 0 DAILY 7 FULL
table_liveness "$TMP/h1.md" 1 FULL 7 >/dev/null 2>&1
chk "today's row + a FULL 7d ago is healthy" "$?" "0"

# 7 days is the largest age a healthy WEEKLY organ can show (Sunday morning,
# before that day's run). It must NOT read as a fault, or the check cries wolf
# once a week forever.
mklog "$TMP/h2.md" 1 DAILY 7 FULL
table_liveness "$TMP/h2.md" 1 FULL 7 >/dev/null 2>&1
chk "yesterday's row is healthy (the daily runs at 06:37)" "$?" "0"

# THE SCAR ITSELF: newest row 2 days old. This is the state the repo was in for
# two days with nothing anywhere going red.
mklog "$TMP/h3.md" 2 DAILY 5 FULL
OUT=$(table_liveness "$TMP/h3.md" 1 FULL 7 2>&1); RC=$?
chk "a 2-day-old newest row is STALE" "$RC" "1"
case "$OUT" in *"2d old"*) ok "and it names the age" ;;
  *) bad "and it names the age" "got [$OUT]" ;; esac

printf '\n--- table_liveness: the weekly mode ---\n'

# 8 days means a Sunday came and went and produced nothing.
mklog "$TMP/h4.md" 0 DAILY 8 FULL
OUT=$(table_liveness "$TMP/h4.md" 1 FULL 7 2>&1); RC=$?
chk "a FULL 8d ago is a missed Sunday" "$RC" "1"

# THE FINDING THE REAL FILE IS IN: 11 DAILY rows, zero FULL, ever. "Never" and
# "stale" want different sentences, so the checker must not collapse them.
mklog "$TMP/h5.md" 0 DAILY 1 DAILY 2 DAILY
OUT=$(table_liveness "$TMP/h5.md" 1 FULL 7 2>&1); RC=$?
chk "a mode that has NEVER run is a fault" "$RC" "1"
case "$OUT" in *EVER*) ok "and it says EVER, not an age" ;;
  *) bad "and it says EVER, not an age" "got [$OUT]" ;; esac

printf '\n--- history_newest_mode_date: the matcher must see the HEALTHY state ---\n'

# THE 56th-AUDIT SCAR: the first-ever FULL row was written "**FULL**" and the
# exact-string compare went blind on the night it was first told good news — a
# stuck alarm that stamped a truthful report STALE. The mode field is prose;
# the matcher must recognise every reasonable emphasis of the same word. A
# liveness watch with no test for its own matcher is a ratchet whose wiring a
# formatting choice can disconnect.
for v in 'FULL' '**FULL**' ' FULL ' '_FULL_'; do
  mklog "$TMP/hm.md" 3 DAILY 5 "$v"
  chk "mode written as [$v] is still FULL" \
    "$(history_newest_mode_date "$TMP/hm.md" FULL)" "$(date -u -d '5 days ago' +%F)"
done

printf '\n--- table_liveness: unknown is not zero ---\n'

table_liveness "$TMP/does-not-exist.md" 1 FULL 7 >/dev/null 2>&1
chk "a missing history file is a fault, not a pass" "$?" "1"

{ echo "| date | mode |"; echo "|---|---|"; } > "$TMP/h6.md"
table_liveness "$TMP/h6.md" 1 FULL 7 >/dev/null 2>&1
chk "a header-only history is a fault, not a pass" "$?" "1"

# An organ cannot look alive by appending an OLD row after a fresh one: every
# check takes a max over the dates rather than reading the last line.
mklog "$TMP/h7.md" 2 DAILY 0 DAILY 9 DAILY
table_liveness "$TMP/h7.md" 1 "" 7 >/dev/null 2>&1
chk "out-of-order rows are maxed, not tailed" "$?" "0"

printf '\n--- stale_output: the stamp and the shared tree ---\n'

# A real repo, because the dirty-refusal and the path-scoped commit ARE the
# behaviour under test.
WORK="$TMP/repo"; mkdir -p "$WORK"
git -C "$WORK" init -q
git -C "$WORK" config user.email t@t; git -C "$WORK" config user.name t
mkdir -p "$WORK/docs"
printf '# PROGRESS\n\nLadder: 84/187.\n' > "$WORK/docs/P.md"
printf 'other\n' > "$WORK/docs/OTHER.md"
git -C "$WORK" add -A; git -C "$WORK" commit -q -m init
( cd "$WORK" && stale_output docs/P.md review "the run died" say ) >/dev/null 2>&1
chk "a clean stale page gets a banner" \
  "$(head -1 "$WORK/docs/P.md" | grep -c 'STALE — ')" "1"
chk "and it is committed, not left dirty" \
  "$(git -C "$WORK" status --porcelain | wc -l)" "0"
chk "and the original content survives under it" \
  "$(grep -c '84/187' "$WORK/docs/P.md")" "1"

# Idempotence: the overseer runs 4x/day and a Review can be dead for days.
( cd "$WORK" && stale_output docs/P.md review "the run died again" say ) >/dev/null 2>&1
chk "a second call does not stack a second banner" \
  "$(grep -c 'STALE — ' "$WORK/docs/P.md")" "1"

# THE `git add -A` LESSON, one surface over: a writer on a shared tree must
# bound itself to its own edits. Someone else's uncommitted work in the same
# file means REFUSE — do not stamp, do not commit, say so.
printf 'owner is editing this\n' >> "$WORK/docs/OTHER.md"
( cd "$WORK" && stale_output docs/OTHER.md scout "the run died" say ) >/dev/null 2>&1
chk "a DIRTY output file is refused, not stamped" \
  "$(head -1 "$WORK/docs/OTHER.md" | grep -c 'STALE — ')" "0"
chk "and the other author's work is left uncommitted for them" \
  "$(git -C "$WORK" status --porcelain -- docs/OTHER.md | wc -l)" "1"

printf '\n--- seal_output: the clean branch respects the cadence ---\n'

# A young clean file is the CURRENT report even when this run died. Stamping it
# would be noise that teaches the reader to skip banners.
printf 'fresh report\n' > "$WORK/docs/Y.md"
git -C "$WORK" add -A; git -C "$WORK" commit -q -m fresh
( cd "$WORK" && seal_output 1 docs/Y.md overseer say 7 ) >/dev/null 2>&1
chk "a 0h-old clean file is NOT stamped (cadence 7h)" \
  "$(head -1 "$WORK/docs/Y.md" | grep -c 'STALE — ')" "0"

# The same file against a cadence it HAS outlived.
( cd "$WORK" && seal_output 1 docs/Y.md overseer say -1 ) >/dev/null 2>&1
chk "the same file IS stamped once it outlives its cadence" \
  "$(head -1 "$WORK/docs/Y.md" | grep -c 'STALE — ')" "1"

# rc=0 must never stamp anything, ever.
printf 'ok report\n' > "$WORK/docs/Z.md"
git -C "$WORK" add -A; git -C "$WORK" commit -q -m z
( cd "$WORK" && seal_output 0 docs/Z.md overseer say -1 ) >/dev/null 2>&1
chk "a SUCCESSFUL run stamps nothing" \
  "$(head -1 "$WORK/docs/Z.md" | grep -c 'STALE — ')" "0"

# The DRAFT branch must survive the change: a dirty file on rc!=0 is still a
# draft, not a stale page.
printf 'VERDICT: ON TRACK\n' > "$WORK/docs/D.md"
git -C "$WORK" add -A; git -C "$WORK" commit -q -m d
printf 'VERDICT: ON TRACK\nhalf-written\n' > "$WORK/docs/D.md"
( cd "$WORK" && seal_output 1 docs/D.md overseer say 7 ) >/dev/null 2>&1
chk "a DIRTY file on rc!=0 is still sealed as a DRAFT" \
  "$(head -3 "$WORK/docs/D.md" | grep -c 'INCOMPLETE RUN')" "1"

printf '\n--- seal_output: the run'"'"'s OTHER dirty files (74th audit B1) ---\n'

# THE 74th-AUDIT SCAR: the 09-05 Review died at max turns with FIVE files
# dirty. One (the report) got the banner; the other four — a live owner
# decision, a shrink-only ratchet move, the week's only queue disposal, the
# builder's priority block — went out six hours later as ordinary work with
# nothing marking their provenance. The seal must commit the dying run's
# whole dirty set, marked, and NAME those paths inside the sealed report.
# Assert on the class (N>1 files, including an untracked one), not the tidy
# example.
W2="$TMP/repo2"; mkdir -p "$W2/docs"
git -C "$W2" init -q
git -C "$W2" config user.email t@t; git -C "$W2" config user.name t
printf 'report v1\n' > "$W2/docs/R.md"
printf 'decision v1\n' > "$W2/docs/DEC.md"
printf 'queue v1\n' > "$W2/docs/Q.md"
printf 'owner draft\n' > "$W2/docs/OWNER.md"
git -C "$W2" add -A; git -C "$W2" commit -q -m init
# The owner's edit PREDATES the run start; the dying run's own edits follow it.
printf 'owner edited this before the run began\n' >> "$W2/docs/OWNER.md"
touch -d '2 hours ago' "$W2/docs/OWNER.md"
RUN_START=$(( $(date +%s) - 60 ))
printf 'report half-written\n' >> "$W2/docs/R.md"
printf 'a new owner decision with a live clock\n' >> "$W2/docs/DEC.md"
printf 'the week'"'"'s only disposal\n' >> "$W2/docs/Q.md"
printf 'brand new file from the dying run\n' > "$W2/docs/NEW.md"
( cd "$W2" && seal_output 1 docs/R.md review say 25 "$RUN_START" ) >/dev/null 2>&1
chk "the report itself is still sealed as a draft" \
  "$(head -3 "$W2/docs/R.md" | grep -c 'INCOMPLETE RUN')" "1"
chk "the run's OTHER dirty files are committed, not abandoned" \
  "$(git -C "$W2" status --porcelain -- docs/DEC.md docs/Q.md docs/NEW.md | wc -l)" "0"
chk "their commit message names the rc, so git log joins them to the death" \
  "$(git -C "$W2" log -1 --format=%s -- docs/DEC.md | grep -c 'rc=1')" "1"
chk "the sealed report NAMES the unbannered files" \
  "$(head -12 "$W2/docs/R.md" | grep -c 'docs/DEC.md')" "1"
chk "a dirty file that PREDATES the run is refused (the git add -A lesson)" \
  "$(git -C "$W2" status --porcelain -- docs/OWNER.md | wc -l)" "1"
chk "and the report says it was LEFT, so a reader still knows it exists" \
  "$(head -12 "$W2/docs/R.md" | grep -c 'docs/OWNER.md')" "1"

printf '\n--- review_liveness: the paused organ ---\n'

# A paused organ is a DECISION, not a fault. Shouting about it would train the
# reader to ignore the banner.
( cd "$WORK" && mkdir -p docs && touch .review-paused \
    && REPO="$WORK" review_liveness say ) >/dev/null 2>&1
chk "a paused Review is not reported dead" "$?" "0"

printf '\n'
[ "$FAIL" -eq 0 ] && { printf 'all green\n'; exit 0; }
printf '%s test(s) FAILED\n' "$FAIL"; exit 1
