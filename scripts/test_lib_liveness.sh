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

printf '\n--- seal_output: swept ledger rows carry provenance (79th audit 1.1) ---\n'

# THE 79th-AUDIT SCAR: the Sunday FULL hit its 40-minute wall and the sweep
# committed `experiments/ledger.json` carrying ME.1's FAIL row. The row was a
# true measurement, but only `review.log` said a dying organ committed it —
# the ledger itself carried nothing, and every reader reads the FILE. The
# sweep must stamp exactly the rows that differ from HEAD with an additive
# `seal_provenance` key, and leave untouched rows byte-identical.
W3="$TMP/repo3"; mkdir -p "$W3/docs" "$W3/experiments"
git -C "$W3" init -q
git -C "$W3" config user.email t@t; git -C "$W3" config user.name t
printf 'report v1\n' > "$W3/docs/R.md"
printf '%s\n' '{"_comment":"c","results":{"A.1":{"attempt":1,"status":"PASS","commit":"aaa"},"B.2":{"attempt":2,"status":"FAIL","commit":"bbb"}}}' \
  > "$W3/experiments/ledger.json"
git -C "$W3" add -A; git -C "$W3" commit -q -m init
RUN_START=$(( $(date +%s) - 60 ))
printf 'report half-written\n' >> "$W3/docs/R.md"
/data/venvs/jackthelearner/bin/python - "$W3/experiments/ledger.json" <<'PYEOF'
import json, sys
p = sys.argv[1]
d = json.load(open(p))
d["results"]["B.2"]["status"] = "PASS"; d["results"]["B.2"]["attempt"] = 3
d["results"]["C.3"] = {"attempt": 1, "status": "VOID", "commit": "ccc"}
json.dump(d, open(p, "w"), indent=2, sort_keys=True)
PYEOF
( cd "$W3" && seal_output 124 docs/R.md review say 25 "$RUN_START" ) >/dev/null 2>&1
chk "the swept ledger is committed" \
  "$(git -C "$W3" status --porcelain -- experiments/ledger.json | wc -l)" "0"
chk "the MODIFIED row is stamped with seal_provenance" \
  "$(/data/venvs/jackthelearner/bin/python -c "import json;d=json.load(open('$W3/experiments/ledger.json'));print('seal_provenance' in d['results']['B.2'])")" "True"
chk "the ADDED row is stamped too" \
  "$(/data/venvs/jackthelearner/bin/python -c "import json;d=json.load(open('$W3/experiments/ledger.json'));print('seal_provenance' in d['results']['C.3'])")" "True"
chk "the UNTOUCHED row is left clean" \
  "$(/data/venvs/jackthelearner/bin/python -c "import json;d=json.load(open('$W3/experiments/ledger.json'));print('seal_provenance' in d['results']['A.1'])")" "False"
chk "the stamp names the organ, the rc and the seal" \
  "$(/data/venvs/jackthelearner/bin/python -c "import json;d=json.load(open('$W3/experiments/ledger.json'));n=d['results']['B.2']['seal_provenance'];print(all(s in n for s in ('review','rc=124','lib_seal.sh')))")" "True"
chk "nothing but the key changed: status and attempt are the runner's" \
  "$(/data/venvs/jackthelearner/bin/python -c "import json;d=json.load(open('$W3/experiments/ledger.json'));r=d['results']['B.2'];print(r['status'],r['attempt'])")" "PASS 3"

printf '\n--- seal_output: the tail receipt (D25, armed default 2026-09-14) ---\n'

# THE 2026-09-06 SCAR, re-enacted. The Sunday FULL wrote its whole page,
# committed its dispositions as it made them, appended its own PROGRESS_LOG row
# at 07:12 — the last item on its checklist — and `timeout` killed it at 07:17.
# The seal, reading only rc!=0, gave that finished page the banner it correctly
# gives a run that wrote nothing. The discriminator is the organ's own tail
# receipt, and all four cases below are asserted because three of them are the
# ways this branch could go wrong: firing when there is no receipt, refusing to
# fire when there is one, and — the one that would quietly undo it — accepting
# the organ's OWN dead-run fallback row as if the agent had written it.
mkrepo() {  # <dir> -> a repo with a committed report and history file
  local d="$1"
  mkdir -p "$d/docs"; git -C "$d" init -q
  git -C "$d" config user.email t@t; git -C "$d" config user.name t
  printf 'report v1\n' > "$d/docs/R.md"
  { echo "| date | mode | one line |"; echo "|---|---|---|"; } > "$d/docs/HIST.md"
  git -C "$d" add -A; git -C "$d" commit -q -m init
  printf 'report v1\nthe whole page, written\n' >> "$d/docs/R.md"
}
TODAY=$(date -u +%F)
PAT="^\| $TODAY "

# (a) The run finished its checklist: its own row is in the history file.
W4="$TMP/repo4"; mkrepo "$W4"
printf '| %s | FULL | the desk read the board and ruled |\n' "$TODAY" >> "$W4/docs/HIST.md"
( cd "$W4" && seal_output 124 docs/R.md review say 25 "$(( $(date +%s) - 60 ))" \
    docs/HIST.md "$PAT" ) >/dev/null 2>&1
chk "a run with its tail receipt is sealed COMPLETE, not a draft" \
  "$(head -3 "$W4/docs/R.md" | grep -c 'CHECKLIST COMPLETE')" "1"
chk "and the DRAFT wording is nowhere on the page" \
  "$(grep -c 'THIS IS A DRAFT, NOT A FINDING' "$W4/docs/R.md")" "0"
chk "and the banner quotes the receipt row as its evidence" \
  "$(head -8 "$W4/docs/R.md" | grep -c 'the desk read the board and ruled')" "1"
chk "and it still says the page is UNAUDITED" \
  "$(head -12 "$W4/docs/R.md" | grep -c 'UNAUDITED')" "1"
chk "and git log says COMPLETE, so a log reader need not open the file" \
  "$(git -C "$W4" log -1 --format=%s -- docs/R.md | grep -c 'sealed as COMPLETE')" "1"
chk "and the page is committed, not left dirty" \
  "$(git -C "$W4" status --porcelain -- docs/R.md | wc -l)" "0"
# A second dying run must not stack a second banner on top of the first.
printf 'a later run appended this\n' >> "$W4/docs/R.md"
( cd "$W4" && seal_output 124 docs/R.md review say 25 "$(( $(date +%s) - 60 ))" \
    docs/HIST.md "$PAT" ) >/dev/null 2>&1
chk "a second call does not stack a second COMPLETE banner" \
  "$(grep -c 'CHECKLIST COMPLETE' "$W4/docs/R.md")" "1"

# (b) NO receipt — today's wording, byte-for-byte. This is what D25's default
# requires of a run that committed neither, and it is the 09-05 death.
W5="$TMP/repo5"; mkrepo "$W5"
( cd "$W5" && seal_output 1 docs/R.md review say 25 "$(( $(date +%s) - 60 ))" \
    docs/HIST.md "$PAT" ) >/dev/null 2>&1
chk "no receipt: the INCOMPLETE RUN banner is unchanged" \
  "$(head -3 "$W5/docs/R.md" | grep -c 'INCOMPLETE RUN — THIS IS A DRAFT, NOT A FINDING')" "1"
chk "no receipt: nothing claims the checklist completed" \
  "$(grep -c 'CHECKLIST COMPLETE' "$W5/docs/R.md")" "0"

# (c) THE ONE THAT WOULD QUIETLY UNDO IT: review.sh writes an INCOMPLETE row
# itself when the agent died before its own append (76th audit B4). That row
# matches the date pattern exactly. A receipt an organ's own dead-run fallback
# can satisfy is not a receipt — it is the organ certifying itself.
W6="$TMP/repo6"; mkrepo "$W6"
printf '| %s | FULL | — | INCOMPLETE — the FULL run exited rc=124 before appending its own row |\n' \
  "$TODAY" >> "$W6/docs/HIST.md"
( cd "$W6" && seal_output 124 docs/R.md review say 25 "$(( $(date +%s) - 60 ))" \
    docs/HIST.md "$PAT" ) >/dev/null 2>&1
chk "an INCOMPLETE fallback row is NOT a completion receipt" \
  "$(head -3 "$W6/docs/R.md" | grep -c 'INCOMPLETE RUN — THIS IS A DRAFT, NOT A FINDING')" "1"

# (d) An organ that passes no receipt at all (the overseer, the field watch)
# must behave exactly as it did before this branch existed.
W7="$TMP/repo7"; mkrepo "$W7"
( cd "$W7" && seal_output 1 docs/R.md overseer say 7 "$(( $(date +%s) - 60 ))" ) >/dev/null 2>&1
chk "an organ with no receipt configured is untouched by this branch" \
  "$(head -3 "$W7/docs/R.md" | grep -c 'INCOMPLETE RUN — THIS IS A DRAFT, NOT A FINDING')" "1"

# (e) The swept artefacts must not contradict the page. A dying run's routed
# decision used to be stamped "its report is sealed as an INCOMPLETE RUN draft"
# unconditionally — the same falsehood, one file over.
W8="$TMP/repo8"
mkdir -p "$W8/docs"; git -C "$W8" init -q
git -C "$W8" config user.email t@t; git -C "$W8" config user.name t
printf 'report v1\n' > "$W8/docs/R.md"
{ echo "| date | mode | one line |"; echo "|---|---|---|"; } > "$W8/docs/HIST.md"
printf '# decisions\n\n## D98 — an older entry\n\n- class: goal\n' > "$W8/docs/DECISIONS_NEEDED.md"
git -C "$W8" add -A; git -C "$W8" commit -q -m init
printf 'report v1\nthe whole page, written\n' >> "$W8/docs/R.md"
printf '| %s | FULL | ruled |\n' "$TODAY" >> "$W8/docs/HIST.md"
printf '\n## D99 — a live decision this run routed\n\n- class: goal\n' >> "$W8/docs/DECISIONS_NEEDED.md"
( cd "$W8" && seal_output 124 docs/R.md review say 25 "$(( $(date +%s) - 60 ))" \
    docs/HIST.md "$PAT" ) >/dev/null 2>&1
chk "a swept decision entry is stamped, as always" \
  "$(grep -c 'PROVENANCE' "$W8/docs/DECISIONS_NEEDED.md")" "1"
chk "and its provenance does NOT call the complete page a draft" \
  "$(grep -c 'INCOMPLETE RUN draft' "$W8/docs/DECISIONS_NEEDED.md")" "0"
chk "and it still says the entry is unverified" \
  "$(grep -c 'unverified until an audit re-measures them' "$W8/docs/DECISIONS_NEEDED.md")" "1"

printf '\n--- review_liveness: the page asserts its own age (119th audit F3) ---\n'

# THE 119th-AUDIT SCAR: the Review died after writing its own B4 INCOMPLETE
# disclosure row into PROGRESS_LOG.md. That row is dated TODAY, so
# table_liveness read the organ as alive while docs/PROGRESS.md sat 42 h old,
# unbannered, headlined as a finished report. The honest disclosure in one
# file suppressed the alarm that owns the other. The check must read the page
# it stamps, not only the table the dying run writes.
W9="$TMP/repo9"; mkdir -p "$W9/docs"
git -C "$W9" init -q
git -C "$W9" config user.email t@t; git -C "$W9" config user.name t
printf '# PROGRESS\n\n**2026-09-24 DAILY.** THE BUILDER HAS NOTHING TO DO.\n' > "$W9/docs/PROGRESS.md"
{ echo "| date | mode | one line |"; echo "|---|---|---|"
  printf '| %s | FULL | ruled |\n' "$(date -u -d '3 days ago' +%F)"
  printf '| %s | DAILY | — INCOMPLETE — the run exited rc=124 before appending its own row |\n' "$(date -u +%F)"
} > "$W9/docs/PROGRESS_LOG.md"
git -C "$W9" add -A
GIT_COMMITTER_DATE="$(date -u -d '2 days ago' -Iseconds)" \
  git -C "$W9" commit -q -m old
( cd "$W9" && REPO="$W9" review_liveness say ) >/dev/null 2>&1
chk "a fresh B4 disclosure row does NOT vouch for a 48h-old page" "$?" "1"
chk "and the page got its banner" \
  "$(head -1 "$W9/docs/PROGRESS.md" | grep -c 'STALE — ')" "1"
chk "and the banner is committed" \
  "$(git -C "$W9" status --porcelain | wc -l)" "0"

# The stamp commit itself refreshes the page's git age. A banner nobody has
# cleared is still a missed schedule — the check must not read its own stamp
# as the Review coming back.
( cd "$W9" && REPO="$W9" review_liveness say ) >/dev/null 2>&1
chk "after the stamp, the check still reports the miss (not OK)" "$?" "1"
chk "and no second banner is stacked" \
  "$(grep -c 'STALE — ' "$W9/docs/PROGRESS.md")" "1"

# A page rewritten by a COMPLETED run (fresh commit, no banner) is healthy.
W10="$TMP/repo10"; mkdir -p "$W10/docs"
git -C "$W10" init -q
git -C "$W10" config user.email t@t; git -C "$W10" config user.name t
printf '# PROGRESS\n\ncurrent state, written today\n' > "$W10/docs/PROGRESS.md"
{ echo "| date | mode | one line |"; echo "|---|---|---|"
  printf '| %s | FULL | ruled |\n' "$(date -u -d '3 days ago' +%F)"
  printf '| %s | DAILY | wrote the page |\n' "$(date -u +%F)"
} > "$W10/docs/PROGRESS_LOG.md"
git -C "$W10" add -A; git -C "$W10" commit -q -m fresh
( cd "$W10" && REPO="$W10" review_liveness say ) >/dev/null 2>&1
chk "a fresh row AND a fresh page is healthy" "$?" "0"
chk "and the healthy page is not stamped" \
  "$(head -1 "$W10/docs/PROGRESS.md" | grep -c 'STALE — ')" "0"

printf '\n--- review_liveness: the paused organ ---\n'

# A paused organ is a DECISION, not a fault. Shouting about it would train the
# reader to ignore the banner.
( cd "$WORK" && mkdir -p docs && touch .review-paused \
    && REPO="$WORK" review_liveness say ) >/dev/null 2>&1
chk "a paused Review is not reported dead" "$?" "0"

printf '\n'
[ "$FAIL" -eq 0 ] && { printf 'all green\n'; exit 0; }
printf '%s test(s) FAILED\n' "$FAIL"; exit 1
