# The 90% stop, and the resume it always promised.
#
# Owner's rule (2026-08-09): pause ALL agents at 90% weekly usage until the
# owner resumes them. The gate was built and worked — it has been logging
# "STOPPED at 92% ... paused until the owner resumes" hourly. What was never
# built is the RESUME. The only exit was the weekly reset, so the message was
# writing a cheque the code could not cash, and on 2026-08-11 the owner said
# "make it continue / all the agents" and nothing in the system could act on it.
#
# So: the 90% limit is UNCHANGED and still the default. An owner resume writes
# .usage-resumed with a ceiling and an expiry, and that is the only thing that
# lifts it.
#
# THE EXPIRY IS THE WHOLE DESIGN. An override with no end is not a resume, it is
# a deletion of the limit that nobody remembers making. It expires at the weekly
# reset, so next week starts back at 90% and the owner has to mean it again.
# UNKNOWN IS STILL NOT ZERO: unreadable usage aborts whether or not an override
# exists — an override lifts a KNOWN ceiling, never a blind one.
#
# A meter that blinks is not a meter that is broken. `claude -p /usage` spawns a
# CLI and has failed twice in ~500 reads (2026-08-19T04:07, 2026-08-23T14:07),
# each costing a whole iteration to "ABORT: usage unreadable". The abort is the
# right direction and stays; one retry is what separates a blink from an outage.
_usage_pct() {
  local p
  p=$(/data/venvs/jackthelearner/bin/python "$REPO/scripts/claude_usage.py" --pct 2>/dev/null)
  case "$p" in ''|*[!0-9]*)
    sleep 5
    p=$(/data/venvs/jackthelearner/bin/python "$REPO/scripts/claude_usage.py" --pct 2>/dev/null);;
  esac
  printf '%s' "$p"
}

# PACING — the second half of the limit, added 2026-08-24 after the audit.
#
# THE MEASUREMENT. The 90% stop works and is not in question. What it cannot do
# is decide WHEN the 90% is spent, and the record says: too early, every week.
#
#     week   loop went dark        dark for   Kaggle GPU-h expired unspent
#     W32    Fri 08-14 15:07          ~4.5 d   8.82 of 30
#     W33    Fri 08-21 12:07          ~2.7 d  22.11 of 30
#
# Both blackouts began on a FRIDAY, and Kaggle's free 30 h expire on the SUNDAY
# inside them. 30.9 free GPU-hours have died in two weeks with no agent awake to
# dispatch them — on a project whose owner has ruled free compute only.
#
# THE CAUSE IS NOT OVERSPENDING. `week:all models` is a SHARED pool: the owner's
# interactive sessions draw on the same meter that stops the loop (2026-08-24
# 10:31 — week:all-models 16%, week:Fable 25%, i.e. the builder's own model is
# metered separately and the gate reads the total). So the loop is stopped by
# consumption it does not control, and being the only consumer with a gate, it
# is the one that starves. At that reading it was 5x ahead of an even pace:
# 16% spent into 3% of the week.
#
# THE FIX IS A LINE, NOT A LOWER CEILING. Allowance rises from PACE_FLOOR at the
# reset to the unchanged 90% at the end of the week. Ahead of the line, skip the
# iteration; the next one is an hour away and the budget is still there.
#
#   PACE_FLOOR=25 buys the week's opening burst; by Friday the line is ~62%,
#   Sunday ~81%, and the loop is still awake when the GPU quota expires.
#
# THIS NEVER LIFTS THE 90% STOP. It is strictly tighter, it is checked AFTER
# usage_gate has already said yes, and it cannot return 0 where usage_gate
# returned 1. An owner resume (.usage-resumed) suspends pacing — an explicit
# "make it continue" outranks a smoothing heuristic.
#
# IT FAILS OPEN, AND THAT IS DELIBERATE. If the week's position is unreadable,
# pacing steps aside rather than inventing a second limit nobody set. The real
# limit already refused to run on an unreadable meter one function above.
PACE_FLOOR=${JACK_PACE_FLOOR:-25}
PACE_CAP=90

# pace_gate <log-fn>  -> 0 = proceed, 1 = skip this iteration (not a stop)
pace_gate() {
  local say_fn="${1:-say}" pct elapsed allow
  [ -n "${JACK_NO_PACE:-}" ] && return 0
  [ -f "$REPO/.usage-resumed" ] && return 0
  elapsed=$(/data/venvs/jackthelearner/bin/python "$REPO/scripts/claude_usage.py" --week-elapsed 2>/dev/null)
  case "$elapsed" in ''|*[!0-9]*) return 0;; esac          # unknown -> fail open
  pct=$(_usage_pct)
  case "$pct" in ''|*[!0-9]*) return 0;; esac              # usage_gate already ruled
  # Round the line UP: truncating divides the last point away, so the line
  # would top out at 89 and the final point of an untouched 90% ceiling would
  # be unreachable — a pace line must converge ON the limit, not beside it.
  allow=$(( PACE_FLOOR + ((PACE_CAP - PACE_FLOOR) * elapsed + 99) / 100 ))
  if [ "$pct" -ge "$allow" ]; then
    # Both meters, and the gate named (27th audit, B3): the old line printed one
    # bare percent and read as "the loop is ${pct}% spent" when the builder's own
    # model meter can sit 20+ points hotter, ungated. The extra CLI read costs a
    # few seconds and only on the skip path — the path where no iteration runs.
    local mdl mpct extra
    mdl="${JACK_LOOP_MODEL:-opus}"; mdl="${mdl^}"
    mpct=$(/data/venvs/jackthelearner/bin/python "$REPO/scripts/claude_usage.py" --model "$mdl" --pct 2>/dev/null)
    case "$mpct" in ''|*[!0-9]*) extra="week:${mdl} unreadable";; *) extra="week:${mdl} ${mpct}% (not the gate)";; esac
    "$say_fn" "PACING: acting on 'week:all models' ${pct}% at ${elapsed}% of the week (line ${allow}%); ${extra} — skipping, budget held for later in the week"
    return 1
  fi
  return 0
}

# usage_gate <log-fn>  -> 0 = proceed, 1 = stop
usage_gate() {
  local say_fn="${1:-say}"
  local pct override_ceiling override_until now
  pct=$(_usage_pct)
  case "$pct" in ''|*[!0-9]*)
    "$say_fn" "ABORT: usage unreadable — refusing to run"; return 1;; esac
  if [ "$pct" -lt 90 ]; then return 0; fi

  local f="$REPO/.usage-resumed"
  if [ -f "$f" ]; then
    override_ceiling=$(awk -F= '/^ceiling=/{print $2}' "$f" | tr -dc '0-9')
    override_until=$(awk -F= '/^until=/{print $2}' "$f" | tr -dc '0-9')
    now=$(date +%s)
    if [ -n "$override_until" ] && [ "$now" -ge "$override_until" ]; then
      "$say_fn" "owner resume EXPIRED at $(date -d @"$override_until" -Iseconds) — back to the 90% stop at ${pct}%"
      rm -f "$f"
      return 1
    fi
    if [ -n "$override_ceiling" ] && [ "$pct" -lt "$override_ceiling" ]; then
      "$say_fn" "RESUMED BY OWNER — ${pct}% weekly (ceiling ${override_ceiling}%, expires $(date -d @"$override_until" -Iseconds))"
      return 0
    fi
    "$say_fn" "STOPPED at ${pct}% — past the owner's resume ceiling of ${override_ceiling}%"
    return 1
  fi
  "$say_fn" "STOPPED at ${pct}% weekly usage — all agents paused until the owner resumes"
  return 1
}

# THE MODEL FLOOR — D14 option (b), the LOUD REFUSAL, in the (b-effective)
# reading. Implemented 2026-08-30 by the builder, one day before the armed
# default fired, on OVERSIGHT.md B1 (51st audit).
#
# THE PROBLEM. `usage_gate` and `pace_gate` both read `week:all models`. The
# builder's own model has a SEPARATE weekly line (`week:Fable`), and nothing
# read it. So the loop would burn ~3 s on a model that was 100% exhausted, log
# `LIMITED on fable`, and silently run the whole hour on the Opus fallback.
#
# THE READING TAKEN, and it is a deliberate departure from the default's literal
# words. D14's default says "a pre-flight check ... before run_claude". Read
# literally that aborts the SLOT whenever the PRIMARY is exhausted. The 51st
# audit measured what that would have cost on 2026-08-30 alone: 19 of 19
# iterations aborted, 4 registered verdicts lost (W.1, W.2, PL.00, LG.01), the
# ladder's 84 -> 91 movement lost — because `week:Fable` read 100% for the whole
# day and EVERY iteration that shipped science ran on the fallback. So the check
# is applied to the model that will ACTUALLY RUN, and the slot is refused only
# when every model in the chain is exhausted.
#
# IT IS STILL ONLY A NARROWING, which is the constraint an armed default is
# under. Against today's behaviour it refuses strictly more (a capped model is
# never attempted at all) and permits strictly nothing new — running on Opus
# after `LIMITED on fable` is already permitted and is current behaviour. It
# moves no threshold, cannot return 0 where `usage_gate` returned 1, and is
# reverted by reverting one commit.
#
# WHAT IT GIVES UP versus the literal reading, said plainly: the shared
# `all models` pool still gets spent on Opus when Fable is capped. That is real,
# and it is the cost D14's points 1 and 2 cared about — but the instrument for
# the shared pool is `usage_gate`'s 90% stop and `pace_gate`'s line, both of
# which stay at full strength above this one. A per-model floor was never able
# to govern a pool it cannot see.
#
# THE LIMITATION THAT MATTERS, and it must not be discovered later as a
# surprise: only Fable HAS a per-model weekly line. `claude_usage.py --model
# Opus --pct` and `--model Sonnet --pct` exit 2 with no output (verified
# 2026-08-30 20:2x) because those models roll into `all models`. An unreadable
# line FAILS OPEN here — see below — so with the stock chain `fable opus sonnet`
# the all-exhausted abort is currently UNREACHABLE. The guard has teeth on
# exactly one model. That is honest and it is the whole of its effect today.
#
# FAILING OPEN IS DELIBERATE AND IS NOT THE `usage_gate` PRECEDENT. `usage_gate`
# refuses on an unreadable meter because UNKNOWN IS NOT ZERO for the gating pool
# — nothing else guards it. Here the unreadable case means the model has no
# separate line at all, so its spend is already inside the pool that `usage_gate`
# and `pace_gate` DO refuse on. Refusing here on unknown would invent a second
# stop nobody set and would abort the loop permanently on Opus.
#
# Lowering the floor refuses MORE, never less, so an override cannot widen
# anything; a floor above 100 makes this a no-op, i.e. exactly today's behaviour.
MODEL_FLOOR=${JACK_MODEL_FLOOR:-95}

# model_gate <model> [say-fn]  -> 0 = this model may be attempted, 1 = refuse it
model_gate() {
  local mdl="$1" say_fn="${2:-say}" cap mpct
  cap="${mdl^}"                       # claude_usage.py labels are capitalised
  mpct=$(/data/venvs/jackthelearner/bin/python "$REPO/scripts/claude_usage.py" \
           --model "$cap" --pct 2>/dev/null)
  case "$mpct" in ''|*[!0-9]*) return 0;; esac   # no separate line -> fail open
  [ "$mpct" -lt "$MODEL_FLOOR" ] && return 0
  "$say_fn" "REFUSING ${mdl} — week:${cap} ${mpct}% is at or past the ${MODEL_FLOOR}% model floor (D14 option (b), effective reading); not attempting it"
  return 1
}

# model_chain <say-fn> <primary> [fallback...]
#   -> prints, one per line, the models that may be attempted, in order.
#      EMPTY OUTPUT means every model in the chain is weekly-exhausted and the
#      slot must be refused without consuming it.
# Deduplicates, because FALLBACK_MODELS may name the primary (the old inline
# walk skipped it with `[ "$FB" = "$MODEL" ] && continue`).
model_chain() {
  local say_fn="$1" m seen=""; shift
  for m in "$@"; do
    case " $seen " in *" $m "*) continue;; esac
    seen="$seen $m"
    model_gate "$m" "$say_fn" && printf '%s\n' "$m"
  done
  return 0
}

# chain_reading <reading> <primary> <chain>  -> the chain that reading permits.
#
# D14's LITERAL reading kept live and switchable, because the entry promises
# "if the owner prefers the other reading, one line settles it" and the builder
# took the other reading a day before the default fired. That promise is only
# true if the alternative is a crontab variable rather than a rewrite:
#   JACK_MODEL_READING=literal  -> refuse the SLOT whenever the primary is
#                                  capped, never walk to a fallback.
# Measured cost of `literal` on 2026-08-30 (51st audit): 19 aborts, 4 lost
# verdicts, the ladder's 84 -> 91 movement. It is not the default and it is not
# deleted; a reading nobody can select is a reading the owner cannot choose.
chain_reading() {
  local reading="$1" primary="$2" chain="$3"
  [ -z "$chain" ] && return 0
  case "$reading" in
    literal) [ "${chain%%
*}" = "$primary" ] || return 0;;
  esac
  printf '%s\n' "$chain"
}

# usage_ledger <organ> <phase> — D15's clause (d), default fired 2026-09-06
# (builder; DECIDE block in docs/DECISIONS_NEEDED.md, full record in
# docs/DECISIONS_RESOLVED.md). Every organ appends
#   {"organ","ts","pct","model_pct","phase"}
# to /data/jack-logs/usage_ledger.jsonl at the start and end of its run, so
# the next audit reads spend ATTRIBUTION instead of inferring it from
# co-occurrence — the inference that produced three falsified price models in
# one week (42nd/44th audits; "do not model the meter").
#
# One CLI read per append, both meters parsed from the same invocation.
# NEVER BLOCKS THE RUN: an unreadable meter writes null (valid JSON), a failed
# write is swallowed. Attribution is worth one line, not an abort — the gates
# above already own refusal. Detectors on this file must treat it as a shared
# log (lib_credits.sh precedent): match structure, not prose.
usage_ledger() {
  local organ="$1" phase="$2" mdl out pct mpct
  mdl="${JACK_LOOP_MODEL:-opus}"
  out=$(/data/venvs/jackthelearner/bin/python "$REPO/scripts/claude_usage.py" 2>/dev/null)
  pct=$(printf '%s\n' "$out" | grep -im1 'week:all models' | grep -oE '[0-9]+%' | head -1 | tr -d '%')
  mpct=$(printf '%s\n' "$out" | grep -im1 "week:${mdl}" | grep -oE '[0-9]+%' | head -1 | tr -d '%')
  case "$pct" in ''|*[!0-9]*) pct=null;; esac
  case "$mpct" in ''|*[!0-9]*) mpct=null;; esac
  printf '{"organ":"%s","ts":"%s","pct":%s,"model_pct":%s,"phase":"%s"}\n' \
    "$organ" "$(date -u -Iseconds)" "$pct" "$mpct" "$phase" \
    >> /data/jack-logs/usage_ledger.jsonl 2>/dev/null || true
}
