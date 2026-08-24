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
    "$say_fn" "PACING: ${pct}% spent at ${elapsed}% of the week (line ${allow}%) — skipping, budget held for later in the week"
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
