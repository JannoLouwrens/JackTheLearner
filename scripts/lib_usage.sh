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
# usage_gate <log-fn>  -> 0 = proceed, 1 = stop
usage_gate() {
  local say_fn="${1:-say}"
  local pct override_ceiling override_until now
  pct=$(/data/venvs/jackthelearner/bin/python "$REPO/scripts/claude_usage.py" --pct 2>/dev/null)
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
