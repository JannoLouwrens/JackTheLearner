# Shared credit-exhaustion detection for the cron organs.
#
# Scar: the organs append to one long-lived log, and each checked
# `tail -5 "$LOG" | grep -qi "out of usage credits"` AFTER its claude run. On
# 2026-08-09 the builder's fable run died on credits at 22:07, opus then ran for
# 28 minutes and exited with `Error: Reached max turns (120)` — appending ONE
# line, so the fable credit message from 28 minutes earlier was still inside
# tail -5. The loop announced "OUT OF CREDITS on opus", burned its last fallback,
# and was one step from credit-pausing every agent for 4h over a run that had
# plenty of credit.
#
# A detector that reads a shared append-only log must bound itself to the bytes
# ITS OWN run wrote. Same failure as reading a directory's mtime to decide
# whether a session is live: the signal was real, it just belonged to something
# else. Call mark_log immediately before each claude invocation.
mark_log() { MARK=$(wc -c < "$LOG" 2>/dev/null || echo 0); }
credits_out() {
  tail -c "+$(( ${MARK:-0} + 1 ))" "$LOG" 2>/dev/null | grep -qi "out of usage credits"
}
