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
# "You've hit your session limit · resets 1pm (UTC)" is NOT "out of usage
# credits": three iterations died on it in 3 s each on 2026-08-13 (10:07,
# 11:07, 12:07) and nothing detected, retried or counted them — 12.5% of a
# day, invisible (14th audit, B4). Same bounded-read discipline as
# credits_out: only the bytes this run wrote.
session_limited() {
  tail -c "+$(( ${MARK:-0} + 1 ))" "$LOG" 2>/dev/null | grep -qi "hit your session limit"
}
# THE THIRD WORDING, and it is the WEEKLY PER-MODEL one — the exact case the
# fallback chain exists for. "You've reached your Fable 5 limit." matches
# NEITHER detector above: not "out of usage credits", not "hit your session
# limit". On 2026-08-21 at 10:07 and 11:07 the builder hit it, exited rc=1 in
# three seconds, `limit_hit` returned false, the fallback loop `break`ed on its
# first test so opus was never tried, and NO lost-iteration marker was written.
# Two dead slots, uncounted, every organ reporting health. It was recorded in
# prose in ladder_prompt.md and routed to the owner as "an organ script"; it sat
# unfixed for six days while the condition that fires it became certain, so the
# Review fixed it (2026-08-27, PROGRESS.md § THE FINDING). The change is
# monotone: it can only ADD a fallback attempt (~3 s) and ADD a marker — it can
# never suppress a run or make anything look better, which is why this desk
# judged it safe to make rather than route a second time.
#
# Anchored to line start, exactly like api_overloaded and for exactly the same
# reason: the organs' own reports quote this string in prose (ladder_prompt.md
# does, in backticks), and an unanchored match would fire on the post-mortem
# instead of the event. Verified against both CLI wordings and both prose forms.
model_limited() {
  tail -c "+$(( ${MARK:-0} + 1 ))" "$LOG" 2>/dev/null |
    grep -qiE "^[[:space:]]*(you've|you have) reached your .{0,40}limit"
}
limit_hit() { credits_out || session_limited || model_limited; }
# "API Error: 529 Overloaded" is server-side and transient — nothing about our
# credits. The 2026-08-24 daily Review died on exactly this line (rc=1, 8 min)
# and never ran; one retry would have cost nothing and saved the day's audit
# (27th audit, B2). Anchored to line start because the organs' own OUTPUT
# discusses these incidents in prose ("died on an API 529") and an unanchored
# match on this shared log would fire on the post-mortem of a previous failure.
# Same bounded-read discipline: only the bytes this run wrote.
api_overloaded() {
  tail -c "+$(( ${MARK:-0} + 1 ))" "$LOG" 2>/dev/null | grep -qE "^API Error: 5[0-9][0-9]"
}
