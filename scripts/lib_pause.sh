# A pause that actually pauses everything.
#
# SCAR, 2026-08-11. Each organ had its OWN pause file — .loop-paused,
# .overseer-paused, .review-paused, .fieldwatch-paused. The owner said "pause
# for now all agents", .loop-paused was written, the builder stopped, the pause
# was reported as done — and the other three organs kept running. The overseer
# was still going a minute later.
#
# A control that LOOKS global and is local is worse than no control at all: it
# produces confident false belief about what is running, which is the same
# failure as the reaper reading a directory mtime and the credit detector
# reading a shared log's tail. Nobody re-checks a switch they believe they
# already threw.
#
# .paused stops EVERY organ. The per-organ files still work, for stopping one.
# Neither self-expires: a human's stop means stop (the builder's credit-pause
# self-expiry is a separate mechanism and stays inside ladder_loop.sh).
pause_gate() {
  local say_fn="${1:-say}" own="${2:-}"
  if [ -f "$REPO/.paused" ]; then
    "$say_fn" "PAUSED — all agents stopped by $REPO/.paused: $(head -1 "$REPO/.paused")"
    return 1
  fi
  if [ -n "$own" ] && [ -f "$own" ] && ! grep -q "^credits" "$own" 2>/dev/null; then
    "$say_fn" "paused ($own)"
    return 1
  fi
  return 0
}
