#!/usr/bin/env bash
# Prune old Claude CLI binaries from the root disk (ORDERS S11 c). Run by hand.
# Keeps: the launcher's target, the newest version, and any version a running
# process has as exe, mapped, or open. Deletes the rest. --dry-run prints only.
# Test: scripts/test_prune_cli_versions.sh
set -u
VDIR=${CLI_VERSIONS_DIR:-$HOME/.local/share/claude/versions}
LAUNCHER=${CLI_LAUNCHER:-$HOME/.local/bin/claude}
DRY=0; [ "${1:-}" = "--dry-run" ] && DRY=1
[ -d "$VDIR" ] || { echo "no $VDIR"; exit 1; }
keep=$(readlink -f "$LAUNCHER" 2>/dev/null)
[ -n "$keep" ] && [ -e "$keep" ] || { echo "REFUSED: launcher $LAUNCHER does not resolve"; exit 2; }
newest=$(ls -t "$VDIR" | head -1)
inuse=$(for p in /proc/[0-9]*; do
          readlink "$p/exe" 2>/dev/null; ls -l "$p/fd" 2>/dev/null | sed 's/.*-> //'
          awk '{print $6}' "$p/maps" 2>/dev/null
        done | sed 's/ (deleted)$//' | grep -F "$VDIR/" | sort -u)
freed=0
for f in "$VDIR"/*; do
  [ -f "$f" ] || continue
  why=""
  [ "$(readlink -f "$f")" = "$keep" ] && why="launcher"
  [ "$(basename "$f")" = "$newest" ] && why="${why:-newest}"
  printf '%s\n' "$inuse" | grep -qxF "$f" && why="${why:-in use}"
  if [ -n "$why" ]; then echo "KEEP   $(basename "$f") ($why)"; continue; fi
  sz=$(stat -c %s "$f"); freed=$((freed + sz))
  echo "DELETE $(basename "$f") ($((sz / 1048576)) MB)"
  [ $DRY = 1 ] || rm -f -- "$f"
done
echo "freed $((freed / 1048576)) MB$([ $DRY = 1 ] && echo ' (dry run)')"
