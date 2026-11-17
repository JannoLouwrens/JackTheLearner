#!/usr/bin/env bash
# One "device" tab in the terminal: hold an SSH session to a host open, and
# reconnect by itself whenever it drops — laptop sleep, a network change, or
# v2ray/sing-box switching on or off (the tunnel re-establishes within a minute
# either way). The tab stays put and comes back on its own; close it with the
# × on the tab.
#
# Usage: device.sh <ssh-host-alias> [label]
#   the alias is defined in ~/.ssh/config (e.g. "laptop" -> DELL@localhost:2222).
set -u
host="${1:?usage: device.sh <ssh-host-alias> [label]}"
label="${2:-$host}"

while true; do
  clear
  printf '\033[2m→ connecting to %s …\033[0m\n' "$label"
  # -tt forces a remote pseudo-terminal, so the Windows shell paints its prompt
  ssh -tt -o ConnectTimeout=10 -o ServerAliveInterval=15 -o ServerAliveCountMax=3 "$host"
  rc=$?
  printf '\n\033[2m— %s closed (exit %s). Reconnecting in 5s… press Enter to retry now, or use the × on the tab to leave. —\033[0m\n' "$label" "$rc"
  # a 5s pause the viewer can skip with Enter; stdin is the tab's terminal
  read -r -t 5 _ || true
done
