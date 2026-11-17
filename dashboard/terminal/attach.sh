#!/bin/bash
# What every way in runs — the web terminal, and `jt` from an SSH app: attach
# to the one shared session, creating it on the first visit after a reboot.
# One session means one Claude process, seen from the phone and the laptop
# alike, that keeps running when either of them disconnects.
cd /home/opc 2>/dev/null
TMUX="/usr/bin/tmux -u -L jack -f /home/opc/jackthelearner/dashboard/terminal/tmux.conf"

# Make sure the shared session exists before we add anything to it.
$TMUX has-session -t jack 2>/dev/null || $TMUX new-session -d -s jack -c /home/opc

# Ensure the always-on device tabs are present (idempotent — never duplicated).
# A "device" tab holds a reconnecting SSH session to another machine of the
# owner's; it survives sleeps, network changes and v2ray switching. See device.sh.
ensure_device() { # <window-name> <ssh-alias> <label>
  $TMUX list-windows -t jack -F '#{window_name}' 2>/dev/null | grep -qx "$1" && return 0
  $TMUX new-window -d -t jack: -n "$1" "exec /home/opc/jackthelearner/dashboard/terminal/device.sh $2 $3"
  $TMUX set-window-option -t "jack:$1" automatic-rename off
  $TMUX set-window-option -t "jack:$1" allow-rename off
}
ensure_device laptop laptop Laptop

# Attach (or create-and-attach) for this visitor.
exec $TMUX new-session -A -s jack -c /home/opc
