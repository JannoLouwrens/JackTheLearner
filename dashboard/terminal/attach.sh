#!/bin/bash
# What every way in runs — the web terminal, and `jt` from an SSH app: attach
# to the one shared session, creating it on the first visit after a reboot.
# One session means one Claude process, seen from the phone and the laptop
# alike, that keeps running when either of them disconnects.
cd /home/opc 2>/dev/null
exec /usr/bin/tmux -u -L jack -f /home/opc/jackthelearner/dashboard/terminal/tmux.conf new-session -A -s jack -c /home/opc
