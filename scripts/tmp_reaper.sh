#!/bin/bash
# Keep root from filling with dead scratch.
# Scar 2026-08-09: / reached 3.6 GB free against the loop's 3 GB floor — the
# builder was ~0.6 GB from silently refusing to run overnight. Cause: 9.6 GB of
# finished research-agent scratch (downloaded ASR models, test checkpoints) on
# the SMALL disk. Same shape as the /data crisis that morning: space consumed by
# work already done, with nothing watching.
# Organs now write scratch to /data/tmp (TMPDIR); this reaps whatever still
# lands on root, plus /data/tmp itself so the big disk cannot creep either.
set -uo pipefail
LOG=/data/jack-logs/tmp_reaper.log
before_root=$(df -BG --output=avail / | tail -1 | tr -dc '0-9')

# Dead Claude session scratch older than 2 days. Live sessions keep their dirs
# mtime-fresh, so an untouched dir is a finished one.
find /tmp/claude-1000 -maxdepth 2 -type d -mtime +2 -exec rm -rf {} + 2>/dev/null
find /data/tmp        -maxdepth 2 -type d -mtime +5 -exec rm -rf {} + 2>/dev/null
# Loose scratch this project leaves behind.
find /tmp -maxdepth 1 -type f -mtime +3 \( -name "*.log" -o -name "*.json" -o -name "*.py" \) -delete 2>/dev/null
find /data -maxdepth 1 -type d -name "tmp??????*" -mtime +2 -exec rm -rf {} + 2>/dev/null  # gpu.py job dirs

after_root=$(df -BG --output=avail / | tail -1 | tr -dc '0-9')
echo "$(date -Iseconds) reaped: / ${before_root}G -> ${after_root}G free, /data $(df -BG --output=avail /data | tail -1 | tr -d ' G')G free" >> "$LOG"
