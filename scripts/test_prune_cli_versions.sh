#!/usr/bin/env bash
# Planted test for prune_cli_versions.sh: a fake versions dir with five
# versions; v1 held open by a live process, v3 the launcher target, v5 newest.
# Must delete exactly v2 and v4; a dangling launcher must refuse everything.
set -u
T=$(mktemp -d /data/tmp/prunetest.XXXX); trap 'kill $H 2>/dev/null; rm -rf "$T"' EXIT
mkdir -p "$T/v" "$T/bin"
for i in 1 2 3 4 5; do head -c 1024 /dev/zero > "$T/v/$i"; touch -d "2026-01-0$i" "$T/v/$i"; done
ln -s "$T/v/3" "$T/bin/claude"
sleep 60 3<"$T/v/1" & H=$!
S=$(dirname "$0")/prune_cli_versions.sh
fail=0
CLI_VERSIONS_DIR=$T/v CLI_LAUNCHER=$T/bin/claude bash "$S" --dry-run >/dev/null
[ "$(ls "$T/v" | tr -d '\n')" = 12345 ] || { echo "FAIL dry run deleted"; fail=1; }
CLI_VERSIONS_DIR=$T/v CLI_LAUNCHER=$T/bin/claude bash "$S"
[ "$(ls "$T/v" | tr -d '\n')" = 135 ] || { echo "FAIL left: $(ls "$T/v" | tr '\n' ' ')"; fail=1; }
CLI_VERSIONS_DIR=$T/v CLI_LAUNCHER=$T/bin/nope bash "$S"; rc=$?
[ $rc = 2 ] && [ "$(ls "$T/v" | tr -d '\n')" = 135 ] || { echo "FAIL dangling launcher rc=$rc"; fail=1; }
[ $fail = 0 ] && echo PASS || echo FAIL; exit $fail
