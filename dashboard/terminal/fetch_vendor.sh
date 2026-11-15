#!/bin/bash
# fetch_vendor.sh — the terminal page's one library (xterm.js, MIT), fetched
# once from the npm registry and checked against the hashes pinned below.
# The page loads it from this server only: a page that can open a shell here
# must not run a line of script from anyone else's machine.
set -euo pipefail
OUT="${1:-/data/jack-dashboard/vendor}"
mkdir -p "$OUT"; T=$(mktemp -d); trap 'rm -rf "$T"' EXIT
get() {  # url, pinned sha512 (base64), file inside the tarball, name to save as
  curl -fsSL --max-time 120 -o "$T/p.tgz" "$1"
  got="sha512-$(openssl dgst -sha512 -binary "$T/p.tgz" | openssl base64 -A)"
  [ "$got" = "$2" ] || { echo "INTEGRITY MISMATCH for $1" >&2; exit 1; }
  tar -xzf "$T/p.tgz" -C "$T" "package/$3"
  install -m 644 "$T/package/$3" "$OUT/$4"
}
X="sha512-hqJHYaQb5OptNunnyAnkHyM8aCjZ1MEIDTQu1iIbbTD/xops91NB5yq1ZK/dC2JDbVWtF23zUtl9JE2NqwT87A=="
F="sha512-UFYkDm4HUahf2lnEyHvio51TNGiLK66mqP2JoATy7hRZeXaGMRDr00JiSF7m63vR5WKATF605yEggJKsw0JpMQ=="
get https://registry.npmjs.org/@xterm/xterm/-/xterm-5.5.0.tgz "$X" lib/xterm.js xterm.js
get https://registry.npmjs.org/@xterm/xterm/-/xterm-5.5.0.tgz "$X" css/xterm.css xterm.css
get https://registry.npmjs.org/@xterm/addon-fit/-/addon-fit-0.10.0.tgz "$F" lib/addon-fit.js addon-fit.js
echo "vendor files in $OUT:"; ls -la "$OUT" | awk 'NR>1 && $9 !~ /^\.+$/ {print "  " $5, $9}'
