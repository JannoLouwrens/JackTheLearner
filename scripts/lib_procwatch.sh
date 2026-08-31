#!/bin/bash
# lib_procwatch.sh — did this iteration leave compute running on a shared box?
#
# THE SCAR, 2026-08-30 (52nd audit, FOR THE BUILDER B2). PID 3749514 —
# `python -c "x=0 / while 1: x+=1"`, a verification aid for the `_cpu_fraction`
# fix in `1296ca3` — was orphaned to `ppid 1` with `cwd=/home/opc/jackthelearner`
# and burned **1.26 core-hours** of a 4-core box shared with paying tenants
# before an auditor found it by hand. `SYSTEM.md` says "leave no process
# running" and "stay under ~1.5 GB RAM"; until this file, that rule was enforced
# by NOTHING. `tmp_reaper.sh` reaps scratch *directories* and explicitly avoids
# processes; `ladder_loop.sh` had no process check on any exit path.
#
# WHY PROSE COULD NEVER BE THE GUARD. Most iteration reports carry a sentence
# like *"no leftover compute — the only `pgrep` match is the grep's own shell"*.
# It is voluntary, and both iterations straddling the scar simply omitted it.
# Worse, the 00:07 iteration printed a **full `ps` dump** to prove its own
# detached run was alive and did not notice a second project python at 99.7% CPU
# in the same output: **a liveness check that scans for one known pid cannot see
# an unknown one.** It looks for presence; this file looks for EXCESS. That is
# the generalisation, and it is the same shape as the review-liveness scar one
# level up — the organ that would notice is not the organ that failed.
#
# THE DETECTOR MUST BOUND ITSELF TO ITS OWN KIND, or it becomes the bug it
# reports. `pgrep -u opc -f '/data/venvs/jackthelearner'` — the auditor's own
# instrument — matches the **builder's `claude` process**, because the venv path
# is written in `ladder_prompt.md` and the whole prompt sits in that process's
# argv. That is `lib_credits.sh`'s `tail -5` lesson in a third costume: a
# substring detector over a shared surface. So the predicate here is
# START-ANCHORED on argv[0] (a path *quoted inside* a command line can never
# trip it), with one deliberate second arm for a python whose argv[0] is bare
# (`python3 -c ...`) but whose cwd is the repo — which is exactly the scar.
#
# WHAT IT DELIBERATELY DOES NOT DO:
#   - it never kills. A detached registered run is legitimate and must survive
#     the iteration that launched it (that is the whole point of dispatch.sh).
#     An unattributable pid is NAMED, with its cpu time and command line, and
#     the next reader decides. The audit asked for exactly this.
#   - it only sees python-shaped compute. A stray `ffmpeg` is out of scope;
#     every leak this project has actually had was a project python.
#   - it cannot tell the owner's interactive session from a leak, when that
#     session runs python from the repo. It says so in the line it prints.
#
# ATTRIBUTION, and why a bare pid is not enough. `launch_detached.sh` and
# `dispatch.sh` legitimately leave a process running, so they DECLARE it here.
# A declaration is `pid:starttime`, never a bare pid: pids are recycled, and a
# stale declaration that silently adopts an unrelated process would turn this
# guard into a laundering service. Children of a declared pid are attributed
# through the ppid chain, because `run.py` forks the actual work.
#
# Run:  bash scripts/test_lib_procwatch.sh     (exit 0 = all green)

JACK_REPO="${JACK_REPO:-/home/opc/jackthelearner}"
JACK_VENV="${JACK_VENV:-/data/venvs/jackthelearner/}"
JACK_PROC_DECL="${JACK_PROC_DECL:-/data/jack-logs/declared_pids}"

# ------------------------------------------------------------------ /proc ---
# Field 22 of /proc/PID/stat is start time in clock ticks since boot: unique
# per pid *incarnation*, which is what makes a declaration forgery-proof under
# pid reuse. Everything before ") " is skipped because field 2 (comm) may
# contain spaces and parens; after the split, ppid is field 2 and starttime 20.
_proc_stat_field() {
  local pid="$1" n="$2" s rest
  s=$(cat "/proc/$pid/stat" 2>/dev/null) || return 1
  rest=${s##*") "}
  [ "$rest" = "$s" ] && return 1
  # shellcheck disable=SC2086
  set -- $rest
  [ $# -ge "$n" ] || return 1
  eval "printf '%s' \"\${$n}\""
}

proc_starttime() { _proc_stat_field "$1" 20; }
_proc_ppid()     { _proc_stat_field "$1" 2; }

# CPU seconds actually consumed (utime+stime, fields 14+15 -> 12+13 here).
proc_cpu_seconds() {
  local pid="$1" u s hz
  u=$(_proc_stat_field "$pid" 12) || return 1
  s=$(_proc_stat_field "$pid" 13) || return 1
  hz=$(getconf CLK_TCK 2>/dev/null || echo 100)
  echo $(( (u + s) / hz ))
}

# proc_key PID -> "pid:starttime", empty when the pid is gone.
proc_key() {
  local st
  st=$(proc_starttime "$1") || return 1
  [ -n "$st" ] || return 1
  printf '%s:%s' "$1" "$st"
}

proc_cmdline() {
  tr '\0' ' ' < "/proc/$1/cmdline" 2>/dev/null | cut -c1-160
}

# ------------------------------------------------------------- predicate ---
# Is this process THIS PROJECT'S compute? Start-anchored, never a substring of
# a whole command line — see the header.
_proc_is_ours() {
  local pid="$1" argv0 exe cwd
  [ -O "/proc/$pid" ] || return 1
  argv0=$(tr '\0' '\n' < "/proc/$pid/cmdline" 2>/dev/null | head -1)
  [ -n "$argv0" ] || return 1          # kernel thread: no argv at all
  case "$argv0" in "$JACK_VENV"*) return 0 ;; esac
  # Second arm: the scar itself. A bare `python -c ...` started from the repo
  # resolves its exe to the system python (the venv's bin/python is a symlink
  # to it, so exe alone can never tell the two apart) — the repo cwd is what
  # marks it as this project's.
  exe=$(readlink "/proc/$pid/exe" 2>/dev/null)
  case "$exe" in *python*) ;; *) return 1 ;; esac
  cwd=$(readlink "/proc/$pid/cwd" 2>/dev/null)
  case "$cwd" in "$JACK_REPO"|"$JACK_REPO"/*) return 0 ;; esac
  return 1
}

# ----------------------------------------------------------- declaration ---
# proc_declare PID [LABEL] — record a process this system MEANT to leave running.
proc_declare() {
  local pid="$1" label="${2:-undeclared-purpose}" key
  key=$(proc_key "$pid") || { echo "proc_declare: pid $pid is already gone" >&2; return 1; }
  mkdir -p "$(dirname "$JACK_PROC_DECL")" 2>/dev/null
  printf '%s\t%s\t%s\n' "$key" "$(date -Iseconds)" "$label" >> "$JACK_PROC_DECL"
}

_proc_declared() {
  local key="$1"
  [ -f "$JACK_PROC_DECL" ] || return 1
  cut -f1 "$JACK_PROC_DECL" 2>/dev/null | grep -qxF "$key"
}

# A child of a declared process is declared: `run.py` forks the real work, and
# `dispatch.sh` declares the watcher it can see. Bounded walk — /proc can lie
# about ancestry mid-teardown and this must never spin.
_proc_attributed() {
  local pid="$1" hops=0 key
  while [ "$hops" -lt 24 ] && [ "$pid" -gt 1 ] 2>/dev/null; do
    key=$(proc_key "$pid") || return 1
    _proc_declared "$key" && return 0
    pid=$(_proc_ppid "$pid") || return 1
    hops=$((hops + 1))
  done
  return 1
}

# Drop declarations whose process is gone, so the file cannot grow without
# bound and cannot accumulate keys that a recycled pid might one day match.
# (It cannot — the starttime differs — but a file nobody prunes is a file
# nobody reads.)
proc_prune_declarations() {
  local tmp
  [ -f "$JACK_PROC_DECL" ] || return 0
  tmp=$(mktemp) || return 0
  while IFS=$'\t' read -r key rest; do
    [ -n "$key" ] || continue
    [ "$(proc_key "${key%%:*}" 2>/dev/null)" = "$key" ] || continue
    printf '%s\t%s\n' "$key" "$rest"
  done < "$JACK_PROC_DECL" > "$tmp"
  mv "$tmp" "$JACK_PROC_DECL" 2>/dev/null || rm -f "$tmp"
}

# ------------------------------------------------------------- snapshot ----
# proc_snapshot -> one "pid:starttime" per line, sorted.
proc_snapshot() {
  local d pid key
  for d in /proc/[0-9]*; do
    pid=${d#/proc/}
    _proc_is_ours "$pid" || continue
    key=$(proc_key "$pid") || continue
    printf '%s\n' "$key"
  done | sort
}

# proc_leaks BEFORE_SNAPSHOT [sayfn]
#
# Names every project process that (a) was not running before, and (b) nobody
# declared. Sets PROC_LEAK_N. Returns 0 when the box is clean, 1 when it is not,
# so a caller can refuse to write a silent `rc=0`.
PROC_LEAK_N=0
proc_leaks() {
  local before="$1" sayfn="${2:-:}" d pid key n=0
  for d in /proc/[0-9]*; do
    pid=${d#/proc/}
    _proc_is_ours "$pid" || continue
    key=$(proc_key "$pid") || continue
    case "
$before" in *"
$key"*) continue ;; esac
    _proc_attributed "$pid" && continue
    n=$((n + 1))
    "$sayfn" "LEFTOVER PROCESS $key — $(proc_cpu_seconds "$pid")s CPU, cwd=$(readlink "/proc/$pid/cwd" 2>/dev/null), cmd: $(proc_cmdline "$pid")"
  done
  PROC_LEAK_N=$n
  if [ "$n" -gt 0 ]; then
    "$sayfn" "LEFTOVER: ${n} project process(es) started during this iteration and still running, declared by nobody. NOT killed — a declared detached run is legitimate and an undeclared one may be the owner's session. Attribute it or kill it: SYSTEM.md forbids leaving compute on a box with paying tenants (52nd audit B2; the scar cost 1.26 core-hours)."
    return 1
  fi
  return 0
}
