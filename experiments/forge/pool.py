"""The forge's candidate pool (docs/FORGE.md "The machine" 1, order F2).

`docs/FORGE_POOL.md` holds one line per candidate:

    - **ID** · source: <tag> · wall: reach|keep · change: … · prior: … · ready: knob|code [· note: …]

`check()` refuses a pool that breaks the charter: a candidate with no `prior:`
(law 3a) or a prior that cites nothing, an unknown source tag, a repeated id,
more than 20 candidates, or no `control` candidate (plain variation near the
baseline, the yardstick every other source is scored against).

    python -m experiments.forge.pool          # exit 0 = the pool file is well formed
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

from experiments.forge import SOURCES

REPO = Path(__file__).resolve().parents[2]
POOL = REPO / "docs" / "FORGE_POOL.md"
MAX_CANDIDATES = 20
WALLS = ("reach", "keep")
READY = ("knob", "code")
REQUIRED = ("source", "wall", "change", "prior", "ready")
_LINE = re.compile(r"^- \*\*(?P<id>[A-Za-z0-9_.-]+)\*\* · (?P<rest>.+)$")
# A prior must point at something a reader can open: a section of
# PUBLISHED_PRIORS (§N), an arXiv id, or one of the lab's evidence files.
_CITES = re.compile(r"§\s?\d+|\b\d{4}\.\d{4,5}\b|FORGE\.md|LAB\.md|discovery evidence|as C\d+")


class PoolError(ValueError):
    pass


def parse(text: str) -> list:
    """Candidate lines -> [{id, source, wall, change, prior, ready, note?}].
    A line that starts like a candidate but has a field without `key: value`
    is an error, never skipped."""
    out = []
    for n, line in enumerate(text.splitlines(), 1):
        if not line.startswith("- **"):
            continue
        m = _LINE.match(line)
        if not m or " · " not in line or "source:" not in line:
            continue                      # an ordinary bold bullet, not a candidate
        c = {"id": m.group("id")}
        for part in m.group("rest").split(" · "):
            key, sep, val = part.partition(":")
            if not sep or not key.strip().isidentifier():
                raise PoolError(f"line {n} ({c['id']}): field without `key: value`: {part[:40]!r}")
            if key.strip() in c:
                raise PoolError(f"line {n} ({c['id']}): field {key.strip()!r} twice")
            c[key.strip()] = val.strip()
        out.append(c)
    return out


def check(text: str) -> list:
    cands = parse(text)
    if not cands:
        raise PoolError("no candidate lines")
    if len(cands) > MAX_CANDIDATES:
        raise PoolError(f"{len(cands)} candidates, the pool holds at most {MAX_CANDIDATES}")
    seen = set()
    for c in cands:
        i = c["id"]
        if i in seen:
            raise PoolError(f"{i}: id used twice")
        seen.add(i)
        for k in REQUIRED:
            if not c.get(k):
                raise PoolError(f"{i}: no `{k}:`" + (" (law 3a: no candidate without a prior)"
                                                     if k == "prior" else ""))
        if c["source"] not in SOURCES:
            raise PoolError(f"{i}: source {c['source']!r} is not one of {SOURCES}")
        if c["wall"] not in WALLS:
            raise PoolError(f"{i}: wall {c['wall']!r} is not one of {WALLS}")
        if c["ready"] not in READY:
            raise PoolError(f"{i}: ready {c['ready']!r} is not one of {READY}")
        if not _CITES.search(c["prior"]):
            raise PoolError(f"{i}: prior cites nothing a reader can open (§N, an arXiv id, "
                            "FORGE.md, LAB.md or the discovery evidence)")
    if not any(c["source"] == "control" for c in cands):
        raise PoolError("no control-source candidate (plain variation near the baseline)")
    return cands


def counts(cands: list) -> dict:
    return {s: sum(c["source"] == s for c in cands) for s in SOURCES}


def main(argv=None) -> int:
    path = Path((argv or sys.argv[1:] or [POOL])[0])
    try:
        cands = check(path.read_text())
    except (OSError, PoolError) as e:
        print(f"POOL REFUSED: {e}")
        return 2
    print(f"pool ok: {len(cands)} candidates; "
          + ", ".join(f"{s} {n}" for s, n in counts(cands).items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
