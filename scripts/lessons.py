#!/usr/bin/env python3
"""Counted lessons (docs/FORGE.md, "Second engine"; order F4).

A lesson is one line: a rule, a count, and the thing that happened here that
taught it. The count is the whole instrument:

    add   a new lesson enters at 1 (refused without a source, or if the same
          rule is already there: use `up`)
    up    +1: it bit again, or following it saved a run
    down  -1: it was followed and misled, or it did not apply where it claimed
    prune delete every lesson standing at zero (the charter's "delete at zero")
    list  print them, highest count first
    check exit 1 if the block is malformed

The lessons live in ONE marked block of docs/LESSONS.md. That file is also an
older prose archive cited by line number all over the repository, so this tool
never touches a byte outside its own block and creates the block at the END of
the file (appending moves no earlier line). Ids are never reused: a pruned
lesson's number is gone, so an old citation of L07 can never point at a
different rule.

    python scripts/lessons.py add "rule text" --src "ruling 76"
    python scripts/lessons.py up L03 --src "what happened this time"
    python scripts/lessons.py down L03 --src "where it misled"
    python scripts/lessons.py prune
    python scripts/lessons.py list | check
"""
from __future__ import annotations

import argparse
import datetime as _dt
import os
import re
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_FILE = REPO / "docs" / "LESSONS.md"

BEGIN_RE = re.compile(r"^<!-- COUNTED-LESSONS:BEGIN next=L(\d+) -->$")
END = "<!-- COUNTED-LESSONS:END -->"
HEADING = "## COUNTED LESSONS"
NOTE = (
    "Written by `scripts/lessons.py` (add / up / down / prune); do not edit by "
    "hand. One line a lesson: the count goes up when it bites again or saves "
    "a run, down when it misleads; at zero it is pruned. Ids are never reused."
)
LINE_RE = re.compile(
    r"^- \*\*L(\d+)\*\* \(n=(\d+), added (\d{4}-\d{2}-\d{2}), "
    r"last (\d{4}-\d{2}-\d{2})\) (.+) — src: (.+)$"
)
SRC_SEP = " — src: "
MAX_LESSONS = 40      # the cheap end of self-improvement stays one screen
MAX_RULE_CHARS = 280
MAX_SRC_CHARS = 200


class LessonError(ValueError):
    """A refusal. The file is not written."""


@dataclass
class Lesson:
    num: int
    n: int
    added: str
    last: str
    rule: str
    src: str

    @property
    def id(self) -> str:
        return f"L{self.num:02d}"

    def render(self) -> str:
        return (f"- **{self.id}** (n={self.n}, added {self.added}, "
                f"last {self.last}) {self.rule}{SRC_SEP}{self.src}")


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def _clean(text: str, what: str, cap: int) -> str:
    text = " ".join((text or "").split())
    if not text:
        raise LessonError(f"refused: empty {what}")
    if SRC_SEP in text or "<!--" in text or "-->" in text:
        raise LessonError(f"refused: {what} contains a reserved marker")
    if len(text) > cap:
        raise LessonError(f"refused: {what} is {len(text)} chars, cap {cap}")
    return text


def _date(value: str | None) -> str:
    if value is None:
        return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%d")
    try:
        _dt.date.fromisoformat(value)
    except ValueError:
        raise LessonError(f"refused: {value!r} is not a date (YYYY-MM-DD)") from None
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        raise LessonError(f"refused: {value!r} is not a date (YYYY-MM-DD)")
    return value


def split(text: str) -> tuple[str, list[str] | None, str]:
    """(before, block lines or None, after). Refuses a malformed block."""
    lines = text.split("\n")
    begins = [i for i, ln in enumerate(lines) if "COUNTED-LESSONS:BEGIN" in ln]
    ends = [i for i, ln in enumerate(lines) if "COUNTED-LESSONS:END" in ln]
    if not begins and not ends:
        return text, None, ""
    if len(begins) != 1 or len(ends) != 1 or ends[0] < begins[0]:
        raise LessonError(
            f"refused: malformed block ({len(begins)} BEGIN, {len(ends)} END)")
    b, e = begins[0], ends[0]
    if not BEGIN_RE.match(lines[b]) or lines[e] != END:
        raise LessonError("refused: a block marker line has been edited")
    before = "\n".join(lines[:b])
    after = "\n".join(lines[e + 1:])
    return before, lines[b:e + 1], after


def parse(text: str) -> tuple[list[Lesson], int]:
    """(lessons, next id number). No block = ([], 1)."""
    _, block, _ = split(text)
    if block is None:
        return [], 1
    next_num = int(BEGIN_RE.match(block[0]).group(1))
    lessons: list[Lesson] = []
    for ln in block[1:-1]:
        if not ln.startswith("- "):
            if ln in ("", HEADING, NOTE):
                continue
            raise LessonError(f"refused: unknown line in the block: {ln[:60]!r}")
        m = LINE_RE.match(ln)
        if not m:
            raise LessonError(f"refused: unreadable lesson line: {ln[:60]!r}")
        lessons.append(Lesson(int(m.group(1)), int(m.group(2)), m.group(3),
                              m.group(4), m.group(5), m.group(6)))
    nums = [x.num for x in lessons]
    if len(set(nums)) != len(nums):
        raise LessonError("refused: a lesson id appears twice")
    if nums and max(nums) >= next_num:
        raise LessonError("refused: an id is at or past `next` (ids are never reused)")
    if len(lessons) > MAX_LESSONS:
        raise LessonError(f"refused: {len(lessons)} lessons, cap {MAX_LESSONS}")
    norms = [_norm(x.rule) for x in lessons]
    if len(set(norms)) != len(norms):
        raise LessonError("refused: the same rule appears twice")
    return lessons, next_num


def render(text: str, lessons: list[Lesson], next_num: int) -> str:
    """The file's text with the block replaced (or appended at the end)."""
    before, block, after = split(text)
    body = [f"<!-- COUNTED-LESSONS:BEGIN next=L{next_num:02d} -->", HEADING, "",
            NOTE, ""]
    body += [x.render() for x in sorted(lessons, key=lambda x: x.num)]
    body.append(END)
    new_block = "\n".join(body)
    if block is None:
        sep = "" if before.endswith("\n\n") or not before else (
            "\n" if before.endswith("\n") else "\n\n")
        return before + sep + new_block + "\n"
    return before + "\n" + new_block + "\n" + after if before else (
        new_block + "\n" + after)


def _find(lessons: list[Lesson], ident: str) -> Lesson:
    m = re.fullmatch(r"L0*(\d+)", ident.strip())
    if not m:
        raise LessonError(f"refused: {ident!r} is not a lesson id (L07)")
    for x in lessons:
        if x.num == int(m.group(1)):
            return x
    raise LessonError(f"refused: no lesson {ident} (pruned ids are not reused)")


def add(text: str, rule: str, src: str, date: str | None = None) -> str:
    lessons, next_num = parse(text)
    rule = _clean(rule, "rule", MAX_RULE_CHARS)
    src = _clean(src, "source (a lesson is something that happened here)",
                 MAX_SRC_CHARS)
    for x in lessons:
        if _norm(x.rule) == _norm(rule):
            raise LessonError(f"refused: already there as {x.id}; use `up {x.id}`")
    if len(lessons) >= MAX_LESSONS:
        raise LessonError(
            f"refused: {MAX_LESSONS} lessons already; `down` or `prune` first")
    day = _date(date)
    lessons.append(Lesson(next_num, 1, day, day, rule, src))
    return render(text, lessons, next_num + 1)


def bump(text: str, ident: str, delta: int, src: str,
         date: str | None = None) -> str:
    """up (+1) or down (-1). The source of the latest change replaces the old
    one only on `up`; a `down` keeps the lesson's own evidence (history is in
    git) but must still say why."""
    if delta not in (1, -1):
        raise LessonError("refused: a count moves by one")
    lessons, next_num = parse(text)
    x = _find(lessons, ident)
    src = _clean(src, "source (say what happened)", MAX_SRC_CHARS)
    if x.n + delta < 0:
        raise LessonError(f"refused: {x.id} is already at zero; `prune` deletes it")
    x.n += delta
    x.last = _date(date)
    if delta > 0:
        x.src = src
    return render(text, lessons, next_num)


def prune(text: str) -> tuple[str, list[Lesson]]:
    lessons, next_num = parse(text)
    gone = [x for x in lessons if x.n == 0]
    if not gone:
        return text, []
    return render(text, [x for x in lessons if x.n > 0], next_num), gone


def _write(path: Path, text: str) -> None:
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".lessons.")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
        if path.exists():
            os.chmod(tmp, path.stat().st_mode & 0o777)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--file", type=Path, default=DEFAULT_FILE)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("add")
    a.add_argument("rule")
    a.add_argument("--src", required=True)
    a.add_argument("--date")
    for name in ("up", "down"):
        b = sub.add_parser(name)
        b.add_argument("id")
        b.add_argument("--src", required=True)
        b.add_argument("--date")
    sub.add_parser("prune")
    sub.add_parser("list")
    sub.add_parser("check")
    args = ap.parse_args(argv)

    path: Path = args.file
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    try:
        if args.cmd == "add":
            new = add(text, args.rule, args.src, args.date)
            print(f"added {parse(new)[0][-1].id}")
        elif args.cmd in ("up", "down"):
            new = bump(text, args.id, 1 if args.cmd == "up" else -1,
                       args.src, args.date)
            x = _find(parse(new)[0], args.id)
            print(f"{x.id} n={x.n}" + ("  (at zero: `prune` deletes it)"
                                       if x.n == 0 else ""))
        elif args.cmd == "prune":
            new, gone = prune(text)
            for x in gone:
                print(f"pruned {x.id}: {x.rule}")
            if not gone:
                print("nothing at zero")
        else:
            lessons, next_num = parse(text)
            if args.cmd == "list":
                for x in sorted(lessons, key=lambda x: (-x.n, x.num)):
                    print(f"{x.id}  n={x.n}  {x.rule}  [{x.src}]")
            print(f"{len(lessons)} lessons, next id L{next_num:02d}, "
                  f"{sum(1 for x in lessons if x.n == 0)} at zero")
            return 0
    except LessonError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    if new != text:
        _write(path, new)
    return 0


if __name__ == "__main__":
    sys.exit(main())
