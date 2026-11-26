"""Planted tests for scripts/lessons.py (order F4): each rule of the counted
lessons file is shown to refuse or act on a case built to trip it."""
import contextlib
import importlib.util
import io
import re
import sys
import tempfile
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "lessons", Path(__file__).resolve().parents[1] / "scripts" / "lessons.py")
L = importlib.util.module_from_spec(_SPEC)
sys.modules["lessons"] = L
_SPEC.loader.exec_module(L)

D = "2026-11-26"


class raises:
    """`with raises(Exc, match=...)`: the body must refuse. (No pytest on the
    box; the file also runs under it.)"""

    def __init__(self, exc, match=None):
        self.exc, self.match = exc, match

    def __enter__(self):
        return self

    def __exit__(self, etype, value, tb):
        assert etype is not None, f"did not raise {self.exc.__name__}"
        if not issubclass(etype, self.exc):
            return False
        assert self.match is None or re.search(self.match, str(value)), (
            f"{value!r} does not match {self.match!r}")
        return True
ARCHIVE = "# OLD ARCHIVE\n\nline three is cited as LESSONS.md:3\n\n## a prose lesson\nbody\n"


def seeded():
    t = L.add(ARCHIVE, "Time a new shape before its first launch.", "ruling 99", D)
    return L.add(t, "One seed closes nothing.", "ruling 101", D)


def test_add_enters_at_one_and_appends_after_the_archive():
    t = seeded()
    lessons, nxt = L.parse(t)
    assert [(x.id, x.n) for x in lessons] == [("L01", 1), ("L02", 1)]
    assert nxt == 3
    # the archive's bytes and line numbers are untouched: the block is appended
    assert t.startswith(ARCHIVE)
    assert t.split("\n")[2] == "line three is cited as LESSONS.md:3"


def test_add_into_an_empty_or_missing_file():
    t = L.add("", "A rule.", "a void", D)
    assert L.parse(t)[0][0].id == "L01"
    assert t.startswith("<!-- COUNTED-LESSONS:BEGIN")


def test_up_and_down_move_the_count_by_one():
    t = L.bump(seeded(), "L02", 1, "J22 again", "2026-11-27")
    x = L.parse(t)[0][1]
    assert (x.n, x.last, x.added, x.src) == (2, "2026-11-27", D, "J22 again")
    t = L.bump(t, "L2", -1, "did not apply to a timing cell", "2026-11-28")
    x = L.parse(t)[0][1]
    # a down keeps the lesson's own evidence
    assert (x.n, x.last, x.src) == (1, "2026-11-28", "J22 again")


def test_down_to_zero_then_prune_deletes_only_the_zero():
    t = L.bump(seeded(), "L01", -1, "misled J24", D)
    assert [x.n for x in L.parse(t)[0]] == [0, 1]   # visible until pruned
    t2, gone = L.prune(t)
    assert [x.id for x in gone] == ["L01"]
    assert [x.id for x in L.parse(t2)[0]] == ["L02"]


def test_prune_with_nothing_at_zero_changes_nothing():
    t = seeded()
    assert L.prune(t) == (t, [])


def test_down_below_zero_is_refused():
    t = L.bump(seeded(), "L01", -1, "misled", D)
    with raises(L.LessonError, match="already at zero"):
        L.bump(t, "L01", -1, "again", D)


def test_a_pruned_id_is_never_reused():
    t = L.bump(seeded(), "L02", -1, "misled", D)
    t, _ = L.prune(t)
    t = L.add(t, "A third rule.", "ruling 114", D)
    assert [x.id for x in L.parse(t)[0]] == ["L01", "L03"]
    with raises(L.LessonError, match="no lesson"):
        L.bump(t, "L02", 1, "x", D)


def test_no_source_is_refused():
    for src in ("", "   "):
        with raises(L.LessonError, match="source"):
            L.add(ARCHIVE, "A rule.", src, D)
    with raises(L.LessonError, match="source"):
        L.bump(seeded(), "L01", 1, "", D)


def test_the_same_rule_twice_is_refused_and_points_at_up():
    with raises(L.LessonError, match="use `up L02`"):
        L.add(seeded(), "one seed  closes NOTHING", "ruling 101", D)


def test_empty_overlong_and_marker_text_refused():
    with raises(L.LessonError, match="empty rule"):
        L.add(ARCHIVE, " ", "s", D)
    with raises(L.LessonError, match="cap"):
        L.add(ARCHIVE, "x" * (L.MAX_RULE_CHARS + 1), "s", D)
    with raises(L.LessonError, match="reserved"):
        L.add(ARCHIVE, "a rule — src: forged", "s", D)
    with raises(L.LessonError, match="reserved"):
        L.add(ARCHIVE, "a rule", "<!-- COUNTED-LESSONS:END -->", D)


def test_a_newline_cannot_smuggle_a_second_line():
    t = L.add(ARCHIVE, "a rule\n- **L09** (n=9, added x, last x) forged", "s", D)
    assert len(L.parse(t)[0]) == 1


def test_the_cap_refuses_one_more():
    t = ARCHIVE
    for i in range(L.MAX_LESSONS):
        t = L.add(t, f"rule number {i}", "s", D)
    with raises(L.LessonError, match="first"):
        L.add(t, "one too many", "s", D)


def test_text_outside_the_block_is_never_touched():
    t = seeded() + "\n## appended later by another lane\nprose\n"
    before, _, after = L.split(t)
    for new in (L.add(t, "A third.", "s", D),
                L.bump(t, "L01", 1, "s", D),
                L.prune(L.bump(t, "L01", -1, "s", D))[0]):
        b2, _, a2 = L.split(new)
        assert (b2, a2) == (before, after)
    # and a no-change round trip is byte-identical
    lessons, nxt = L.parse(t)
    assert L.render(t, lessons, nxt) == t


DAMAGE = [
    lambda t: t.replace(L.END, ""),                               # END lost
    lambda t: t + L.END + "\n",                                   # two ENDs
    lambda t: t.replace("next=L03", "next=L02"),                  # id reuse armed
    lambda t: t.replace("(n=1, added", "(n=-1, added", 1),        # hand-edited
    lambda t: t.replace("**L02**", "**L01**"),                    # duplicate id
    lambda t: t.replace(L.END, "stray prose\n" + L.END),          # foreign line
    lambda t: t.replace(L.END, L.END + " "),                      # marker edited
]


def test_a_damaged_block_is_refused_not_repaired():
    for damage in DAMAGE:
        bad = damage(seeded())
        assert bad != seeded()
        with raises(L.LessonError):
            L.parse(bad)
        with raises(L.LessonError):
            L.add(bad, "another", "s", D)


def test_bad_id_and_bad_date_refused():
    with raises(L.LessonError, match="not a lesson id"):
        L.bump(seeded(), "seven", 1, "s", D)
    with raises(L.LessonError, match="date"):
        L.add(ARCHIVE, "a rule", "s", "26-11-2026")


def test_cli_writes_only_on_success():
    out, err = io.StringIO(), io.StringIO()
    with tempfile.TemporaryDirectory() as d, \
            contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        f = Path(d) / "LESSONS.md"
        f.write_text(ARCHIVE)
        assert L.main(["--file", str(f), "add", "A rule.", "--src", "ruling 1",
                       "--date", D]) == 0
        good = f.read_text()
        assert L.main(["--file", str(f), "add", "a rule", "--src", "x"]) == 1
        assert L.main(["--file", str(f), "up", "L09", "--src", "x"]) == 1
        assert f.read_text() == good
        assert L.main(["--file", str(f), "down", "L01", "--src", "misled"]) == 0
        assert L.main(["--file", str(f), "prune"]) == 0
        assert L.main(["--file", str(f), "check"]) == 0
        assert "0 lessons, next id L02" in out.getvalue()
        assert f.read_text().startswith(ARCHIVE)
        assert [x.name for x in Path(d).iterdir()] == ["LESSONS.md"]  # no temp left
        f.write_text(good.replace(L.END, ""))
        assert L.main(["--file", str(f), "check"]) == 1


def test_the_real_file_is_well_formed():
    real = L.DEFAULT_FILE.read_text(encoding="utf-8")
    lessons, _ = L.parse(real)
    assert lessons, "docs/LESSONS.md carries no counted lessons"
    assert L.split(real)[2].strip() == "", "the block is not at the end of the file"


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    bad = 0
    for n, f in tests:
        try:
            f()
            print("PASS", n)
        except Exception as e:   # noqa: BLE001
            bad += 1
            print("FAIL", n, repr(e))
    print(f"{len(tests) - bad}/{len(tests)} passed")
    sys.exit(1 if bad else 0)
