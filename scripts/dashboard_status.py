#!/data/venvs/jackthelearner/bin/python
"""dashboard_status.py — one JSON snapshot of Jack and of the machine that builds him.

Owner, 2026-11-15: a simple page of bars that QUANTIFIES what Jack can do, shows
what is running, and gives the detail on a tap ("we need to quantify the stuff
man you know but well"). This script is the only place those numbers are made;
the page (`dashboard/page.html`) only draws them. No model; the only network
call is asking Kaggle, at most every ten minutes, which sent jobs still run.

Every number is read from something the project already keeps honest:

  proven    experiments/coverage.py — each promise in GOAL.md and the tests that
            declare they prove it. A promise's score is
                passing CLAIM tests / registered CLAIM tests.
            Fixtures, rules and sensors are groundwork: listed, never counted
            (coverage.report() makes the same split, for the same reason).
            A bar is the same ratio over every promise filed under it, and the
            headline over all of them; a promise with no proof written yet
            owes one. The grouping lives in dashboard/abilities.json. Each
            proof is also marked by where it was earned (`where_ran`): in the
            living Jack, on a test bench, or with the earlier brain. The page
            leads with "in Jack himself".
  measured  dashboard/vitals.json — a number from a ledger row, with its floor
            (what an untrained or do-nothing Jack scores) and its pass mark:
                score = (value - floor) / (pass mark - floor), clipped 0..100.
            The same normalisation the field uses to compare agents against a
            random baseline and a reference.
  route     docs/ROADMAP.md — the nine milestones and the tests each one names.
  machine   the organs' logs, the crontab, ps, the GPU/CPU journals, the usage
            ledger.

Run by cron every two minutes; writes status.json and index.html atomically to
$JACK_DASH_OUT (default /data/jack-dashboard). Reads only; never touches the
ledger, the repo, or a running job.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(os.environ.get("JACK_REPO", "/home/opc/jackthelearner"))
CFG = Path(os.environ.get("JACK_DASH_CFG", str(REPO / "dashboard")))
OUT = Path(os.environ.get("JACK_DASH_OUT", "/data/jack-dashboard"))
# Where the web server can see the page. Only index.html, status.json and the
# film are copied there; the working files above (history, the Kaggle cache)
# are never served.
PUBLISH = os.environ.get("JACK_DASH_PUBLISH") or None
LOGS = Path(os.environ.get("JACK_DASH_LOGS", "/data/jack-logs"))
sys.path.insert(0, str(REPO))

UTC = dt.timezone.utc
NOW = dt.datetime.now(UTC)


# ── small helpers ───────────────────────────────────────────────────────────
def iso(t: dt.datetime | None) -> str | None:
    return t.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ") if t else None


def parse_ts(s: str | None) -> dt.datetime | None:
    """Timestamps in this project are UTC, with or without an offset."""
    if not s:
        return None
    try:
        t = dt.datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None
    return t if t.tzinfo else t.replace(tzinfo=UTC)


def read(path: Path, tail_bytes: int | None = None) -> str:
    try:
        if tail_bytes is None:
            return path.read_text(errors="replace")
        with open(path, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - tail_bytes))
            return f.read().decode(errors="replace")
    except OSError:
        return ""


def clean(md: str) -> str:
    """Markdown to plain words: drop code ticks and link syntax, keep **bold**."""
    md = re.sub(r"`([^`]*)`", r"\1", md)
    md = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", md)
    return re.sub(r"\s+", " ", md).strip()


def sh(cmd: list[str], timeout: int = 10) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True,
                              timeout=timeout).stdout
    except (OSError, subprocess.SubprocessError):
        return ""


def guarded(fn, default):
    """One broken section must not blank the page."""
    try:
        return fn()
    except Exception as e:  # noqa: BLE001 — reported, never swallowed silently
        ERRORS.append(f"{getattr(fn, '__name__', 'section')}: {type(e).__name__}: {e}")
        return default


ERRORS: list[str] = []


# ── the tests ───────────────────────────────────────────────────────────────
def load_tests():
    from experiments import registry
    by_id = getattr(registry, "BY_ID", None) or {s.id: s for s in registry.ALL}
    results = json.load(open(REPO / "experiments" / "ledger.json")).get("results", {})
    return by_id, results


_TEST_FILES: dict = {}
_WHERE: dict = {}


def where_ran(sid: str) -> str | None:
    """Where a test's proof was earned, read from the test's own code:
    'world'  it runs the living Jack — today's brain inside his world;
    'old'    it runs the earlier, larger brain;
    'bench'  it tests a part by itself on a test bench.
    The owner asked how hearing can read as proven before a brain is chosen:
    it cannot, in him. Only 'world' counts as Jack himself."""
    if sid in _WHERE:
        return _WHERE[sid]
    if not _TEST_FILES:
        _TEST_FILES.update({f.stem: f for f in (REPO / "experiments" / "tests").glob("*.py")})
    fam, _, num = sid.partition(".")
    key = (fam + "_" + num.replace(".", "_")).lower()
    f = next((p for b, p in _TEST_FILES.items() if b == key or b.startswith(key + "_")), None)
    out = None
    if f is not None:
        src = read(f)
        if "UnifiedBrain" in src:
            out = "old"
        elif re.search(r"run_survival|\bW0\(|experiments\.(survival|w0)\b"
                       r"|from\s+\.+\s*import[^\n]*\b(survival|w0)\b|from\s+\.+(survival|w0)\s+import", src):
            out = "world"
        else:
            out = "bench"
    _WHERE[sid] = out
    return out


def test_row(sid: str, by_id: dict, results: dict, kind: str | None = None) -> dict:
    spec, res = by_id.get(sid), results.get(sid) or {}
    return {
        "where": where_ran(sid) if spec else None,
        "id": sid,
        "title": getattr(spec, "title", None) or "(a test named in the plan, not written yet)",
        "status": res.get("status") or ("NONE" if spec else "UNWRITTEN"),
        "ran": (res.get("ran_at") or "")[:16] or None,
        "kind": kind,
    }


# ── proven: the goal's promises ─────────────────────────────────────────────
def abilities(by_id: dict, results: dict) -> dict:
    from experiments import coverage
    rep = {r["commitment"]: r for r in coverage.report(by_id, results)}
    cfg = json.load(open(CFG / "abilities.json"))
    names = cfg.get("promise_names", {})

    def promise(name: str) -> dict:
        r = rep[name]
        kinds = r["kinds"]
        claims = [i for i in r["specs"] if kinds.get(i) == "claim"]
        ground = [i for i in r["specs"] if kinds.get(i) != "claim"]
        frac = (r["n_pass"] / len(claims)) if claims else 0.0
        in_him = sum(1 for i in claims if (results.get(i) or {}).get("status") == "PASS" and where_ran(i) == "world")
        return {
            "passed_world": in_him,
            "id": name, "name": names.get(name, name[:1].upper() + name[1:]),
            "why": r["why"], "frac": frac, "pct": round(100 * frac),
            "claims": len(claims), "passed": r["n_pass"],
            "ground": len(ground), "ground_passed": len(r["support_pass"]),
            "tests": [test_row(i, by_id, results, kinds.get(i)) for i in claims + ground]
                     + [test_row(i, by_id, results, "parked") for i in r["parked"]],
        }

    def bar(ident, name, blurb, ps):
        # One rule for every bar: proofs that pass / proofs there are. A promise
        # nobody has written a proof for yet still owes one, so it counts as one.
        need = sum(max(p["claims"], 1) for p in ps)
        got = sum(p["passed"] for p in ps)
        him = sum(p["passed_world"] for p in ps)
        return {"id": ident, "name": name, "blurb": blurb, "proofs": need, "passed": got, "passed_world": him,
                "pct": round(100 * got / need) if need else 0,
                "pct_world": round(100 * him / need) if need else 0, "promises": ps}

    aspects, used = [], set()
    for a in cfg["aspects"]:
        ps = [promise(n) for n in a["promises"] if n in rep]
        used.update(p["id"] for p in ps)
        aspects.append(bar(a["id"], a["name"], a["blurb"], ps))
    rest = [promise(n) for n in rep if n not in used]
    if rest:   # a promise added to GOAL.md shows up here until someone files it
        aspects.append(bar("other", "Other promises",
                           "Promises in the goal that the page has not been told where to put.", rest))
    n_prom = sum(len(a["promises"]) for a in aspects)
    n_claims = sum(a["proofs"] for a in aspects)
    n_passed = sum(a["passed"] for a in aspects)
    overall = (n_passed / n_claims) if n_claims else 0.0
    n_world = sum(a["passed_world"] for a in aspects)
    started = sum(1 for a in aspects for p in a["promises"] if p["passed"] > 0)
    for a in aspects:
        for p in a["promises"]:
            del p["frac"]
    return {"overall": round(100 * overall, 1), "aspects": aspects,
            "promises": n_prom, "promises_started": started,
            "proofs": n_claims, "proofs_passed": n_passed, "proofs_passed_world": n_world,
            "overall_world": round(100 * n_world / n_claims, 1) if n_claims else 0.0,
            "senses_for_m5": cfg.get("milestone_promises", {})}


# ── measured: vital signs ───────────────────────────────────────────────────
def vitals(by_id: dict, results: dict) -> list:
    out = []
    for v in json.load(open(CFG / "vitals.json")):
        res = results.get(v["spec"])
        if not res:
            continue
        pool = {}
        pool.update(res.get("control_metrics") or {})
        pool.update(res.get("metrics") or {})

        def val(x):
            if isinstance(x, (int, float)):
                return float(x)
            if isinstance(x, str):
                y = pool.get(x)
                return float(y) if isinstance(y, (int, float)) else None
            if isinstance(x, dict):
                if "sum" in x:
                    parts = [val(p) for p in x["sum"]]
                    return None if None in parts else sum(parts)
                y = val(x.get("key"))
                return None if y is None else y * x.get("scale", 1.0)
            return None

        scale = v.get("scale", 1.0)
        value, floor, target = val(v["value"]), val(v["floor"]), val(v["target"])
        if value is None or floor is None or target is None or target == floor:
            continue
        value *= scale
        score = max(0.0, min(1.0, (value - floor) / (target - floor)))
        refs = []
        for r in v.get("refs", []):
            y = val(r["value"])
            if y is not None:
                refs.append({"label": r["label"], "value": round(y, 2)})
        out.append({
            "id": v["id"], "label": v["label"], "unit": v.get("unit", ""),
            "digits": v.get("digits", 1), "value": round(value, 3),
            "floor": round(floor, 3), "target": round(target, 3),
            "floor_label": v.get("floor_label", "floor"),
            "target_label": v.get("target_label", "pass mark"),
            "score": round(100 * score), "where": v.get("where", "living"),
            "note": v.get("note", ""), "refs": refs,
            "test": test_row(v["spec"], by_id, results),
        })
    return out


def lives(results: dict) -> dict | None:
    """The fifteen lives of the last 'does living teach him' run, one run shown."""
    res = results.get("LF.04")
    if not res:
        return None
    m, c = res.get("metrics") or {}, res.get("control_metrics") or {}

    def arr(key):
        try:
            a = json.loads(m.get(key) or "[]")
            return [round(float(x), 1) for x in a] if a and not isinstance(a[0], list) else None
        except (ValueError, TypeError):
            return None

    learner, twin = arr("spans_learner"), arr("spans_twin")
    if not learner:
        return None
    return {"learner": learner, "twin": twin,
            "nothing": round(float(c.get("statue_mean_life_s", 600.0)), 1),
            "test": "LF.04", "ran": (res.get("ran_at") or "")[:16]}


# ── the route: ROADMAP.md ───────────────────────────────────────────────────
ID = r"[A-Z][A-Z0-9]{0,3}\.[0-9A-Za-z][0-9A-Za-z.]*"


def _num(sid: str) -> float | None:
    try:
        return float(sid.split(".", 1)[1])
    except ValueError:
        return None


def named_tests(text: str, by_id: dict) -> list[str]:
    fam = {}
    for k in by_id:
        fam.setdefault(k.split(".")[0], []).append(k)
    out: list[str] = []

    def add(i):
        if i not in out:
            out.append(i)

    pat = re.compile(rf"`({ID}|[A-Z][A-Z0-9]{{0,3}}\.\*)`(?:\s*[–-]\s*`({ID})`)?(\+)?")
    for a, b, plus in pat.findall(text):
        f = a.split(".")[0]
        if a.endswith(".*"):
            for i in fam.get(f, []):
                add(i)
        elif b or plus:
            lo, hi = _num(a), (_num(b) if b else None)
            hit = [i for i in fam.get(f, []) if _num(i) is not None and lo is not None
                   and _num(i) >= lo and (hi is None or _num(i) <= hi)]
            for i in hit or [a] + ([b] if b else []):
                add(i)
        else:
            add(a)
    return out


def milestones(by_id: dict, results: dict, ab: dict) -> list:
    """A milestone that names no test (M5: "a passing claim for every sense")
    is scored on the promises abilities.json files under it."""
    promise_map = ab.get("senses_for_m5", {})
    promise_state = {p["id"]: {"name": p["name"], "passed": p["passed"], "claims": p["claims"]}
                     for a in ab["aspects"] for p in a["promises"]}
    text = read(REPO / "docs" / "ROADMAP.md")
    blocks = re.split(r"(?m)^(?=\*\*M\d+ — )", text)[1:]
    out = []
    for b in blocks:
        b = b.split("\n---")[0]
        m = re.match(r"\*\*M(\d+) — (.+?)\.\*\*\s*(.*)", b, re.S)
        if not m:
            continue
        n, title, body = int(m.group(1)), m.group(2), m.group(3)
        parts = re.split(r"·\s*status(?:\s*\(([^)]*)\))?:", body, maxsplit=1)
        pre = parts[0]
        when, status = (parts[1] or "", parts[2]) if len(parts) == 3 else ("", "")
        desc, _, done = pre.partition("done when:")
        # "(to be registered; replaces the Stack-B `T6.*`)" names tests that are
        # on their way OUT; they are not this milestone's gates.
        live = re.sub(r"\([^)]*replaces[^)]*\)", " (to be registered) ", pre)
        ids = named_tests(live, by_id)
        tests = [test_row(i, by_id, results) for i in ids]
        unwritten = len(re.findall(r"\b(?:unregistered|to be registered|none registered)\b", live))
        tests += [{"id": None, "title": "A test the roadmap names that nobody has written yet",
                   "status": "UNWRITTEN", "ran": None, "kind": None}] * unwritten
        proms = []
        for name in promise_map.get(str(n), []) if not ids else []:
            p = promise_state.get(name)
            if p:
                proms.append(p)
        passed = sum(1 for t in tests if t["status"] == "PASS")
        total = len(tests) + len(proms)
        got = passed + sum(1 for p in proms if p["passed"] > 0)
        s = status.strip()
        if re.search(r"\bMET\b", s):
            state = "met"
        elif re.search(r"\bnot started\b", s, re.I) and not got:
            state = "not started"
        elif got or re.search(r"\b(orders?|running|in work|in flight)\b", s, re.I):
            state = "in work"
        else:
            state = "not started"
        out.append({"n": n, "title": title, "about": clean(desc),
                    "done_when": clean(done), "status": clean(s), "as_of": when,
                    "state": state, "tests": tests, "promises": proms,
                    "passed": got, "total": total,
                    "pct": 100 if state == "met" else (round(100 * got / total) if total else 0)})
    return out


# ── what the Director says and what the workers are on ──────────────────────
def headline() -> dict:
    text = read(REPO / "docs" / "PROGRESS.md")
    m = re.search(r"(?ms)^## The short version\s*\n(.*?)(?=^## )", text)
    paras = [clean(p) for p in re.split(r"\n\s*\n", m.group(1).strip())] if m else []
    w = re.search(r"(?m)^Written (.+?)(?:\.|$)", text)
    return {"written": clean(w.group(1)) if w else None, "paras": [p for p in paras if p]}


def orders() -> list:
    text = read(REPO / "docs" / "ORDERS.md")
    out = []
    for m in re.finditer(r"(?m)^### ([JS])(\d+) — (.+)$", text):
        sec = text[m.end(): m.end() + 6000].split("\n### ")[0]
        st = re.search(r"(?m)^state:\s*(.+)$", sec)
        out.append({"id": m.group(1) + m.group(2),
                    "lane": "Jack" if m.group(1) == "J" else "The machine",
                    "title": clean(m.group(3)),
                    "state": clean(st.group(1))[:240] if st else ""})
    return out


def recent(by_id: dict, results: dict, n: int = 8) -> list:
    rows = sorted(results.items(), key=lambda kv: kv[1].get("ran_at") or "", reverse=True)
    return [test_row(i, by_id, results) for i, _ in rows[:n]]


# ── the team: logs and the timetable ────────────────────────────────────────
def _field(expr: str, lo: int, hi: int) -> set:
    out = set()
    for part in expr.split(","):
        step = 1
        if "/" in part:
            part, s = part.split("/")
            step = int(s)
        if part == "*":
            a, b = lo, hi
        elif "-" in part:
            a, b = (int(x) for x in part.split("-"))
        else:
            a = b = int(part)
        out.update(range(a, b + 1, step))
    return out


def cron_lines() -> list:
    out = []
    for line in sh(["crontab", "-l"]).splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "jackthelearner" not in line:
            continue
        f = line.split(None, 5)
        if len(f) < 6:
            continue
        try:
            sched = (_field(f[0], 0, 59), _field(f[1], 0, 23), _field(f[2], 1, 31),
                     _field(f[3], 1, 12), {d % 7 for d in _field(f[4], 0, 7)})
        except ValueError:
            continue
        out.append({"sched": sched, "dom_any": f[2] == "*", "dow_any": f[4] == "*",
                    "cmd": f[5]})
    return out


def next_fire(c: dict, after: dt.datetime) -> dt.datetime | None:
    mi, ho, dom, mon, dow = c["sched"]
    t = after.replace(second=0, microsecond=0) + dt.timedelta(minutes=1)
    for _ in range(60 * 24 * 8):
        d_ok = t.day in dom
        w_ok = ((t.weekday() + 1) % 7) in dow
        day_ok = (d_ok and w_ok) if (c["dom_any"] or c["dow_any"]) else (d_ok or w_ok)
        if t.minute in mi and t.hour in ho and t.month in mon and day_ok:
            return t
        t += dt.timedelta(minutes=1)
    return None


def _alive(pattern: str) -> bool:
    return bool(sh(["pgrep", "-f", pattern]).strip())


def team() -> dict:
    crons = cron_lines()

    def nxt(*needles, skip: str | None = None) -> str | None:
        ts = [next_fire(c, NOW) for c in crons if all(n in c["cmd"] for n in needles)
              and not (skip and skip in c["cmd"])]
        ts = [t for t in ts if t]
        return iso(min(ts)) if ts else None

    # the worker launcher: one log, one writer at a time
    lad = read(LOGS / "ladder.log", 600_000)
    runs, cur = [], None
    for line in lad.splitlines():
        m = re.match(r"(\S+) iteration start — lane (\w+), (\d+)/(\d+) demonstrated, model (\w+)", line)
        if m:
            cur = {"start": m.group(1), "lane": m.group(2), "model": m.group(5), "end": None, "rc": None}
            runs.append(cur)
            continue
        m = re.match(r"(\S+) iteration end rc=(\S+)", line)
        if m and cur and cur["end"] is None:
            cur["end"], cur["rc"] = m.group(1), m.group(2)
    worker_alive = _alive("scripts/ladder_loop.sh")
    day_ago = NOW - dt.timedelta(hours=24)

    default_model = {"jack": "opus", "system": "opus", "lab": "fable", "outsider": "fable"}

    def runs_text(n: int) -> str:
        return "No runs in the last day" if not n else f"{n} run{'s' if n != 1 else ''} in the last day"

    def lane(name: str) -> dict:
        mine = [r for r in runs if r["lane"] == name]
        last = mine[-1] if mine else None
        running = bool(last and last["end"] is None and worker_alive and runs and runs[-1] is last)
        recent_runs = [r for r in mine if (parse_ts(r["start"]) or day_ago) >= day_ago and r["end"]]
        bad = [r for r in recent_runs if r["rc"] not in ("0",)]
        return {"running": running, "since": last["start"] if running else None,
                "model": last["model"] if last else default_model.get(name),
                "last_end": next((r["end"] for r in reversed(mine) if r["end"]), None),
                "last_ok": (next((r["rc"] for r in reversed(mine) if r["end"]), None) == "0") if mine else None,
                "runs_24h": len(recent_runs), "bad_24h": len(bad)}

    def simple(log: str, start_pat: str, end_pat: str, proc: str) -> dict:
        text = read(LOGS / log, 200_000)
        starts = re.findall(rf"(?m)^(\S+) {start_pat}", text)
        ends = re.findall(rf"(?m)^(\S+) {end_pat}", text)
        s_last = starts[-1] if starts else None
        s_ts = s_last[0] if isinstance(s_last, tuple) else s_last
        e_last = ends[-1] if ends else None
        e_ts = e_last[0] if isinstance(e_last, tuple) else e_last
        running = bool(s_ts and (not e_ts or e_ts < s_ts) and _alive(proc))
        return {"running": running, "since": s_ts if running else None,
                "last_end": e_ts, "starts": starts, "ends": ends}

    rev = simple("review.log", r"review start — mode (\w+), model (\w+)", r"sweep end rc=(\d+)", "scripts/review.sh")
    aud = simple("overseer.log", r"audit start — model (\w+)", r"audit end rc=(\d+) — verdict: ([A-Z][A-Z ]*)", "scripts/overseer.sh")
    sco = simple("field_watch.log", r"sweep start — model (\w+)", r"sweep end rc=(\d+) — (\d+) nomination", "scripts/field_watch.sh")
    reg_ts = re.findall(r"(?m)^(\S+) committed and pushed", read(LOGS / "regate.log", 60_000))

    def pack(d: dict, **kw) -> dict:
        d = {k: v for k, v in d.items() if k not in ("starts", "ends")}
        d.update(kw)
        return d

    jack, system, lab, outsider = lane("jack"), lane("system"), lane("lab"), lane("outsider")
    verdict = aud["ends"][-1][2].strip() if aud["ends"] else None
    members = [
        pack(rev, id="director", name="Director", role="Reads everything each morning, writes the orders and your progress page.",
             model=(rev["starts"][-1][2] if rev["starts"] else "fable"),
             last_ok=(rev["ends"][-1][1] == "0") if rev["ends"] else None,
             detail=("Full weekly sitting" if rev["starts"] and rev["starts"][-1][1] == "FULL" else "Daily sitting"),
             next=nxt("review.sh", skip="--retry")),
        pack(aud, id="auditor", name="Auditor", role="Checks the others' work against the goal every night.",
             model=(aud["starts"][-1][1] if aud["starts"] else "fable"),
             last_ok=(aud["ends"][-1][1] == "0") if aud["ends"] else None,
             detail=(f"Last verdict: {verdict.title()}" if verdict else ""), verdict=verdict,
             next=nxt("overseer.sh")),
        pack(jack, id="workers", name="Workers", role="Build and test Jack, his world and his curriculum, one order at a time.",
             detail=runs_text(jack["runs_24h"]) + (f", {jack['bad_24h']} cut short or failed" if jack["bad_24h"] else (", all finished cleanly" if jack["runs_24h"] else "")),
             next=min(filter(None, [nxt("ladder_loop.sh jack"), nxt("gap_fill.sh")]), default=None)),
        pack(system, id="repair", name="Repair crew", role="Work on the machine itself: tools, guards, the cloud routes.",
             detail=runs_text(system["runs_24h"]), next=nxt("ladder_loop.sh system")),
        pack(lab, id="inventor", name="Inventor", role="Tries a new idea against whatever holds a seat in his brain or world.",
             detail=runs_text(lab["runs_24h"]), next=nxt("ladder_loop.sh lab")),
        pack(outsider, id="outsider", name="Outsider", role="Reads the goal and the results cold, and says what the team is getting wrong.",
             detail="Twice a week", next=nxt("ladder_loop.sh outsider")),
        pack(sco, id="scout", name="Paper scout", role="Reads new research twice a week and nominates challengers.",
             model=(sco["starts"][-1][1] if sco["starts"] else "fable"),
             last_ok=(sco["ends"][-1][1] == "0") if sco["ends"] else None,
             detail=(f"{sco['ends'][-1][2]} nominations last sweep" if sco["ends"] else ""),
             next=nxt("field_watch.sh")),
        {"id": "clerk", "name": "Clerk", "role": "Re-runs cheap tests whose code changed. No model.",
         "running": _alive("scripts/regate.sh"), "since": None, "model": None,
         "last_end": reg_ts[-1] if reg_ts else None, "last_ok": True if reg_ts else None,
         "detail": "Every two hours", "next": nxt("regate.sh")},
    ]
    paused = (REPO / ".paused").exists()
    return {"members": members, "paused": paused,
            "running": sum(1 for m in members if m.get("running"))}


# ── running now: this box and the free cloud ────────────────────────────────
WHERE = {"kaggle": "Kaggle GPU", "kaggle-cpu": "Kaggle processor", "colab": "Colab GPU"}
KAGGLE = "/data/venvs/kaggle/bin/kaggle"
KAGGLE_RECHECK_S = 540          # ask Kaggle about a running job at most every ~10 minutes
LIVE, GONE = ("running", "queued"), ("complete", "error", "cancel")


def kaggle_states(slugs: list[str]) -> dict:
    """slug -> 'running' | 'queued' | 'complete' | 'error' | 'cancel' | None.

    The submission journal knows a job was SENT; only Kaggle knows whether it
    is still running. One status call per unfinished job, cached, so the page
    costs Kaggle a handful of calls every ten minutes and nothing otherwise.
    """
    path = OUT / "kaggle_cache.json"
    try:
        cache = json.load(open(path))
    except (OSError, ValueError):
        cache = {}
    user = cache.get("_user")
    if not user and os.path.exists(KAGGLE):
        m = re.search(r"(?m)^- username:\s*(\S+)", sh([KAGGLE, "config", "view"], 15))
        user = cache["_user"] = m.group(1) if m else None
    now = NOW.timestamp()
    for slug in slugs:
        c = cache.get(slug) or {}
        if c.get("state") in GONE or now - c.get("at", 0) < KAGGLE_RECHECK_S or not user:
            continue
        txt = sh([KAGGLE, "kernels", "status", f"{user}/{slug}"], 20).lower()
        m = re.search(r"(running|queued|complete|error|cancel)", txt)
        cache[slug] = {"state": m.group(1) if m else None, "at": now}
    keep = set(slugs) | {"_user"}
    cache = {k: v for k, v in cache.items() if k in keep}
    try:
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(cache))
        os.replace(tmp, path)
    except OSError:
        pass
    return {k: (v or {}).get("state") for k, v in cache.items() if k != "_user"}


PLAIN: dict = {}     # test id -> a name in plain words (dashboard/config.json "plain_names")


def jobs(by_id: dict) -> list:
    out = []
    # ── the cloud: an attempt in the journal with no result row yet
    rows = []
    for line in read(REPO / "experiments" / "gpu_submissions.jsonl", 400_000).splitlines():
        try:
            rows.append(json.loads(line))
        except ValueError:
            continue
    done = {r.get("attempt_id") for r in rows if r.get("phase") != "attempt"}
    open_rows = []
    for r in rows:
        if r.get("phase") != "attempt" or r.get("attempt_id") in done or not r.get("ts"):
            continue
        t0 = dt.datetime.fromtimestamp(r["ts"], UTC)
        if (NOW - t0).total_seconds() > 13 * 3600:      # a free session cannot outlive 12 h
            continue
        prefix = {"kaggle-cpu": "jack-cpu-", "kaggle": "jack-ladder-"}.get(r.get("backend"))
        open_rows.append((r, t0, f"{prefix}{int(r['ts'])}" if prefix else None))
    states = kaggle_states([slug for _, _, slug in open_rows if slug])
    cloud_tests = set()
    for r, t0, slug in open_rows:
        age_h = (NOW - t0).total_seconds() / 3600
        est = float(r.get("est_hours") or 0) or None
        watched = bool(r.get("pid") and os.path.exists(f"/proc/{r['pid']}"))
        state = states.get(slug) if slug else None
        if state in GONE:
            continue
        if state not in LIVE and not (watched or (est and age_h < 1.2 * est)):
            continue                                   # Kaggle could not say, and nothing here vouches for it
        spec = r.get("spec") or ""
        cloud_tests.add(spec)
        out.append({"where": WHERE.get(r.get("backend"), r.get("backend")),
                    "name": PLAIN.get(spec) or getattr(by_id.get(spec), "title", None) or "Lives for a contest",
                    "test": spec or None, "since": iso(t0), "est_hours": est,
                    "progress": round(min(0.97, age_h / est), 2) if est else None,
                    "state": "waiting to start" if state == "queued" else "running",
                    "confirmed": state in LIVE})
    # ── this box: one row per process tree that is actually computing
    procs = {}
    for line in sh(["ps", "-eo", "pid,ppid,etimes,pcpu,args"]).splitlines()[1:]:
        f = line.split(None, 4)
        if len(f) == 5 and "experiments" in f[4] and "dashboard_status" not in f[4]:
            procs[int(f[0])] = {"ppid": int(f[1]), "age": int(f[2]), "cpu": float(f[3]), "args": f[4]}
    kids: dict[int, list] = {}
    for pid, p in procs.items():
        kids.setdefault(p["ppid"], []).append(pid)

    def tree(pid):
        yield pid
        for k in kids.get(pid, []):
            yield from tree(k)

    for pid, p in procs.items():
        if p["ppid"] in procs:
            continue                                   # not the root of its tree
        members = [procs[x] for x in tree(pid)]
        if max(m["cpu"] for m in members) < 5.0:
            continue                                   # a watcher waiting on the cloud, not work
        blob = " ".join(m["args"] for m in members)
        m = re.search(r"-m experiments\.run (\S+)", blob)
        spec = by_id.get(m.group(1)) if m else None
        if spec is not None and spec.id in cloud_tests:
            continue
        label = re.search(r"wrap detached:(\S+?)(?:\.log)?\s", blob + " ") or re.search(r"-m experiments\.(?:tests\.)?(\w+)", blob)
        out.append({"where": "This box",
                    "name": (PLAIN.get(spec.id) if spec else None) or getattr(spec, "title", None)
                            or (label.group(1).replace("_", " ").capitalize() if label else "A test"),
                    "test": spec.id if spec else None,
                    "since": iso(NOW - dt.timedelta(seconds=p["age"])),
                    "est_hours": None, "progress": None, "state": "running", "confirmed": True})
    return out


# ── resources ───────────────────────────────────────────────────────────────
def _const(path: Path, name: str, default: float) -> float:
    m = re.search(rf"(?m)^{name}\s*=\s*([0-9.]+)", read(path))
    return float(m.group(1)) if m else default


def resources(job_list: list) -> dict:
    out: dict = {}
    usage_rows = []
    for line in read(LOGS / "usage_ledger.jsonl", 20_000).splitlines():
        try:
            usage_rows.append(json.loads(line))
        except ValueError:
            continue
    if usage_rows:
        last = usage_rows[-1]
        stop = re.search(r"USAGE_STOP=\$\{JACK_USAGE_STOP:-(\d+)\}", read(REPO / "scripts" / "lib_usage.sh"))
        fable = next((r for r in reversed(usage_rows) if r.get("model") == "fable" and r.get("model_pct") is not None), None)
        out["claude"] = {"pct": last.get("pct"), "stop": int(stop.group(1)) if stop else 95,
                         "resets": last.get("resets"), "read": last.get("ts"),
                         "fable_pct": fable.get("model_pct") if fable else None}
    day = NOW.strftime("%Y-%m-%d")
    try:
        cpu = json.load(open(REPO / "experiments" / "cpu_budget.json"))
        out["cpu_day"] = {"used_s": round(cpu.get("days", {}).get(day, {}).get("used_s", 0.0)),
                          "ceiling_s": round(_const(REPO / "experiments" / "cpu_budget.py", "CPU_DAY_CEILING_S", 57600))}
    except (OSError, ValueError):
        pass
    try:
        gpu = json.load(open(REPO / "experiments" / "gpu_budget.json"))
        y, w, _ = NOW.isocalendar()
        wk = gpu.get("weeks", {}).get(f"{y}-W{w:02d}", {})
        out["gpu_week"] = {"kaggle_h": round(wk.get("kaggle", 0.0) + wk.get("kaggle_failed", 0.0), 1),
                           "colab_h": round(wk.get("colab", 0.0) + wk.get("colab_failed", 0.0), 1),
                           "kaggle_cap_h": _const(REPO / "experiments" / "gpu.py", "KAGGLE_WEEKLY_HOURS", 30.0)}
    except (OSError, ValueError):
        pass
    out["cloud_cpu"] = {"running": sum(1 for j in job_list if j["where"] == "Kaggle processor"), "cap": 5}
    out["disk"] = []
    for label, path in (("System disk", "/"), ("Data disk", "/data")):
        try:
            u = shutil.disk_usage(path)
            out["disk"].append({"label": label, "used_gb": round((u.total - u.free) / 2**30, 1),
                                "total_gb": round(u.total / 2**30, 1), "free_gb": round(u.free / 2**30, 1)})
        except OSError:
            pass
    try:
        out["load"] = {"now": round(os.getloadavg()[0], 2), "cores": os.cpu_count(),
                       "ceiling": _const(REPO / "experiments" / "cpu_budget.py", "LOAD_CEILING", 6.0)}
        mem = dict(re.findall(r"(?m)^(\w+):\s+(\d+) kB", read(Path("/proc/meminfo"))))
        out["memory"] = {"total_gb": round(int(mem["MemTotal"]) / 2**20, 1),
                         "used_gb": round((int(mem["MemTotal"]) - int(mem["MemAvailable"])) / 2**20, 1)}
    except (OSError, KeyError, ValueError):
        pass
    return out


# ── history ─────────────────────────────────────────────────────────────────
def ladder_history(days: int = 400) -> list:
    last: dict[str, int] = {}
    for m in re.finditer(r"(?m)^(\d{4}-\d\d-\d\d)T\S+ iteration (?:start — lane \w+, (\d+)/\d+|end rc=\S+ — \d+ -> (\d+))",
                         read(LOGS / "ladder.log")):
        last[m.group(1)] = int(m.group(2) or m.group(3))
    keys = sorted(last)[-days:]
    return [{"d": k, "v": last[k]} for k in keys]


def score_history(ab: dict, passing: int, total: int) -> list:
    """One row a day, written here so the page can show the score moving."""
    path = OUT / "score_history.jsonl"
    rows = []
    for line in read(path).splitlines():
        try:
            rows.append(json.loads(line))
        except ValueError:
            continue
    today = NOW.strftime("%Y-%m-%d")
    row = {"d": today, "overall": ab["overall"], "overall_world": ab.get("overall_world"), "passing": passing, "total": total,
           "aspects": {a["id"]: a["pct"] for a in ab["aspects"]}}
    rows = [r for r in rows if r.get("d") != today] + [row]
    tmp = path.with_suffix(".tmp")
    tmp.write_text("".join(json.dumps(r) + "\n" for r in rows[-400:]))
    os.replace(tmp, path)
    return [{"d": r["d"], "v": r["overall"]} for r in rows[-60:]]


# ── looking back: one entry a day ───────────────────────────────────────────
def journal() -> list:
    """The Director's log line for every day since August, and — from the day
    this page was built — that day's plain-words note, kept here because
    PROGRESS.md itself is rewritten daily. A few kilobytes a month."""
    rows: dict = {}
    for line in read(REPO / "docs" / "PROGRESS_LOG.md").splitlines():
        m = re.match(r"\|\s*(\d{4}-\d\d-\d\d)\s*\|\s*\w+\s*\|\s*(\d+)/(\d+)\s*\|[^|]*\|[^|]*\|[^|]*\|\s*(.+?)\s*\|?\s*$", line)
        if m:
            rows[m.group(1)] = {"d": m.group(1), "passing": int(m.group(2)), "total": int(m.group(3)),
                                "line": clean(m.group(4))[:340], "paras": []}
    path, kept = OUT / "journal.jsonl", {}
    for line in read(path).splitlines():
        try:
            r = json.loads(line)
            kept[r["d"]] = r
        except (ValueError, KeyError):
            continue
    h = headline()
    m = re.match(r"(\d{4}-\d\d-\d\d)", h.get("written") or "")
    if m and h["paras"] and kept.get(m.group(1), {}).get("paras") != h["paras"]:
        kept[m.group(1)] = {"d": m.group(1), "paras": h["paras"]}
        tmp = path.with_suffix(".tmp")
        tmp.write_text("".join(json.dumps(kept[k], ensure_ascii=False) + "\n" for k in sorted(kept)[-800:]))
        os.replace(tmp, path)
    for d, r in kept.items():
        rows.setdefault(d, {"d": d, "passing": None, "total": None, "line": "", "paras": []})["paras"] = r["paras"][:3]
    return sorted(rows.values(), key=lambda r: r["d"], reverse=True)[:120]


# ── worth a look: what the owner might want to act on ───────────────────────
def attention(st: dict) -> list:
    out = []

    def add(level, text, tab="machine"):
        out.append({"level": level, "text": text, "tab": tab})

    if st["team"].get("paused"):
        add("crit", "All agents are paused. Nothing new starts until they are resumed.")
    c = st["resources"].get("claude")
    if c and c.get("pct") is not None:
        if c["pct"] >= c["stop"]:
            add("crit", f"The agents have stopped: this week's Claude usage reached {c['pct']}%. It resets {c.get('resets') or 'at the weekly reset'}.")
        elif c["pct"] >= c["stop"] - 15:
            add("warn", f"Claude usage is at {c['pct']}% for the week. The agents pause at {c['stop']}%; it resets {c.get('resets') or 'at the weekly reset'}.")
    for m in st["team"]["members"]:
        if m["id"] == "auditor" and (m.get("verdict") or "").upper() not in ("", "ON TRACK"):
            add("warn", f"The Auditor's last verdict was {m['verdict'].title()}. The Director's note on the Jack page says what is being done.", "jack")
        if m.get("last_ok") is False and not m.get("running"):
            add("info", f"{m['name']}: the last run was cut short or failed.")
    year = NOW.year
    for i, o in enumerate(st["orders"]):
        m = re.match(r"(\d\d)-(\d\d) (\d\d):(\d\d)", o["state"])
        when = None
        if m:
            try:
                when = dt.datetime(year, int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4)), tzinfo=UTC)
                if when > NOW + dt.timedelta(days=2):
                    when = when.replace(year=year - 1)
            except ValueError:
                when = None
        asked = re.search(r"owner request (\d\d)-(\d\d)", o["title"])
        if asked and not re.search(r"\bDONE\b", o["state"]):
            try:
                since = dt.datetime(year, int(asked.group(1)), int(asked.group(2)), tzinfo=UTC)
                days = (NOW - since).days
                title = re.sub(r"\s*\(owner request[^)]*\)", "", o["title"]).split(" — ")[0]
                add("warn" if days >= 2 else "info",
                    f"You asked for this {days} day{'s' if days != 1 else ''} ago and it is not done: {title}.")
            except ValueError:
                pass
        elif i == 0 and when and (NOW - when).total_seconds() > 24 * 3600 and not re.search(r"\bDONE\b", o["state"][:40]):
            add("warn", f"The first job has not moved for {int((NOW - when).total_seconds() // 3600)} hours: {o['title']}.")
    for d in st["resources"].get("disk", []):
        if d["label"].startswith("System") and d["free_gb"] < 4.5:
            add("warn", f"The system disk has {d['free_gb']} GB free. The workers stop below 3 GB.")
    if st.get("errors"):
        add("info", f"{len(st['errors'])} part(s) of this page could not be read this time.")
    order = {"crit": 0, "warn": 1, "info": 2}
    return sorted(out, key=lambda a: order[a["level"]])


# ── films kept to look back on, inside a fixed amount of disk ───────────────
VIDEO_DIR = Path(os.environ.get("JACK_VIDEO_DIR", "/data/jack-video"))
ARCHIVE_CAP_MB = 400      # the most disk the film archive may ever hold
KEEP_ALL_DAYS = 14        # every film this recent is kept
WEEKLY_UNTIL_DAYS = 90    # older than that: one a week; older still: one a month


def archive_films() -> list:
    """Each new latest.mp4 is linked (same disk, no copy) under a dated name.
    Recent films are all kept, older ones thin to one a week, then one a month,
    and the whole archive never exceeds ARCHIVE_CAP_MB — oldest go first."""
    arc = OUT / "media" / "archive"
    for ext in ("mp4", "webm"):
        src = VIDEO_DIR / f"latest.{ext}"
        if not (src.is_file() and src.stat().st_size > 0):
            continue
        arc.mkdir(parents=True, exist_ok=True)
        st = src.stat()
        if any(f.stat().st_ino == st.st_ino or (f.stat().st_size, int(f.stat().st_mtime)) == (st.st_size, int(st.st_mtime))
               for f in arc.glob(f"*.{ext}")):
            break
        stamp = dt.datetime.fromtimestamp(st.st_mtime, UTC).strftime("%Y%m%d-%H%M%S")
        dst = arc / f"{stamp}.{ext}"
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)
        side = VIDEO_DIR / "latest.json"
        if side.is_file():
            shutil.copyfile(side, arc / f"{stamp}.json")
        break
    if not arc.is_dir():
        return []
    films = sorted((f for f in arc.iterdir() if f.suffix in (".mp4", ".webm")), key=lambda f: f.name, reverse=True)

    def drop(f):
        f.unlink(missing_ok=True)
        f.with_suffix(".json").unlink(missing_ok=True)

    seen, keep = set(), []
    for f in films:                                   # newest first
        try:
            t = dt.datetime.strptime(f.stem, "%Y%m%d-%H%M%S").replace(tzinfo=UTC)
        except ValueError:
            continue
        age = (NOW - t).days
        if age <= KEEP_ALL_DAYS:
            keep.append(f)
            continue
        y, w, _ = t.isocalendar()
        slot = ("w", y, w) if age <= WEEKLY_UNTIL_DAYS else ("m", t.year, t.month)
        if slot in seen:
            drop(f)
        else:
            seen.add(slot)
            keep.append(f)
    total = sum(f.stat().st_size for f in keep)
    while keep and total > ARCHIVE_CAP_MB * 2**20:
        f = keep.pop()                                # the oldest
        total -= f.stat().st_size
        drop(f)
    out = []
    for f in keep[:60]:
        info = {}
        try:
            info = json.loads(f.with_suffix(".json").read_text(errors="replace"))
        except (OSError, ValueError):
            pass
        t = dt.datetime.strptime(f.stem, "%Y%m%d-%H%M%S").replace(tzinfo=UTC)
        out.append({"video": f"media/archive/{f.name}", "made": iso(t),
                    "info": f"media/archive/{f.stem}.json" if f.with_suffix(".json").exists() else None,
                    "title": clean(str(info.get("title") or ""))[:120] if isinstance(info, dict) else "",
                    "summary": clean(str(info.get("summary") or ""))[:400] if isinstance(info, dict) else "",
                    "mb": round(f.stat().st_size / 2**20, 1)})
    return out


# ── watch: a recorded life, if a worker has made one ────────────────────────
def watch() -> dict:
    """ORDERS S6 writes the newest recorded life to /data/jack-video/latest.mp4.
    The page can only play what sits beside it, so the file is linked (same
    disk, no copy) into the dashboard's own folder. Nothing is rendered here."""
    media = OUT / "media"
    for src_dir in (VIDEO_DIR, media):
        for ext in ("mp4", "webm"):
            src = src_dir / f"latest.{ext}"
            if not (src.is_file() and src.stat().st_size > 0):
                continue
            dst = media / f"latest.{ext}"
            if src != dst:
                media.mkdir(exist_ok=True)
                a, b = src.stat(), (dst.stat() if dst.exists() else None)
                if b is None or (a.st_ino, a.st_size, a.st_mtime) != (b.st_ino, b.st_size, b.st_mtime):
                    tmp = dst.with_suffix(".tmp")
                    tmp.unlink(missing_ok=True)
                    try:
                        os.link(src, tmp)
                    except OSError:
                        shutil.copy2(src, tmp)
                    os.replace(tmp, dst)
            note = ""
            for side in (src_dir / "latest.txt", src_dir / "latest.md"):
                if side.is_file():
                    note = clean(side.read_text(errors="replace"))[:300]
                    break
            # The description that travels with the film (owner 11-15: "will it
            # have description thingy"). Written by whoever renders the video,
            # from the life's own record; this script only passes it on.
            info = {}
            try:
                info = json.loads((src_dir / "latest.json").read_text(errors="replace"))
            except (OSError, ValueError):
                pass
            if not isinstance(info, dict):
                info = {}

            def txt(x, n):
                return clean(str(x))[:n] if isinstance(x, (str, int, float)) else ""

            facts = [{"label": txt(f.get("label"), 40), "value": txt(f.get("value"), 40)}
                     for f in (info.get("facts") or []) if isinstance(f, dict)][:8]
            events = [{"t": round(float(e["t"]), 1), "label": txt(e.get("label"), 120), "at": txt(e.get("at"), 40)}
                      for e in (info.get("events") or [])
                      if isinstance(e, dict) and isinstance(e.get("t"), (int, float))][:60]
            return {"video": f"media/latest.{ext}", "note": note,
                    "made": iso(dt.datetime.fromtimestamp(src.stat().st_mtime, UTC)),
                    "title": txt(info.get("title"), 120), "summary": txt(info.get("summary"), 900),
                    "facts": facts, "events": events, "source": txt(info.get("source"), 160)}
    return {"video": None, "made": None, "note": "", "title": "", "summary": "",
            "facts": [], "events": [], "source": ""}


# ── assemble ────────────────────────────────────────────────────────────────
def build() -> dict:
    by_id, results = load_tests()
    counts: dict[str, int] = {}
    for sid in by_id:
        s = (results.get(sid) or {}).get("status") or "NONE"
        counts[s] = counts.get(s, 0) + 1
    passing, total = counts.get("PASS", 0), len(by_id)
    ab = guarded(lambda: abilities(by_id, results), {"overall": 0, "aspects": [], "promises": 0, "promises_started": 0, "proofs": 0, "proofs_passed": 0, "senses_for_m5": {}})
    cfg = guarded(lambda: json.load(open(CFG / "config.json")), {})
    PLAIN.update(cfg.get("plain_names", {}))
    job_list = guarded(lambda: jobs(by_id), [])
    status = {
        "generated": iso(NOW), "refresh_s": 120,
        "head": sh(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"]).strip() or None,
        "ladder": {"passing": passing, "total": total, "failed": counts.get("FAIL", 0),
                   "no_verdict": counts.get("VOID", 0) + counts.get("ERROR", 0) + counts.get("BLOCKED", 0),
                   "not_run": counts.get("NONE", 0), "history": guarded(ladder_history, [])},
        "abilities": ab,
        "score_history": guarded(lambda: score_history(ab, passing, total), []),
        "vitals": guarded(lambda: vitals(by_id, results), []),
        "lives": guarded(lambda: lives(results), None),
        "milestones": guarded(lambda: milestones(by_id, results, ab), []),
        "headline": guarded(headline, {"written": None, "paras": []}),
        "orders": guarded(orders, []),
        "recent": guarded(lambda: recent(by_id, results), []),
        "team": guarded(team, {"members": [], "paused": False, "running": 0}),
        "jobs": job_list,
        "resources": guarded(lambda: resources(job_list), {}),
        "watch": guarded(watch, {"video": None, "made": None, "note": "", "title": "", "summary": "", "facts": [], "events": [], "source": ""}),
        "chat": {"url": cfg.get("chat_url") or None, "label": cfg.get("chat_label") or None,
                 "terminal": cfg.get("terminal_url") or None},
        "site": cfg.get("site_url") or None,
        "journal": guarded(journal, []),
        "errors": ERRORS,
    }
    films = guarded(archive_films, [])
    status["watch"]["archive"] = films
    status["watch"]["archive_mb"] = round(sum(f["mb"] for f in films), 1)
    status["watch"]["archive_cap_mb"] = ARCHIVE_CAP_MB
    status["attention"] = guarded(lambda: attention(status), [])
    return status


SKELETON = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<meta name="robots" content="noindex,nofollow">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-title" content="Jack">
<meta name="theme-color" content="#f2f3f7" media="(prefers-color-scheme: light)">
<meta name="theme-color" content="#000000" media="(prefers-color-scheme: dark)">
<link rel="manifest" href="manifest.webmanifest">
<link rel="apple-touch-icon" href="icon-180.png">
<link rel="icon" type="image/png" sizes="192x192" href="icon-192.png">
<style>:root{color-scheme:light;padding:env(safe-area-inset-top,0px) 0 env(safe-area-inset-bottom,0px)}
body{margin:0;font:14px system-ui,sans-serif}img{max-width:100%}[hidden]{display:none!important}</style>
</head><body>
__PAGE__
</body></html>
"""


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    status = build()
    blob = json.dumps(status, ensure_ascii=False, separators=(",", ":"))
    try:        # the page names jobs, never the account they run under
        handle = json.load(open(OUT / "kaggle_cache.json")).get("_user")
    except (OSError, ValueError):
        handle = None
    if handle:
        blob = re.sub(re.escape(handle) + r"/?", "", blob, flags=re.I)
    page = read(CFG / "page.html")
    inlined = page.replace("__SNAPSHOT__", blob.replace("</", "<\\/"))
    # The preview copy lives on another site, so its link to the terminal must be whole.
    away = status
    if status["chat"].get("terminal") and status.get("site"):
        away = dict(status, chat=dict(status["chat"], terminal=status["site"].rstrip("/") + "/" + status["chat"]["terminal"]))
    away_blob = json.dumps(away, ensure_ascii=False, separators=(",", ":"))
    if handle:
        away_blob = re.sub(re.escape(handle) + r"/?", "", away_blob, flags=re.I)
    preview = page.replace("__SNAPSHOT__", away_blob.replace("</", "<\\/"))
    for name, text in (("status.json", blob), ("index.html", SKELETON.replace("__PAGE__", inlined)),
                       ("artifact.html", preview), ("artifact-status.json", away_blob)):
        tmp = OUT / (name + ".tmp")
        tmp.write_text(text)
        os.replace(tmp, OUT / name)
    if PUBLISH:
        try:
            publish(Path(PUBLISH))
        except OSError as e:
            ERRORS.append(f"publish: {e}")
    if ERRORS:
        print(f"{iso(NOW)} dashboard: {len(ERRORS)} section(s) failed: {'; '.join(ERRORS)}", file=sys.stderr)
    return 0


def publish(pub: Path) -> None:
    (pub / "media").mkdir(parents=True, exist_ok=True)
    for name in ("index.html", "status.json"):
        tmp = pub / (name + ".tmp")
        shutil.copyfile(OUT / name, tmp)
        os.replace(tmp, pub / name)
    live = set()
    for f in (OUT / "media").glob("latest.*") if (OUT / "media").is_dir() else []:
        if f.suffix not in (".mp4", ".webm"):
            continue
        live.add(f.name)
        dst = pub / "media" / f.name
        a, b = f.stat(), (dst.stat() if dst.exists() else None)
        if b is None or (a.st_ino, a.st_size, a.st_mtime) != (b.st_ino, b.st_size, b.st_mtime):
            tmp = dst.with_suffix(".tmp")
            tmp.unlink(missing_ok=True)
            try:
                os.link(f, tmp)
            except OSError:
                shutil.copy2(f, tmp)
            os.replace(tmp, dst)
    for old in (pub / "media").glob("latest.*"):
        if old.name not in live:
            old.unlink(missing_ok=True)
    # the film archive, mirrored by hard link; what was thinned out goes here too
    arc, parc = OUT / "media" / "archive", pub / "media" / "archive"
    have = {f.name for f in arc.iterdir()} if arc.is_dir() else set()
    if have or parc.is_dir():
        parc.mkdir(parents=True, exist_ok=True)
        for name in have:
            dst = parc / name
            if not dst.exists():
                try:
                    os.link(arc / name, dst)
                except OSError:
                    shutil.copy2(arc / name, dst)
        for old in parc.iterdir():
            if old.name not in have:
                old.unlink(missing_ok=True)
    # The dashboard's sign-in page, manifest and icons (dashboard/app).
    app = CFG / "app"
    if app.is_dir():
        for src in sorted(f for f in app.iterdir() if f.is_file()):
            dst = pub / src.name
            if (not dst.exists() or dst.stat().st_size != src.stat().st_size
                    or dst.stat().st_mtime < src.stat().st_mtime):
                tmp = dst.with_suffix(dst.suffix + ".tmp")
                shutil.copyfile(src, tmp)
                os.replace(tmp, dst)
    # The terminal page and the one library it loads, both from this server.
    www, vendor = CFG / "terminal" / "www", OUT / "vendor"
    if www.is_dir() and vendor.is_dir():
        (pub / "terminal").mkdir(exist_ok=True)
        for src in sorted(f for d in (www, vendor) for f in d.iterdir() if f.is_file()):
            dst = pub / "terminal" / src.name
            if (not dst.exists() or dst.stat().st_size != src.stat().st_size
                    or dst.stat().st_mtime < src.stat().st_mtime):
                tmp = dst.with_suffix(dst.suffix + ".tmp")
                shutil.copyfile(src, tmp)
                os.replace(tmp, dst)


if __name__ == "__main__":
    raise SystemExit(main())
