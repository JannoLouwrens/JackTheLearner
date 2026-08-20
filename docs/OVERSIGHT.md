# OVERSIGHT — 21st audit, 2026-08-20 07:00 UTC

## VERDICT: ON TRACK — but the loop has been idling for three hours on work that is already dead

Ledger integrity is spotless on every hard check I can run, and it is the
cleanest reading this audit has produced: 82/82 PASS name a live commit and a
present implementation, 80/82 declare a control whose metrics were actually
recorded, **zero** PASS stamped from a dirty tree, **zero** seed shortfalls,
**zero** stale PASS entries, and `run verify` re-judges the record with 0
failures on all five probes. No silent loosening in 7 days across 35 commits
touching the registry and tests. The day's work — TA.02 (taste), SM.02 (smell),
the reattach guard — traces cleanly to GOAL.md. Nothing here is drift.

The finding that outranks all of it is not about the ledger's honesty but about
the loop's *liveness*, and it is live right now:

> **The builder has spent its last three iterations (04:07, 05:07, 06:07)
> waiting to be "notified" by three background CPU checks that are dead, hold
> no process, and have written zero bytes.** The loop is not blocked, not
> paused, and not out of credits. It is waiting on a message that can never
> arrive, and it did not check.

This is the fourth instance in 24 hours of the same class (a detached job dying
silently) and the first one that was not caught inside the launching iteration.
LESSONS.md already carries both rules that would have prevented it — "verify
the artifact, not the exit code" (line 92) and "detach it at birth… ask who its
parent is and whether the parent is guaranteed to outlive the wait" (line 3769)
— written by this builder, about this failure.

---

## 0. Coverage — is the ladder the RIGHT ladder?

`python -m experiments.coverage` exits **0**. **Zero commitments with no
declared spec.** The tool's own guard holds; no commitment is invisible.

**15 of 23 commitments have specs but nothing passing** — unchanged since
yesterday (taste crossed at 23:15 on 08-19; smell has not yet). The eight that
carry a claim-kind PASS are: damage/nociception, taste, memory-across-lives,
generality, language (parent), hearing, curiosity, one-brain/unison.

Two numbers from that table deserve to be read out loud, because they are the
thesis and not a side quest:

| commitment | declared specs | claim PASS |
|---|---|---|
| one brain / all senses in unison | **21** | **1** |
| curiosity | **12** | **1** |
| sight | 5 | **0** (3 support specs pass; no claim) |
| fast/slow | 6 | **0** |

Sight having five specs, three passing support specs (T2.03, PG.6, PG.9) and
**zero claim-kind passes** is the sharpest instance of the pattern the coverage
tool exists to expose: a commitment can look busy and still have nothing
falsifiable behind it. This is not a new finding — the 20th audit decomposed it
and established that most of these are *not implemented* rather than *lost* —
and no action is owed this iteration. It is restated because the honest summary
in §8 turns on it.

---

## 1. Integrity of the ledger — CLEAN, no findings

90 rows: **82 PASS, 4 FAIL, 4 VOID, 0 ERROR, 0 NOT_RUN**.

Checked mechanically over all 82 PASS entries:

| check | result |
|---|---|
| `commit` field present | 82/82 |
| commit still resolves in git (`cat-file -e`) | **82/82** |
| implementation file present in `experiments/tests/` | **82/82** |
| spec present in `registry.BY_ID` | **82/82** |
| declares a `control` | 80/82 (T0.01, T0.10 declare none **by design**) |
| control declared **and** `control_metrics` non-empty | **80/80** |
| ran with ≥ the spec's declared seed count | **82/82** |
| stamped DIRTY / from a modified tree | **0** |

`run verify`, which re-judges each entry from the record alone, reports
**0 verdicts that no longer re-derive, 0 gates that ignore their control,
0 controls declared but never run, 0 gates that could not be replayed, 0
entries that could not be audited.** T0.18 self-excludes correctly (a spec
cannot re-judge its own entry).

The stale set is **6 entries and not one is a PASS**: T2.05 (VOID), LC.03
(VOID), BA.02 (VOID), T2.02 (VOID, pre-`impl_sha` content check), T3.07 (FAIL),
T4.02 (FAIL). This is exactly the declared lower-priority tail the 19th audit's
RANK 1 left behind after closing the PASS cascade. **No capability on this
ledger is currently claimed by code that has since changed.**

Yesterday's dirty stamp on T0.24 was cleared as promised, and the four specs
staled by the `gpu.py`/`protocol.py` edits (T0.12, T0.17, T0.27, T0.24) were all
re-run PASS from a clean tree in the same iteration. That is the right discipline
and it was followed without being asked.

---

## 2. Thresholds and controls over time — NO SILENT LOOSENING

`git log -p --since="7 days ago"` over `registry.py`, `registry_expansion.py`
and `experiments/tests/`: **35 commits**. I scanned every removed line carrying a
constant, an assertion, a comparison operator or a seed count. Five commits move
something that already existed; I read each one against its message.

| commit | change | verdict |
|---|---|---|
| `265e683` | `ALIEN_MIN_DIST` 1.5 → **2.0** | **STRENGTHENED**, message quotes the pilot min_dists (2.6–3.7) that made it constructible |
| `1861b18` | BA.02's five `[PILOT-FINAL]` gates de-tagged | **values byte-identical** (0.60 / 0.80 / 0.20 / 0.50 / 0.20); only the comments changed, to record where the pilot landed inside each gate |
| `c030106` | `seeds=3` line reflowed | cosmetic; seed count unchanged |
| `7112515` | LC.03 `Budget.CPU_LONG` → `CPU_DAYS` | a cost class, not a gate |
| `87590a4` | LC.03 control (e): `darkroom_margin <= -SIGMA_GATE` **inverted** to `> -SIGMA_GATE` | **examined — justified, and not a loosening of any claim** |

The last one is the only genuine control-side change in the window and it earns
a paragraph rather than a table row. The darkroom control was pre-registered to
come out strongly *negative*; the seed-90 pilot measured it strongly *positive*
(`life_gain` margin **+49.7 s** over its paired null, mean life 183.5 s — it
**learned passivity** and prospered on a length ruler, which is T2.08's known
inversion and which LC.03's own docstring had flagged as a risk). The side was
flipped **before registration**, in the registering commit, with the measurement
quoted, the old side left in git history per the T1.02 precedent, and the
message stating explicitly that **claim gates are UNMOVED**. LC.03 currently
reads VOID, so no ledger claim rests on it. This is the amend-openly path
working as designed, not silent loosening.

**No `_check` gained an `or`. No assertion was deleted. No seed count was
reduced. No claim gate moved in the loosening direction in seven days.**

---

## 3. Drift from the goal — none in the work; the gap is in what is *not* worked on

What the builder did since 07:00 yesterday (42 commits), and the GOAL.md
sentence each serves:

| work | GOAL.md sentence |
|---|---|
| TA.02 — one-trial conditioned taste aversion, **PASS** | *"gustation drives conditioned taste aversion, one-trial learning with long delay tolerance, the fastest learning in biology"* (GOAL.md:46–48) — and fast/slow axis 3, "fast SPECIALISED learners" (GOAL.md:246) |
| SM.02 — smell finds occluded food (in pilot) | *"olfaction finds food, fire and decay at a distance and through occlusion — the sense that works when sight fails"* (GOAL.md:45–46) |
| Reattach guard (`kernel_sha256`), T0.24 P7 | *"protects the honesty of watching what happens when the three meet"* (GOAL.md:8) |
| `spec_phase` receipt attribution | same — compute honesty is a precondition for the ledger meaning anything |
| Staleness re-runs (T0.12/T0.17/T0.27/T0.24) | *"every capability claimed only by an experiment that could have failed"* (GOAL.md:58) |

**Nothing in the last day serves no GOAL.md sentence.** Both new specs are
constitutional senses that had zero claims, picked by the standing rule. The
reattach guard closed the 20th audit's RANK 1 in the same iteration it was
raised. This is a good day's work and I am not going to manufacture a
complaint about it.

The converse question is where the discomfort lives, and §0 has the numbers:
curiosity carries **1 claim PASS across 12 specs**, one-brain/unison **1 across
21**. Those are the two claims GOAL.md is actually about, and they are the two
most likely to be quietly deferred in favour of a sense that can be certified in
a day. I am not calling the last day drift — the picks were correct under the
standing rule and both closed real holes. I am recording that **the ladder is
getting broader faster than it is getting deeper**, and that §8 is where that has
to be answered.

---

## 4. Is the builder alive and productive? — ALIVE, PRODUCTIVE FOR 18 HOURS, THEN STALLED

**24 iterations** in the last 24 h. **22 ended `rc=0`**, one `rc=1` (11:07),
one `rc=124` (14:57, timeout). PASS delta over the window: **80 → 82**.

The honest decomposition of that +2: one is a genuinely new claim (**TA.02**,
taste, the first zero-pass commitment to cross in this window); the rest of the
window's green went to **re-certifications** of claims that already existed
(T2.03, T2.04) and to closing the stale cascade. That is legitimate,
necessary work — a stale PASS is a lie the ledger is telling — but it should be
read as *repair*, not *advance*.

**The stall.** The last six iterations produced **82 → 82**, and the last three
are the finding:

| iteration | ended | reported | reality |
|---|---|---|---|
| 04:07 | 04:15 | *"full-budget CPU learnability checks … are running in the background — about 10 minutes total"* | `/data/sm02_geo_check.log` created 04:13, **0 bytes** |
| 05:07 | 05:10 | *"Waiting on the two CPU checks — I'll be notified when both have written their `ratio` lines."* | `/data/sm02_geo_occ.log`, `/data/sm02_geo_vis.log` created 05:08, **0 bytes each** |
| 06:07 | 06:12 | **empty journal entry** — no summary written at all | nothing relaunched, nothing noticed |

Evidence that these processes are definitively dead, not slow:

1. `pgrep -af sm02` → **no process.** Nothing is running.
2. All three logs are **0 bytes**, hours after creation.
3. Decisive: the script's final act is
   `json.dump(out, open("/data/sm02_learnability_{cond}.json","w"))`. Those two
   files exist — but with mtimes **03:13 and 03:15**, and they contain the
   **Euclidean** results (`vis` ratio 0.7166, `occ` ratio 1.0). If either
   geodesic run had reached its last line it would have overwritten them. It did
   not. The geodesic checks never produced a number.
4. The script itself is healthy: `python -c "import
   experiments.sm02_learnability_check"` runs to the expected
   `IndexError: sys.argv[1]` — so this is not an import failure of the kind that
   killed the SH.01 probe yesterday.

**Cost so far: three loop-hours and three iterations of Claude credits, spent
waiting on nothing** — plus SM.02 held at the gate while **26.35 of 30 Kaggle
hours expire on Sunday** (§5).

**Why it happened (leading hypothesis, offered as a lead, not a conclusion).**
The 03:0x checks *succeeded*, writing their JSON at 03:13/03:15 — **after** the
03:07 iteration ended at 03:13:04. So they outlived their session; they were
properly detached. The 04:13 and 05:08 launches did not. Either (a) they were
started as session-children rather than `setsid`-detached and were reaped when
the iteration ended ~2 minutes later, or (b) they were detached but launched
with `> log` and no `2>&1`, so a traceback went to a severed stderr and left
exactly the 0-byte log observed. Both are consistent with every fact above;
(a) is favoured by the two-minute gap between launch and iteration end in both
cases. The builder should determine which before re-launching, because the two
have different fixes.

**This class is recurring.** Yesterday at 10:42, commit `a51686c`: *"SH.01
handoff correction: live oracle-probe pid is 3963630 (**first two launches died
at import**)"*. That was caught inside the launching iteration. This one was
not, for three iterations — because the builder trusted a notification instead
of asserting on the artifact. The rule it broke is its own, LESSONS.md line 92:
*"assert on the product — the file exists … never on the absence of an error.
When automating, ask: 'if this failed right now, would anything say so?'"*
Here, nothing said so.

**Secondary observation:** 2 of the last 5 iterations (02:07, 06:07) wrote **no
journal summary at all**, both `rc=0`. An iteration that produces no journal
entry is invisible to the review organ that reads journals — and the 06:07
iteration is precisely the one that should have caught the dead checks.

---

## 5. Compute honesty — accounted, attributed, and one honestly-declared overspend

`experiments/gpu_budget.json`, week 2026-W33: **3.6473 h Kaggle used, 0.0935 h
failed. 26.35 of 30 h remain, and they expire Sunday 2026-08-23** (~65 hours
from this writing).

Every hour is attributed to a job id, and the 20th audit's B2 is **closed**:
receipts now carry `spec_phase`, and the SM.02 pilot's receipt
(`1787185633739-4170993-kaggle`) carries `spec_phase: "pilot"` — I verified it in
`gpu_submissions.jsonl`. There are **no unattributable GPU hours this week.**

W33's 3.65 h bought three PASS certificates (T2.03 0.322 h, T2.04, TA.02 0.51 h
— the last recovered for **zero new quota** via `JACK_REUSE_KERNEL` after a
harvest bug) and one pilot.

**The pilot is the honest question.** SM.02's pilot cost **1.5583 h** and
produced **no ledger entry** — by design; a pilot exists to price gates. But it
did not price gates: it measured all six DQN arms at 83–100 % timeout on all
three seeds and returned a non-learning verdict. The builder's own new LESSONS
entry (uncommitted, in the tree now) states the finding plainly: *"the
learnability half of its verdict was predictable locally, in minutes, before any
kernel was pushed."*

I want to be fair about this rather than score it as waste. The pilot **worked**
— it caught the non-learning *before* a registered run recorded a VOID, which is
exactly what `_GATES_FROZEN` was built to force. And the builder did not
rationalise the spend; it wrote the lesson against itself, then applied the new
rule twice in the same day, the second time catching its own Euclidean repair
before it could burn a second 2-hour pilot. That is the machine getting better
in the way this project is supposed to get better. **1.56 h is the tuition, it
is recorded, and the rule it bought is now in LESSONS.md.** No finding.

---

## 6. Stuck decisions — one, and it is correctly escalated

`docs/DECISIONS_NEEDED.md`: **D1 — does the 57M trunk stay in the control
path?** — open **11 days**, evidence complete, blocking T2.01 (*locomotion beats
a random policy*), which has read **FAIL since 2026-08-12T12:59 — eight days
with no attempt 3**.

The 20th audit put a full cost update on the owner's desk **six hours ago**
(00:45 UTC) with the fork stated in one line. **I am deliberately not appending
another.** Nothing material has changed since: the only moved number is the
quota (27.81 → 26.35 h remaining, still five times T2.01's 5.58 h billing, still
expiring Sunday). Re-raising an unchanged decision every six hours trains the
owner to skim the file, which costs more than the update is worth.

Checked and clean:
- **Nothing blocked on the owner is resolvable by bakeoff.** The 15th audit
  established D1 is constitutional — what the PLASTIC-ONLY decree admits — not a
  measurement question. That ruling still holds.
- **No owner-decision was quietly acted on.** D2 (does VOID block dependents?)
  was resolved by the *loop*, but only after the 11th audit explicitly ruled it
  the loop's to resolve, and it is recorded in `DECISIONS_RESOLVED.md` with its
  method and its losing arm. That is the correct path, taken correctly.

---

## 7. Bakeoff hygiene — CLEAN, no findings

`docs/DECISIONS_RESOLVED.md` holds three entries.

- **PS.01/J — VOID, and reported as VOID.** Three arms sat below the 3.0-σ
  learning gate and the record says so in its first line: *"An arm that has not
  demonstrably learned cannot arbitrate the decision."* **A VOID is not being
  treated as a verdict** — the exact failure this section exists to catch.
- **PS.01/J2 — WINNER `impact_speed`.** 10.32 σ over null, beating the
  runner-up by **2.66 σ**. Nowhere near the noise margin. Eliminated arms are
  listed by name, and the `screen` gate mode carries a written rationale for why
  these arms are observables rather than learners.
- **D2 — WINNER BLOCK.** Not run through `run_bakeoff`, and the entry explains
  why in its own method section (two readings of a dependency graph: no seeds,
  no null, nothing that could have failed to train). The metric — retraction
  exposure — was pre-stated, measured by replaying the ledger's recorded history
  (exposure 9 vs 0, benefit 0), the losing semantics is recorded, and a
  **re-open trigger** is attached to the quantity the decision rests on.

**No decision made without a learning gate. No VOID promoted to a verdict. No
winner chosen inside the noise margin.**

---

## 8. The honest summary — are we closer to a creature, or to a longer list of green ticks?

**Both, and this week the honest answer is: more genuinely closer than the raw
count suggests, and still not closer on the two things that matter most.**

The case for real progress is not the +2. It is **TA.02**. One-trial conditioned
taste aversion is a capability *nothing else in Jack's design has* — the DQN null
ate 196–218 toxic meals across 150 lives and never learned the discrimination,
while the aversion path got it in one trial and carried it across death. The
Garcia double dissociation came out textbook in both directions. That is not a
green tick; that is a creature-shaped capability with a null that genuinely
failed. And it landed alongside the reattach guard, which closed a hole through
which a local code edit could have been laundered into a certificate — the
ledger is measurably harder to fool today than it was yesterday.

The case against is §0's arithmetic, and it has not moved. **Curiosity: 1 claim
PASS across 12 specs. One-brain-in-unison: 1 across 21.** Those two lines *are*
GOAL.md — *"one brain, all senses in unison, that learns its world by living in
it"*, and *"he explores because he wants to."* Meanwhile the ladder-and-apple
standard the owner wrote — *"climbing the ladder on attempt 40 after falling on
attempts 1–39, without anyone telling him to"* — requires locomotion, and
**T2.01 has read FAIL for eight days** behind a one-line owner decision. We are
certifying senses for a creature that cannot yet be shown to walk. The senses
are the right work (they are constitutional, they were empty, and the standing
rule picked them correctly), but a wider base is not the same as a taller
ladder, and it would be dishonest to let 82 stand in for progress toward the
thesis.

And then there is the thing that is true *right now, as I write this*: **the
builder is asleep at a mailbox waiting for a letter from a dead process.** Three
iterations, three hours, one of them not even leaving a journal entry. The
system's instruments are in excellent shape — coverage exits 0, `verify` is
clean on five probes, the stale detector reads zero PASS, the reattach guard
shipped within an hour of being asked for. What it is missing is far more
ordinary than any of that: **nothing in this system checks whether the thing the
builder said it was waiting for is still alive.** Every organ here is good at
judging results. None of them is watching for the absence of one.

That is the gap I would close next, and it is worth more than the 83rd PASS.

---

## FOR THE BUILDER

**B1 (RANK 1, do this first — you are blocked on nothing).** Your SM.02
learnability checks are **dead**. `/data/sm02_geo_check.log` (04:13),
`/data/sm02_geo_occ.log` and `/data/sm02_geo_vis.log` (05:08) are all **0
bytes**, `pgrep -af sm02` returns nothing, and `/data/sm02_learnability_*.json`
still hold the **Euclidean** run from 03:13/03:15 (`vis` ratio 0.7166, `occ`
ratio 1.0) — proof the geodesic runs never reached their final `json.dump`.
Stop waiting. Before relaunching, establish *which* failure this was, because
the fixes differ:
  - **(a) reaped with the session.** Both launches sat ~2 minutes before their
    iteration ended; the 03:0x pair, which wrote its JSON at 03:13/03:15 *after*
    the 03:07 iteration ended, evidently was detached. Re-launch under `setsid`,
    out of the session's process tree, exactly as `dispatch.sh` does for
    watchers.
  - **(b) a traceback into a severed stderr.** Redirect with `> log 2>&1` — a
    0-byte log with no process is what an uncaptured stderr looks like.

Whichever it is, **add the assertion your own LESSONS.md line 92 demands**: ~10 s
after any detached launch, verify the *product* — process alive **and** log
non-empty — and fail loudly in the same iteration if not. This is the fourth
instance in 24 h of this class (`a51686c` yesterday: *"first two launches died
at import"*), and the first that cost three iterations instead of one. Consider
making it mechanical rather than advisory, the way you made `dispatch.sh`
pre-flight the GPU lock.

**B2.** Two of your last five iterations (02:07, 06:07) ended `rc=0` with **no
journal summary at all**. The 06:07 one is exactly the iteration that should
have caught B1. An iteration that writes nothing is invisible to the review
organ and to me. If a run has nothing to report, say *"nothing to report and
here is why"* — a one-line journal entry is not overhead, it is the only
evidence the iteration happened.

**B3.** `experiments/tests/sm_02_smell_finds_occluded.py` currently carries a
literal `[PENDING — numbers land here from the check before any dispatch]` in
its docstring. `_GATES_FROZEN` is still `False` and `run()` still hard-refuses,
so **the guard is holding and this is not yet a problem** — I checked. Keep it
that way: do not commit that placeholder in the same change that flips
`_GATES_FROZEN`, and treat the placeholder itself as a second tripwire.

**B4 (credit where due, and one thing to preserve).** I read the shaping repair
closely and it is correct science: potential-based (`r + γφ(s') − φ(s)`),
`φ(terminal) = 0`, training-only, identical in every arm, and absent from every
observation — so the twins still differ *only* in the smell channel and the
must-fail controls keep their meaning. The geodesic potential's edge test reuses
the *same* `mj_ray` + `R_AGENT` check that `step()` moves by, which is the right
way to guarantee it cannot dead-end on geometry the agent cannot cross. When
you freeze the gates, record pilot 1's table and the shaping change as a
**declared deviation between pilots** in the docstring, so a later reader knows
the rig changed underneath the two pilots and does not compare their numbers
naively.

**B5.** 26.35 of 30 W33 Kaggle hours expire **Sunday 2026-08-23** (~65 h). SM.02
pilot 2 is the queued spender. If B1 resolves quickly the quota is ample; if the
learnability checks come back negative *again*, say so and stop — a third repair
attempt on the same rig needs a pre-registered reason to expect a different
answer, not another 10-minute check.

---

## FOR THE OWNER

**Nothing new is owed by you this audit, and I am deliberately not re-raising
what is already on your desk.**

**D1 remains the only thing blocked on you** — *does the 57M trunk stay in the
control path?* The 20th audit laid out the full case six hours ago, and nothing
material has changed since: the fork is still one line (strike option A, or keep
it and say where the PLASTIC-ONLY decree narrows), T2.01 (*locomotion beats a
random policy*) has still read **FAIL for eight days**, and the quota that would
run it has ticked from 27.81 to **26.35 hours, expiring Sunday**. I have not
appended another cost update to `DECISIONS_NEEDED.md`, because re-raising an
unchanged decision every six hours is how a file stops being read.

**One thing worth your attention that is not a decision.** The system's
integrity instruments are, as of this morning, genuinely excellent — the ledger
survived every hard check I could design, and a hole I reported yesterday was
closed within the hour. But for the last three hours the builder has been idling
against three background jobs that died without a word, and no organ noticed.
The gap is not sophistication; it is that **nothing here watches for the absence
of a result.** I have asked the builder to make that check mechanical (B1). If
you see the PASS count sit flat for several hours again, that — rather than
anything on the ledger — is the first thing worth asking about.

---

*21st audit. Integrity: clean on every hard check. Loosening: none in 7 days.
Drift: none in the work. Liveness: stalled 3 iterations, cause identified,
handed to the builder as RANK 1.*
