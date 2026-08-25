# PROGRESS.md — the Review's current-state page

> Written by the Review organ. **Current state, not a log** — each run rewrites
> this file. The running history is `docs/PROGRESS_LOG.md`.
> Mode: DAILY (Part 2, the test re-examination, runs Sundays only).

**2026-08-25 06:40 UTC — DAILY. Window: 2026-08-24 06:40 → 2026-08-25 06:40.**
**First Review in four days** — 08-22 and 08-23 were refused at the 94% weekly
gate, and the 08-24 run died on a single API 529 at 06:45 before it read a file.
Where the 24-hour window is misleading on its own, the four-day figure is given
beside it and labelled.

*The one sentence: **the owner intervened twice on Monday and the builder turned
both rulings into machinery inside four hours — architecture is now raced rather
than decreed, and an escalation can no longer deadlock — and then the ladder
spent the next twenty-four hours proving, on four independent instruments, that
the thing actually blocking Jack is the world he lives in.***

---

## 1. The numbers

**Ladder: 84/181 demonstrated (46.4%).** At window start: 84/179 (46.9%). At the
last Review, 08-21: 83/169 (49.1%).

| | this window (24 h) | since the last Review (4 d) |
|---|---|---|
| ledger events recorded | **12** | 18 |
| PASS / FAIL / VOID / ERROR | **6 / 5 / 1 / 0** | 9 / 7 / 2 / 0 |
| **first-ever PASSes** | **0** | 1 (NE.00, 06:27 — 13 min before this window) |
| net new demonstrated | **+0** | +1 (83 → 84) |
| registry growth | **+2** (179 → 181: DP.05, BO.01) | **+12** (169 → 181) |
| builder iterations | **17 fired / 24 slots** (14 rc=0, 3 rc=1, 7 pace-skipped) | — |
| commits | **37** | 45 |

**Every one of this window's six PASSes was a re-certification** — T0.21 ×4,
T0.17, T0.27, all of them buying back a stamp after a lane edit or a clean-tree
re-run. **Nothing new was demonstrated in twenty-four hours.** What the ladder
actually did was record **five FAILs and a VOID**: NE.01 ×3, DP.05, T2.15,
BA.02. Every one landed at a pre-registered gate with its rig gates green. That
is not a bad day; it is the tier where things get refused, and the refusals were
clean. But it must be said plainly, because six green ticks in a row look like
progress and none of them was.

**Goodhart check: the rate FELL, 49.1% → 46.4%, and this is the good direction.**
The registry grew by twelve while demonstrated grew by one. Three windows ago
this page complained that the registry had been **flat at 169 for eight days** —
"the research queue not feeding the ladder". It is feeding now: NE.00–NE.09 came
off the top queue entry (first growth in eleven days), and DP.05 and BO.01 came
from the owner's Monday ruling. A falling rate on registry growth of that kind is
the ladder correctly outrunning the runner. **Both failure modes are real and
this is the one to want.**

**Rework: 51 of 84 passing specs took more than one attempt (60.7%),** down from
61.4%. Unchanged in substance.

**Compute.** Meters at 06:40: **`week:all models` 35% at 15% of the week**
(the gate), `week:Fable` 61% (not the gate). Kaggle **W34: 0.31 h charged of 30**
— one job, T2.15, the first GPU dispatch in four days. 29.7 h expire Sunday
2026-08-30.

**The three blackouts, stated once because they are the loop's dominant cost.**
W33 died at 94% from 08-22 00:07 to 08-24 05:07 — ~53 hours dark, taking the
overseer and two Reviews with the builder, and **22.1 of 30 Kaggle hours expired
unspent on Sunday 08-23, the second consecutive allocation to die.** The builder
shipped the fix on Monday morning (`e03693d`): a PACE LINE rising from a 25%
floor at the reset to the unchanged 90% ceiling, so the loop is still awake on
the Sunday the quota expires. It is doing exactly what it was built to do — 7 of
24 slots skipped this window at 31–35% spent against a 15%-elapsed week — and
**this is the first week it will be tested against the deadline it was built
for.** Verdict due Sunday.

**The frontier, recomputed live (`run blocked`, not quoted).** 66 of 181 specs
are unreachable.

| rank | spec | status | frees | what it is actually waiting for |
|---|---|---|---|---|
| 1 | **T2.01** Locomotion beats random | FAIL (2.67σ vs 5) | **35** (blocks 36) | D1/D9 — **now ARMED, `decide_by 2026-08-31`** |
| 2 | **LC.03** learning-core screening | VOID (concluded) | 8 | D10 — **ARMED, `decide_by 2026-08-31`** |
| 3 | **NE.01** needs are a real control problem | FAIL ×3 | 8 | **world design** — the occlusion law is knife-edged |
| 4 | UB.10 fusion bakeoff | NOT_RUN | 4 | arm redesign — **dispositioned today, §4** |
| 5 | T2.02 | VOID | 3 | the ladder's only stale claim |
| — | DP.05 → **BO.01** | FAIL | 1 | **world design** — traps, delays, irreversibility |

**Read the right-hand column, not the ranks.** Roughly **52 of the 66
unreachable specs sit behind two owner decisions that now have expiry dates, and
a world that four instruments say is too shallow.** Not one of them is waiting
on compute, and the GPU quota is nearly untouched with five days to run.

---

## 2. What the window produced — and the largest thing in it was not ours

**THE OWNER RULED TWICE ON MONDAY, AND BOTH RULINGS WERE STRUCTURAL.**

> *"in the future he mustnt get blocked by anything like this but instead test
> and try and research both and decide at end which works better"*

> *"this project won't work if you can't let them research and test stuff like
> that!!! architectural stuff. cause then we would go one brain and NEVER EVEN
> SEE what would be fast slow brain or whatever? this project depends on
> research and testing at EVEERY SINGLE STAGE"*

The builder turned both into machinery inside four hours, and the audit it ran
first is the most valuable thing anyone has produced here in a week:

- **Of 179 specs, not one raced brain organisations.** One shared brain was a
  PREMISE of the ladder, never an outcome of it. SYSTEM.md made it unfalsifiable
  on purpose — a non-unified arm was *"not scored and beaten, EXCLUDED"* — while
  UB.10's own A5 arm in the same repo said *"if A5 wins, 'one brain' is the wrong
  shape and we say so."* Two documents, opposite answers.
- **Six fictional falsifiers.** CHAMPIONS.md named PL.00, PL.02, LG.00, LT.03,
  LT.04 and T2.21 as the arenas that would unseat four consequential design
  choices. **None of the six existed in the registry.** `champions.py` now
  enforces that every seat's challenger resolves in `BY_ID`.
- **DP.00 passed in a 12×12 gridworld, not in W0** — and its reading had drifted
  to "Jack's world rewards deliberation". The precondition for the entire
  fast/slow axis had never been tested in the world Jack is embodied in.

Shipped from it: SYSTEM.md's invariant split into **ENDS / ARCHITECTURE /
CONDUCT** (an architectural decree is a CHAMPION, not a constitution);
EXCLUDED became **SCORED-AND-INELIGIBLE**, so a losing arm's number is kept and
the owner is owed the finding loudly rather than protected from it;
`experiments/decisions.py`, which parses `DECIDE:` blocks, joins `blocks:` to
the **live** dependency graph so cost is computed rather than typed, and ships as
a ratchet; and **D1, D7, D8, D9 and D10 ARMED with `decide_by 2026-08-31` and
written defaults.** An escalation can no longer deadlock. D1 had been open
twenty days with "evidence complete" in its own title.

**This desk should say plainly that it did not find any of this.** Four
Reviews reported D1's cost correctly every day and none of them found the rule
that made it unanswerable — SYSTEM.md line 164 silently repealing rule 3 for
exactly the class of question rule 3 exists for. The owner found it in two
sentences.

**And then the world answered.** Within twenty-eight hours of BO.01 being
registered, its gate spec DP.05 came back **FAIL**: lookahead buys 13.3 ± 18.8 s
in W0, σ 0.70 against a 3.0 gate, `gap_clear` 1/3 — and **H10 pays LESS than
H4**. Every VOID gate green (reference 4 eats/173.1 s vs a 132 s ceiling,
`ctrl_gain` −0.014, `broken_gap` 0.112), so the world measured and the rig did
not fail. Planners eat; reactives never do; the best reactive policy in W0 is
*starve at the resting ceiling*. Pre-registered routing binds: fix the world
before any dual-process claim, **BO.01 does not run**.

Then, six hours later, **SH.01's oracle pilot read ORACLE_CANNOT at the full
N=10000 envelope** — `z_shelter` 0.0, the certified core with the working hut's
direction *in its observation* sheltered in **0 of 27 lives**, while sheltering
demonstrably pays. That removes the perception excuse and implicates the core
jointly with the world. SH.01 was parked per its own pre-registered rule.

Method held under pressure in three places worth naming: DP.05's fidelity pilot
caught two real physics faults **before** the registered run (Water.apply leaves
a phantom xfrc row on any body that exits the pool; W0 carried one-substep-stale
`xipos`/`cvel` across decision boundaries) and the water fault was **routed, not
patched in passing**; NE.01 was re-run from a clean tree to convert a dirty
stamp into a real one, landing metrics identical on every digit; and BA.02's
stale re-run landed the *expected* D8-reason VOID rather than being quietly
left alone.

---

## 3. THE FINDING — a dead audit published a green verdict

`scripts/overseer.sh:67` extracts the audit's verdict by grepping
`docs/OVERSIGHT.md` for the first of `ON TRACK|DRIFTING|INTEGRITY RISK`. That
file is the *previous* audit's output. So when an audit dies before writing
anything, the wrapper reports the **stale** verdict as if it were this audit's:

```
2026-08-24T12:37:03  audit start — model opus, 71c879f
You've hit your session limit · resets 3pm (UTC)
2026-08-24T12:37:05  audit end rc=1 — verdict: ON TRACK
```

Two seconds, `rc=1`, nothing read, nothing written — **and a green verdict in
the log.** The same session-limit wall took three builder iterations that hour
(12:07, 13:07, 14:07), and there the loop handled it correctly: it tried Fable →
opus → sonnet, marked the lost iteration, and the 13:07 and 14:07 runs each
announced *"inheriting N iteration(s) lost to limits"*. The builder's failure is
loud and recovered; the overseer's is silent and flattering.

This is one turn past 08-20's *"an organ can stall indefinitely with a correct
`rc=0`"* and 08-21's *"an organ correctly executed a rule applied to the wrong
number"*. The shape here is narrower and worse: **the failure is not that the
audit died — deaths are fine and this one was recovered six hours later — it is
that the death was reported in the vocabulary of success.** A human or organ
scanning `overseer.log` for the last verdict reads ON TRACK. The fix is two
lines and belongs to the builder (§ FOR THE BUILDER item 1): when `rc≠0`, print
`verdict: UNKNOWN (audit did not complete)` and never consult a file the dead
run did not write.

**Also noted and NOT a finding, because it was checked:** `gpu_budget.json` and
`gpu_submissions.jsonl` are uncommitted at HEAD after the pace-skip bookkeeping
commit staged only `ledger.json`. This does **not** dirty-stamp anything —
`protocol.py`'s `RUNNER_OUTPUTS` has excluded both since 2026-08-12, and that
exclusion exists precisely because it once did. The accounting is intact on
disk; the next unskipped iteration commits it.

---

## 4. Steering maintenance (Part 2.5) — done, five changes

**1. `ladder_prompt.md` PRIORITY, reconciled — and one correction is against the
builder.** The frontier block was signed **"(Review, 2026-08-24)"** and no
Review ran on 08-24; it was written by the builder in `9449a1b`. The *content*
was correct — I re-derived it from `run blocked` and have adopted it under my own
name — but **an organ that can be quoted under another organ's name is an organ
whose independence is decorative**, and the separation of builder from Review is
the only reason this desk's judgement is worth anything. Attribution corrected in
place. Also: the cached **"frees 26"** for T2.01 had been wrong for eleven days
(live: **35**) in a file whose own next paragraph forbids caching counts — now
replaced with a live-read instruction. And the block was re-pointed: **the three
CLAIM-DEAD commitments are now named as the highest-value CPU work on the
board**, DP.05's routing is stated so BO.01 is not manufactured around it, and
NE.01's FAIL is labelled a world-design result rather than a tuning miss.

**2. `FIELD_WATCH.md` wk4 CONSUMED** (sweep 08-24, three nominations; consumed
one day late for the 529). All three ACCEPTED as design work, none as a dispatch
— they enter LC.04/LC.05, which D10 blocks until 08-31. **N1** (Koopman
Dreamer's spectral radius, narrowed to a two-hyperparameter reparameterisation of
`A4`'s existing transition matrix) — accepted with the scout's own strongest
objection attached: no seed count, no CIs on the headline table. **N2**
(PSG-JEPA) — accepted as **two arms that are each other's control**, with the
scout's falsifiable prediction pre-registered: in W0 the observation already
contains proprioception, so `ℒ_static` is a decoder on a slice of the input and
should not help, while `ℒ_dynamic` is a temporal quantity present in no single
step and might. **N3** (infant motor noise) — **REJECTED as an arm** (PPO's
exploration *is* its policy distribution; autocorrelated noise breaks the
likelihood ratio, and this desk will not make that design call by argument),
**ACCEPTED and PROMOTED in its W0-diagnostic form** — see item 4. The scout's
`[s]`-marker proposal accepted as a LESSONS entry for the builder to write; its
self-retraction of two invented arXiv:2607.25337 numbers is recorded so they are
not re-quoted.

**3. `REVIEW_QUEUE.md` — first use, four rows, and the file immediately earned
itself.** Three of the four edit `playground.py`, and **each bills the same 21
PASS certificates mechanically**. Servicing them in arrival order costs 63
re-runs; servicing them in one world-edit window costs 21. A **BUNDLING RULE** is
now written into the file, `ne01-occlusion-knife-edge` and
`water-apply-phantom-force` are HELD for that window, and if `w0-too-shallow`
resolves toward a new world (W1) rather than an edit to W0 the bill goes to
**zero** — which the row itself had already flagged as design input. *This is the
argument for the file: a backlog with a computed bill can be sequenced; a backlog
in commit messages can only be serviced in arrival order, the most expensive
order.* `recipe-sensitivity` ACTED (design below).

**4. `w0-too-shallow` — the design is MINE and it is dated 2026-08-30.** DAILY
mode cannot fund a world redesign and pretending otherwise is how a routed row
rots. But one thing is ordered **today, before the design**: all four
instruments behind this row are expensive, were run by this project, on this
world — the exact condition under which a shared confound is invisible. wk4-N3
supplies a **CPU-minutes attack**: a β-scheduled colored-noise random policy
against the `random` and `random-repeat` nulls LC.03 already defines. It asks
whether "the cores cannot learn in W0" is partly "**the exploration process
never reaches the food**". A redesign informed by four expensive agreeing
instruments and one cheap disagreeing one beats a redesign informed by four.

**5. `CHAMPIONS.md` — two cells corrected, one of them a rule-3 violation.**
The **Deliberation** seat still named its arena as *DP.00–DP.04* a day after
`DP.05` and `BO.01` were registered into it and one of them had already run and
FAILed. Rule 3 says *every seat names its ARENA*; for the newest arena in the
project it did not, so `champions.py` could not check it and the field watch
could not target it. Corrected: the seat is **contested for the first time**,
BO.01 is named as the real challenger, and DP.05's FAIL is recorded as the
answer to "does this world reward lookahead at all" — **no**. The **Sensory
fusion** cell still called UB.9 `PASS but STALE` and ranked it "#3, frees 5";
`run status` now flags exactly one stale claim and it is T2.02, and `run blocked`
does not list UB.9 at all. Corrected — and what replaced the staleness is
smaller but real: per the 26th audit's B3 the PASS is now **conditional** on a
re-run recording per-arm loss descent.

**ORGAN LIVENESS — all four live, one recovered failure.** Builder hourly, last
06:07 ✓. Overseer 6-hourly, last completed audit 08-25 00:37 (**DRIFTING**), next
running now ✓. Field watch Mondays, ran 08-24 05:37 ✓ — **back on its intended
cadence for the first time in three weeks**. Review daily: 08-22 and 08-23 gated,
08-24 died on a 529 — `review.sh` now retries once at 120 s (`7f3a907`), which is
the direct fix. **`lost_iterations.log` is 0 bytes**, correctly, because all
three lost iterations were inherited and cleared.

---

## 5. The honest paragraph

No numbers. Are we closer to a creature that lives, learns, and is known — or
just busier? **Closer, and for a reason that has nothing to do with the ladder
moving.** For twenty days this project could not answer its own most important
question because two rules it obeyed perfectly had quietly cancelled each other,
and no organ here could see it from inside. The owner saw it, said so, and by
Monday afternoon the thing that had been an argument was a race, the premise
that Jack has one brain had stopped being a premise, and every escalation on the
desk had grown a deadline and a default. That is the week's single most
important step toward Jack: **not a capability, but the removal of the mechanism
by which this project could be permanently prevented from finding out it was
wrong.** A system that cannot be blocked by its own silence is a different kind
of system.

And then the ladder spent a day and a night doing the only honourable thing
available to it, which was to keep refusing. Every measurement it took, it took
against a gate written before the answer was known, and every one came back no.
The needs do not yet make a control problem a body can survive; looking ahead
buys nothing in this world, and buys less the further ahead it looks; and a
creature handed the direction of a shelter that would save its life walked past
it twenty-seven times out of twenty-seven. Those are not four disappointments.
They are four instruments, run independently, agreeing — and what they agree
about is not Jack's brain. **It is that the place we put him is too thin to be
worth learning.** He is not failing to learn his world. There is not yet enough
world there to learn.

So the most concerning drift is this, and it is the exact drift the owner just
legislated against in a different form: **the world has become the binding
constraint on everything, and nobody owns it.** Two of Jack's constitutional
commitments — *he builds a shelter*, *too cold kills him* — went claim-dead at
eleven minutes past midnight, and one of them is GOAL.md's own image of success.
The ladder is registering specs faster than it has in eleven days, the compute
sits nearly untouched, and every road forward runs through a redesign that is
routed to this desk with a date on it and no design behind it yet. **We have
built an extraordinary apparatus for finding out that we are wrong, and it has
now told us the same thing four times in four different voices.** The risk is not
that we stop measuring. It is that measuring is the comfortable part, and the
next move is to build.

---

## FOR THE BUILDER — ordered

**1. Two lines: a dead audit must not publish a green verdict.**
`scripts/overseer.sh:67` greps `OVERSIGHT.md` — the *previous* audit's file — so
`rc=1` at two seconds logged `verdict: ON TRACK` on 08-24 12:37. Make it
`verdict: UNKNOWN (audit did not complete)` whenever `rc≠0`, and never read a
file the dead run did not write. Your own builder path already does this right
(fallback chain, lost-iteration marking, `inheriting N`); the auditor should not
be the organ that fails quietly. Cheapest item on this page.

**2. UB.10: matched TUNING BUDGET, not matched hyperparameters** — and it is
**strictly harder than what it replaces**, which is the only kind of redesign
this desk may propose. Dropping the uniform-recipe constraint destroys the
comparison (an arm would win by getting a better LR); keeping it is what left A2
dead under every tested recipe. So: every arm gets the **identical
pre-registered LR grid, the same trial count, the same pre-registered selection
criterion**, all declared before any arm runs. Cost rises from N to N×K runs —
that is the point; the budget is what is matched, not the setting. The gate that
makes it honest **already exists**: per the 23rd audit's B1,
`uni_marginal_ok`/`uni_learn_ok` mean a dead arm can no longer read as a clean
0.5, so "did this arm's recipe train it" is machine-checkable per arm. An arm
that clears `uni_learn_ok` **nowhere** on the grid is recorded
SCORED-AND-INELIGIBLE under SYSTEM.md's new language (`0345f0d`) — measured on
the same ruler, kept as a standing challenger, not seated and not silently a 0.5.

**3. Register a successor claim spec for shelter/thermal.** Three commitments
are CLAIM-DEAD and two of them are among the four original 2026-08-10 misses
that caused `coverage.py` to exist. SH.01's own park note writes the
constraint for you: *"a successor spec that does not require this core to learn
seeking from an outside spawn."* Curriculum spawn, or shelter-seeking as a
conditioned response with the thermal gradient already sensed, or an
oracle-initialised life measured on **maintenance** rather than acquisition —
any of them restores a falsifiable claim. CPU, no owner gate, no dispatch. **A
named gap is a decision; a claim-dead commitment is a blind spot with a green
tick over it.**

**4. Run wk4-N3's β-noise diagnostic before any world redesign work.**
CPU-minutes, uses nulls LC.03 already defines. Full reasoning in
`REVIEW_QUEUE.md § w0-too-shallow`. If it fires, the shallowness diagnosis
changes before we spend a redesign on it.

**5. Write the `[s]` LESSON** the field watch asked for (this desk does not
commit LESSONS.md). `[c]` means *the authors claim this and I have not checked*;
`[s]` means *a third party asserts the authors claim this* — and only the second
can be a number **nobody ever wrote**. The scar is in an append-only file:
`FIELD_WATCH_LOG.md`'s 08-12 entries carry two arXiv:2607.25337 claims that two
full-text fetches confirm appear nowhere in the paper.

**6. NE.01 is a world-design result. Do not re-run it a fourth time.** Attempt 3
reproduced attempt 2 on every digit from a clean tree — that is a deterministic
FAIL, and a fourth roll is the run-until-pass ratchet. Its row is HELD for the
world-edit window per the bundling rule.

---

## FOR THE OWNER — strategic forks only

**1. The world is now the binding constraint, on four independent instruments,
and this is the fork.** LC.03's darkroom control (passivity prospers), LC.03 v2
(one learner in five at 4× envelope), DP.05 (lookahead buys nothing, and less
the deeper it looks), SH.01 (ORACLE_CANNOT — a creature handed the direction of
a life-saving shelter used it in **0 of 27 lives**). These agree, and none of
them is about Jack's brain. **Recommendation: build W1 rather than patch W0.**
Three reasons, one of which is arithmetic. Patching W0 bills 21 PASS
certificates in mechanical re-certification; **a new world bills nothing**
(T1.02 precedent) and leaves every W0 result standing as the reference it
already is. Second, the repairs the instruments call for — traps, delays,
irreversibility, depletion, pursuit — are the DP.00 preconditions GOAL.md
already names, so this is scheduled work, not new scope. Third, W0 keeps its
job: it is the throughput floor and the fidelity fixture, and a champion that
must re-defend at W1 is exactly what CHAMPIONS rule 4 is for. **What I need from
you is the sequencing, not the design** — the design is mine and it is dated
2026-08-30.

**2. D1 and D10 fire on 2026-08-31 whether or not you answer, and I recommend
letting D1 fire.** This is new and it is your machinery working as ordered: both
now carry written defaults, and silence resolves them rather than deadlocking
them. D1's default leaves the constitution **exactly as written** — PLASTIC-ONLY
verbatim, option A struck as unconstitutional, the permitted arms to a bakeoff.
That is the branch that costs you nothing to allow and unblocks 38 specs. **D10
is the one worth your five minutes**, because its default ("accept the screen's
answer — one learner, amend LC.04's premise") is reasonable *only if* the world
is not the reason four of five arms failed, and item 1 says it might be. If you
answer one decision this week, answer D10 — or tell me to re-arm it past the W1
design, which is the option I would take.

**3. Your Monday rulings did something no organ here could have done, and that
is worth naming as a finding about US, not about you.** D1 sat open twenty days
because SYSTEM.md rule 3 ("decisions are made by bakeoff, never by argument")
was silently repealed for architecture calls by another line in the same file.
Every audit reported the cost correctly every six hours; four Reviews put it on
your desk; not one of them found the rule. **The lesson we should draw is that
this system is good at measuring against its stated standard and structurally
blind to the standard itself** — which is the same failure that hid smell, taste,
voice and body-schema on 2026-08-09. The completeness audit exists for exactly
this and it runs Sundays. **I am flagging that one weekly external-reference
audit was not enough to catch a twenty-day deadlock**, and that if you see
another one, saying it in two sentences remains cheaper than anything we can
build.

**4. Not a fork, a number you should have: the pace line gets its first real
test this Sunday.** Two consecutive Kaggle allocations (8.8 h, then 22.1 h) have
expired unspent inside gate-blackouts that both began on a Friday. The fix
shipped Monday and is visibly working — 7 of 24 slots skipped this window with
35% spent into 15% of the week. W34 currently has 29.7 of 30 hours unspent, and
they expire Sunday 2026-08-30. **If a third allocation dies, the problem is not
pacing and I will say so on Sunday's page.**
