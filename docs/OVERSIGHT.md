# OVERSIGHT — 142nd audit, 2026-10-06 ~00:4x UTC

> Written by the overseer organ. **Current state, not a log** — each audit
> rewrites this file. Every number below was read live during this sitting; the
> closing block re-runs every instrument AFTER my last act.

## VERDICT: INTEGRITY RISK — one armed-default firing in this repository's history has never been diff-audited, by either of the two channels built so that could not happen, and every page reports the audit complete. Separately: the builder measured the shipped creature at runtime for the first time, found he can neither move nor see, and routed it to nobody.

The ledger is in good order and that is a measured result, not a courtesy:
**106 PASS certificates, 0 whose `commit` is absent from git, 2 declaring no
control at all (`T0.01`, `T0.10`, both owned with a clock), 0 thresholds moved
in any direction over seven days, 0 controls weakened, 0 seed counts reduced.**
Section 2 has no finding.

The builder's second live day is **nine slots, nine `rc=0`** and real work:
three of the 141st audit's five FOR THE BUILDER items discharged (`gpu_unattributed_jobs`
back to its floor, `XL.01`'s additive conjunct written *and* run, the reset-time
correction), `LG.14` implemented and honestly FAILed, and `T6.01` — the standing
freeze's own lift condition — given a harness for the first time since its
2026-08-04 registration. `demonstrated` was 106 at 16:07 yesterday and is 106 now.

| # | finding | age | floor |
|---|---|---|---|
| **1** | `D36`'s firing commit `c10a128` is in **NEITHER** identification channel; `D37`'s is in **one**. 29 of 31 recorded firings are on the resolved page; the last two are not. `SYSTEM.md` calls this gap closed | 9 d | unfloored |
| **2** | Three runtime facts about the **shipped** Jack — 17-wide action into a 57-actuator body, refused 177/177 frames; no `"eye"` camera in his scene; a worldless session that reports completion — on **no desk's dated page**, visible to **no instrument**, and absent from `D41`, the live owner decision they are evidence for | 1 slot | n/a |
| **3** | `proc_memory_report` can only see a RAM breach that **outlives the slot**. `T6.01`'s child peaked at **2129 MB** against the 1536 MB tenant ceiling inside the 00:07 slot and `ladder.log` has no `MEMORY` line for it | 1 slot | unfloored |
| **4** | `T0.06` "Env/policy dimension contract" is PASS with `kills = "Every locomotion result"`, certified at `EXPECTED_NU = 17` on Gymnasium `Humanoid-v5` — a venue the shipped body is not | 2 mo | n/a |

Standing reds, re-derived and not re-litigated: `decisions_default_action_expired`
1 (`D33`, the owner's, 13 days past and MOOT rather than merely expired),
`pass_on_dead_dependency` 6 vs floor 3 (140th audit RANK 3), `unreachable` 96 of
257 vs baseline 95 (`LT.02`'s honest demotion's downstream; the argument for
raising the floor is written at `coverage.py:~1186` and is the desk's).
`gpu_unattributed_jobs` is **back to 21, AT floor** — the 141st's RANK 1 is
discharged.

---

## RANK 1 — A firing act in this repository's history has never been diff-audited, and the two-channel guard that exists to make that impossible is blind to it in both channels

`SYSTEM.md` (lines ~190–206) says, about this exact mechanism:

> **Of the two smaller gaps named beside the check when it shipped, ONE IS
> CLOSED AND IT COST A MEASUREMENT TO FIND.** … A firing is now identified by
> **two independent channels**: its subject, and the `RESOLVED BY ARMED DEFAULT`
> record it has to write whatever it calls the commit. `T0.28`'s P17 is the
> certificate. … **The residual on identification is now a firing that declares
> itself in NEITHER channel — named here rather than called closed.**

**That residual is not hypothetical. It has a live instance and it is nine days
old.** Measured this sitting against `experiments/decisions.py`'s own functions,
on the real git log and the real pages:

    D36 firing c10a128: channel1(subject)=False  channel2(record page)=False
    D37 firing 85cb0ee: channel1(subject)=True   channel2(record page)=False

`c10a128` is the commit that wrote `## D36 — RESOLVED BY ARMED DEFAULT` into
`docs/DECISIONS_NEEDED.md` on 2026-09-27 00:50 (123rd audit, this organ).
Channel 1 (`_FIRING_SUBJECT`, `decisions.py:741`) requires all three of
`\bD\d+\b`, `\bfired?\b`, `\bdefaults?\b` in the subject; that commit's subject
is *"123rd audit — DRIFTING: the Review's deaths stopped being max-turns
deaths…"* — it contains no `D<n>` and neither keyword. Channel 2 anchors on
`^##\s+(D\d+)\s+\S*\s*RESOLVED BY ARMED DEFAULT` over
`docs/DECISIONS_RESOLVED.md` (`decisions.py:942`), and **`D36` has no such
heading on that page.** So the commit is absent from the 44-commit population
`firing_diff_hazards` audits, and its diff has never been checked against the
two safety clauses the `FIRING-DIFF` ratchet exists to enforce.

**THE PROXIMATE CAUSE IS ONE MISSING ACT, AND THIS ORGAN NAMED IT NINE DAYS AGO.**
`docs/DECISIONS_NEEDED.md:9035`, inside `D36`'s own firing record:

> **OWED BY THE REVIEW, NOT BY ME:** the `## D36 — RESOLVED BY ARMED DEFAULT`
> transcription onto `docs/DECISIONS_RESOLVED.md`, per the `D13` rule that the
> overseer stays inside its own file set and the `D31`/`D32`/`D34` precedent.
> **Until that lands, `decisions.py`'s second identification channel
> (`RECORD_PAGE`/`RECORD_MARKER`) cannot see this firing.**

The 123rd audit got the diagnosis exactly right, named the owner, and named the
cost in terms. Sixteen audits and one Review sitting later it has not landed,
nobody re-derived it, and a second instance has stacked on top:

    firings recorded on docs/DECISIONS_NEEDED.md : 31
    firings declared on docs/DECISIONS_RESOLVED.md: 29
    recorded-as-fired, ABSENT from the resolved page: D36, D37

**The discipline held for twenty-nine consecutive firings and broke on the last
two.** `D31`'s transcription was itself an ordered repair (119th audit FTB 3+4,
landed `251ebcd`, citing *"D13 rule, D32/D34 precedent 6aaed9b"*), so the act is
a known, cheap, precedented one-commit job.

**WHY `T0.28`'s P17 DOES NOT CATCH IT, AND THIS IS THE PART THAT GENERALISES.**
P17 reads PASS and it is not lying. Its battery
(`t0_28_decisions_tool_is_honest.py:872-915`) replays the mechanism against a
**synthetic page literal** —

    page = ("## D22 — RESOLVED BY ARMED DEFAULT (fired 2026-09-12): x\n"
            "## D25 — RESOLVED BY ARMED DEFAULT (fired 2026-09-14): y\n" …)

— and asserts nothing about `docs/DECISIONS_RESOLVED.md` as it stands. The
certificate is true of the *code* and silent about the *documents*. Worse, the
one reading that could have surfaced this — `firing_coverage`, which P17 itself
describes as *"names a declared firing no commit reached, which is the reading
that says whether the audit found them ALL"* — is defined over firings
**declared on the resolved page**. A firing that never reaches that page is
invisible to the completeness reading too. **No instrument in this repository
asks the converse question: which firings recorded on `DECISIONS_NEEDED.md` have
no record on `DECISIONS_RESOLVED.md`.** It is three lines of set arithmetic over
two files, and it is the whole of RANK 1.

**EXPOSURE TODAY IS ZERO AND I CHECKED RATHER THAN ASSUMED IT.** I ran
`decisions.firing_diff_hazards` over both unaudited diffs:

    c10a128  hazards: CLEAN   files: docs/DECISIONS_NEEDED.md, docs/OVERSIGHT.md
    85cb0ee  hazards: CLEAN   files: docs/DECISIONS_NEEDED.md, docs/OVERSIGHT.md

No `GOAL.md` edit, no numeric bar moved or deleted, in either. **This is the
same shape as `3b2e38b`** — the precedent `SYSTEM.md` records as costing a
measurement to find, of which it says *"It audits clean, so the exposure was
zero — which is why closing it was cheap."* It is cheap again, today, for the
same reason. That is why this is RANK 1 and not a filing note: it is ranked on
the false completeness claim, not on damage done. A guard that reports 100 %
coverage of a population it does not fully enumerate is making a capability
claim, and law 1 binds it like any other.

**And the `D37` half is the weaker failure in the same mechanism.** Its firing
*is* identified — by channel 1 alone, the commit's own subject, which
`decisions.py:746` describes verbatim as *"the author's word about their own act
— the exact form of evidence this file exists to distrust."* Channel 2 exists
precisely so no firing rests on that alone, and for `D37` it does.

## RANK 2 — The first runtime measurement of the shipped Jack says he cannot move and cannot see. It is on no desk's dated page and no instrument can see it.

The 00:07 slot implemented `T6.01` ("Full episode completes") and drove the
shipped companion loop on the shipped embodied scene. **It is the first time
this project has measured the creature it ships, rather than the rig that judges
him.** Three facts came back. I verified each at source rather than taking the
commit message's word:

**(a) The companion cannot drive his own body.** `apply_action` refuses the
brain's 17-wide action against the humanoid's 57 actuators on **every frame —
177 of 177**. Static corroboration: `assets/humanoid_full.xml` carries **59
`<motor>` declarations** (the model compiles to `nu = 57`), against a policy
width of 17.

**(b) He is blind by a name.** The scene has **four** cameras — `track`,
`left_eye`, `right_eye`, `head_cam` — and **none called `"eye"`**.
`VirtualWorld.py:714` looks up `mj_name2id(…, mjOBJ_CAMERA, "eye")`
specifically, so `_get_eye_image()` returns `None` and the embodied brain runs
proprioception and touch only. The sharper version of the builder's own finding:
the eyes are *there*, under three names, and the consumer asks for a fourth —
while `playground.py:537` emits `<camera name="eye" …>`, which is why the rig's
visual certificates are fine and the shipped scene's are not. This is **not**
covered by `two-eyes-one-certified` (OPEN, DUE 10-11), which is about render
*quality* (offsamples/shadowsize) inside the playground.

**(c) A session that never built a world runs to completion and looks healthy.**
At `main()`'s default 800×600 the renderer exceeds the model's 640×480
offscreen framebuffer, `_init_mujoco` fails closed, and a full half-minute
session reported `completed 1`, 173 frames, autosave fired, monologue thinking —
with `construction_ok 0` and `nu 0`. The companion loop swallows brain, physics,
emotional and autonomous exceptions at `logger.debug`, so **a session that
crashed on every frame reads identically to one that lived.**

**The builder's conduct here is exemplary and I want it on the record before the
complaint.** The harness was pre-registered before any recording run; the
registry gained `T6.01`'s **first control declaration** (two injected-fault
twins must be CAUGHT or the run is VOID — strengthen-only, no certificate
exists), and the control is genuinely load-bearing: `_check` returns `VOID` when
the planted faults go unseen, and both twins were proven alive (NaN twin 1242
non-finite actions, raise twin 2479 crash frames). The expected verdict was
pre-registered as **FAIL**. The slot summary refuses the cheap repair in terms —
*"the repair routes through the Review, not a pad/truncate."* That is right, and
the generalisable half is in `LESSONS.md` as a strong entry (*gate what was
BUILT and what was CAUGHT, never what was logged*).

**THE FINDING IS ROUTING, AND IT IS THE `VANISHED-OWNER-ASK` SCAR'S EXACT
SHAPE.** Those three facts exist in a commit message, a journal line, a test
docstring and a methodological lesson. They exist in **no** `ROUTED:` row —
`git log -p -- docs/REVIEW_QUEUE.md` over the last ten hours adds **zero**
`ROUTED:` lines — and grepping `REVIEW_QUEUE.md` and `DECISIONS_NEEDED.md` for
`nu 57`, `action width`, `eye camera`, `eye_alive`, `crash_frames` returns
**nothing**. Finding (b), the missing camera, appears in no desk file at all.

**No instrument can see any of it, and the reasons are structural:**

- `fail_unowned` is blind, because `T6.01` has **no ledger row** — the recording
  run is `BLOCKED` behind `T4.05`, so there is no FAIL for an owner to attach to.
- `review-queue` is blind, because a row that was never routed has no clock.
- `decisions` is blind, because **`D41`'s entry does not mention this.** `D41` —
  *which artefact is Jack, the ladder's rig or `TrainingPipeline.py`* — is armed
  on the owner's desk with `decide_by 2026-10-18`, and the builder's own commit
  calls finding (a) *"runtime-side corroboration of `D41`'s bridge drift."* It is
  the **fourth** instance of that drift and the **only one measured at runtime**,
  and it reached neither the decision it strengthens nor the page the owner reads.

If the 06:37 Review has a short sitting — and it has died `rc=124` **seven
consecutive times** — this is gone. That is the precise failure mode my own
instructions were written around.

## RANK 3 — The tenant RAM guard can only see a breach that outlives the slot, and one slipped past it last night

`scripts/lib_procwatch.sh:310` claims:

> Names **every** project python whose PEAK rss (VmHWM) exceeds the ceiling. …
> a leak is defined by when a process started; **a memory breach is defined by
> what it did**.

The implementation (`proc_memory_report`, `:318-335`) iterates `/proc/[0-9]*`
and reads `VmHWM` from `/proc/$pid/status`. It is called once per iteration,
from `leftover_report` at slot end. **A process that peaks over the ceiling and
exits before the slot ends has no `/proc` entry, so its high-water mark is
unrecoverable and the watcher reports nothing.** The guard is structurally
scoped to breaches that survive the slot — which is the `proc_leaks` definition
the docstring explicitly distinguishes itself from.

**It happened last night and the evidence is one slot old.** `T6.01`'s smoke ran
a child process inside the 00:07 slot (00:07:09 → 00:29:02) that peaked at
**2129 MB — 38 % over the 1536 MB ceiling** — measured by the spec's own
`resource.getrusage(RUSAGE_SELF).ru_maxrss` at `t6_01_full_episode.py:285`.
`/data/jack-logs/ladder.log` carries **no `MEMORY` line for that slot** and the
iteration-end line carries no `MEMORY=` suffix. The last `MEMORY` line in the
file is 2026-10-05T22:22:36, for `lg_14_structured_decode --llm-pass` at 2424 MB
— and that one was caught **because it was a detached pass still alive at
sampling time**, which is the whole asymmetry.

**Why this is ranked rather than filed.** `SYSTEM.md`'s hard constraints say
*"This box serves paying tenants … stay under ~1.5 GB RAM"*, and the box hosts
four customer agents plus the WorldTwin ingress. The guard exists, per its own
comment, because the 63rd audit found a 7.57 GB process live at 5× the ceiling
and the header was claiming a capability it did not have — *"a guard's comment
is a capability claim and law 1 binds it."* **That is the same defect, one layer
in.** And the number that escaped is not hypothetical: the builder disclosed it
honestly in its commit and slot summary — *"Peak child RSS 2129 MB is recorded
and disclosed for whoever schedules the 30-minute recording run"* — so the
builder's prose is currently doing the instrument's job, which is exactly the
inversion the watcher was built to end (*"named so the excess is a number
instead of an anecdote"*). The 30-minute recording run, when `T4.05` clears,
will hold that footprint for sixty times as long as the smoke did.

I am **not** proposing a gate. The ceiling's own validity is an open owner
question (63rd audit B2) and the discipline is NAME, NEVER KILL. The repair is
that the name gets written.

## RANK 4 — `T0.06` certifies the dimension contract at `nu = 17`, and the shipped body has 57

`T0.06` "Env/policy dimension contract" is **PASS**, with the strongest `kills`
field in Tier 0: *"Every locomotion result."* Its hypothesis is *"MuJoCo model
`nu` equals the policy action dim, asserted at startup."* Read at source
(`t0_06_dimension_contract.py`):

    Decision D2 standardises on Gymnasium Humanoid-v5: nu = 17.
    EXPECTED_NU = 17          # Humanoid-v5, per decision D2
    def _env(): … return gym.make("Humanoid-v5")

So the certificate is bought in the rig's venue, on a 17-actuator body, where
the policy's 17 outputs match by construction. RANK 2(a) is the identical
mismatch the spec was written to make impossible, occurring in the venue the
spec does not test, on every frame.

**`T0.06` did not fail and its control is not weak — the control is the thing
that caught the shipped mismatch.** Its control reads *"Every WRONG width driven
through the real write path (`VirtualWorld.apply_action`) must be REFUSED"*, and
`apply_action` duly refused all 177 frames. The defect is that the refusal lands
in a `logger.debug` swallow in the shipped loop, so `T0.06`'s own
`null_baseline` — *"Current code silently writes a wrong-width tensor to
`mj_data.ctrl`"* — is present in the shipped artefact in a mutated form: not a
silent wrong write, a **silent no-op write**, with nobody told.

This is the `T1.11` shape (certified loss, no shipped caller) on a Tier-0
certificate, and it is the strongest single piece of evidence for `D41` that
exists. **It is a reading, not an accusation of loosening.** A venue-scoped
certificate is honest; reading it as global is what is not. The repair is not to
demote `T0.06` and it is certainly not to pad the action vector — it is for
`D41` to carry this and for whoever rules it to know the contract spec is
venue-local.

---

## The audit, section by section

**1. Integrity of the ledger — CLEAN, re-derived rather than cited.**
`experiments/ledger.json` holds 158 keyed specs; **106 carry a PASS as their
latest verdict**. Of those: **0** have a `commit` absent from git (all 106
resolve under `git cat-file -e <sha>^{commit}`); **2** declare no control at all
— `T0.01` (repo imports clean) and `T0.10` (Kaggle round-trip), both harness
plumbing and both owned with a clock at
`t018-explicit-no-control-reads-as-an-unrun-promise` (OPEN, DUE 2026-10-19).
The mechanised half of this section, `T0.18`, reads PASS. I additionally checked
that `T6.01`'s brand-new control is CALLED and not decorative: `_control` is
implemented at `t6_01_full_episode.py:382` and `_check` returns `Status.VOID` on
`c` at `:409` when a planted fault goes unseen — a declared control that the
verdict genuinely depends on. `pass_on_dead_dependency` 6 vs floor 3 is the
standing exception, unchanged in membership: `LF.02←T6.03 BLOCKED`,
`T0.18←T0.13 FAIL`, `T0.19←T0.13 FAIL`, `T1.12←T1.11 FAIL`, `T2.03←T1.08 FAIL`,
`T2.14←T1.08 FAIL`.

**2. Thresholds and controls over time — NO FINDING, and over the full seven
days rather than since my predecessor.** `git diff @{7 days ago}..HEAD` over
`registry.py`, `registry_expansion.py` and `experiments/tests/`: **no `control=`
line removed**; **no seed count reduced** (the only `seeds=` changes are three
additions, all `seeds=3`); **no assertion removed**; **no `_check` gained an
`or`** in a loosening direction. Exactly two module-level numeric constants were
removed, and both moved in the **demanding** direction with an in-source
measurement and date: `CLF_EPOCHS` 300 → **900** (`T2.11`, more training) and
`_SEC_PER_SEED` 355.0 → **531.1** (re-measured on the redesigned rig, an honest
cost increase). The three edits since the 141st audit, reviewed line by line:
`T6.01` **gained** a control where none existed (strengthen-only, no certificate
to protect); `XL.01` v3 **added** the equal-N pooled ratio as a conjunct
*beside* the all-3-seeds per-seed gate — `RATIO_MAX` 0.5 byte-unmoved, the
pooled-for-per-seed substitution and the `2 of 3` form both pre-refused in the
docstring, and `pooled_search_time_ratio` written into `m` **before** any gate
can return so both statistics land whatever the verdict; `LG.14` imports every
bar from `lg_10`/`lg_12` with no local redefinition — `MATCH_MIN`,
`UNANIMITY_MIN`, `SWAP_AGREE_MIN` 0.90, `NULL_MATCH_MAX` 0.35, `VARIETY_MIN`
0.30, `LIVENESS_MIN` 0.80, `UTTER_MIN` 0.50 — which is the strongest available
form of "unmoved". Saying "no finding" here is the result, not a shrug.

**3. Drift from the goal.** Nine slots, every one traced to a `GOAL.md` sentence
and each checked rather than accepted from the journal. `XL.01` v3 + attempt 4 →
*"he dies, he remembers"* and *"Really learning, not appearing to learn"*: a
strengthened gate run to an honest FAIL. `LG.14` → *"The LLM is his mouth, never
his mind"*: structured decode, FAIL at 0.7778/0.7722 against 0.90 with **both**
nulls crushed (0.0444 free-generation, 0.1389 mismatched-constraint), so the
constraint demonstrably carries meaning and the verdict is FAIL rather than
VOID. `T6.01` → *"A living Jack … one life, start to finish."* The
`gpu_unattributed` receipt repair and the reset-time correction → the first
principle's fourth clause. **Nothing here is drift in the sense of serving no
sentence**, and unlike the previous four weeks the day's largest unit was aimed
directly at the creature. On the converse and harder question: `coverage` reads
**0 commitments with no declared spec**, **3 CLAIM-DEAD** (smell,
shelter/building, thermal-kills — the subject of `D42`), and **14 with live
claim specs and nothing passing**, including touch/contact, tool use,
proprioception, plasticity, sleep, fast/slow and hunger/thirst. Curiosity 12
specs / 2 passing; one-brain-unison 28 specs / **1** passing. Those two remain
the thinnest on the board relative to their spec count, and that has not changed
this week. The new `coverage` item worth naming: **4 NEW unrunnable `GOAL.md`
citations** (`GEN.02`, `GEN.03`, `GEN.06`, `GEN.09`, all `welded<-LC.07`) —
owned at `gen-four-revival-needs-an-affordable-lc07-successor` (OPEN, DUE
2026-10-11), so it ages in public.

**4. Is the builder alive and productive? — ALIVE AND THE MOST PRODUCTIVE DAY IN
FIVE WEEKS, with the PASS delta still zero.** 24-hour window: **9 iterations,
9 `rc=0`, PASS delta 0** (106 → 106 across all nine). `dark_slots` **0**, at
floor. No repeated identical failures, no paused loop, no load aborts, no
credit exhaustion — the meter was acted on at **15 %** in the 00:07 slot
(Fable 12 %, pace line ~32 %), so the week reset held and there is headroom. Two
slots lost detached children at session end (19:07, 20:07) and **both were
recovered and journalled by the next live slot**, which is the inherited-work
discipline working. One real complaint about the builder's own self-report, and
it is RANK 3: the 00:07 slot's 2129 MB child peak reached its commit message and
its slot summary but not `ladder.log`'s `MEMORY` channel, because the instrument
could not see it. The creature gate reads **`T6.01` MOVED**, resetting the
`NONE` streak that had reached its declared maximum of two — and the builder's
reasoning for that choice, written out in the slot summary, is correct on the
arithmetic: all three gates were closed to *runs* (`XL.01` already recorded at
21:07, `T2.01`/`T6.01` behind `T1.08`'s FAIL, and `T1.08` Step 2 explicitly a
desk's to route off Step 1's number, which no desk has sat on since it landed).

**5. Compute honesty — the receipt defect is repaired and the repair is the
right shape.** `gpu_unattributed_jobs` is **back to 21, AT its declared floor**.
The fix at `1ac24e4` is what the 141st's FTB 1 asked for and not the cheap
version: `experiments/gpu_submissions.jsonl` gains an **append-only
`"phase": "attribution"` record** joined to the existing two by `attempt_id`,
naming `spec T1.08 / spec_phase probe` with its evidence and its reason — **the
original `attempt` and `result` lines are byte-untouched**, and the dispatch
path was fixed in the same commit so it cannot drop the name again.
`gpu_hours_no_verdict` moved `TOTAL 49.49 → 49.77 h` with `T1.08` going
`0.36 h / 1 attempt` → `0.63 h / 2 attempts / 1 verdict`, i.e. the hours landed
where they were spent. **`2026-W40`: 0.2708 h drawn of 30 free Kaggle GPU-hours,
expiring Saturday 2026-10-10** — ~29.73 h with four days left and a *designed*
buyer (`T1.08` Step 2b). The three expired weeks should keep being said out
loud: W37 1.379 h, W38 0.918 h, W39 1.072 h of 30 each — **~86 free GPU-hours
lost in three weeks.** There is no GPU hour this week without something to show
for it.

**6. Stuck decisions.** `decisions --check` exits 1 on one class.
**`MEANS-ESCALATED`: none** — nothing a measurement could settle sits on the
owner's desk. **`UNDECLARED`: 0, AT floor** — so there is nothing to arm, and I
am **not** manufacturing an entry to satisfy the quota; the ratchet may shrink
and never grow, and 0 is where it stops. **`OVERDUE — DEFAULT DUE TO FIRE`:
none.** `D41` and `D42` stay armed with monotone defaults, `decide_by
2026-10-18`; I am not re-arguing either, but **`D41` gained material evidence
last night that is not in its entry** — RANK 2. Three `CONDUCT-DESK` entries are
past their dates and all three are desk-executable, not the owner's: `D33` (13 d,
MOOT — no desk can clear it by firing anything, re-derived and agreed), `D35`
(12 d, owned via `d35-none-quota-has-no-satisfying-move`, DISPOSITIONED DUE
10-14), and **`D38` (2 d, and the desk that owns it sat on 10-05 for seven acts
and did not take it)** — its subject binds only on Sundays, so the earliest it
can bite is 2026-10-11. Nothing was quietly acted on without being recorded: all
4 owner-asks the tool finds on `PROGRESS` resolve to register entries (`D37`,
`D33`, `D41`, `D42`), and `decisions_unrouted_owner_ask` /
`decisions_vanished_owner_ask` both read **0 at floor**.

**7. Bakeoff hygiene — and this is where RANK 1 lives.** `champions --check`
exits **0** with all 10 violations **at their declared floors**; no seat lost a
door in this window; `champions_unwinnable` 4, `champions_trigger_debt` 3. The
standing bad news is unchanged and not re-litigated: the Learning-core seat is
held **BY VERDICT off a VOID** (`LC.03`) with every re-open trigger a closed
door, and the World seat is held BY VERDICT declaring neither a deciding row nor
a rematch trigger. On `docs/DECISIONS_RESOLVED.md` itself I found **no decision
made without a learning gate, no VOID treated as a verdict, and no winner chosen
inside a noise margin** in this window. What I found instead is that the page is
**missing its last two entries** — `D36` and `D37` — which is RANK 1, and which
also means a `DECISIONS_RESOLVED.md` reader doing this very section sees `D34`
as the most recent firing while two more have happened.

**8. The honest summary — closer to a curious humanoid, or only to a longer list
of green ticks?** **Neither, again — but for the first time in weeks the reason
changed, and the change is worth more than a tick.** The list did not grow:
`demonstrated` was 106 at 16:07 yesterday and 106 after nine `rc=0` slots. What
happened instead is that this project **looked at the thing it is building, for
the first time**, and the thing looked back with three facts: he emits
seventeen numbers into a body with fifty-seven joints and every one is thrown
away; his scene has three cameras and his code asks for a fourth that does not
exist; and a session in which no world was ever built reports that it completed.
For four weeks the finding has been *"the apparatus is improving while Jack sits
still."* It is now sharper and worse: **the apparatus and Jack were never
connected, and a harness was the only instrument that could tell us.** That is
the substance of `D41`, it is exactly the question the owner has on their desk,
and the right thing happened to it in part — the builder measured it, refused
the cheap repair, and wrote the generalisable lesson down. The wrong thing
happened to it in the part that decides whether it survives: no row, no clock,
no entry in the decision it belongs to. Honesty that reaches no page is
indistinguishable, in a week's time, from not having looked. **Jack did not
climb anything last night. But for the first time we know why, in a number, and
the number is 17 against 57.**

---

## FOR THE BUILDER

0. **THE 138th–141st AUDITS' ORDERS AND `PROGRESS` FTB 1–7 ARE STILL LIVE, AND
YOU DISCHARGED THREE MORE LAST NIGHT.** Credit where owed: 141st FTB 1 (the
`gpu_unattributed` receipt — done in the right shape, append-only, path fixed in
the same commit), 141st FTB 2 (`XL.01`'s additive conjunct — written *and* run
to its pre-registered FAIL), and 141st FTB 4 (the reset-time correction at
`9e8f762`, committed as inherited work). **Still open and not re-ranked by me:**
141st FTB 3 (`lib_seal.sh` marks a page by whether the FILE is dirty, never by
who wrote the CONTENT — `docs/PROGRESS.md` still wears an `UNVERIFIED` draft
banner over the completed 2026-10-04 FULL page, and it is now two days stale as
well), 140th FTB 2 (the `pace-forecast` reading off `claude_usage.py`'s own
`resets` field), 140th FTB 3, 138th FTB 1 (the steering-size growth fit, still
the cheapest item on your board), `PROGRESS` FTB 2 (`cpu<8h` rung), FTB 4 (the
ladder-wide per-seed-boolean audit, report-only), FTB 5 (`review_prompt.md` —
write the page first).

1. **TRANSCRIBE `D36` AND `D37` ONTO `docs/DECISIONS_RESOLVED.md`, AND THEN MAKE
THE GAP IMPOSSIBLE TO REPEAT.** This is RANK 1 and it is two commits.
**(a)** Write `## D36 — RESOLVED BY ARMED DEFAULT …` and `## D37 — RESOLVED BY
ARMED DEFAULT …` onto `docs/DECISIONS_RESOLVED.md`, quoting each firing record
from `docs/DECISIONS_NEEDED.md` (`D36` at `:8961`, `D37` at `:9953`) with its
options-not-taken, per the **`D13` rule** and the `D31`/`D32`/`D34` precedent —
`251ebcd` is the worked example and it was itself an ordered audit repair
(119th FTB 3+4). This is yours and not mine: the overseer stays inside its own
file set, which is why `D36`'s own record says *"OWED BY THE REVIEW, NOT BY ME"*
and why it has sat nine days. **Measured before you start: 29 of 31 recorded
firings are on that page; `D36` and `D37` are the two missing.** Check
`decisions.firing_commits_by_record` sees both afterwards.
**(b)** Add the set-arithmetic reading that would have caught it: the firings
recorded on `DECISIONS_NEEDED.md` (`^##\s+D\d+\s+\S*\s*RESOLVED BY ARMED
DEFAULT`) **minus** those declared on `RECORD_PAGE`, reported beside
`FIRING-DIFF` on every `decisions --check`. **Reporting-only and unfloored for
now** — do not invent a ratchet for it in the same commit that creates its first
reading, and note that freeze clause 2 forbids a *new organ*, not a new reading
inside `decisions`. Name in your commit that `c10a128` was in **neither**
channel and that both unaudited diffs audit **CLEAN**, so the exposure was zero
and this shipped at floor — which is the only cheap moment, exactly as
`SYSTEM.md` says of `3b2e38b`.
**(c)** `T0.28`'s P17 asserts the two-channel property against a **synthetic
page literal** and never reads `docs/DECISIONS_RESOLVED.md`. Say so in P17's
comment. **Do NOT re-scope P17 to the live page in the same slot** — that is a
Tier-0 certificate change with a staleness bill and a design question about
whether a battery should depend on document state, and the design is the
Review's under `seven-instrument-readers-are-gated-by-no-spec` (DUE 10-19).
Price the bill, state it, stop.

2. **ROUTE LAST NIGHT'S THREE RUNTIME FINDINGS, AND PUT THEM IN `D41`.** You
measured them, you refused the cheap repair, and you said the repair *"routes
through the Review"* — then wrote no row, so nothing can route. Open **one**
`ROUTED:` row (they share a cause), carrying all three with the numbers:
`crash_frames 177/177` at action width 17 vs `nu` 57; `humanoid_full.xml` has
cameras `track`/`left_eye`/`right_eye`/`head_cam` and **no `"eye"`** while
`VirtualWorld.py:714` looks up `"eye"` and `playground.py:537` emits it; and the
800×600-default worldless session that reports `completed 1`. Use
`review-queue`'s own **"next date with room"** (it printed `2026-10-15`), not a
hand-picked date, and declare `WAITS-ON`. **Then add the measurement to `D41`'s
entry as evidence** — `D41` is armed on the owner's desk with `decide_by
2026-10-18` and your own commit calls this *"runtime-side corroboration of
`D41`'s bridge drift"*; it is the fourth instance and the only runtime one, and
it is not in the entry the owner will read. **Pre-refused, and your own slot
summary already refuses them:** padding or truncating the action vector, and
renaming a camera to `"eye"` to turn a red green. Record in the row that
`T0.06`'s certificate is venue-local (`EXPECTED_NU = 17`, Gymnasium
`Humanoid-v5`) with `kills = "Every locomotion result"` — **and do not demote
`T0.06`**; that is a desk judgement about certificate scope, not yours, and its
control is the thing that caught this.

3. **MAKE THE TENANT RAM GUARD SEE A BREACH THAT DIES WITH ITS SLOT.**
`proc_memory_report` (`scripts/lib_procwatch.sh:318`) walks `/proc/[0-9]*` once
at slot end, so a child that peaks over the ceiling and exits is invisible —
while the function's own docstring claims it names *"every project python"* and
that *"a memory breach is defined by what it did."* Last night's instance:
`T6.01`'s child peaked at **2129 MB against the 1536 MB ceiling** inside the
00:07 slot, and `ladder.log` has no `MEMORY` line for that slot. The cheapest
honest repair is to have the loop harvest the **child's own reported peak** —
`T6.01` already writes `maxrss_mb` via `resource.getrusage` at
`t6_01_full_episode.py:285`, and `run_spec` children can report the same — so an
exited process's high-water mark reaches the same `MEMORY` channel the live scan
uses. **Keep the discipline exactly as it is: NAME, NEVER KILL, and gate
nothing** — the ceiling's own validity is an open owner question (63rd audit
B2), so do not touch `JACK_MEM_CEILING_MB` and do not make a breach refuse a
slot. If the harvest cannot be made reliable this slot, the **minimum** is to
correct the docstring so it claims only what it does — a guard's comment is a
capability claim and law 1 binds it, which is the 63rd audit's own sentence. And
before the 30-minute `T6.01` recording run is ever scheduled, that footprint
needs a number: the smoke held 2129 MB for half a minute.

4. **SEVEN QUEUE ROWS FALL DUE TODAY AGAINST A MEASURED CAPACITY OF SIX, AND ONE
CANNOT BE DISCHARGED BY THIS CYCLE.** `run review-queue`'s `IMMINENT` block says
so before the dates pass, which is the only time anything can be done about it.
`OVERDUE` is currently **0** — the first clean reading in weeks, bought by the
10-04 FULL's disposal work — and it will not survive midnight untouched. **This
is the desk's to disposition and not yours to act on**; it is here so the
builder's slot does not land on top of it unaware, and so the reading is in a
committed document before the dates break. Also unchanged and not re-ranked:
the drain reads **UNBOUNDED** at 85 live rows.

---

## FOR THE OWNER

**1. Your creature was measured for the first time last night, and he cannot
move or see. Nothing is asked; this is the one thing on this page you should
read.** Implementing the harness for `T6.01` ("one life, start to finish") meant
driving the companion you actually ship, on the body you actually ship, and it
produced three facts: his brain emits a **17-wide action into a 57-actuator
body** and every frame is refused — **177 of 177**; his scene has three cameras
(`left_eye`, `right_eye`, `head_cam`) and his code asks for one called `"eye"`
that does not exist, so he runs on proprioception and touch alone; and at the
shipped default window size the world fails to build and the session **reports
that it completed**, because the companion loop swallows its own errors at debug
level. The builder found all three, refused the cheap repairs by name, and wrote
the lesson down. **This is the direct, runtime answer to `D41` — the question
already on your desk about which artefact is Jack, the ladder's rig or the
shipped pipeline — and it is the fourth instance of that drift and the first
measured in a running creature.** `D41` stays armed, `decide_by 2026-10-18`,
with a monotone default (iii) that holds both artefacts and repairs each
instance on its own row; I have **not** touched it, and I have ordered the
builder to put this evidence into the entry so you are not reading `D41` without
it. The Review's standing recommendation is (ii): declare the ladder's rig the
system of record and put `TrainingPipeline.py` on the ablation block. **This
measurement strengthens that recommendation and does not change it** — and it
adds one thing the Review did not have: `T0.06`, a Tier-0 PASS whose own `kills`
field says *"Every locomotion result"*, certifies the dimension contract at
`nu = 17` in a venue your shipped body is not.

**2. A safety check on your own governance reports 100 % coverage of a
population it does not fully enumerate. Nothing is asked and the measured
exposure is zero.** Every armed default that fires unattended is supposed to
have its commit diff checked for two things: no `GOAL.md` edit, no numeric bar
moved. `SYSTEM.md` says a firing is found by **two independent channels** so
that it cannot rest on the author's own word, and calls that gap closed.
Measured at source this sitting: **`D36`'s firing commit is in neither channel,
and has been for nine days** — so its diff has never been checked — and
**`D37`'s is in one**, the author's-word channel the second one exists to
backstop. Both diffs **audit clean**, so nothing bad happened; what is wrong is
the claim of completeness. The cause is one missing clerical act — two firing
records that were never copied onto `docs/DECISIONS_RESOLVED.md`, where the
second channel looks — and **this organ named that debt nine days ago inside
`D36`'s own record, named who owed it, and named exactly this consequence.**
Sixteen audits and a Review sitting passed over it. The repair is two commits
and it is ordered. **The reason this is on your page rather than only the
builder's:** the thing that makes an unattended firing safe for you is this
check, and for nine days it was quietly smaller than it said it was.

**3. NO-DECISION — the builder is back and productive, and the tick count still
did not move.** Nine slots, nine `rc=0`, **106 → 106 demonstrated**;
`dark_slots` **0**; the meter at 15 % with the pace line clear, so there is
headroom. The day's work was real — a strengthened `XL.01` run to an honest
FAIL, `LG.14` implemented and FAILed with both its nulls crushed (so the FAIL
means something), the first harness for `T6.01`, and three of my predecessor's
five orders discharged. **`2026-W40`: 0.2708 h drawn of 30 free Kaggle
GPU-hours, expiring Saturday 2026-10-10** — ~29.73 h left, four days, and a
*designed* buyer in `T1.08` Step 2b whose routing is owed by the Review off
Step 1's `eval_cv_pct 0.52`. W37/W38/W39 lost **~86 free GPU-hours** unbought;
that number stops growing only if Step 2b is routed this week. All four organs
fired within cadence; the Review's 10-05 DAILY died `rc=124` for the **seventh
consecutive time**, which is why `docs/PROGRESS.md` is still the 2026-10-04 page
wearing a false `UNVERIFIED` banner (141st RANK 2, repair ordered, undischarged).

**4. A tenant-safety guard on this shared box can only see a RAM breach that
outlives the slot it happened in. Nothing is asked.** This box serves four
paying customer agents, and `SYSTEM.md` holds us under ~1.5 GB. The watcher that
names an excess reads live processes only, so a child that peaks and exits is
invisible — and one did, last night, at **2129 MB, 38 % over the ceiling**,
named in the builder's commit message and in no log. Repair ordered, **as a
report and never as a gate**: whether the ceiling itself is right remains your
open question from the 63rd audit, and the discipline stays NAME, NEVER KILL.
One thing to know before the eventual 30-minute `T6.01` recording run is
scheduled: the half-minute smoke held that 2129 MB.

**5. NO-DECISION — `D33`, `D41`, `D42` are pointers, deliberately.** `D33`:
`decide_by` was 2026-09-23, now **13 days** past, sole cause of the one broken
ratchet class on your register (`decisions_default_action_expired` 1 against
floor 0); its default is **MOOT rather than merely expired** — its object went
terminal when `w1-world-edit-window` was `DECLINED` — so **no desk can clear it
by firing anything.** I re-derived that and agree. `D42` (which of three
claim-dead commitments gets a successor) is unchanged: `claim_dead` has read
**3 for eleven days** — smell, shelter/building, thermal (*"too cold kills
him"*) — and `coverage` counts **6 distinct commitments or seats with no live
path at all**. Both armed, monotone, `decide_by 2026-10-18`. I am not re-arguing
either. Also past its date and **not yours**: `D38`, `CONDUCT-DESK`, 2 days
stale, desk-executable, and the desk that owns it sat on 10-05 and did not take
it; its subject binds only on Sundays, so nothing is at risk before 10-11.

**6. NO-DECISION — what this sitting did not do, named rather than omitted.** I
did not audit the **cognitive half** of the capability-completeness list —
attention, working memory, imagination, self-model, theory of mind, teaching.
This is the **third consecutive audit** to defer it and I am saying so plainly;
it is owned with a clock at
`completeness-audit-2026-09-13-the-cognitive-half-is-the-hole` (OPEN, DUE
2026-10-11), so it ages in public rather than inside this paragraph. I spent the
clock on the six builder slots my predecessor could not see and on the firing
population, and I would make the same call again: RANK 1 is nine days old and
RANK 2 is one slot old, and neither was going to be found by looking harder at
the ladder. I also did **not** arm a new decision, because
`decisions_undeclared` reads **0 and AT floor** — manufacturing an entry to
satisfy the quota would invert the rule's purpose. **One disclosure about my own
conduct:** `scripts/ladder_prompt.md` stands at **123,246 bytes, 1,754 below its
self-imposed 125,000 ceiling** and 7,826 below the 131,072 exec cliff, growing
+94 B/day — **19 days to the ceiling.** I added nothing to it.

---

## CLOSING BLOCK — every instrument re-run AFTER my last act, not quoted from the top of the sitting

| instrument | exit | reading |
|---|---|---|
| `coverage` | **2** | 0 commitments uncovered; 3 CLAIM-DEAD; 14 live-but-nothing-passing; `unreachable` 96 of 257 vs baseline 95; `pass_on_dead_dependency` 6 vs baseline 3; 4 NEW unrunnable `GOAL.md` citations (GEN.02/03/06/09, owned, DUE 10-11); queue depth 6 dispatchable, **all 6 VOID — 0 FRESH** |
| `decisions --check` | **1** | `D33` alone, `DEFAULT-ACTION-EXPIRED` against floor 0 — the class no desk can clear by firing anything; `MEANS-ESCALATED` 0, `UNDECLARED` 0 at floor, `OVERDUE-DUE-TO-FIRE` 0; `D35` stale 12 d, `D38` stale 2 d, both CONDUCT-DESK |
| `champions --check` | **0** | 32 seats, 10 violations **all at declared floors**; `champions_unwinnable` 4, `champions_trigger_debt` 3; no seat lost a door |
| `run review-queue` | **2** | 52 OPEN / 3 HELD / 85 live; **7 violations, all `HOLD-ON-A-RESOLVED-BLOCKER`**, deliberately not laundered; **`OVERDUE` 0**; oldest live 43 d; **7 rows due 10-06 against measured capacity 6, 1 undischargeable**; drain UNBOUNDED |
| `run status` | **2** | 106/257 demonstrated; floors **3 ABOVE** (`decisions_default_action_expired`, `pass_on_dead_dependency`, `unreachable`), 0 BELOW, 0 UNVERIFIED |

**ABOVE-floor MEMBERSHIP, never just its count** (141st RANK 1's lesson, applied):
yesterday 18:4x was `gpu_unattributed_jobs · decisions_default_action_expired ·
pass_on_dead_dependency · unreachable`; now it is
`decisions_default_action_expired · pass_on_dead_dependency · unreachable`.
**The count fell 4 → 3 and the departure is a real repair** —
`gpu_unattributed_jobs` 22 → 21, back at floor, by the builder naming the
receipt rather than by anyone raising a constant. Nothing rose into the vacancy.

`ratchets vs committed readings (HEAD): 7 MOVED` —
`fail_unowned_owned_forms` queue-row `31 -> 33`, `gpu_hours_no_verdict`
`TOTAL 49.49 -> 49.77 h` (`T1.08` `0.36 h / 1 attempt` -> `0.63 h / 2 attempts /
1 verdict`, `PROBE` `3.59 -> 3.86 h`), `pass_on_dead_dependency 5 -> 6`,
`review_queue_net_arrivals 32 -> 1`, `review_queue_piled_on 4 -> 9`,
`review_queue_violation_forms` (`OVERDUE` 6 -> 0, `HOLD-ON-A-RESOLVED-BLOCKER`
8 -> 7), `review_queue_violations 14 -> 7`. No counter refused to compute.
Quoted BEFORE any `ratchets record`.

**Ledger integrity, re-derived this sitting rather than cited:** 158 keyed
specs, **106 PASS latest**, **0** with an absent `commit`, **2** with no
declared control (`T0.01`, `T0.10` — both owned, DUE 10-19). Verdict census
`{PASS 106, FAIL 35, VOID 16, BLOCKED 1, NOT_RUN 0, ERROR 0}`.

**Builder:** 9 iterations / 9 `rc=0` / PASS delta 0 in 24 h; `dark_slots` 0;
last `rc=0` 2026-10-06T00:29:02; creature gate `T6.01` MOVED, streak reset.
**Organs:** builder hourly (00:07, live), overseer 6-hourly (this sitting),
Review daily (10-05 06:37, `rc=124` — **7th consecutive `INCOMPLETE`**; next
06:37 today), field watch weekly (10-05 05:59, its Monday). **Silence is never
success, and there is none today.**

Working tree was clean before this file was written; `docs/OVERSIGHT.md` is a
`PROSE_DOCS` member (`experiments/protocol.py:139`) and carries no per-spec
staleness bill, which is why this page was written in place rather than staged
through `/tmp`.
