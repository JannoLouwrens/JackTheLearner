# OVERSIGHT.md — the overseer's current-state report

> Written by the overseer organ. **Current state, not a log** — each audit
> rewrites this file. Findings are ranked by how much damage they do to the
> trustworthiness of the ledger, not by how interesting they are.

**2026-09-23 18:37–19:0x UTC — the 110th audit.** Six hours after the 109th
(12:37 today). The window is the builder's six live slots 13:07–18:07, four
commits (`096a8ab`, `f7900b5`, `a4bae41`, `623b16d`, `9663f23`, `415a94d`) and
two ledger events (`HR.1` attempts 3 and 4, both FAIL). Every instrument here
was re-run against the tree immediately before committing.

---

## VERDICT: ON TRACK — with one spec whose registry text says the opposite of the run that is on its ledger row

Six slots, all `rc=0`, every one of the 109th audit's `FOR THE BUILDER` items
discharged, no threshold moved in any direction, and `run verify` still
re-judges 109 PASS entries with **0 verdicts that no longer re-derive and 0
gates that ignore their control**. The builder spent the day executing two
pre-registered arms of a Review disposition and reporting both refutations
without re-tuning anything. That is the job.

What I do not endorse is where the venue swap was written down. `HR.1`'s binding
FAIL was measured on **VCTK-Corpus-0.92**, and the registry text that `spec_sha`
hashes — **byte-identical across all four attempts** — names LibriSpeech, says
`REJECTED: VCTK … does not fit on this box at any observed free-space level`,
and states a hypothesis conjunct (`CROSS-SESSION test material`) that the run
deliberately did not deliver. Everything was disclosed; none of it was disclosed
in the field an auditor greps. Nothing is bought today because the verdict is a
FAIL — but `HR.1` is the FIXTURE spec whose PASS would certify the corpus
`HR.2`/`HR.3`/`HR.4` get scored on, and the sentence that PASS would buy is
currently false about the run.

---

## RANK 1 — `HR.1`'s LEDGER ROW AND `HR.1`'s SPEC TEXT NOW DESCRIBE DIFFERENT EXPERIMENTS, AND `spec_sha` SAYS NOTHING MOVED

This is a **record-location** finding, not an honesty finding. The builder
declared the venue swap before the run, in the test docstring, in
`LOOP_JOURNAL.md`, in the queue row and in the commit message, and it recorded
the one provenance discrepancy against itself rather than reconciling it
silently. Nobody hid anything. The defect is that the registry — the only text
`spec_sha` covers and the only text `run verify` and `coverage` read — was never
amended, so the project's own integrity hook reports "the claim text did not
move" across a total replacement of the corpus.

### Three sentences in the live registry entry, against the run on the ledger

| registry `HR.1` (unchanged, `spec_sha` `769b55d0` on attempts 1–4) | attempt 4, `ran_at` 2026-09-23T17:11:38, commit `9663f23` |
|---|---|
| `Corpus: LibriSpeech dev-clean … split 20 enrolled / 20 impostor` | VCTK-Corpus-0.92, `n_enrolled` 20, **`n_unknown` 40** |
| `REJECTED: VCTK … does not fit on this box at any observed free-space level` | VCTK fetched, extracted (88,328 flac) and scored |
| `/data free space was observed swinging between 725 MB and 4.8 GB` | `/data` measures **67 G free** of 100 G today |
| HYPOTHESIS: `… disjoint enrolment/test utterances, CROSS-SESSION test material …` | cross-**MICROPHONE**, same booth, **simultaneous** (mic1 DPA 4035 → mic2 MKH 800) |

The fourth row is the load-bearing one. VCTK documents no session boundaries;
the builder said so, argued that cross-microphone is the sharper venue for a
claim about *channel* cues, and disclosed in the queue row that "cross-mic
controls for equipment, not occasion — the residual is speaker-**or**-session-
borne." I agree with the argument. It is not in the hypothesis, and the
hypothesis is what a PASS would certify.

### Why this matters even though the verdict is a FAIL

1. **`spec_sha` is the auditor's only hook for "did the claim move?"** `run
   status` prints `6 PASS row(s) predate spec_sha: whether the claim text moved
   since cannot be answered from the record` — the whole point of the field.
   Here the field is present, unchanged, and answering the question **wrongly**:
   the claim text should have moved three times and did not.
2. **`HR.1` is a fixture spec.** Its PASS is the certificate `HR.2`–`HR.4` would
   be scored under. The docstring's arm-(c) branch says so explicitly: *"PASS →
   the VCTK fixture is honest; HR.2–HR.4 unblock on VCTK."* On the day that
   happens, the registry sentence quoted into every downstream row says
   LibriSpeech, cross-session, VCTK-rejected.
3. **The Review rules the family's fate off this row.** The
   `hr1-clean-stratum-is-a-microphone-measurement` disposition (DISPOSITIONED,
   `DUE 2026-09-30`) now carries all three spent arms. It will be read beside a
   registry entry that rejects the corpus the result came from.

### What is NOT wrong, said plainly because it would be easy to read this the other way

- **No gate moved.** `git diff 5283aad..HEAD` over the test shows `LEAK_EXCESS`
  0.05, `CONTROL_FLOOR_EXCESS` 0.15, `SEEDS`, `PROBE_*`, `N_BANDS`,
  `NOISE_SNR_DB`, `RT60_S` and every bar byte-identical across all three arms.
- **The provenance delta was self-reported.** The registry recorded
  `Content-Length 11,749,118,645` on 2026-08-09; the live server served
  **11,747,302,977** post-DSpace-migration, a 1,815,668-byte difference, and the
  builder wrote that delta into the source rather than calling it a match.
- **The disk claim it broke was the registry's, and it was stale.** The Review
  had already confirmed that on 09-23; the fetch was legal under `D19`.

**The repair is an AMENDMENT, not a re-run** — and it must land *before* the next
`HR.1` attempt, because after a PASS it is a correction to a standing
certificate instead of a correction to a spec. See `FOR THE BUILDER` 1.

**And the instrument gap under it:** nothing in this project compares the venue a
run actually used against the venue its spec declares. `run verify` checks the
gate replays and the control ran; `STEERING-METRIC-MISMATCH` checks quoted
numbers; `run stale` checks code shas. A corpus swap is invisible to all three.

---

## RANK 2 — `D33`'s DEADLINE IS TONIGHT, ITS DEFAULT IS ALREADY SPENT, ITS RECOMMENDATION WAS WITHDRAWN NINE HOURS AGO, AND `decisions --check` CANNOT EVER REPORT IT OVERDUE

`D33` carries `decide_by: 2026-09-23`. In 5 hours 20 minutes it passes. Here is
the state of every moving part:

| part | state |
|---|---|
| default **(i) RE-DATE ONCE MORE, TO 2026-09-23** | **already executed** 2026-09-20 in `c9aca70`, before the entry was written |
| the row's stop-rule | *"if 2026-09-23 breaks, W1 is not re-dated again by this desk"* — a fifth re-date is **forbidden** |
| recommendation **(ii) move design authority** | **withdrawn by its own author** at 09:35 today (`d9f568f`), central fact falsified |
| the owner | has not ruled |
| `decisions --check` tomorrow | prints the **identical line** it printed today |

**There is nothing left to fire.** The default is spent, the re-date is barred,
and the ask has been retracted. `w1-world-edit-window` goes `OVERDUE` at 00:00
(the queue reader *does* catch that half), leaving the Review exactly two legal
dispositions at 06:37: **ACT** or **DECLINE**. The addendum makes ACT the cheap
one — it prices registering `W1.01`/`W1.03`/`W1.04` from design text published
2026-09-06 as *"desk-shaped, small, and mine"*, with the honest cost that
`unreachable` rises above its floor of 96 and needs a growth-log entry.

### The instrument half, and it is the part no other organ covers

`experiments/decisions.py:1291`:

```python
if cls == "conduct":
    violations.append(("CONDUCT-DESK", did,
                       f"desk-executable, not the owner's (due {d.get('decide_by')}) "
                       "— execute it, report it, do not ask. Listed so a stale "
                       "conduct entry cannot silently self-approve."))
    continue
```

The `continue` lands **before** `decide_by` is parsed (line 1310) and **before**
`rows.append(... "overdue": (today - due).days ...)` at line 1371 — the only
place an overdue day-count exists. `OVERDUE — DEFAULT IS DUE TO FIRE` is printed
off that field alone (line 2124). **A conduct entry can therefore never go red.**
The due date is interpolated into a static string and never compared to anything.

The branch's own comment says it exists *"so a conduct entry that has gone stale
is still visible rather than silently self-approving."* **The mechanism it uses
cannot tell stale from fresh.** `D33` will print `(due 2026-09-23)` tomorrow, next
week and next month in exactly the same ink.

Blast radius today is **one entry** — `D33` is the only live conduct decision
(`D28`, reclassified to conduct on 09-21, is closed). That is why this is RANK 2
and not RANK 1. But it is the live one, and its date is tonight.

---

## RANK 3 — THE INSTRUMENT BUILT TODAY TO PRICE THE 109th AUDIT'S FINDING HAS ZERO CALLERS

Yesterday's RANK 1 ended with the diagnosis that `T4.06`'s binding conjunct was
decided at 6.9% of its own anchor spread and *"nothing computes the ratio and
prints it."* The 18:07 slot built `experiments/resolution.py:anchor_margin`,
208 lines, with a self-test that mechanically re-derives both of that audit's
calibration points from the committed ledger row (+6.9% with 1/3 seeds
regressing on the certified arm; −56.7%, 8.2× that margin, 3/3 agreeing on the
refuted one). The arithmetic is right and I checked it.

**Nothing calls it.** Grepped across `experiments/`, `scripts/` and every `.md`:

- the only `anchor_margin(` call sites in the repository are **inside its own
  `_selftest`**;
- `run status` does not print it — I read the full tail;
- no `Spec` names `resolution.py` in `IMPL_DEPS`, so no certificate stales on it
  and no re-buy exercises it;
- the builder's own journal says it: *"no consumers for `anchor_margin` — wire
  it where a disposition asks, do not [pre-empt]."*

So the next bakeoff decided against an in-run anchor will be decided exactly as
`T4.06` was, unless a human remembers to invoke a module by hand. This project
has an OPEN queue row for this class of defect
(`oversight-for-the-builder-has-no-reader`, `DUE 2026-09-30`), and the Review's
own sentence from this morning is the right epitaph: **"Printing a number is not
watching it."** This one does not print.

**And the diagnosis now reads as discharged.** The builder added an `UPDATE
2026-09-23` line to the `LESSONS.md` entry — deliberately, *"so the diagnosis
does not outlive its repair"* — which says the check *"computes and PRINTS the
margin … for any conjunct decided against an in-run anchor."* The verb is true
of the function and false of the system: no path reaches it. A future reader of
that lesson will see a scar marked healed. The `UPDATE` line was the right
instinct and it is now the thing that has to be earned.

**What I explicitly endorse, so the repair is not over-read.** The builder
declined to declare a cutoff, declined to make it a verdict, and declined to
edit `bakeoff.py` because that file sits in `LG.13`'s `IMPL_DEPS` — it checked
the stale cost (0 standing PASS staled) before choosing a new module. All three
choices are right and none of them should be reversed. **Declining a verdict is
not the same as declining a reader.** The repair costs one reporting-only print
and arms nothing: see `FOR THE BUILDER` 2.

---

## RANK 4 — `HR.1`'s FAIL IS ITSELF ONE CLIP WIDE, AND THE METRIC ITS ROW LEADS WITH UNDERSTATES IT BY 12×

The verdict the Review is about to rule a whole sense-family on:

| | seed 0 | seed 1 | seed 2 | bar |
|---|---|---|---|---|
| `clean_acc` (cross-mic stratum) | **0.125** | 0.075 | **0.10625** | 0.10 |
| in clips of 160 | 20 (over by 4) | 12 (under by 4) | **17 (over by 1)** | 16 |
| `margin` (what `_check` reads) | −0.025 | **+0.025** | −0.00625 | ≥ 0 |

Binomial sd at chance (p = 0.05, n = 160) is **0.0172**. The bar sits 2.9 sd
above chance; the worst seed sits **1.45 sd above the bar**; one seed passes.
The builder flagged this itself as an `UNSATURATED-NULL` and routed it rather
than reading it as a clean refutation — that is the right conduct and it is why
this is RANK 4.

**Two things are worth pinning anyway.**

**(a) The gate is correct and I want that on the record.** `_check` reads
`min(margins) >= 0.0` — worst-seed, the conservative direction. No aggregate
hides anything here.

**(b) The headline metric is not what the gate reads, and it is 12× smaller.**
`min_channel_leak_margin` on the ledger row is **−0.00208**. That is the
seed-MEAN of the per-seed min-over-strata margin. The gate's actual statistic is
**−0.025**. Anyone quoting the row's headline understates the failure by a factor
of twelve, and the name says `min`. This is the live class of two OPEN queue rows
— `gates-that-measure-something-other-than-what-they-say` (`DUE 2026-10-04`) and
`aggregate-hides-worst-seed` (`DUE 2026-09-29`) — arriving in a metric nobody
routed.

**(c) And the design's guard is good, so say so.** Across the three arms the
claim statistic fell 0.427 → 0.274 → 0.125 against a fixed 0.10 bar, while the
planted-leak control fell 0.78 → 0.54 → 0.32 against a fixed 0.20 floor —
headroom 3.9× → 2.7× → **1.6×**. Both numbers approach their bars from opposite
sides, which is precisely the pincer that stops "the corpus is honest" being
manufactured by destroying the channel. A fourth arm would probably buy the PASS
and might VOID the instrument in the same run. **The disposition pre-registered
three arms and the builder invented no arm (d).** That is the discipline working.

---

## The audit, section by section

**1. Integrity of the ledger — CLEAN, with RANK 1's caveat about where a
venue lives.** `run verify`: **109 PASS re-judged, 107 controls probed, 0
verdicts that no longer re-derive, 0 gates that IGNORE their control, 0 gates
that could not be replayed, 0 entries that could not be audited.** Every
implementation on disk; every `commit` present in git. `HR.1` attempt 4
specifically: impl at `experiments/tests/hr_1_voice_corpus_honest.py`, commit
`9663f23` present, `control` declared in the Spec, `control_fn=_control` wired
into `run_spec`, and `control_metrics` recorded — planted same-mic leak
0.3625/0.28125/0.30625 against its 0.20 floor on **every** seed, so the VOID lane
was tested and did not fire. **The control was run and it testified.** Carried
and unchanged: 2 PASSes with no control (`T0.01`, `T0.10`); 3 UNBACKED
CERTIFICATES (`LF.02`, `T2.03`, `T2.14`) at the declared floor of 3; 6 PASS rows
predating `spec_sha`; 2 predating `impl_sha`, of which `T2.02` is stale by
content.

**2. Thresholds and controls over 7 days — CLEAN, and I looked at the channel
the grep cannot see.** No registry change since `aa7d49c` (the `T4.06`
registration the 109th audit already cleared). The window's only gate-adjacent
edits are `HR.1`'s two arms, and the diff across all of them
(`5283aad..HEAD`) touches **no threshold, no seed count, no assertion, no
control**: `LEAK_EXCESS` 0.05, `CONTROL_FLOOR_EXCESS` 0.15, `SEEDS (0,1,2)`,
`PROBE_STEPS/LR/WD`, `N_BANDS`, `QUIET_FRAC`, `NOISE_SNR_DB`, `RT60_S` all
byte-identical; `spec_sha` `769b55d0` across all four attempts proves the
registry bars did not move either. No `_check` gained an `or`; no baseline was
raised. **The channel worth naming: three successive FIXTURE edits each moved
the measured statistic toward a fixed bar** (0.427 → 0.274 → 0.125 against
0.10). That is the shape silent loosening takes when the threshold is
untouchable, and it is **legal here on three counts** — each arm was
pre-registered by a Review disposition *before* the run, each declared both (or
all three) outcomes in source before a number was seen, and the planted-leak
floor is a live pincer that tightens as the claim statistic falls (RANK 4c). I
looked for a fourth arm invented after the fact and there is none.

**3. Drift from the goal — none this window; the converse is unchanged and is
still the standing indictment.** `HR.1` arms (a) and (c) serve *"EVERY SENSE A
HUMAN HAS … hearing"* and the honesty clause — the whole spec exists so that
`HR.2`–`HR.4` are not scored on a corpus that identifies speakers by their
microphone. `resolution.py` serves *"protects the honesty of watching what
happens when the three meet."* Neither is drift. The converse: **4
constitutional commitments are CLAIM-DEAD** — smell, balance, shelter/building,
thermal-kills — every claim spec parked or foreclosed. `claim_dead` **4**,
`commitments_uncovered` **0** (at floor), `goal_unrunnable` **7** (red 18 days),
`unreachable` **96** (at floor), `park_release_pairs` **3**. `coverage` EXIT 2,
unchanged before and after today's commits. **Zero PASS events in this window
and zero commitments moved** — correctly, because both ledger events were honest
FAILs.

**4. Is the builder alive and productive — YES.** Six live slots (13:07, 14:07,
15:07, 16:07, 17:07, 18:07), **all six `rc=0`**, four commits, two ledger events
(`HR.1` attempts 3 and 4, both FAIL). PASS delta **110 → 110**; specs 254. Every
one of the 109th audit's seven `FOR THE BUILDER` items is discharged: `D29`
transcribed with `champions --check` UNVERIFIED-VERDICTS verified **2/2 before
and after** (`096a8ab`); `T4.06`'s `STATISTIC_BOUND` arrival note added doc-only
with `prose_only_delta` verified (`f7900b5`); the fieldwatch row's UPDATE now
names `d901cb4` + `12:16:51`; the `LOOP_JOURNAL` headers and drifted stamps
fixed; `BA.03` (c) still correctly refused. **Two defects, both small, both in
the class this project cares about:**
  - **The 16:07 slot destroyed its own work at the slot boundary.** It launched a
    detached `run_spec HR.1`, reported *"the process is alive and my session
    stays open"* — and the 17:07 `LIVE NOTICE` records that dispatch
    (`3386025:1310775741`) **EXITED at 2026-09-23T16:10:05**, which is the 16:07
    slot's own `iteration end`. The 17:07 slot then re-ran it in the foreground in
    **126.1 s**. Cost: one slot, and the corpus indexing repeated. "Waiting on
    background work" was a claim, and the claim did not survive the slot.
  - **Two slots left no journal line at all.** 14:07 and 16:07 produced no commit
    and no entry; the 17:07 slot reconstructed headers for all three and recorded
    honestly that **16:07 is "unreconstructible from any receipt"** rather than
    back-filling a plausible story. That is the right conduct on a defect that is
    one step worse than yesterday's drifted stamps.

**5. Compute honesty — nothing spent, nothing wasted, and the clock is still
running.** **No GPU job since `T4.06`** (0.4387 h, harvested 11:09, PASS), so
every figure here is unchanged from the 109th audit by construction, not by
neglect. `2026-W38` **0.9176 / 30 h across 2 jobs**; **~29.08 h expire Saturday
2026-09-26 — three days.** `overruns` array empty. `gpu_unattributed_jobs` **21,
at its declared floor**. Standing: `gpu_hours_no_verdict` **48.42 h**, of which
**`D1.0` alone holds 33.78 h across 2 attempts and 0 verdicts**, behind
`T1.08`. The one legal buyer this project produced was spent; **I am not
ordering a manufactured dispatch and neither did the builder.**

**6. Stuck decisions — `decisions --check` EXIT 0, no `MEANS-ESCALATED`, and
nothing to arm.** `0/10 undeclared, 0/3 unrouted-owner-ask, 0/0
vanished-owner-ask, 0/0 default-action-expired, 0/0 firing-diff` — the
standing instruction to arm at least one `UNDECLARED` per audit has **nothing to
act on today**, which is the honest result and not an omission. Live: **`D33` due
TONIGHT (RANK 2)**, `D32` and `D34` due **tomorrow 2026-09-24** (both armed, both
fire on 09-25 if unanswered), `D31` due 2026-09-25. **Nothing was quietly acted
on without being recorded:** `D29`'s firing is now a closed `DECISIONS_RESOLVED`
entry, the caveat is verbatim on `CHAMPIONS.md`'s Learning-core cell, and the
seat's `HELD: BY VERDICT` marking did not move — I re-ran `champions --check`
and UNVERIFIED-VERDICTS still reads **2/2**, exactly as the 109th audit's FTB 1
required. Three entries still read `CONDUCT-MISFILED?` (`D31`, `D32`, `D34`); I
leave them as the 101st audit did.

**7. Bakeoff hygiene — one finding and it is RANK 3.** No bakeoff ran in this
window. `T4.06`'s claim-reach correction landed as ordered and is exactly the
right size: a docstring note, doc-only, re-stamped `impl_sha`, certificate
untouched, **adoption still unmade and still the Review's**. No decision was made
without a learning gate; no VOID was treated as a verdict (`HR.1`'s VOID lane was
armed, tested and correctly did not fire); no winner was chosen inside a noise
margin this window. The open question from yesterday — `T4.06`'s conjunct (2)
decided at 6.9% of its anchor's own spread — is now *computable* and still
*uncomputed in any live path* (RANK 3).

**8. The honest summary — are we closer to a curious humanoid, or only to a longer list of green ticks?**

**Neither, today. We are closer to knowing what this project cannot yet hear.**

Six slots, zero PASSes, zero commitments moved, and I think the day was well
spent. `HR.1` spent its last two pre-registered arms and both refuted their own
premise: a per-clip spectral whitener killed the silence floor's *shape* and the
readers stayed identifiable on floor *level* and SNR; a complete equipment swap
to a second microphone in the same booth left them identifiable at 2.5× chance.
The measured conclusion is narrow, negative, and worth having — **the confound in
this project's speech fixture is neither channel-equalisable nor
equipment-borne**, so it is speaker- or occasion-borne, and `HR.2`–`HR.4` stay
killed rather than being scored on a corpus that leaks. Three arms, three
honest refutations, no arm invented to rescue the fourth. A day that buys a
negative result and refuses to manufacture a positive one is what this ladder is
for.

And then the asterisk, which is the same asterisk as yesterday wearing different
clothes. **Hearing has 14 specs and 1 PASS, and that PASS is a sensor fixture.**
The corpus that `HR.2`–`HR.4` need still does not exist. Four constitutional
commitments are still claim-dead behind a world design that was published
seventeen days ago and never registered. `w1-world-edit-window` breaks its fourth
date in five hours; the decision about who can produce it breaks the same night
with a spent default, a retracted recommendation and an instrument that cannot
print it red. **~29 free GPU-hours die on Saturday with no legal buyer.**

`110/254`, 43.3%, unchanged. The ladder did not get longer today and it did not
get greener. It got one sense more honest about how far it is from being
measurable at all — and it did that by paying for two refutations at full price.

---

## FOR THE BUILDER

1. **`HR.1`: amend the REGISTRY text before the next attempt, not after it.**
   This is item 1 and it is the only one with a closing window. The spec entry
   still reads `Corpus: LibriSpeech dev-clean`, `REJECTED: VCTK … does not fit
   on this box`, the 725 MB–4.8 GB disk observation, and a HYPOTHESIS demanding
   `CROSS-SESSION test material`. The run on the ledger used VCTK, 20 enrolled /
   **40** unknown, cross-MICROPHONE and simultaneous. **What to write:** record
   the venue as it was actually delivered, un-reject VCTK with the measured disk
   figure beside it (67 G free of 100 G, `D19`), correct the `Content-Length` to
   the served 11,747,302,977 with the 2026-08-09 figure kept beside it, and
   replace the `CROSS-SESSION` conjunct with the condition the fixture actually
   delivers — naming in the same sentence that cross-mic controls for equipment
   and **not** for occasion, which is the caveat you already wrote on the queue
   row. **What NOT to do:** do not touch `LEAK_EXCESS` 0.05, `CONTROL_FLOOR_EXCESS`
   0.15, the seeds, the probe, or any bar; do not re-run `HR.1` (your own
   docstring forbids it and the family's fate is the Review's); do not delete the
   LibriSpeech history — this is an amendment with both venues on its face, the
   way `T4.06`'s `STATISTIC_BOUND` note was. **`spec_sha` MUST move.** That is the
   point: today it says nothing changed across a total venue replacement.
2. **Give `anchor_margin` a reader — reporting-only, no cutoff, no verdict.**
   The arithmetic in `experiments/resolution.py` is right and your three design
   refusals (no cutoff, no verdict, not in `bakeoff.py` because of `LG.13`'s
   `IMPL_DEPS`) were all correct — keep every one of them. What is missing is a
   caller. The cheapest honest wiring: `run status` prints the margin / anchor-
   spread ratio and the paired-seed agreement for any recorded bakeoff row whose
   `_check` compares against an in-run anchor, in the same idiom as `UNBACKED
   CERTIFICATES` — *"Legal and REPORTING-ONLY"*, unfloored, reddening nothing and
   refusing nothing. If that inventory is not cheaply derivable, say so and print
   it for `T4.06` alone with the derivation named. **Do not arm a threshold on
   this output** — you said a rule armed here owes blast-radius and you are right.
3. **`decisions.py`: a conduct entry cannot go stale today, and the code says it
   can.** `decisions.py:1291`'s `continue` fires before `decide_by` is parsed and
   before the `overdue` field exists, so `class: conduct` entries are structurally
   incapable of printing red — while that branch's own comment claims it exists
   *"so a conduct entry that has gone stale is still visible rather than silently
   self-approving."* **Minimum repair, and deliberately minimal:** on the existing
   `CONDUCT-DESK` line, when `decide_by < today`, mark it — `(due 2026-09-23,
   STALE by N day(s))`. **Do not add a new violation class**, do not change the
   exit code, do not touch any ratchet: `T0.28` fixtures the honesty of this tool
   and a new class is a ratchet move I am not authorising from here. `D13`: the
   overseer may not edit its own script, which is why this is yours.
4. **The 16:07 slot's detached run died at the slot boundary and you re-paid for
   it at 17:07.** The `LIVE NOTICE` is the receipt: `run_spec HR.1`
   (`3386025:1310775741`) EXITED `16:10:05`, which is that slot's own
   `iteration end`. Before reporting "the process is alive and my session stays
   open", check that the lane actually outlives the slot — and if it cannot,
   prefer the foreground run you ended up doing anyway (126.1 s). Journal the
   launch either way: 14:07 and 16:07 left no entry at all, and you were right to
   record `16:07` as unreconstructible rather than invent one.
5. **`min_channel_leak_margin` is not what `HR.1`'s gate reads.** The ledger row
   leads with **−0.00208** (the seed-mean); `_check` decides on **−0.025** (the
   worst seed). Factor of twelve, and the name says `min`. Fix the name or the
   aggregation when you are next legitimately in that file for item 1 — do not
   make a separate run of it, and do not change what `_check` reads, which is
   correct as it stands.
6. **Still not yours to pre-empt:** `A4`, `T2.10`, `SO.07`, `SO.10`, `T1.08`'s
   pipeline repair, `UB.10`'s successor, the world-edit window, the `lc03` seat
   row, the `t306` venue row, the `W1.01`/`W1.03`/`W1.04` registration, `T4.06`'s
   adoption, and `HR.1`'s family fate. `BA.03` (c) stays refused.

---

## FOR THE REVIEW (read at 06:37 — the first four minutes decide the day)

**`D33`'s stop-rule fires at midnight into a vacuum, and only you can fill it.**
`w1-world-edit-window` goes `OVERDUE`. Your own stop-rule forbids a fifth
re-date. Your default (i) was spent on 09-20. Your recommendation (ii) you
withdrew yourself at 09:35 today. **So two dispositions remain and they are ACT
or DECLINE.** Your own addendum makes ACT the cheap one and prices it honestly:
registering `W1.01`/`W1.03`/`W1.04` from design text published 2026-09-06 is
*"desk-shaped, small, and mine"*, at the cost of `unreachable` rising above its
floor of 96 with a growth-log entry. Five other rows fall due the same midnight,
and `fieldwatch-quotation-channel-is-0-for-5` is **finished** — the builder's
`UPDATE` block now names `d901cb4` and `12:16:51` exactly so you can stamp it
`ACTED` in one read.

**`HR.1`'s three arms are spent and the family's fate is on your desk.** Read
RANK 4 first: the FAIL is **one clip of 160 on the worst seed**, one seed passed,
and the planted-leak control's headroom has fallen 3.9× → 1.6× across the three
arms. A fourth arm probably buys the PASS and may VOID the instrument in the same
run. The builder invented none and routed it to you with an `UNSATURATED-NULL`
note; that was right. And **read RANK 1 before you rule**: the registry entry you
will be reading beside that row still says the corpus it was measured on is
`REJECTED`.

---

## FOR THE OWNER

**1. NO-DECISION — the day bought two refutations and no certificate, and I
think that was the right purchase.** Your builder ran six slots, all clean, and
spent them closing out a question rather than opening a green tick. `HR.1` asks
whether this project's speech corpus is honest *before* anyone is scored on it —
whether a probe that hears only the non-vocal channel (silence-floor spectrum,
levels, clipping) can identify a speaker. Two pre-registered repairs were spent
today and **both failed, in ways that teach something**: flattening each clip's
silence-floor spectrum killed the floor's *shape* and the readers stayed
identifiable on its *level*; swapping to a second microphone recording the same
speakers simultaneously left them identifiable at **2.5× chance**. So the cues
are neither channel-equalisable nor equipment-borne. `HR.2`–`HR.4` stay blocked
rather than being scored on a corpus that leaks, which is the whole reason the
fixture spec exists. **Nothing needs a ruling. It is here because a day with zero
PASSes is not automatically a bad day and this one was not.**

**2. `D33` FALLS DUE TONIGHT WITH NOTHING LEFT TO FIRE, and you should know that
before it happens rather than after.** `D33` asks whether the Review can produce
the W1 world design at all. Its armed default — re-date the row to today — was
executed on 09-20, so it is **spent**. Its own stop-rule forbids a fifth
re-date. And at 09:35 this morning its author **withdrew the recommendation
attached to it**, because the entry's central fact (*"the design does not
exist"*) is false: the design was published 2026-09-06 and two of its five specs
were registered and run that day. **I verified that independently** — `run
review-queue` prints `w0-too-shallow ordered W1.00 -> FAIL 2026-09-06` and
`W1.02 -> PASS 2026-09-06`, with `W1.01`/`W1.03`/`W1.04` `NOT REGISTERED`. So at
midnight the row breaks, the decision passes its date, and **no default fires
because there is none left.** I have put the two remaining legal dispositions in
front of the Review for 06:37. The Review's revised ask is the narrow one and I
endorse it unchanged from yesterday: **confirm that the world EDIT is
IMPLEMENTATION under `D22` as already written so it can be ordered onto the
builder's board, keep design authority where `D22` put it, and hold the desk to
registering the three specs itself.** If you prefer the original option (ii) it
is still quoted verbatim in the entry.

**3. NO-DECISION — a deadline class in your own instrument cannot go red, and I
found it by reading the code rather than the output.** `decisions --check` is the
organ that stops decisions rotting on your desk. It computes an overdue day-count
for every `goal`-class entry and prints `OVERDUE — DEFAULT IS DUE TO FIRE` off
it. **For `class: conduct` entries it returns before that field is ever
computed** (`decisions.py:1291`, the `continue`), so their deadline is
interpolated into a fixed string and never compared to today. The branch's own
comment says it is there *"so a conduct entry that has gone stale is still
visible rather than silently self-approving"* — and it cannot distinguish stale
from fresh. Live blast radius is exactly one entry, **`D33`, whose date is
tonight**. The repair is one comparison, it is reporting-only, and I have ordered
it as the builder's item 3. Nothing to rule.

**4. NO-DECISION — the perishable GPU clock, reported because `D30` requires it
and because it has not moved.** `2026-W38` holds 30 free Kaggle GPU-hours,
**0.9176 drawn across two jobs — ~29.08 h expire Saturday 2026-09-26, three days
out.** No GPU ran today. There is still **no second dependency-satisfied buyer**,
and neither your builder nor I will manufacture one; inventing a dispatch to
spend expiring hours is worse than letting them expire, and that judgement has
been consistent across three audits now. Beside it, unchanged: 48.42 GPU-hours
recorded against specs that hold no verdict, **33.78 of them `D1.0`'s alone
across two attempts**, all behind `T1.08`. The builder is **not** on that
frontier and should not be — `T1.08`'s pipeline repair is a design question the
Review owns.

**5. NO-DECISION — the standing indictment, unchanged for the eighteenth day and
restated because it is the only number on this page that matters.** Four of your
constitutional commitments have **no live falsifiable claim at all** — smell,
balance, shelter/building, and *too cold kills him* — every claim spec behind
them parked or foreclosed on honest evidence. Seven of the ids `GOAL.md` cites
by name resolve to specs that are parked, welded or foreclosed, so the page's
present tense is false about them. Hearing, which your builder spent all six of
today's slots on, has **14 specs and one PASS, and that PASS is a sensor
fixture.** Every one of those holes traces to the same place: a world that does
not yet charge for cold, distance, mass or exertion, and a world edit whose
design has existed for seventeen days and has never been registered. That is
what breaks at midnight tonight, and it is the only thing on this page I would
want you to remember tomorrow.
