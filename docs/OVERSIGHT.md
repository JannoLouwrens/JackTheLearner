# OVERSIGHT — 85th audit, 2026-09-08 06:37–07:0x UTC (at `750f90e`, tree clean, 10 unpushed — all the Review's)

## VERDICT: ON TRACK — **the ledger is sound, nothing was loosened, and the Review had the strongest single morning this organ has recorded (5 queue violations → 0, six rows disposed inside one sitting). The findings are all one shape: three instruments are now calibrated against a system that no longer exists, and the most consequential of them is about to push every new queue row fifteen days into the future.**

**This audit ran concurrently with the Review** — `crontab` puts `overseer.sh`
at `37 */6` and `review.sh` at `37 6`, so both fired at 06:37:0x. I took a
reading at 06:37 (pre-Review), waited for `review.sh` to exit at **06:55:14**,
and **every number below is the settled post-Review state.** Where the two
differ I say so.

Sections 1, 2, 4, 5, 6 and 7 are clean and that goes on the record before the
findings.

- **Ledger integrity: 108 PASS, zero dead commits.** I resolved all **352
  unique commit shas** across all **676 rows including history** against
  `git cat-file` — every one alive. Every PASS resolves to an implementation on
  disk; every PASS spec declares a `control`; the only two rows without
  `control_metrics` are `T0.01`/`T0.10`, which argue `"NONE, BY DECISION"` on
  their own specs. **No PASS row is stale** — all 12 entries in `status`'s
  STALE CLAIMS block are FAIL or VOID.
- **Nothing was loosened.** I enumerated every `CONST = value` change in
  `registry.py`, `registry_expansion.py` and `experiments/tests/` over seven
  days by diffing each commit's constant table. **Every numeric move is a
  tightening or an assertion count growing**: `MIN_DISTRACTOR_EVAL` 9 → 12,
  `N_LIVES` 16 → 32, `N_PROPERTIES` 10 → 17 across six commits. No `_check`
  gained an `or`; no seed count fell; no control was deleted.
- **Builder alive: 24 iterations in 24 h, 24 × `rc=0`, PASS delta +2**
  (`ME.1`, then `PL.00`/`T0.35` at 11:22). The demonstrated count has not moved
  in **19 hours** and that is correct, not a stall — see §4.
- **Compute honest.** No overrun, every GPU hour attributed, `UNATTRIBUTED` at
  its declared floor of 21 jobs.

---

## 1. THE FINDING — `MEASURED_DISCHARGE_CAPACITY = 1` was refuted at 06:55 this morning, and the instrument built to stop piling is now scheduling rows fifteen days out

`experiments/review_queue.py:219`, with its own raising rule in the docstring
directly above it:

```
#: The consumer's only MEASURED discharge capacity, in rows per cycle: the
#: 2026-08-30 FULL run died at eleven minutes owing exactly ONE dated row
#: (`w0-too-shallow`), and no cycle since has demonstrably discharged more.
#: Raise it only by citing a cycle that actually discharged N.
MEASURED_DISCHARGE_CAPACITY = 1
```

**A cycle that discharged N happened this morning.** Between 06:39 and 06:55
the Review DAILY disposed **six dated rows**, each in its own commit:
`pl02-dependency-on-pl00-verdict-vs-table` (ACTED, `102a47a`),
`d10-successor-rerun-under-adopted-gate` (`9924291`),
`ub10-seed-fragility-and-saturated-battery` (`ad9bced`),
`lg10-mouth-fidelity-vs-freedom` (`66dcd86`),
`t309-control-clears-the-claims-own-margin` (`da202e1`),
`cpu48h-class-self-forecloses-the-day-meter` (`16f7eb8`) — plus four STALE
rows re-armed (`a985f05`). `review-queue` went **EXIT 2 → EXIT 0**, violations
**5 → 0**. The citation the constant asks for now exists, and it is 6×.

**This is not a cosmetic mis-calibration. The constant is load-bearing in three
places and the third one does damage:**

| consumer | effect at `CAPACITY = 1` | effect at a capacity of 6 |
|---|---|---|
| the `!! AMBER: pile` print | **8 of 14** dated days flagged, including **four days carrying exactly 2 rows** (09-12, 09-14, 09-15, 09-16) | 2 days flagged: **09-09 (8)** and **09-13 (10)** — the two real piles |
| `piled_on` (ratcheted, = 25) | a row is "dated onto a full day" if **one** other row was there first | counts only genuine crowding |
| **`next_free_due`** | **`2026-09-23`** — the first date carrying *zero* rows | `2026-09-17` |

**`next_free_due` is the one that matters,** because the renderer hands it to
the reader as an instruction: *"the mechanical answer for the next router,
instead of defaulting onto Sunday."* A desk that just demonstrated six
disposals in eighteen minutes is being told the next honest date it may promise
anything is **fifteen days away**. Every row routed today that obeys that
answer lands on 09-23 or later. **An instrument built to prevent piles is now
the mechanism that manufactures them** — it pushes routine work past the
horizon, and the pressure comes back as exactly the Sunday pile it exists to
forbid (`2026-09-13` still carries **10** rows this morning, untouched).

The amber half is the cry-wolf disease the 84th audit found in
`cpu_foreclosed_now` one night ago, arriving from a different door: a warning
that fires on 8 of 14 days, four of them for two rows, teaches its reader to
skim — and the 65th audit built this print *because* nobody saw the 09-06 pile
coming.

**Two honesty conditions on the repair, and I want them stated before the
number moves, because this is a threshold going UP:**

1. **This is not a spec bar and no verdict depends on it.** It decides when a
   report prints amber. `SYSTEM.md` law 4 is not in play. But the direction is
   the loosening direction, so the raise must cite this cycle's six commits by
   hash in the constant's own docstring, in the commit that moves it. If it
   cannot cite them it may not move.
2. **It will SHRINK a ratcheted counter, and that must be recorded as a
   definitional change, never banked as progress.** `review_queue_piled_on`
   reads 25 today against a committed reading of 26; raising the denominator
   drops it further for reasons that have nothing to do with anyone routing
   better. This is the disease `T0.31`'s P4/P5/P6 exist to catch. There is
   clean precedent for doing it honestly: the 73rd audit B1 re-baselined the
   same counter 17 → 22 and wrote *"22 is the honest baseline; 17 was never a
   real reading."* Same move, opposite direction, same discipline.

**What I am NOT claiming.** I am claiming **1 is refuted**, not that **6 is
right**. One DAILY facing a docket it could clear is one cycle; the defensible
value is an argument from the recent record, and that argument is the
builder's to make with the numbers in `throughput()`. If the answer is 3, the
finding is discharged just as well.

---

## 2. `champions.py` currently rewards NOT declaring what a spec is — and the soft spot grew 1 → 7 with nothing counting it

`_challenger_runs` (`experiments/champions.py:860`) filters out arena
completions whose declared `COVERS:` kind is `fixture`/`rule`/`sensor`, because
those seat no challenger. A spec that declares **nothing** counts as a
challenger. That is a deliberate, documented rule (45th audit B4) and the
docstring names it as *"this filter's remaining soft spot."* My finding is not
the rule. It is what the rule now costs, measured:

```
T2.03   kinds=['fixture']   ->  Vision encoder            UNCONTESTED  (violation)
PL.00   kinds=['rule']      ->  PLASTIC-ONLY decree       UNCONTESTED  (violation)
T2.12   NONE DECLARED  \
T3.07   NONE DECLARED  /    ->  Emotion (affect)          ok
LC.00   NONE DECLARED  \
LC.02   NONE DECLARED  /    ->  Learning core             discharged by these two
ME.11.A NONE DECLARED  \
ME.11.B NONE DECLARED   |
ME.11.C NONE DECLARED   |   ->  Episodic retrieval        ok  (held BY VERDICT)
ME.11.D NONE DECLARED  /
```

**The only two seats `champions --check` calls UNCONTESTED are precisely the
two whose arena specs declared their kind honestly.** Three seats resting on
seven undeclared specs read clean, including `Episodic retrieval`, which holds
the file's **strongest marking — BY VERDICT** — on an arena where not one of
the four completed arms says what it is.

**And the growth was silent.** The docstring, written when the rule was made,
says *"`Learning core` is discharged by exactly one such spec, `LC.02`."* Live
today: **seven specs across three seats.** Nothing counts them — the champions
ratchet tracks phantom arenas (0/0), unfalsifiable (2/3), uncontestable (2+1/4),
unverified verdicts (2/2) and trigger debt (3/3), and the kindless set is
printed but has **no counter and no floor**, so 1 → 7 happened under a green
`ratchet ok` line. That is the shape the brief sends me looking for: a class
with no id, no gate and no number.

**The repair is the one the docstring already prescribes and costs no
threshold:** declare the seven kinds in the registry (notes are not
claim-hashed; zero certificates stale), and add `kindless_arena_discharges` to
the champions ratchet with today's **7** as its declared floor, so it can only
shrink. Note the honest consequence up front: declaring `LC.02` (a throughput
feasibility gate) or `T2.12` truthfully may turn `ok` seats red. **That is the
repair working, not the repair failing** — the ratchet counting a class it was
blind to is exactly the `T0.31` precedent.

---

## 3. The `ORDERED:` lane cannot tell "commissioned and not yet run" from "commissioned and never registered"

The 79th audit built the ORDERED join so a commissioned measurement that never
came back would be as visible as one that refuted its own disposition. It
resolves ids against the **ledger only** — `review_queue.py:582-596`, `lrow =
(ledger or {}).get(sid) or {}` — and never against `BY_ID`. Live output:

```
w0-too-shallow (DISPOSITIONED)  ordered W1.01 -> NO ROW YET
w0-too-shallow (DISPOSITIONED)  ordered W1.03 -> NO ROW YET
w0-too-shallow (DISPOSITIONED)  ordered W1.04 -> NO ROW YET
```

**None of the three is registered.** `BY_ID` holds 245 specs; `W1.00` and
`W1.02` are in it, `W1.01`/`W1.03`/`W1.04` are not, and that is *deliberate* —
`83c75f3` states *"W1.01/W1.03/W1.04 deliberately NOT registered — they wait on
`w1-world-edit-window`."* So the lane prints a scope decision recorded in a
commit message as though it were three outstanding measurements, in the same
words it would use for a spec that exists and has not run.

Damage today is small and the ceiling is what matters: **a typo'd `ORDERED:`
id prints `NO ROW YET` forever** and can never be distinguished from an honest
pending commission. This project already treats that exact failure as
first-class twice over — `coverage.py`'s dangling-citation class, and
`champions.py`'s `!` marker for *"named as this seat's arena, absent from the
registry."* The newest of the three instruments is the one that does not check.

**Repair: a third state.** Where the spec id does not resolve in `BY_ID`, print
`NOT REGISTERED` instead of `NO ROW YET`. Still a READING, never a violation;
no count moves; one `elif`.

---

## 4. A minor note on last night's `DAY-ROLLED` suppression — I tried to break it and could not

The 84th audit's B3 ordered a fix and the builder shipped `DAY_SCOPED_COUNTERS`
(`3881ac4`). Suppressing an alarm deserves adversarial reading, so: **I tried to
show that the banner it preserves is unreachable in practice, and the data
refutes me.** The guard suppresses only when `prev_at != today`, so I checked
whether `cpu_foreclosed_now` is ever re-recorded within a day. Reconstructed
from every revision of `ratchet_readings.json`: **09-04 three times (36, 39,
41), 09-05 four times (0, 41, 40, 39), 09-07 twice (0, 39).** A same-day
baseline is routine, so *"movement within today would still banner"* is true.
The design holds. Two small residuals, neither urgent:

- `today` comes from `time.strftime("%Y-%m-%d")` — **local** time — while the
  banner it prints asserts *"resets at 00:00 UTC"*. The box is `GMT` so the two
  coincide today; it is latent, not live. `time.gmtime()` closes it.
- The branch fires on any `cur != prev` across the boundary, so a day-scoped
  meter that **failed to reset** is invisible either way (39 → 39 reads
  UNCHANGED; 39 → 20 reads DAY-ROLLED). The stronger form asserts the new value
  is consistent with a reset having happened. Worth one line, not a rewrite.

---

## The audit, item by item

**1. Integrity of the ledger — clean.** 108 PASS; 352 unique commit shas across
676 rows including history, all alive; every PASS has an implementation on disk
and a declared `control`; `T0.01`/`T0.10` the only rows without
`control_metrics`, by their own argued decision. **No PASS is stale.** The 12
stale-claim entries are all FAIL/VOID, which is the low-risk direction. I also
checked the overnight re-buy discipline: `T0.31` is stamped `commit=021d302` —
the commit carrying the B2 edit it certifies — and committed in the *next*
commit (`50b8a23`), not `+dirty`. Correct order.

**2. Thresholds and controls — clean, and I looked mechanically rather than by
eye.** See the preamble. The one move worth naming because it *looks* like a
loosening and is not: `LG.10`'s `TEMP` 0.25 → 1.0 (six days ago) raises a
sampling temperature to clear an unmoved 0.30 variety floor — that changes the
arm, not the bar.

**3. Drift — no drift, and the converse is the uncomfortable half.** Everything
the builder shipped in 24 h traces to `GOAL.md`'s honesty clause ("really
learning, not appearing to learn"): the `D17` armed default fired without
touching `GOAL.md:76`, the AGEING-IN forecast, the DAY-ROLLED fix, the
pre-registration lesson. **None of it touches a sense, a need, or a
capability** — correctly, because there was nothing to run. The converse:
**13 of 25 constitutional commitments have zero passing claim.** Four are
CLAIM-DEAD (smell, balance, shelter/building, thermal-kills) and nine have live
claim specs with nothing passing (touch, tool use, told world, proprioception,
death & retry, plasticity, sleep, hunger/thirst, fast/slow). Unmoved for six
days.

**4. Builder — alive and correctly idle.** 24 iterations, 24 × `rc=0`, PASS
delta +2, and **ten consecutive slots** that verified an empty board and
stopped early. The last eight ran 3 minutes each. This is the empty-board rule
working, not a stall: `coverage` prints **7 dispatchable today, 6 of them VOID
→ 1 FRESH**, and that one is `PL.02`, whose registered run is legitimately
blocked pending tomorrow's `pl02-eye-gate-reads-the-encoder-not-the-eye`
ruling. I checked the builder's central claim rather than inheriting it for the
sixth slot running, and it holds.

**5. Compute honesty — the instrument worked and the finding travelled.** W36
(opened Sunday 09-06): **17.724 h used of 30, 12.276 h remain**, week ends
09-13 00:00 UTC. `D1.0` attempt 2 spent **17.611 h across four kernels** and
returned VOID; cumulative `gpu_hours_no_verdict` reads `D1.0: 33.78 h / 2
attempts / 0 verdicts` — the largest compute expenditure in the ladder, and it
has bought nothing. **The 84th audit routed that arithmetic six hours ago and I
can confirm it reached the ruling:** this morning's `d10-successor` disposition
dates the successor gate 09-14 *"because W36 has ~12.4 GPU-h left against
attempt 2's measured 17.61 h, so attempt 3 cannot fit this window anyway and
W37 opens 09-13."* A finding published at 00:38 changed a decision at 06:45.
That chain is the point of this desk and it is worth recording when it works.
No overruns; `UNATTRIBUTED` at its declared floor (6.32 h / 21 jobs); CPU
budget coherent.

**6. Stuck decisions — nothing stuck.** `decisions --check` EXIT 0, ratchet
clean (0/10 undeclared, 0/3 unrouted-owner-ask, 0/0 vanished, 0/0
default-action-expired). **No `MEANS-ESCALATED`** — no fork a measurement could
settle is sitting on the owner's desk. Six entries armed with live clocks
(`D18` 09-09, `D19` 09-14, `D20` 09-18, `D22` **today**, `D23`/`D24` 09-11,
`D25` 09-13). Nothing has become decidable that is not. **No owner decision was
acted on without being recorded**: `D17` fired at 01:10 with its record in
`DECISIONS_RESOLVED.md`, its OVERDUE notice in `DECISIONS_NEEDED.md`, and
`GOAL.md` untouched — I verified all three. **I armed nothing this audit
because there was nothing undeclared to arm**, which the brief permits and I
would rather say plainly than manufacture an entry.

**7. Bakeoff hygiene — clean, including the item the 84th audit routed.** Its
B4 asked that `PL.00/RENDER`'s tie-break stop calling itself *"pre-declared"*
when the record cannot support it; `DECISIONS_RESOLVED.md:735` and
`REVIEW_QUEUE.md:1669` now both read *"declared at `b7324ba`"* and zero
`pre-declared` phrases remain. A checkable pointer where an adjective stood. No
decision was made without a learning gate; the one VOID-treated-as-verdict
remains `Learning core` held off `LC.03`, at its ratchet floor and routed. No
winner was chosen inside a noise margin.

**8. The honest summary.** **No — we are not closer to a curious humanoid than
we were yesterday, and I want that unhedged.** Jack did not change in 24 hours.
The demonstrated count has not moved since 11:22 yesterday; every one of the
six things decided this morning was about an instrument, which the Review says
about itself in its own last sentence. What *did* improve is real and I will
not undersell it: the desk that has been this project's measured bottleneck for
a week cleared its whole docket in eighteen minutes, took the queue red-to-green
for the first time since it went red, and refused three separate offers to buy
a verdict by relaxing what counts as one — `lg10` (c), `ub10`'s
training-budget cut, `t309` (a). Refusing an easy green three times before
breakfast is the culture this project is actually made of. But the ladder-and-
apple standard is indifferent to that. Thirteen of twenty-five commitments
still have no passing claim, four of them have no live path in at all, and the
honest reading of today is that we got better at knowing that, not at fixing
it.

---

## FOR THE BUILDER

1. **`MEASURED_DISCHARGE_CAPACITY` is refuted by this morning's cycle — raise
   it, with the citation its own docstring demands, and re-baseline
   `piled_on` in the same commit as a DEFINITIONAL change.** Cite the six
   disposal commits by hash (`102a47a`, `9924291`, `ad9bced`, `66dcd86`,
   `da202e1`, `16f7eb8`) in the constant's docstring. Argue the *value* from
   `throughput()`'s record rather than taking 6 because today was 6 — the
   finding is that **1 is refuted**, not that 6 is right. Then, in the same
   commit: record the new `review_queue_piled_on` reading with the words *"the
   denominator changed; this is not routing improving"*, in the 73rd audit
   B1 idiom (17 → 22, *"22 is the honest baseline"*). **No violation class is
   added or removed, no row's status changes, `review-queue` stays EXIT 0.**
   State in the commit what `next_free_due` reads before and after — today it
   is **2026-09-23**, fifteen days out, and that number is the reason this is
   item 1.
2. **Declare the seven undeclared arena kinds, and give the class a counter.**
   `LC.00`, `LC.02`, `T2.12`, `T3.07`, `ME.11.A`, `ME.11.B`, `ME.11.C`,
   `ME.11.D` — registry notes, which are not claim-hashed, so **zero
   certificates stale**. Then add `kindless_arena_discharges` to the champions
   ratchet with **7** as its declared shrink-only floor. **Declare what each
   spec actually is, not what keeps its seat green** — if `LC.02` is a
   throughput feasibility gate, saying so may flip `Learning core` or
   `Emotion (affect)` red, and that is the repair succeeding. Update the
   `_challenger_runs` docstring, which still says *"exactly one such spec,
   `LC.02`"* against a live seven.
3. **Give the `ORDERED:` lane a third state.** In `review_queue.py`'s
   ordered-join renderer, when the spec id does not resolve in `BY_ID`, print
   `NOT REGISTERED` rather than `NO ROW YET`. `W1.01`/`W1.03`/`W1.04` are the
   live known-positives (deliberately unregistered per `83c75f3`); `W1.00`/
   `W1.02` and `D1.0` are the known-negatives. **METRIC only — no violation
   class, no count moves.** Add the known-positive to `T0.31`.
4. **Two small residuals on last night's B3, neither urgent.** `today` in
   `print_ratchet_block` reads local time via `time.strftime` while the banner
   it prints asserts *"00:00 UTC"* — latent on this box (`TZ=GMT`), so fix it
   with `time.gmtime()` when you are next in the file. And note in the
   docstring that the branch cannot distinguish a reset from a *failed* reset.
   **The design itself is sound and I checked it adversarially** — see §4.
5. **Standing prohibitions, restated and unchanged from the Review's own
   list:** no third `D1.0` dispatch until the twin-spread probe is on the row
   AND the successor gate is committed in a non-dispatch commit, and then into
   **W37** (W36 has 12.276 h against a measured 17.61 h need); no `UB.10`
   re-dispatch before its redesign; no `LG.10` re-roll; `HR.1`–`HR.4` stay
   `D19`-held to 09-14; `PL.02`'s registered run stays blocked pending
   tomorrow's ruling; `LF.01` attempt 2 waits for the 09-09 design; no edit to
   `cpu_budget.py`'s or `rtf.py`'s ceilings (`D20`, owner-gated). The
   overseer's own script stays untouched (`D13`).
6. **Hygiene, for whoever runs next:** 10 commits are unpushed, all of them the
   Review's own from this morning — it commits but does not push. Nothing is
   blocked today (item 1 of the Review's list is CPU forward passes), but a GPU
   dispatch refuses an unpushed HEAD, so push before W37 opens.

---

## FOR THE OWNER

1. **`D22`'s `decide_by` is TODAY and its default fires at 09-09T00:00 if you
   are silent. The Review changed its own recommendation this morning and you
   should have that before the clock runs out.** The ask: should *drafting* of
   spec redesigns move to the builder, with ratification staying with the
   Review and this desk. The default is **(i) THE RULE STANDS** — the status
   quo, and correctly the only legal default, because the alternative widens
   what the builder may do and a default may not widen what this project takes.

   > **The Review now asks you to wait a week rather than grant it, and its
   > reason is that its own evidence turned against it this morning.** Its case
   > was that design throughput is the binding constraint. Then it cleared six
   > rows, re-armed four, and took the queue from 5 violations to 0 in eighteen
   > minutes — which is evidence against the thing it asked you to fix. I have
   > verified that independently and it is true: `review-queue` reads **EXIT 0,
   > 0 violations** at HEAD. The file went red for the first time in its life
   > at 2026-09-08T00:00 (one OVERDUE plus four rows crossing the 8-day cycle
   > together); it was clean again inside seven hours.
   >
   > **My own reading, since I am one of the two ratifying organs under the
   > proposal and you should have it separately.** The trend has *not* turned:
   > drain still **UNBOUNDED**, **41 live rows**, 32 arrivals against 3
   > disposals over the trailing week. One morning is one morning. But I would
   > add something the Review did not say about itself: **three of the six
   > rulings it made today refused an easier green** — it declined to lower
   > `LG.10`'s bar, declined `UB.10`'s training-budget cut, declined to let a
   > VOID kill a claim in `t309`. That is the judgement the ratification half
   > of `D22` is actually buying, and it was exercised well today. **My
   > position on 09-04 is unchanged and I will restate it plainly: if you rule
   > (iii), ask for a tripwire, not a veto** — every builder-drafted redesign
   > carries its prior version in the ledger's history and states why the new
   > threshold is HARDER, so that a year from now the drafts can be counted and
   > the ones that got easier can be found. **Silence is a real answer here and
   > it costs the divergence continuing.**

2. **NO-DECISION, and it is the sentence to read if you read one: thirteen of
   your twenty-five constitutional commitments have no passing claim, four of
   them have no live path in at all, and that is a seventh day unmoved.** Too
   cold kills him, he builds a shelter, smell, balance — all four CLAIM-DEAD
   behind foreclosures whose successor specs need redesigns that are dated
   09-11. The nine others (touch, tool use, told world, proprioception, death &
   retry, plasticity, sleep, hunger/thirst, fast/slow) have live specs and
   nothing passing. **This is not a new finding and I am not dressing it as
   one.** I am recording that the builder shipped four correct instrument
   repairs yesterday, the Review made six good rulings this morning, and not
   one of the ten touched any of the thirteen.

3. **NO-DECISION, for your awareness: today's findings are all the same
   shape, and it is a shape worth naming.** Three separate instruments —
   `review_queue.py`'s pile threshold, `champions.py`'s kind filter,
   `run.py`'s day-scoped banner — are calibrated against a system that has
   since changed underneath them, and each was *honest when written*. This is
   the price of an instrument-heavy project going well: the measurements
   improve faster than the constants that read them. Nothing here is a
   loosening and nothing threatens the ledger. I flag it because the Review
   independently reported the same pattern one level down about our *tasks*
   ("too easy for our instruments to say anything about them", four fronts,
   routed to Sunday). **Two organs found the same disease in different
   organs on the same morning.** Nothing for you to rule on.

4. **NO-DECISION: liveness.** All four organs live, verified against
   `/data/jack-logs` mtimes rather than anyone's report: builder **06:11**
   (hourly, 24/24 `rc=0`), overseer **06:37** (this run), field watch
   **2026-09-07 05:56** (Mondays, third consecutive sweep on cadence), Review
   **06:37 → 06:55** (`rc=0` — the DAILY cadence has now exited clean twice
   running after 09-05's `rc=1` and 09-06's `rc=124`). `lost_iterations.log`
   still 0 bytes and still never exercised. Usage `week:all models` **23% at
   this audit's start, 24% at the Review's end** — no pacing pressure and no
   PACING lines. `D24` (09-11), `D25` (09-13), `D18` (09-09), `D20` (09-18)
   and `D23` (09-11) unchanged on your desk.
