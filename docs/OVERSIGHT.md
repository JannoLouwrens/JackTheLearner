# OVERSIGHT — 59th audit, 2026-09-01 18:40 UTC (HEAD `8fae35c`, tree dirty: 3 GPU/ledger files written 18:23 by the D1.0 watcher, harvest not yet committed)

## VERDICT: DRIFTING — nothing was weakened and the ledger is clean, but `LC.03`'s foreclosure has quietly welded **10 specs shut, three of them ids GOAL.md itself cites in the present tense**, and no instrument in this repo says so

The honest headline first, because it is the counterweight to the finding:
sections **1, 2, 6 and 7 have no violations**, checked mechanically rather than
asserted. In particular **section 2 is clean** — one bar moved downward in seven
days, it was a provisional placeholder frozen from pilot seeds *disjoint* from
its scoring seeds, declared in its own commit subject, and justified from a
fixed point rather than shaved to the observation. That is the pattern working.

The 58th audit's B1/B2/B3 were **executed within four hours** (`da9880a`,
`a2748f1`) and the one place the repair diverged from the audit's spec (4
CLAIM-DEAD, not 5) was **declared in the commit message with its reasoning and
routed with a clock**. That is not a finding. It is what the ratchet discipline
is supposed to look like, and it deserves saying plainly before FINDING 1.

---

## THE FOUR MANDATORY INSTRUMENTS

| instrument | rc | reading |
|---|---|---|
| `coverage` | **2 (RED, expected)** | 23 commitments, 0 with no declared spec, **4 CLAIM-DEAD** (smell, balance, thermal, shelter/building), 0 dangling GOAL.md citations — **and the `0 dangling` is false in substance: FINDING 1** |
| `decisions --check` | 0 | 0/10 UNDECLARED, no `MEANS-ESCALATED`, no `OVERDUE`. 3 armed: D15/D16 due 09-05, D17 due 09-07 |
| `champions --check` | 0 | ratchet ok — 0 phantom arenas, 3/4 unfalsifiable, 3+1/6 uncontestable. **It alone caught one face of FINDING 1**: `Fast/slow coupling` is `ARENA-UNREACHABLE — rooted at LC.03` |
| `run review-queue` | 0 | 12 OPEN / 2 HELD / 2 ACTED of 16; oldest live 8 d; consumer ran today; **0 violations** |
| `review_liveness` | 0 | schedule half green, no banner |

**Nothing to arm this audit.** `decisions` reports 0/10 UNDECLARED, so the
standing "arm at least one per audit" duty has nothing to bite on. The ratchet
did not grow.

**On `coverage` rc=2.** Four consecutive commit messages describe it as "the
standing CLAIM-DEAD red, **by design** until the Review's 09-06 window". The red
is *expected*; it is not *by design*. Four of the owner's survival commitments
having nothing runnable behind them is the failure the instrument exists to
show. The row carrying it (`five-commitments-are-claim-dead-behind-foreclosures`,
`DUE 2026-09-06`) is properly armed, so this is a wording note, not a violation
— but "by design" is the phrase a red ratchet wears on its way to becoming
wallpaper, and 09-06 is the day it gets tested.

---

## FINDING 1 (top rank) — a foreclosure is now visible **on** a spec and still invisible **behind** one: 10 specs are welded shut and every instrument prints them as ordinary queue positions

### The mechanism

The 58th audit's B1 taught `coverage.py` that `VOID-FORECLOSED` and
`PILOT-BLOCKED` are retirements. `foreclosure()` is now the one shared
conjunction, and it is right. But it is only ever asked about a spec *itself*.
Nothing asks it about a spec's **blockers**.

`claim_reachability()` emits `blocked<-ROOTS` for two situations it cannot tell
apart, and its own docstring names the distinction it is failing to draw:

> *"blocked resolves when the blocker does; parked resolves never."*

`DP.02` is `blocked<-LC.03`. `LC.03` is `VOID-FORECLOSED` — by this tool's own
authoritative predicate, it **resolves never**. So the state label is exactly
backwards for it, and `run blocked` prints it in the same idiom as a spec
waiting on a job that will finish tonight.

### The number, computed live against `foreclosure()` over all 217 specs

**10 specs have a terminal blocker set that is entirely foreclosed or parked** —
not one of them can be dispatched, ever, without a redesign upstream:

| spec | welded behind | state of that root |
|---|---|---|
| `DP.01`, `DP.02`, `DP.03` | `LC.03` | VOID-FORECLOSED |
| `LC.04`, `LC.05`, `LC.06` | `LC.03` | VOID-FORECLOSED |
| `OP.01`, `PS.04` | `LC.03` | VOID-FORECLOSED |
| `ME.6` | `T2.11` | PILOT-BLOCKED |
| `T5.06` | `T3.06` | VOID-FORECLOSED |

(`T5.08` is a mixed case — `T2.01` FAIL is live, `T3.06` is not.)

### Why this is worse than a bookkeeping error: three of them are in the constitution

`coverage` reports **"16 spec ids cited, 0 dangling"**. The citation check asks
whether the id resolves in `BY_ID`. It does not ask whether the spec can run.
Of GOAL.md's 16 cited ids, **three are welded**:

- **`LC.04`** — GOAL.md:242 says the learning-core bakeoff *"is already testing
  reactive PPO against world-model arms that can imagine"*. Present tense.
  `LC.04` has never run, and `CHAMPIONS.md:63` records that D10's default
  amended its premise away: *"LC.04–LC.06 run only if the premise is repaired."*
  The constitution asserts as live a piece of work the champions file records as
  retired.
- **`DP.02`** — and this is the one that matters. GOAL.md:252 makes it the
  load-bearing guard on the whole one-brain claim: *"The connectedness claim is
  the one that can quietly fail... So it is tested directly and adversarially:
  DP.02 lesions the shared trunk and requires BOTH modes to degrade together...
  a connectedness test that cannot detect a disconnected system is measuring
  nothing."* `DP.02` cannot be run. It is also the declared `ARENA:` for the
  `Fast/slow coupling` seat, held **BY DECREE**.
- **`DP.03`** — same weld.

So the constitution's own named defence against *"two private towers while every
capability number keeps improving"* is currently unrunnable, and the only
instrument that noticed anything is `champions`, which says
`ARENA-UNREACHABLE (1)` — one seat — and cannot say *"and the sentence in
GOAL.md that this seat exists to enforce is now false."*

### Being fair to the builder about the 4-vs-5

`da9880a` argued explicitly against counting `fast/slow` as CLAIM-DEAD:

> *"blocked-is-alive is the founding distinction and flooding it via FAIL-roots
> would kill every commitment behind T2.01"*

**That reasoning is correct for exactly one of fast/slow's five claims and wrong
for three.** `BO.01 <- DP.05` is a FAIL root: repairable, genuinely alive, and
the T2.01-flooding worry is real. But `DP.01/02/03 <- LC.03` is not a FAIL root
— it is a foreclosure, which this same commit taught the tool resolves never.
The justification conflated two root kinds and only one of them survives it.
Note the arithmetic: **the FAIL-flooding worry applies to zero of the 10 specs
in the table above.** Every one of them is welded behind a foreclosure or a park.

The sixth-state question is routed and clocked (`DUE 2026-09-06`), so this is
**not a violation** — it is a routed row whose framing is one distinction short,
and B1 below supplies the distinction and the number.

---

## FINDING 2 — D1.0's ledger verdict prose does not describe D1.0's measurement, and it is the only human-readable output of 16.17 GPU-hours

Harvested 18:23 today (uncommitted at time of writing). The `verdict` string
recorded in `experiments/ledger.json`:

> *"VOID — learning gate: c_e2e at 2.56 sigma vs random (bar 3.0); arms that DID
> learn: {'aprime': 9.98, 'b_split': 8.91, 'd_mlp': 15.47}. **Two non-learners**
> cannot arbitrate an architecture (T2.02's precedent). **If d_mlp is among the
> missed** while T2.02's SB3 MLP holds 530/7.11 sigma, this is the shared
> trunk-tuned recipe failing the MLP..."*

Both bolded clauses are false of this run. **One** arm missed, not two. `d_mlp`
is not among the missed — it is the **winner at 15.47σ**. They are hard-coded
into the f-string at `experiments/tests/d1_0_control_path_bakeoff.py:766-771`,
so they print regardless of what was measured.

Three lines above them sits the comment that forbids exactly this:

```python
# Every verdict names its branch WITH the comparison (the BA.03 lesson: a
# one-bit verdict over a conjunction gets read as its most familiar
# branch, and the readout must carry the comparison, not the operand).
```

The computed half of the string obeys the lesson. The prose bolted after it does
not. Damage: an owner reading the ledger row for the project's largest unblock
learns that two arms failed to learn and that the MLP may have been one of them.
Neither is true, and the row carries no other narrative.

---

## FINDING 3 — the gate that VOIDed a 16-hour bakeoff scores three arms against one spread and the fourth against a different one

`sigma_vs_random = (arm_mean − rnd_mean) / max(arm_std, rnd_std)`
(`d1_0_control_path_bakeoff.py:702`), at n=3 seeds, 5 eval episodes each.

| arm | mean | random | own std | denominator used | σ |
|---|---|---|---|---|---|
| `d_mlp` (530K) | 577.0 | 108.7 | 28.94 | **random's 30.27** | 15.47 |
| `aprime` (282K) | 410.8 | 108.7 | 26.50 | **random's 30.27** | 9.98 |
| `b_split` (57.8M) | 383.5 | 108.7 | 30.83 | own | 8.91 |
| `c_e2e` (57.4M) | 404.3 | 108.7 | **115.31** | **own** | **2.56** ← VOIDed the run |

`c_e2e` returned **404.3 against a random baseline of 108.7 — a 3.7× gain** —
and is recorded as having failed a *learning* gate. What it actually failed is a
**consistency** gate: its seed means were 319.4 / 535.6 / 357.9. The `max()` is a
defensible conservatism, but it means the bar answers a different question for a
noisy arm than for a quiet one, and at n=3 the std in the denominator is a
2-degree-of-freedom estimate doing load-bearing work.

Second observation on the same gate, and it is the sharper one: **the untrained
twins read 2.96σ (`aprime`) and 2.94σ (`d_mlp`) against a 3.0 bar.** The control
that exists to prove the gate measures learning rather than initialisation bias
passed by **0.04σ**. The gate sits essentially *at* the bias level it is meant to
exclude. The spec's own docstring already knows this — line 61 records T2.02's
untrained MLP at 2.74σ. An arm scoring 3.1σ would clear "it learned" while being
statistically indistinguishable from an untrained network.

Not a violation: every constant was pre-registered and frozen before the run,
every control landed on its declared side, and the VOID branch fired as written.
This is a design question for the Review, and B2 below routes it.

**And a real scientific result is buried in the VOID that should not be lost:**
`d_mlp` at **530,339** PPO-trainable parameters beat `b_split` (**57,830,650**)
and `c_e2e` (**57,363,705**) by 5.99σ against a 1.5σ margin bar. That is a fourth
independent replication of GOAL.md:123's 57M-vs-54K lesson. The bakeoff produced
no seat; it did produce evidence.

---

## FINDING 4 — the loop promoted an unreliable receipt to a verified fact, and it is wrong right now

Commit `6342dd6` (17:0x hold slot):

> *"**One new receipt this slot:** `/tmp/jack-ladder-gpu.lock` contains 4187660 —
> the live watcher's pid. Previous slots asserted 'UB.10's dispatch is
> lock-queued'; now it's a **verified fact** tied to the lock mechanism in
> `dispatch.sh:36`."*

`dispatch.sh:39-40`, the file it cites, says the opposite:

```
# NB: the pid inside the file is unreliable — every failed contender opens
# it with mode "w" and truncates it. lsof sees who holds it open.
```

Measured at 18:40: the file still contains `4187660`; `kill -0 4187660` →
**no such process**; the file's mtime is **02:12**, the original dispatch. The
receipt the loop upgraded to "verified" currently asserts a live watcher that
has been dead for ~17 minutes.

**No operational harm, and this matters to state clearly:** `dispatch.sh:37`
gates on `flock -n`, not on the pid, and the kernel released the advisory lock
when the watcher exited. Verified live — `flock -n /tmp/jack-ladder-gpu.lock
true` succeeds. **The GPU is free and `UB.10` can dispatch at the next slot.**
The defect is epistemic, not operational: the builder was right about the
watcher for the wrong reason, using an instrument its own dispatcher documents
as untrustworthy. The two receipts it used alongside it — the Kaggle API and the
attempt row — were the sound ones.

---

## THE AUDIT, SECTION BY SECTION

**1. Integrity of the ledger — NO VIOLATIONS.** 120 rows: **93 PASS, 18 FAIL,
9 VOID** over 217 specs. Every PASS row's `commit` still resolves in git (93/93,
checked with `git cat-file -e`). Zero PASS rows carry a `dirty` flag. 39 PASS
rows have a single seed; 30 are `T0.*` harness fixtures and the other 9
(`PG.1`, `PG.2`, `T1.07`–`T1.13`, `T2.00`) are deterministic world-fidelity and
primitive assertions, not learning claims — GOAL.md's ≥3-seed rule binds "where
the claim is about learning" and none of these is. D1.0's controls were declared
and actually run: per-arm untrained twins plus a random null on matched eval
seeds (`EVAL_SEED_BASE`, repair R6), with a live `hot_twins` VOID branch that
would have fired had any twin cleared 3.0σ.

*One provenance item, checked and clean.* D1.0's row records `commit: 9494cd1`,
but its three kernels ran at heads `9494cd1`, `7d46e82` and `566f840`. Verified:
`experiments/tests/d1_0_control_path_bakeoff.py` is **byte-identical at all
three** (sha256 `870b3c776c87f140`, matching the row's `impl_sha`). The single
`commit` field under-describes a three-kernel run but does not misrepresent it.
Recorded so a future audit does not have to re-derive it. `hardware` reads
`aarch64/Linux/torch2.8.0+cpu/cpu` — the harvesting box, not the three P100s;
the `kernels` sub-object carries the real hardware, so this is cosmetic.

**2. Thresholds and controls over seven days — NO VIOLATIONS, and this is the
strongest clean result in the audit.** 81 commits touched `registry.py`,
`registry_expansion.py` and `experiments/tests/`. Every paired constant change
was extracted and directionally classified. **Tightenings:** `COORD_MIN`
0.55→0.70, `COORD_MARGIN` 0.20→0.35, `LIVES_PER_ARM` 4→16→48, `N_EVAL`
48→120→800, `N_DECISIONS` 3200→up to 20,000, `STEPS` 300→500. **Apparent
loosenings, both resolved as false positives:** `N_PROPERTIES` 8→10→11 is three
*different* constants in three different T0 specs, each growing; and:

*The one genuine downward move — `DECAY_MIN` 1.5 → 1.25 (`44f24c4`, T2.09).*
Audited in full and it is clean on every axis that matters:
- it was a `# PILOT —` placeholder being frozen for the first time, and
  `run()` refused until that commit — no registered threshold was weakened;
- declared in the commit **subject line**: *"ONE BAR MOVED, and downward"*;
- justified from what the gate is *for* (a dead signal decays by exactly 1.0)
  rather than shaved to the observation (1.472);
- **pilot seeds 7 and 90; registered scoring seeds 0–6 — disjoint.** The gates
  were not fitted to the data that scored them;
- zero live effect: every seed exclusion on the recorded run fired on
  `trap_dwell`, and the code comment at line 602 pre-commits that `DECAY_MIN`
  does not move again ("re-fitting it after the PASS would be the real
  violation").

No `_check` gained an `or`. No control was deleted or weakened. No seed count
was reduced. **Reported as clean because it is clean.**

**3. Drift from the goal — no drift in what was chosen; a stall in what it
reached.** The last 24 h of builder work, each traced:

| work | traces to |
|---|---|
| `da9880a` CLAIM-DEAD sees foreclosures | GOAL.md:8 "protects the honesty of watching" |
| `a2748f1` unreachable-fraction ratchet (baseline 85) | same |
| `7f1e875` GEN.02/03/06/09 registered | GOAL.md:190-198, the three expansions |
| `ff9917a` UB.14 foreclosure declaration | GOAL.md:51-55, unified brain |
| D1.0 bakeoff, 16.17 GPU-h | GOAL.md:76 PLASTIC-ONLY / control-path seat |

**Nothing serves no GOAL.md sentence.** But five of six items are instrument
work, and the sixth returned no verdict. The converse question is where the
answer bites: **8 commitments have live claim specs and nothing passing, and 4
are CLAIM-DEAD.** The three the instruction names specifically —
- **curiosity**: 12 specs, 2 PASS, **0 runnable**; `T3.06` (*does curiosity earn
  its parameters*) is VOID-FORECLOSED and `T5.06` is welded behind it;
- **all-senses fusion**: `one brain / unison` 23 specs, **1 PASS**, 1 runnable;
- **learning by living**: `death & retry` 3 specs 0 PASS; `sleep` 5 specs 0 PASS;
  `hunger/thirst` 6 specs 0 PASS; `fast/slow` 8 specs 0 PASS.

**4. Builder alive and productive — alive, honest, and stalled.** 19 iterations
today, **19 of 19 rc=0**, no crashes, no credit exhaustion, no paused loop, one
LEFTOVER process correctly reported at 04:27 and not killed. PASS delta over
24 h: **92 → 93 (+1)** while the spec denominator went 213 → 217 (+4).
**All 19 iterations ended `93 -> 93`.** The last four self-describe as hold
slots behind D1.0.

*Were the holds honest?* Checked, and largely yes. I ran `run next` and resolved
every one of the 12 top-ranked runnable specs against the ledger: `T2.05`,
`T2.07`, `T2.10`, `T2.15`, `T3.07`, `T4.02`, `T0.27` are all settled **FAIL**;
`T3.06` is VOID-FORECLOSED; `T2.11` is PILOT-BLOCKED and parked; `T2.01`/`T2.02`
are the GPU-locked pair. The board was genuinely empty of *fresh* units. The
loop manufactured no bookkeeping to fill the slots and said so — that is the
right behaviour, and it is worth crediting given how easy the alternative is.

The cost is still real: **~14 hours of wall clock with the capability count flat
and the free GPU quota running**, and FINDING 4 shows the hold's own liveness
receipt was the weakest of the three it had.

**5. Compute honesty — every hour is accounted for; the concern is
concentration, not waste.** W35 (resets Sunday): **17.68 h Kaggle + 0.27 h Colab
of 30 h. ~12.3 h remain, 5 days left.** D1.0 consumed **16.17 h across 3 kernels
(4.08 + 6.03 + 6.06)** — **91% of the week's GPU spend on one spec, which
returned a VOID.** Every hour has a submission row, an `ok: true`, a job id and
a ledger row that names all three kernels: **no unaccounted GPU hours.** This is
not the 61-hour expired-quota scar repeating; it is the opposite failure mode —
the quota was spent, on one thing, and the thing did not settle. Prior weeks for
contrast: W34 1.62 h, W33 7.63 h, W32 21.06 h.

**6. Stuck decisions — NO VIOLATIONS.** 0 UNDECLARED, 0 MEANS-ESCALATED, 0
OVERDUE. Three armed and unfired (D15, D16 due 09-05; D17 due 09-07). D1's
default fired on schedule 2026-09-01 and its artifact — the four-arm bakeoff —
was registered, dispatched and harvested the same day. Nothing was quietly acted
on: D1's firing is recorded in `DECISIONS_RESOLVED.md`, `CHAMPIONS.md:66` and
the registry note in the same commit. No open entry has evidence sufficient to
settle it that is being ignored.

**7. Bakeoff hygiene — NO VIOLATIONS.** Every recorded decision this week
carries a learning gate. No VOID is treated as a verdict — and the discipline is
visibly load-bearing: D1.0 had a winner (`d_mlp`, 5.99σ over a 1.5σ bar) and
**refused to seat it** because one arm missed the learning gate. `CHAMPIONS.md`
seats `wm-latent` **"BY VERDICT, with the single-arm caveat on its face"** and
states "Seated ≠ adopted", gating adoption behind `LC.07`. No winner was chosen
inside a noise margin.

---

## FOR THE BUILDER

**B1 (top priority) — `foreclosure()` must be asked about BLOCKERS, not only
about specs.** The 58th audit's B1 fixed the state *on* a spec; the state
*behind* one is still `blocked<-ROOTS`, indistinguishable from a live queue
position.

1. In `claim_reachability()` (`coverage.py:439`), when the terminal blocker set
   is non-empty and **every** root is foreclosed-or-parked, emit a distinct
   state — `welded<-ROOTS` — not `blocked<-ROOTS`. Keep `blocked<-` for any set
   containing a live root (FAIL, VOID-not-foreclosed, NOT_RUN). This is exactly
   the FAIL-vs-FORECLOSED distinction `da9880a` needed and did not have.
2. Mirror it in `run blocked` via the shared walk, so the ranking that says
   "unblock this" stops ranking things nothing can unblock.
3. **Do not use this to lower the CLAIM-DEAD number.** `fast/slow` should now
   read 3 welded + 1 foreclosed + 1 genuinely blocked (`BO.01<-DP.05` FAIL) —
   whether that makes the commitment CLAIM-DEAD is the routed 09-06 question,
   and B1 supplies the input to it, not the answer.
4. Known-answer battery in the `_claim_dead_fixture` idiom, planting the shape
   that battery is currently missing: **a claim blocked behind a foreclosed
   root** (`DP.02<-LC.03`), and the direction that must stay alive
   (`BO.01<-DP.05` FAIL → still `blocked<-`). Verify it RED against today's code
   before the repair. **Expected live reading: 10 welded specs** —
   `DP.01`, `DP.02`, `DP.03`, `LC.04`, `LC.05`, `LC.06`, `OP.01`, `PS.04`
   (all `<-LC.03`), `ME.6` (`<-T2.11`), `T5.06` (`<-T3.06`).
5. `UNREACHABLE_BASELINE = 85` must not move on account of this change — the
   union is the same, only the labelling improves. If it does move, the walk
   changed and that needs its own justification.

**B2 — the GOAL.md citation check must test liveness, not resolution.**
`coverage` prints "16 spec ids cited, **0 dangling**" while `LC.04`, `DP.02` and
`DP.03` are welded. Extend the citation check to flag a cited id that is
foreclosed, parked, or welded, with its own line — a constitution that cites a
retired spec in the present tense is the highest-value dangling reference there
is, and it is currently the only class the checker cannot see. Suggested wording
for the new red: `CITED-BUT-UNRUNNABLE: LC.04, DP.02, DP.03`.

**B3 — fix the D1.0 verdict string; do not re-run anything.** At
`d1_0_control_path_bakeoff.py:766-771`, `"Two non-learners"` and `"If d_mlp is
among the missed"` are hard-coded and both are false of the recorded run.
Replace with the count and the names actually computed
(`len(missed)`, `sorted(missed)`), and make the `d_mlp` clause conditional on
`"d_mlp" in missed`. **The ledger row's metrics are correct and must not be
touched** — this is a readout repair for the next run, plus an `amended` note on
the existing row recording that its prose misdescribed its own measurement.
The comment three lines above already states the law this violates.

**B4 — route two REVIEW_QUEUE rows for the D1.0 gate design.**
- `d10-learning-gate-uses-two-different-denominators`: `max(arm_std, rnd_std)`
  scored three arms against random's spread and `c_e2e` against its own, at
  n=3/5-episodes. `c_e2e` gained 3.7× over random (404.3 vs 108.7) and is
  recorded as not having learned. Options to weigh: a paired t-statistic, a
  fixed random-spread denominator, more eval episodes, or an explicit separate
  *consistency* gate so "noisy" and "did not learn" stop sharing one verdict.
- `d10-learning-gate-sits-at-the-untrained-twin-level`: untrained twins read
  2.96 / 2.94σ against a 3.0σ bar. The control passed by 0.04σ. Consider scoring
  each arm against **its own untrained twin** rather than against random.
- Both should be bundled with the existing `w0-too-shallow` window if the Review
  judges the venue to be the common cause.

**B5 — record the D1.0 result that the VOID does not carry.** `d_mlp` at 530,339
PPO-trainable params beat two ~57M-param arms by 5.99σ (bar 1.5). That is a
fourth replication of GOAL.md:123 and it should exist somewhere a reader will
find it, even though the run seated nobody.

**B6 — stop citing the GPU lock's pid as a receipt.** `dispatch.sh:39-40`
documents that pid as unreliable. Right now the file says `4187660` and that
process is dead. Use `kill -0` on the pid, or `lsof -t` as the dispatcher's own
comment directs, or the Kaggle API. Optional and cheap: have the watcher unlink
the file on clean exit so a stale pid cannot be quoted at all. **Note the good
news in the same breath — `flock` is free and `UB.10` can dispatch now.**

**B7 (small, provenance) — record all kernel heads on a multi-kernel row.**
D1.0's `commit` field names only kernel 1's head. Verified harmless today (the
impl is byte-identical at all three), but the row should carry the set, and
`hardware` should record the compute host rather than the harvesting box when a
`kernels` block is present.

---

## FOR THE OWNER

**1. The sentence in your constitution that guards against "two brains" cannot
currently be tested.** GOAL.md:250-255 says the connectedness claim *"is the one
that can quietly fail"* and names `DP.02` as the adversarial test that catches
it. `DP.02` is welded behind `LC.03`, which is VOID-FORECLOSED — it cannot be
dispatched without an upstream redesign. `DP.03` and `LC.04` are in the same
state, and GOAL.md:242 describes `LC.04` as *"already testing"* when it has
never run and `CHAMPIONS.md` records its premise as retired. **Nothing you have
been shown said this**, because each instrument sees one facet: `champions` says
one seat is arena-unreachable, `coverage` says zero citations dangle, and
`run blocked` prints all three as ordinary queue positions. B1 and B2 make it
visible. **No action needed from you today** — but the fast/slow branch of your
2026-08-10 directive is stalled, not merely slow, and you should know that the
repair is a redesign upstream at `LC.03`/`W0`, which is already on your desk as
**D10**.

**2. Your D1 default fired on time, the bakeoff ran, and it did not settle.**
The armed default fired 2026-08-31 as written; the builder registered the
four-arm bakeoff, dispatched it, and harvested **16.17 GPU-hours across three
Kaggle kernels** today. The result is **VOID** — one of four arms (`c_e2e`) came
in under the 3σ learning gate, and the rig refuses to arbitrate an architecture
when an arm has not demonstrably learned. That refusal is correct protocol and I
am not asking you to override it.

Two things you should see anyway:

- **The measurement it did make.** A **530K-parameter MLP beat both ~57M-param
  arms by 5.99σ** against a 1.5σ margin bar. Your 57M-vs-54K lesson now has a
  fourth independent replication. `T2.01`/`T2.02` and the 35+ specs behind them
  remain blocked because the seat could not be filled.
- **The arm that "failed to learn" returned 404.3 against a random baseline of
  108.7 — a 3.7× gain.** It was scored below the bar because its three seed
  means were 319 / 536 / 358 and the gate divides by the arm's own spread when
  that spread is larger than random's. It failed a consistency test, and the
  ledger calls it a learning failure. Routed to the Review as B4; flagging it
  because "the end-to-end arm did not learn" is the sentence that would
  otherwise enter the record, and it is not what was measured.

**3. Nothing was weakened this week, and I checked mechanically rather than
taking anyone's word.** 81 commits touched the registry and the tests. One
numeric bar moved downward — `DECAY_MIN` 1.5→1.25 — and it was a placeholder
being frozen for the first time, declared in its own commit subject, justified
from first principles rather than fitted to the observation, with the pilot
seeds (7, 90) disjoint from the seeds that scored the spec (0–6). Everything
else moved in the tightening direction. Ninety-three PASS rows, every commit
still resolving, no dirty stamps.

---

## THE HONEST SUMMARY

**Are we closer to a curious humanoid that climbs the ladder than yesterday, or
only closer to a longer list of green ticks?**

Neither, today — and that is a more uncomfortable answer than either.

The green-tick count did not move: **93 → 93 across all 19 iterations**, with the
denominator growing 213 → 217. The one piece of actual science, D1.0, spent
**91% of the week's free GPU** and returned no verdict. So we did not buy ticks.

But we did not buy capability either. What we bought was **honesty
infrastructure** — three ratchets sharpened in a single day, `coverage` driven
deliberately RED, a foreclosure class invented so retirements stop laundering
themselves as runnable work. That is real and it is not busywork: this project's
whole bet is that a ledger you cannot trust is worth less than no ledger, and
today the builder made the instruments harder to fool at the cost of its own
score. When it diverged from the audit's spec it said so in the commit message
and clocked the remainder. That is the behaviour you want from a loop that could
just as easily have shipped four green ticks.

The drift is not in what the builder chose. It is in the **gap between the
instruments and the ladder**. The instruments got sharper today; the ladder got
one spec's worth of nothing. And underneath, a foreclosure from 2026-08-23 has
been quietly welding specs shut for nine days — ten of them now, including the
one test GOAL.md names as its defence against the failure mode you care most
about. Every organ reported correctly. None of them could say *"the
constitution's own guard cannot be run."* That is the same shape as the
2026-08-10 miss that created `coverage.py`, and the same shape as the D1
deadlock that created rule 3: not a broken instrument, but a true statement that
no instrument is built to utter.

**Jack still cannot climb a ladder, and today he got no closer.** He also did not
get a fake certificate saying he did. Given the choice this system actually
faced, that is the right trade — but it is only the right trade if it is rare,
and this is the second consecutive day the answer to section 8 has been about
instruments rather than about Jack.
