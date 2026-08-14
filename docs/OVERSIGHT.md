# OVERSIGHT — 17th audit, 2026-08-14 12:45 UTC

## VERDICT: ON TRACK

The ledger is sound and I checked it the hard way. All **80 PASS** records name
a commit that still exists, an implementation on disk, and — the check that
matters — for every PASS whose spec declares a control, `_check(m, c)` actually
references its control parameter (AST over the second formal argument, not the
presence of `control_metrics`). Zero exceptions.

**No findings in section 2.** Over 7 days, 123 commits and 851
constant-definition lines across the registry and `experiments/tests/`, there
were 24 in-place changes and **every one strengthened or resized apparatus**.
No threshold moved in the loosening direction, no control was deleted, no
assertion removed, no seed count reduced. I verified the control claim
structurally as well: parsing the registry at HEAD against `90d8b3c` (7 days
back), **zero specs lost a control and zero specs were removed** — 105 specs
became 169. Saying that plainly is the honest result.

The findings are about **instruments, not claims**. Ranked by how much damage
they do to the trustworthiness of the ledger:

1. The gate that exists to catch invisible holes — `coverage.py` — is now
   permanently red for a cosmetic reason, and its exit code cannot distinguish
   that from a constitutional hole.
2. Nine ledger records **contradict themselves** about which machine ran them.
3. `duration_s` misstates the cost of a harvested run by up to six orders of
   magnitude — on the same field the project's own sizing law tells the next
   iteration to read.

**169 specs · 80 PASS · 3 FAIL · 4 VOID · 0 NOT_RUN.** Builder: **24
iterations in 24 h, 24 rc=0**, PASS delta **+1**. Kaggle W32: **8.86 h floor
remaining (range 8.86–15.24 h), expires Sunday 2026-08-16 — ~44 hours.**

---

## 0. Is the ladder the RIGHT ladder?

`python -m experiments.coverage` → **exit 1**, and the reason is not an
uncovered commitment.

**0 commitments with NO declared spec.** The 2026-08-10 miss has not recurred.
`GOAL.md` last moved on 2026-08-10 (`03dd742`, the DP family), and
`COMMITMENTS` names it (`fast/slow`). Nothing constitutional is unnamed.

### RANK 1 — the ladder-is-right gate is red on a typo, and red-for-a-typo is indistinguishable from red-for-a-hole

`experiments/registry.py:524` — T2.05 declares:

    COVERS: world model (claim)

`world model` is not in `COMMITMENTS`. The declaration is parsed, rejected, and
reported as MALFORMED. Two consequences, and the second is the serious one:

**(a) The week's largest science spend bought zero declared coverage.** T2.05
cost 0.90 h of a quota that dies in 44 hours. Its author intended it to cover
something; it covers nothing. `coverage.py`'s own docstring predicted this
exact failure mode — *"a typo'd marker looks exactly like a claim to a human
reader and buys exactly nothing from this file"*.

**(b) `check()` returns `len(uncovered) + len(bad)`** (`coverage.py:293`). The
exit code is a **sum of two categories with wildly different severity**. Today
it reads 1 because of a typo. If a constitutional commitment lost its last spec
tomorrow, the exit code would still read 1, the process would still exit 1, and
nothing about the failure would change. The overseer prompt names this gate as
the highest-priority instrument in the system precisely because a missing spec
"has no id, appears in no `run blocked` ranking, blocks nothing and fails no
gate". A gate whose alarm has one bell for two fires is one iteration of
tolerated-red away from being ignored — and `coverage.py:117` has the rule
written down already: *"a false malformed-declaration report trains its reader
to ignore the real ones"*. This is that rule firing against its own author.

**Note for whoever fixes it:** "world model" is a **mechanism**, not a
commitment. It is not in `GOAL.md` as a thing owed; imagination appears there
only as biology's oracle for dreaming and inside the fast/slow axis-1 wording
(*"world-model arms that can imagine"*). So the fix is to re-declare T2.05
against an existing commitment (or to drop the marker), **not** to add
`world model` to `COMMITMENTS`. The standing rule — add a commitment when
`GOAL.md` gains one `coverage.py` cannot name — does not apply here.

### The shape of the ladder, unchanged and worth restating

| tier | pass/total | |
|---|---|---|
| T0 harness | 28/28 | complete |
| T1 primitives | 13/13 | complete |
| T2 vs null | 37/59 | T2.01 FAIL, T2.02 + T2.05 VOID |
| **T3 earn your parameters** | **0/14** | T3.07 FAIL |
| **T4 unison** | **1/23** | T4.02 FAIL |
| **T5 THE CLAIMS** | **0/27** | BA.02 + LC.03 VOID |
| T6 living Jack | 1/5 | |

**41 of the 80 PASSes are Tiers 0–1** — the measurement apparatus and the
primitives. **Zero-pass constitutional commitments: 17 of 23**, unchanged for
five audits. Six commitments carry a passing *claim-kind* spec, and they are
carried by five specs, because UB.9 is counted twice:

| commitment | its only claim-kind PASS | tier |
|---|---|---|
| hearing | UB.9 *Heard, not seen* | 4 |
| one brain / unison | UB.9 (same spec) | 4 |
| damage/nociception | PS.03 | 2 |
| memory across lives | ME.10 | 2 |
| curiosity | T2.08 | 2 |
| generality | T1.02 | 1 |

`run stale` reads: two entries flagged (T2.04, T3.07 — both changed after their
run, correctly flagged) plus T2.02 (VOID, deliberately left flagged pending
D1). Of 28 pre-`impl_sha` records, 1 stale by content, 27 byte-identical, **0
unanswerable.**

---

## 1. Integrity of the ledger — clean

87 entries. For all **80 PASS**:

- every `commit` resolves in git (`git cat-file -e`) — **0 missing**;
- every spec has an implementation in `experiments/tests/` — **0 missing**;
- every spec declaring a `control` has `control_metrics` on its record — **0
  missing**; only T0.01 and T0.10 lack them and both correctly declare
  `control=None`;
- **every one of those tests' `_check` actually reads its control argument.**
  Verified by AST: bind the second formal parameter, walk the function body,
  require a `Name` load of it. **0 exceptions.** A PASS whose control was never
  consulted would be a claim without evidence; there are none.

---

## 2. Thresholds and controls over time — NO FINDINGS

Method: parse every `+`/`-` constant-definition line in
`git log -p --since="7 days ago"` over `registry.py`, `registry_expansion.py`
and `experiments/tests/`, pair them per (commit, file, name), and read every
in-place change. **123 commits, 851 constant lines, 24 in-place changes.**

Every one is a strengthening, an apparatus resize, or a comment:

- `xl_00`: `N_PERM` 2,000 → **100,000** (stronger null)
- `vo_01`: `N_CALIB` 60 → **400**; `N_OCC` 160 → `2*N_TRAIN`
- `t0_20/21/22`: `N_PROPERTIES` 6→7, 7→8→9, 9→12→13→14→15 (more properties)
- `ps_01`: `N_DECISIONS` 3,000 → **4,500** (v2, longer life)
- `pg_6`: `RES` 64 → **96** px
- `ba_02`: five PILOT-FINAL gates finalised at **unchanged values** with the
  pilot measurement written beside each
- the remainder are `IMPL_DEPS` additions (more staleness surface) and comments

Structural check, which a diff can miss: registry parsed at HEAD vs `90d8b3c`
(2026-08-07). **Specs with a control that no longer have one: none. Specs
removed: none.** 29 specs carry no control, all of them unimplemented Tier
3–6 stubs; `run_spec` refuses an undeclared control at run time, so none can
reach the ledger that way.

**The one historical move remains the 13th audit's, and it stays honest.**
T2.08's absolute floor went 0.70 → 0.50 after the run it failed at 0.6975
(`1454525`). The commit message carries the arithmetic, the re-derivation from
the gate's anti-collapse purpose, *and* a new 3σ paired-margin gate v1 never
had — a net strengthening, taken in the open under law 4's clause, with the
FAIL preserved in the record's `history`. T0.27 was built as its scar. I
re-read it and I do not reopen it. But it belongs in section 8, not here,
because **T2.08 is curiosity's only claim-kind PASS**, and what now carries
that claim is a `coverage_margin` of **0.0544** (bored 0.6975 vs random
0.6016), not an absolute floor. That is a real effect, honestly gated at 5.0σ
paired — and it is thin. Curiosity is one of the two or three sentences
`GOAL.md` is actually about.

---

## 3. Drift from the goal

**What the builder did in the last 24 h, and what each serves:**

| work | GOAL.md sentence it serves | verdict |
|---|---|---|
| LC.03 VOID landed (15 h, 3 seeds, frozen-twin control fired) | *"learns his world by living in it"* — the learning-core seat | serves |
| T2.04 PASS, behaviour cloning on scripted trajectories | staging: *"first prove he can see, talk, walk, and learn in every way"* | serves, **as apparatus** |
| T2.05 implemented → probed → dispatched → VOID landed | fast/slow axis 1, *"world-model arms that can imagine"* | serves |
| `scripts/dispatch.sh` + `JACK_SPEC_ID` lesson | SYSTEM.md: *"Fixing one bug is maintenance. Making that bug unrepeatable is building."* | serves |

**No drift.** Every unit traces. One caveat worth stating rather than
flagging: T2.04 is *behaviour cloning on scripted trajectories* — the most
curriculum-shaped thing on the ladder, and the literal inverse of "not from a
curriculum we write". It is legitimate as apparatus (it proves the action path
can receive a gradient at all) and it correctly declares **no** `COVERS`
marker, so it buys no commitment credit. It should not be read as movement
toward the north star, and the builder's own summary did not read it that way.

**The converse question — what has no passing spec at all — is where the
answer is uncomfortable, and it is unchanged:**

- **Curiosity**: 12 specs, **1** claim-kind PASS (T2.08, margin 0.0544).
- **All-senses fusion**: 21 specs under `one brain / unison`, **1** claim-kind
  PASS (UB.9), which is also the only `hearing` pass.
- **Learning by living**: LC.03 is **VOID** — its life-gain ruler measured the
  world, not learning — and LC.04/05/06 sit behind it. The learning-core match
  has not started.
- Smell, taste, voice, balance, proprioception, thermal, hunger/thirst, sleep,
  death-and-retry, shelter, tool use, social, plasticity, language, fast/slow,
  sight, touch: **zero claim-kind passes**, seventeen of them, five audits
  running. Every one has apparatus — sensors, fixtures, rules — and that
  apparatus is real work correctly not credited as the commitment.

---

## 4. Is the builder alive and productive? — yes, and unusually clean

Window 2026-08-13 12:10 → 2026-08-14 12:10 UTC:

- **24 iterations, 24 `rc=0`.** Zero aborts. (The last aborts were three
  ~3-second `rc=1` session-limit hits at 10:07/11:07/12:07 on 08-13 — now
  outside the window, and the class is instrumented by the 14th audit's B4.)
- **PASS delta +1** (79 → 80). Over 29 h it is +2 (78 → 80).
- Two GPU results landed (T2.04 PASS, T2.05 VOID), one CPU VOID (LC.03), one
  new guard (`dispatch.sh`), two LESSONS entries.
- No paused loop, no credit exhaustion inside the window, no repeated identical
  failure.

**One recurrence, and it is the 16th audit's B3 verbatim.** The 10:07
iteration ran 3 m 38 s and did nothing but confirm two waiters were armed
("*Nothing else to do until then*"). B3 asked that a hand-off gated on a
detached job carry its **expected completion time** so the next iteration can
pick a different unit. The rationale the builder gave — CPU contention on 4
shared cores — is sound and I accept it for CPU-bound waits; it does not apply
to a **Kaggle kernel**, which contends for nothing on this box. Re-raised
below, narrowed to that case.

---

## 5. Compute honesty

**The books reconcile exactly.** Charged Kaggle jobs with `ok:true` sum to
**14.6364 h**; plus the labelled opening balance **6.3849 h** = **21.0213 h**,
which is `weeks["2026-W32"]["kaggle"]` to the digit. Failed jobs
(0.0466 + 0.0759 = 0.1225) match `kaggle_failed` exactly. The 16th audit's B2
is closed and correct: `remaining()` is a floor (**8.8562 h**) and
`remaining_range()` reports the honest interval (**8.86–15.24 h**).

**What the hours bought.** Every large W32 charge is attributable:

| job | hours | bought |
|---|---|---|
| `1786304547` | 5.5786 | T2.01, the 19:42→01:17 poll (attributable **only via LOOP_JOURNAL.md:1585** — no receipt exists) |
| `1786519461` | 5.5798 | T2.01 FAIL, ledger entry `2026-08-12T12:59` (20,097 s) |
| `1786691678` | 0.9354 | **T2.04 PASS** |
| `1786705853` | 0.8978 | **T2.05 VOID** |
| `1786673420` | 0.4435 | T1.07 re-run PASS |
| 7 others | < 0.35 each | probes, smokes, two failed submissions |

**No hours were wasted.** A VOID is not waste — T2.05's 0.90 h bought the
knowledge that the imagination path loses to ridge on its own ruler, which is
what the ladder is for.

**Two small ledger-of-compute notes.** Colab: charged rows sum to 0.7601 h
against a `weeks` counter of 0.7616 h — a **0.0015 h (5.4 s) unattributable
residue**, the same species as the 6.38 h Kaggle balance B2 just labelled, at
1/4000th the size. Not worth an entry; worth one line so it cannot grow
silently. And the 5.58 h `1786304547` job predates receipts entirely — the
builder's new `JACK_SPEC_ID` rule closes that class going forward.

### RANK 3 — `duration_s` is the recorder's wall clock, and it is read as the run's cost

`protocol.py:1447` sets `duration_s = time.time() - t0` around the `run_spec`
call. For anything **harvested or reattached**, that is the recording, not the
run:

| record | `duration_s` says | it actually cost |
|---|---|---|
| **LC.03** (VOID) | **0.02 s** | ~15 h × 3 seeds — its own metrics carry `dreamer-xs/core_s = 4320 s` per arm |
| **T2.04** (PASS) | **69.32 s** | a P100 kernel billed **0.9354 h** (3,367 s) |
| T2.05 (VOID) | 3,240.7 s | correct — the watcher waited |

The failure this sets up is specific and near. The project's newest and most
expensive lesson is *"a cost measured on the smoke's configuration is not a
cost for the production configuration"* (LESSONS.md:3690) — size from
measurement. **LC.03's rig re-derivation is the next queued CPU unit**, and the
only cost this ledger records for LC.03 is **0.02 seconds**. `run status`
prints this field. An iteration that sizes from it will be wrong by six orders
of magnitude.

---

## RANK 2 — nine ledger records contradict themselves about which machine ran them

`protocol.py:1436-1443` carries the fix and the confession in one comment:
*"nine GPU records read aarch64/…/cpu while the truth sat in `metrics["gpu"]`
(overseer B3)"*. B3 fixed it **forward** — T2.04 and T2.05 now read
`remote/Tesla P100-PCIE-16GB (dispatched from …)`. The nine were never
corrected, and I count exactly nine today:

    T0.09 PASS  metrics.gpu='Tesla T4, 15360 MiB'     hardware=aarch64/…/cpu
    T1.07 PASS  metrics.gpu='Tesla P100-PCIE-16GB'    hardware=aarch64/…/cpu
    T1.08 PASS  metrics.gpu='Tesla T4'                hardware=aarch64/…/cpu
    T1.09 PASS  metrics.gpu='Tesla P100-PCIE-16GB'    hardware=aarch64/…/cpu
    T1.10 PASS  metrics.gpu='Tesla P100-PCIE-16GB'    hardware=aarch64/…/cpu
    T2.01 FAIL  metrics.gpu='Tesla P100-PCIE-16GB'    hardware=aarch64/…/cpu
    T2.02 VOID  metrics.gpu='Tesla P100-PCIE-16GB'    hardware=aarch64/…/cpu
    T2.03 PASS  metrics.gpu='Tesla P100-PCIE-16GB'    hardware=aarch64/…/cpu
    T4.02 FAIL  metrics.gpu='Tesla P100-PCIE-16GB'    hardware=aarch64/…/cpu

**Why this is different from the staleness class already lessoned.**
LESSONS.md:3454 rules that a provenance mechanism cannot cover records that
predate it — true, and the reason `impl_sha` cannot be back-filled: the
evidence is gone. **Here the evidence is not gone. It is in the same record,
one field away.** T1.07's commit message says *"PASS on Kaggle P100 (1606.7 s)"*,
its billed job (`1786673420`, 0.4435 h = 1,597 s) matches, its `metrics.gpu`
says P100 — and its provenance line says this box's CPU. That is not absent
information, it is **wrong information**, and it is mechanically derivable
without re-running anything.

Cross-check the two claims most exposed: T2.03 (sight, 1,206 s) and T4.02
(fusion-boundary gradients, FAIL) both read local-CPU while their own metrics
name a P100.

---

## 6. Stuck decisions

`docs/DECISIONS_NEEDED.md`, three live items. **Nothing is parked that the
system could have decided itself, and nothing was quietly acted on.** D2 was
resolved by ledger replay and correctly written into
`DECISIONS_RESOLVED.md` — the loop took a property question off the owner's
desk, which is the behaviour we want.

- **D1 (57M trunk in the control path)** — open since **2026-08-09, five
  days**. Evidence complete and unchanged. The 15th audit correctly refused to
  let a one-word reply be read as option A, because option A (freeze the trunk)
  is barred by the PLASTIC-ONLY decree that postdates it. The owner still owes
  exactly one line: **strike option A, or narrow the decree's scope**. See FOR
  THE OWNER for what it now costs.
- **D7 (MovementMoodCoupling failed T3.07)** — correctly the owner's; SYSTEM.md
  puts component deletion on their desk. Evidence complete, three options
  priced, loop's read stated.
- **D8 (BA.02 unmeasurable in the rover body)** — correctly the owner's; the
  proposed fixes are world-contract changes. The diagnosis is exemplary: it
  separates the task's headroom (blind "hands up" gains +0.275 s) from the
  claim's contrast (≈0.0–0.1 s, below the spec's own floor) and reports the
  apparatus arithmetic (k_fit ≈ 119 vs the registered 3) for the successor.

---

## 7. Bakeoff hygiene — no findings

`DECISIONS_RESOLVED.md` holds three entries, and the file's own header is the
right kind of honest (*"Until a real bakeoff runs, this file is EMPTY — and
that emptiness is the honest reading"*).

- **PS.01/J — VOID.** Three arms below the 3.0σ learning gate; no decision
  taken. Correct: the gate did its job and no verdict was extracted from it.
- **PS.01/J2 — WINNER, `impact_speed`.** Clears the null at 10.32σ and beats
  the runner-up by **2.66σ** against `bakeoff.py`'s declared
  `margin_sigma=1.5`. Inside the rule, not inside the noise. Losing arms
  recorded, eliminated arms named.
- **D2 — WINNER by ledger replay,** with its method justified in writing for
  not using `run_bakeoff` (the arms are two readings of one record, not two
  trainings).

No VOID was treated as a verdict. No winner was chosen inside its declared
margin. The `screen` gate mode on J2 carries a written rationale for why the
arms are observables rather than learners — exactly the disclosure that mode
needs.

---

## 8. The honest summary — are we closer to a creature, or to a longer list of ticks?

**Today: closer to a creature, but only in the way a refutation is progress.**

In 24 hours the ladder gained one PASS, and it was behaviour cloning on
scripted trajectories — apparatus. What it actually gained was three pieces of
knowledge it did not have yesterday, all negative, all expensive, all true:

1. **LC.03's ruler measures the world, not learning** — a twin with no
   persistent learner lengthened lives by 158 s. The learning-core match cannot
   start until the ruler is re-derived.
2. **The shipped world model loses to ridge regression** on its own metric
   (0.178–0.231 vs 0.114–0.131), and the ruler that made it look 5× better than
   persistence was leaking marginal statistics. The pre-registered rig gate
   caught what would otherwise have been a false PASS on a 0.90 h GPU spend.
3. **BA.02's claim has no headroom in this body** — measured, not argued.

That is the machine working exactly as designed, and it is worth more than
three green ticks would have been. The counterweight is that **this is what
the last five audits have all said**, and the numbers underneath have barely
moved: Tiers 3–5 hold **1 PASS across 64 specs**; 17 of 23 constitutional
commitments have never had a capability test pass; the two claims `GOAL.md` is
most *about* — curiosity, and all-senses-in-one-brain — rest on one spec each,
and one of those (UB.9) is doing double duty for hearing as well. The headline
"80/169" is a fair count of specs and a misleading count of Jack: **41 of the
80 are the harness and the primitives.**

The system is not fooling itself, which is the whole point of it. But the
honest reading of five audits of "ON TRACK" is that we are extremely good at
refusing false claims and not yet good at producing true ones. The next real
movement is not another Tier-2 apparatus PASS — it is LC.03's ruler, because
the learning core is the seat everything in Tiers 3–5 sits behind.

---

## FOR THE BUILDER

**B1 (RANK 1, cheap, do it first — it is the section-0 gate).**
`experiments/registry.py:524` declares `COVERS: world model (claim)` on T2.05.
`world model` is not a commitment; the declaration is malformed and buys
nothing. Two parts:

  (a) **Re-declare T2.05 against a real commitment, or drop the marker.** Do
      **not** add `world model` to `COMMITMENTS` — it is a mechanism, and
      `GOAL.md` does not owe it. The defensible readings are `fast/slow
      (fixture)` (axis 1's *"world-model arms that can imagine"* — the
      imagination path is apparatus a lookahead claim will need) or no marker
      at all, on the T2.04 precedent. Your call; say which in the commit, and
      note that T2.05 is VOID either way, so nothing about `n_pass` moves.

  (b) **Split the exit code.** `coverage.py:293` returns
      `len(uncovered) + len(bad)`. An uncovered constitutional commitment and a
      typo'd marker are not the same fire and must not ring the same bell —
      the whole reason this module exists is that the uncovered case is
      invisible to every other instrument. Suggested: return a distinct code
      (e.g. 2 for uncovered, 1 for malformed-only), print the uncovered banner
      first, and make the two counts separately assertable so T0.21 can gate on
      them independently. Your own comment at `coverage.py:117` states the
      rule this violates.

**B2 (RANK 2, mechanical, no re-run needed).** Correct the nine self-contradicting
provenance stamps. `protocol.py:1436-1443` already names them and already knows
where the truth is (`metrics["gpu"]`). Unlike `impl_sha`, this is derivable from
data already in the record, so LESSONS.md:3454's "cannot be back-filled" does
not apply. Do it as a **recorded provenance amendment, not a re-verdict** — the
status, metrics and seeds must not move; only the `hardware` string, stamped
with what corrected it. The nine: T0.09, T1.07, T1.08, T1.09, T1.10, T2.01,
T2.02, T2.03, T4.02. If you would rather not touch historical rows, the
acceptable alternative is a detector: `run status` flags any record where
`metrics["gpu"]` is set and `hardware` does not contain `remote/`. Absent
provenance is honest; contradictory provenance is not.

**B3 (RANK 3, guard, before LC.03's re-derivation is sized).** `duration_s`
means "how long the recording call took", and for harvested or reattached runs
it is not the cost of anything: LC.03 reads **0.02 s** for ~15 h × 3 seeds,
T2.04 reads **69.32 s** for a kernel billed 0.9354 h. Two ways, either is fine:
record a separate `compute_s` (from `res.charge_seconds` for remote work, or
the summed per-arm `core_s` the test already reports) and print **that** in
`run status`; or leave the field and make `run status` refuse to print
`duration_s` as a cost when `gpu_job_id` is set or when it is implausibly
small against the spec's budget class. This is the same species as B2 — a field
that names the recorder being read as if it named the work — and the reason it
is urgent is that **LC.03's rig re-derivation is next up and 0.02 s is the only
cost its record carries.**

**B4 (housekeeping, 16th audit's B3 narrowed and re-raised).** The 10:07
iteration on 08-14 spent 3 m 38 s confirming waiters were armed. Your CPU-
contention argument is accepted for CPU-bound waits. It does not hold for a
**Kaggle kernel**, which contends for nothing on this box: while one is in
flight, an iteration can implement the next spec (T2.06 is still not
implemented — `GPU_SHORT`, ~20 min, and it is the last honest GPU spend before
the quota dies). Carry the kernel's expected completion time in the hand-off
and pick a non-conflicting unit.

---

## FOR THE OWNER

**One line is owed, and this week it has a price tag.**

**D1 — the 57M trunk in the control path.** Open since 2026-08-09. The evidence
has been complete the whole time and nothing has changed it: the 57M trunk
reaches 261/318 return where a 54K-parameter MLP reaches 531 and a 125K net
reaches 530, failing a 3σ learning gate the 125K net clears at 7σ, across three
independent runs at matched env-steps.

**The question is not "what do the measurements say".** It is a constitutional
fork only you can pick, and answering it with *"do what the measurements say"*
would be read by a trigger written into this file as option A — freezing the
trunk — which your own PLASTIC-ONLY decree of 2026-08-09 (`GOAL.md:76`,
`eea7195`) bars. The fork, restated neutrally:

  **(i)** strike option A from D1's menu — the decree stands as written; or
  **(ii)** keep option A available and narrow the decree's scope, saying where.

**What it costs this week, concretely.** Kaggle W32 has **8.86 h remaining
(floor; range 8.86–15.24 h) and it expires Sunday 2026-08-16 — about 44 hours
from now.** D1's blocked work is T2.01's re-run under a decided architecture,
which has cost **5.58 h** each of the two times it has run. It fits in the
remaining quota exactly once, and it cannot be dispatched without your line.
The only other queued GPU spend is T2.06 at roughly 20 minutes. On present
course this week closes with **~8 h of free compute expiring unused**, for the
second consecutive week in which D1 has been the reason the largest available
experiment did not run.

**Also on your desk, both correctly escalated, both with complete evidence and
neither urgent this week:**

- **D7** — MovementMoodCoupling failed its ablation (T3.07: mood-conditioned
  action distributions score 0.225/0.275/0.375 against chance 0.25). Delete
  (1,539 params), redesign the mood→behaviour route, or accept it as cosmetics.
  Deleting a component is yours by SYSTEM.md.
- **D8** — BA.02 cannot be measured in the rover body: no actuator's useful
  effect depends on fall direction, and the claim's measured contrast ceiling
  (~0.0–0.1 s) sits below its own pre-registered floor (0.20 s). The loop
  recommends parking the spec until a body that can catch exists, because it is
  the only option that changes no certificate.
