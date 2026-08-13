# OVERSIGHT — 13th audit, 2026-08-13 07:00 UTC

## VERDICT: ON TRACK

The 12th audit's RANK 1 was fixed within **73 minutes** of being written, and
fixed properly: all 86 `COVERS:` pairs kinded, kindless promoted from silent
default to error, honest zero-pass coverage restated 9 → 19 (`a703604`). In the
six hours since, **three constitutional commitments gained their first
claim-kind pass** — curiosity (T2.08), damage/nociception (PS.03), sight
(T2.03) — taking zero-pass commitments 19 → 16 and the ladder 75 → 78.

§1 is clean: all **78 PASS** records name a commit that exists and is
reachable, **none is dirty**, and every PASS whose spec declares a control
recorded control metrics. `run stale` reads zero. §5 reconciles: the 6.38 h gap
between the weekly meter and the per-job register is **fully explained** and I
found no unbilled Kaggle time.

The finding that matters is not a false number. It is that **T2.08's absolute
floor was moved in the loosening direction after the run it failed, and the
repo contains no artifact that can corroborate the account of that move.** The
move was disclosed as loudly as it is possible to disclose something. But the
ledger structurally cannot check it — and that is what makes the next one
cheaper.

---

## 0. Is the ladder the RIGHT ladder?

`python -m experiments.coverage` → **exit 0. Zero commitments with no declared
spec.** The 2026-08-10 miss has not recurred.

**166 specs · 78 PASS · 1 FAIL · 1 VOID.**
T0 27/27 · T1 13/13 · **T2 36/59 · T3 0/14 · T4 1/23 · T5 0/25** · T6 1/5.

Zero-pass commitments **16 of 23** (was 19 at 01:18 today, 9 before the kind
migration corrected the flattery). The three that moved are exactly the three
the standing zero-pass rule pointed at. The rule is working.

*Note: T2.03's ledger record was uncommitted in the working tree at audit time
(the loop is mid-iteration). I verified it against the kernel receipt and count
it as PASS; the builder is expected to commit it.*

---

## RANK 1 — T2.08's floor moved 0.70 → 0.50 after the run it failed, and nothing in the repo can corroborate the account

**Nothing on the ledger is false. The damage is to auditability, and it is
structural.**

### What happened

| | |
|---|---|
| 02:34 | T2.08 attempt 1 recorded **FAIL** at `75a1938+dirty` |
| | Failing quantity: `state_coverage` **0.6975** vs v1's floor **0.70** — a miss of 0.0025 |
| ~02:40 | v2 committed at `1454525`: `COV_MIN` **0.70 → 0.50**, plus a new 3σ gate |
| 02:42 | T2.08 attempt 2 recorded **PASS** |

`experiments/tests/t2_08_curiosity_coverage.py:120` — `COV_MIN = 0.50  # v2
anti-collapse floor; v1's 0.70 was a...`

### What is right about it, stated fairly

This is not silent loosening, and I will not call it that:

- The registry-level falsification criterion — *"Coverage at or below random"* —
  was **decisively rejected**: margin 0.0544 over both nulls, every seed
  positive, paired t = 5.0, shuffled control losing by 0.035. The floor was an
  **implementation-added auxiliary gate**, not a registered one.
- The move is declared in the commit message, the module docstring, the loop
  journal (`docs/LOOP_JOURNAL.md:3757-3763`) and a new LESSONS entry.
- The spec was **strengthened in exchange**: v2 added `MARGIN_TSTAT_MIN = 3.0`,
  a paired 3σ significance gate v1 had none of. Attempt 1 measures 5.0 on it.
- The diagnosis is technically correct, and I can confirm the arithmetic the
  docstring only asserts. v1 called 0.70 *"~6 sigma below the pilot bulk"*. Against
  the pilot's **seed** std (bored 0.772 ± 0.023) that is **3.1σ**, not 6. It is
  ~6σ only against the standard error over 4 lives (0.023/√4 = 0.0115). **v1
  sized a per-seed gate with a standard error instead of a standard
  deviation** — which is precisely why a uniform seed-family shift landed it
  mid-bulk. The builder's conclusion is sound; the number it cites is sized on
  the wrong scale, and that error is the whole story.

### What is wrong about it

**Every artifact corroborating the move was written by the same agent in the
same iteration, and the one artifact that could settle it independently does
not exist.**

- v1's code was **never committed** — the FAIL is stamped `75a1938+dirty`, and
  `t2_08_curiosity_coverage.py` appears in git history **only at `1454525`
  (v2)**. The failing implementation is unrecoverable.
- The FAIL's ledger history entry carries **`{commit, message, ran_at,
  status}`** and nothing else. The number that failed — 0.6975 — is **not in
  the ledger**. It survives only in prose written by the party that moved the
  threshold.

So the repo's answer to *"was the move honest?"* is *"read the commit message of
the commit that made the move."* That is not an audit trail; it is a
self-certification. Today it happens to be true. It is the second use of law 4's
escape clause (T1.02 was the first), and a precedent used twice is a habit
forming.

---

## RANK 2 — the ledger's history is metric-free and code-free, in all 163 entries

RANK 1 generalises. I checked every history entry in `experiments/ledger.json`:

```
history entries total                                     163
carrying metrics                                            0
carrying control_metrics                                    0
carrying impl_sha                                           0
keys present, universally:  commit, message, ran_at, status
```

**Zero of 163.** Every superseded run in this project's life is a status string
and a sentence. That includes every amend-after-FAIL the ladder has ever done:
VO.01 (5 FAILs), XL.00 (3), LC.02 (3), T0.13 (3), PS.01 (2), PS.02 (2), T0.14,
T0.17, T0.21, T0.23, T0.25, T2.01, BA.01 (FAIL + VOID), T2.08.

The current record answers *"did it fail?"* It cannot answer *"by how much, on
what code, against which control?"* — and those are the only questions that
distinguish a justified threshold move from a quiet one. `impl_sha` already
exists on live records; history is where it is missing and where it matters
most.

**33 of 163 history entries are stamped `+dirty`** — code that was never in
git — compounding the same gap.

---

## RANK 3 — ~0.4 T4-hours were consumed and billed nowhere, and T0.12's receipt-pairing property never looks at the live receipt file

`experiments/gpu_submissions.jsonl` has **11 attempts and 10 results**. The
unpaired one:

```
attempt_id 1786594878451-2409873-colab   iso 2026-08-13T04:21:18
est_hours 0.4   timeout_s 2940   head 7783535
```

This is the T2.03 pilot whose watcher was killed by the ladder's 50-minute
timeout at ~46 min. It is charged **nowhere**: no `charged_jobs` key contains
`1786594878`, and both Colab columns reconcile **exactly** without it —
`colab` 0.9914+0.0616 = 1.053 = `colab_failed` to four places, `colab` ok
0.5498+0.2103 = 0.7601 vs 0.7616. So this is not rounding. Real GPU time was
consumed and the meter has no line for it.

**The cause is structural: charging happens only on the result path.** A job
whose watcher dies is free. `1a01e69` prevents the *loss of the result*; it does
not make the *bill* appear, and it exempts Kaggle — where the quota actually
binds — on the reasoning that reattach recovers it. A Kaggle kernel that runs
with no reattach ever attempted burns quota server-side and is billed zero.

**And the gate that should have caught it passed.** T0.12 has property
`receipt_pairs_attempt_with_result`
(`experiments/tests/t0_12_gpu_budget.py:357-363, 402`) — but it asserts pairing
over a **synthetic 2-attempt/2-result fixture**, never over
`gpu_submissions.jsonl`. T0.12 re-ran **PASS, 33/33, at 06:40 today** — two
hours after the unpaired attempt appeared in the live file it certifies.

Minor, same file: T0.12's probe writes `{"phase": "selftest"}` into the
**production** receipt file. A test should not mutate the artifact it audits.

---

## RANK 4 — 11.47 expiring Kaggle hours, and T2.01 is mechanically unblocked for the first time this week

`gpu_budget.json` W32 (%U week, Aug 9–15, resets **Sunday 2026-08-16**):
kaggle **18.5322 / 30 used**, **11.47 h remaining**, kaggle_failed 0.1225.

Today's `643f542`/`c6f2f91` found and fixed something bigger than the pilot they
were chasing: `KAGGLE_TORCH_FIX` had been **silently broken upstream** (the
index dropped `nvidia-cudnn-cu12==9.1.0.70`; `check=False` swallowed the
resolution failure), which had been **killing every torch-on-P100 job all
week — including the planned T2.01 8-hour re-run**. The repair was then
**verified live on the real P100** (`TORCH_PIN 2.5.1+cu121` printed) plus a
matched `torchvision==0.20.1` pin.

So: **T2.01 — the ladder's only FAIL, gating the locomotion branch and owner
decision D1 — has been impossible to re-run all week for mechanical reasons,
and became possible today.** Its 8 h fits inside the 11.47 h that evaporate
Sunday. This is the highest-value use of the expiring budget and there are
~3 days left.

---

## RANK 5 — observation, not a fault: sight's first passing claim credits a frozen borrowed trunk

T2.03 PASS is honest and well-built: margin_min 0.46 vs floor 0.10, null_gap
0.5733 vs 0.15, shuffled control at chance (dev_max 0.0633 vs band 0.10), canary
and param-range VOID guards live, control gated in `_check`. Thresholds were
**finalised unchanged before** the registered run (`42df762`), correctly
anchored to purpose rather than to pilot bulk — the T2.08 lesson applied one
iteration after it was learned.

But `coverage.py` now reads **sight: 1 pass**, and the winner is frozen
DINOv2-L + CLIP-L, which the PLASTIC-ONLY decree bars from ever being inside
Jack. The builder flagged this itself, twice, in the registry note and the
docstring: *"a frozen winner cannot take the vision seat."* The spec measures
**the gap the plastic path must close** — it does not demonstrate that Jack
sees. Nothing is misreported; the concern is that the coverage map is the
builder's work-selection input, and there "sight: started" now reads greener
than the goal's actual state.

---

## 2. Thresholds and controls, over time

`git log -p --since="7 days ago"` over `registry.py`, `registry_expansion.py`,
`experiments/tests/` — 112 commits, +22,210/−141 lines. Reviewed every deleted
line and every numeric constant change.

- **One threshold moved in the loosening direction: T2.08's `COV_MIN`
  0.70 → 0.50.** RANK 1. Disclosed, justified, strengthened in exchange,
  uncorroborable.
- **No control was deleted or weakened. No `_check` gained an `or`. No seed
  count was reduced. No assertion was removed** without a stated measurement.
- **Two controls were ADDED before first run** (strengthen-only): T2.03's
  shuffled-label control, PS.03's global-fear control.
- **BA.01 v1→v4** moved through FAIL → PASS → VOID → PASS with **every gate
  byte-identical from v1 onward** — four rig rewrites, zero threshold moves,
  including a VOID that took back its own prior PASS. That is the standard
  T2.08 should have met.
- **The kind migration (`a703604`) is honest and conservative.** I diffed all 86
  `COVERS:` pairs: every change either kept the `claim` default or **downgraded**
  to fixture/sensor/rule. **No declaration was upgraded toward `claim`.** A
  labelling commit that makes your own scoreboard worse by 10 commitments is
  the good kind.

---

## 3. Drift from the goal

**No drift.** Everything the builder touched in the last 24 h traces to a
GOAL.md sentence:

| Work | GOAL.md sentence |
|---|---|
| T2.08 curiosity coverage | *"He explores because he wants to"* — the north star's first claim |
| PS.03 damage/nociception | *"pain as a fast signal distinct from reward"*; *"he lives, he dies"* |
| T2.03 pretrained vision | *"sight"*; *"components must EARN their parameters"* |
| T0.12/T0.26, deadline guard, torch fixes | *"protects the honesty of watching what happens"* |

**The converse is the real answer.** Tier 3 (*"earn your parameters"*) is
**0/14**. Tier 5 (*"the claims — the thesis itself"*) is **0/25**. **16 of 23
constitutional commitments have zero passing claim specs**, including
shelter/building, tool use, smell, taste, voice, sleep, social/other agents,
plasticity, thermal, hunger/thirst, balance, proprioception, and fast/slow.

Two commitments — **balance** and **hunger/thirst** — have **no claim-kind spec
declared at all**, so `n_pass` for them is structurally frozen at zero no matter
what runs. The kind migration surfaced this and the builder named it; no claim
spec has been queued for either yet.

---

## 4. Is the builder alive and productive?

**Yes, and this was a strong day.**

```
iterations, last 24 h            24
ending rc=0                      24   (100%)
PASS delta                       75 -> 78
zero-pass commitments            19 -> 16
```

No paused loop, no credit exhaustion, no load aborts (load 0.00–0.28
throughout), no repeated identical failures. Iterations are ~5–35 min inside
hourly slots.

One iteration recorded a **negative** delta (19:22, 74 → 73): BA.01 v3 VOIDed
and took back v2's own PASS. **A ladder whose count can go down is a ladder
that is measuring.** That is a health sign, not a fault.

---

## 5. Compute honesty

**Reconciles, with one gap (RANK 3).** I checked the weekly meter against the
per-job register line by line:

| bucket | `weeks[2026-W32]` | Σ `charged_jobs` | delta |
|---|---|---|---|
| kaggle (ok) | 18.5322 | 12.1473 | **+6.3849** |
| kaggle_failed | 0.1225 | 0.1225 | 0.0000 |
| colab (ok) | 0.7616 | 0.7601 | +0.0015 |
| colab_failed | 1.0530 | 1.0530 | 0.0000 |

**The 6.3849 h is explained and is not a fault.** It has been **constant across
every commit** since `92931a6` (2026-08-09 11:04), which is the commit that
introduced `charged_jobs` with an empty register. W32 already held 6.3849 h of
Kaggle spend recorded under the old weeks-only scheme. It is real spend, in the
correct week, counted against the quota — conservative in the safe direction.

Two things I checked and found **correct**, worth stating because both were
prior audit findings:

- **The `%U` week key is right.** I initially read `2026-W32` as a stale ISO key
  (ISO today is W33) and it is not: `_week()` deliberately uses `%U`
  (Sunday-start) to match Kaggle's actual reset, and `%U`-W32 is Aug 9–15.
  Today's charge landed in the correct bucket.
- **The reattach meter fix (10th audit, T0.12 P9) is verified in production
  data, not just in its test.** The 08-12 06:56 reattach returned a result with
  `duration_s` 35,331 s (**9.814 h** wall-clock) against job
  `jack-ladder-1786482462` — and the budget billed it **0.6561 h**, the
  kernel's own window, with no double-charge on the colliding key. Exactly the
  designed behaviour, on a real 15× overbill it would have taken.

Waste this week: 0.1225 h kaggle_failed + 1.053 h colab_failed, every line
honestly marked `ok: false`. Two billed-failed kernels today bought two real
defects (the numpy seed overflow and the upstream cudnn break) — that is waste
that paid for itself, and it was billed anyway.

---

## 6. Stuck decisions

**D1 — "Does the 57M trunk stay in the control path?" remains OPEN with
evidence complete, and it is now the oldest and most expensive blockage.** It
gates T2.01 (the ladder's only FAIL), T2.02 (VOID), and their dependents. The
11th and 12th audits both escalated it; the 12th appended a cost correction at
00:45 today.

**New evidence for the owner, from today (see RANK 4):** the delay has been
partly *mechanical*, not decisional — every torch-on-P100 job was silently
broken all week, so the T2.01 re-run could not have happened even had D1 been
answered. That is now fixed and verified live. The option set is still flagged
stale (option A contradicts PLASTIC-ONLY, raised 2026-08-10 and unanswered).

**Nothing was quietly acted on.** D2 was resolved by the system, properly, and
recorded (§7). No open owner decision has been actioned without a record.

Other open owner items, unchanged and correctly parked: `/data` at 95%, the
owner's-hands design fork, LC-bakeoff-at-scale, physics-first retirement, D4.

---

## 7. Bakeoff hygiene

**Clean — and D2 is a model of how this should go.**

D2 ("does a VOID dependency block its dependents?") was resolved 2026-08-13 by
**replaying the ledger's own recorded history** rather than by argument. It
states its metric in advance (retraction exposure), measures both arms
(BLOCK: 0 dependents exposed; NO-BLOCK: **11 admitted, 9 resting on a T2.01
that FAILed 17 minutes later**), measures the benefit side of the loser
(**exactly 3 specs, none implemented — benefit zero**), **records the loser and
what was right in it**, names a **re-open trigger** tied to the quantity the
verdict rests on, and makes the invariant executable (T0.08 property 6).

It also correctly declined to use `run_bakeoff`, with the reason stated: the
arms are two readings of a dependency graph, not learners — no seeds, no null,
no learning gate to apply. **A VOID was not treated as a verdict; a winner was
not chosen inside a noise margin; no decision was made without a gate.**

---

## 8. The honest summary — are we closer to a curious humanoid, or to a longer list of green ticks?

**Closer to the humanoid, genuinely, and for the first time in several audits I
can point at the specific reason.**

The green ticks got *fewer* today before they got more: the kind migration
deleted 10 commitments' worth of unearned credit at 01:18, and the builder
committed that correction against its own scoreboard within 73 minutes of being
told. Then it spent the next five hours putting real passes where the honest
gaps were — curiosity, damage, sight — and one of those (T2.08) is the
north-star commitment's first claim of any kind.

T2.08's pilot is worth more than its PASS, and this is the part that reads like
science rather than ticking: **every positive-reward curiosity construction
anti-explores** in bootstrapped tabular Q — naive ICM 0.283, normed ICM 0.194,
RND 0.289, additive count 0.327, all far below random's 0.829 — because the
visited core's accumulated value beats one-shot frontier bonuses. What explores
is valuing the frontier *below* the familiar: the boredom form `1/√N − 0.5`.
That is a negative result about four mechanisms the field defaults to,
measured, in Jack's world, and it is real context for T2.09 and the LT bakeoff.
Nobody asked for it and it could not have been guessed.

Against that: **Tier 3 is 0/14 and Tier 5 is 0/25.** Six days in and 47% green,
the ladder has not yet made a single claim of the kind the thesis is *about* —
no component has been ablated and deleted, no continual-learning claim, no
plasticity claim, no curiosity-drives-real-exploration claim in a body. What
passes is Tier 0/1/2: the harness works, the primitives learn, components beat
nulls. That is the right order and it is not drift — but it means the honest
statement is *"the instruments are excellent and the thesis is untested"*, and
that has been true for several audits running.

And the thing to watch is in RANK 1. This system's real asset is that its
numbers can be checked. Today a threshold moved after the run it failed, and I
could not check it against anything except the prose of the agent that moved
it. It happened to be honest — the journal corroborates, the arithmetic works
out, the spec got stricter in exchange. But "it happened to be honest" is a
property of this iteration, not of the ledger. Fixing RANK 2 makes it a
property of the ledger.

---

# FOR THE BUILDER

**B1 (RANK 2, do this first — it is small and it is what makes B2 checkable).**
Ledger history entries must carry the evidence, not just the verdict. When a
record is superseded, copy `metrics`, `control_metrics`, `impl_sha` and `seeds`
into the history entry alongside the existing `{commit, message, ran_at,
status}`. Cost is a few lines in the recorder. Without it, no amend-after-FAIL
in this project — past or future — can be audited by anyone but its author.
Do **not** back-fill the 163 existing entries with invented numbers; leave them
empty and let them be the historical gap they are.

**B2 (RANK 1).** Make law 4's escape clause leave an artifact instead of a
paragraph. When a threshold is moved after a run that failed it:
1. **Commit the failing implementation before re-running.** T2.08's v1 was
   stamped `75a1938+dirty` and its code exists nowhere; a `+dirty` FAIL that is
   then amended is unauditable by construction. The FAIL should be stamped at a
   real commit, exactly as BA.01 v1/v3 were.
2. Record the old and new values and the failing measurement **in the ledger
   record** (B1 gives you the slot), not only in the docstring.
3. Consider a T0 spec asserting the pattern: *a spec whose `_check` constants
   changed between a FAIL and the next PASS must have its FAIL stamped at a
   reachable commit carrying metrics.* This is the executable form of the rule
   you are currently keeping by hand.

**B3 (RANK 3).** Two defects in GPU accounting, one real, one cosmetic:
- `receipt_pairs_attempt_with_result` (`t0_12_gpu_budget.py:357-363`) checks a
  synthetic 2/2 fixture and never reads `gpu_submissions.jsonl`. Add a property
  that reads the **live** receipt file and requires every `attempt` older than
  its own `timeout_s` to have a `result` line — or an explicit
  abandonment/`billed-unknown` line. It fires today on
  `1786594878451-2409873-colab`.
- Charging happens only on the result path, so a dead watcher makes a job free
  (~0.4 T4-h missing right now). Write the estimated charge at **attempt** time
  and reconcile it down on result, rather than writing nothing until a result
  arrives. Note the Kaggle case is the sharp one: quota burns server-side
  whether or not anyone reattaches.
- T0.12's probe writes `{"phase": "selftest"}` into the production
  `gpu_submissions.jsonl`. Point it at a temp file.

**B4 (RANK 4, time-boxed — 3 days).** 11.47 Kaggle hours expire Sunday
2026-08-16. T2.01's 8-hour re-run now fits and is mechanically possible for the
first time this week (your own `c6f2f91` verified `TORCH_PIN 2.5.1+cu121` live
on the P100). T2.01 is the ladder's only FAIL and it gates the locomotion
branch and D1. If the D1 answer is what blocks the *decision*, the *measurement*
is not blocked — a clean re-run at the repaired torch path is evidence D1 needs
either way. Spend the hours or lose them.

**B5 (§3).** **balance** and **hunger/thirst** have no claim-kind spec declared
at all — `n_pass` for them cannot move no matter what runs. You identified this
in `a703604`'s own commit message and did not queue anything. Register one claim
spec for each. (BA.01 and PS.01 are correctly kinded sensor/fixture; they are
the apparatus those claims will use.)

**B6 (RANK 5, judgement call — yours, not mine).** T2.03 is correctly built and
correctly caveated, but `coverage.py` now credits **sight** with a passing claim
whose winner is constitutionally barred from being inside Jack. Either surface
the `notes` caveat in coverage output for specs whose winning arm is frozen, or
reconsider whether "measures the gap the plastic path must close" is a `claim`
or a `fixture`. I am not asserting the label is wrong — I am asserting a reader
of the coverage table cannot currently tell.

**B7 (minor).** The `hardware` field reads
`aarch64/Linux/torch2.8.0+cpu/cpu` on **every** GPU-run record — T2.03, T1.09,
T1.10, T2.01, T0.09, T1.08 — because it captures the ARM orchestrator, not the
executing device. The truth is in `metrics.gpu`/`metrics.backend`, so nothing is
misreported, but the field a reader would trust for *"where did this run"* says
"cpu" for six GPU results. Record the executing device there.

---

# FOR THE OWNER

**1. D1 is still on your desk, and today changed its cost structure.**
Does the 57M trunk stay in the control path? Evidence has been complete for
days: three independent runs at matched env-steps have the 57M trunk at 261–318
return and below its own 3σ learning gate, while a 54K MLP reaches 531 and a
125K net reaches 530 at 7.11σ.

What is new: the delay has been partly **mechanical**, not decisional. An
upstream package removal silently broke every torch-on-P100 job all week, so
T2.01's re-run could not have happened even if you had answered. That is fixed
and verified on real hardware today, and 11.47 Kaggle hours expire Sunday.

Note also that your own PLASTIC-ONLY decree (2026-08-09) **post-dates** D1's
option set, and option A ("freeze the trunk") now contradicts it. That was
raised on 2026-08-10 and has not been answered. **D1 cannot be decided as
written** — the recommended option is barred by a later decree. If you want to
unblock this with one sentence, the useful one is whether PLASTIC-ONLY admits a
*small dedicated plastic policy head reading a plastic trunk's features* as
distinct from freezing the trunk.

**2. Nothing in the ledger is false, and one thing in it cannot be checked.**
You do not need to act on this — B1/B2 are the fix and the builder can do them.
I raise it because it is the only class of problem that would let this project
quietly stop being trustworthy: a threshold was moved after the run it failed,
the move was honest and thoroughly disclosed, and the repo contains no artifact
that could have told me so if it had not been. Today the disclosure was the
safeguard. Disclosure is a property of the agent; the ledger should not depend
on it.

**3. The instruments are excellent; the thesis is still untested.**
Tier 3 (earn your parameters) is 0/14. Tier 5 (the claims themselves) is 0/25.
16 of 23 of your constitutional commitments have no passing claim spec. This is
the correct build order and the loop is moving fast and honestly along it — 24
iterations, all clean, +3 commitments in six hours. But if you are asking *"is
Jack closer to climbing a ladder out of curiosity"*, the honest answer is that
today he got his first measured claim that curiosity beats random at all, in a
2D rover, and that the four curiosity mechanisms the field would have reached
for first all measurably **anti**-explore. That is real and it is a beginning.
It is not yet the humanoid on the ladder.
