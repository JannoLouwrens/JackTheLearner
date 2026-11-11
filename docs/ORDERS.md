# ORDERS — what the workers do. Written by the Director; ≤ 12 KB.

Workers: take the FIRST open order in your lane that you can advance. Update
only your order's `state:` lines (≤ 5). Machine problems go under INBOX, one
line each. The Director rewrites this file daily; git keeps the history.

*This first set was written 2026-11-11 by the acting Director (the owner's
session), from `docs/research/BOTTLENECK_AUDIT_2026-11.md`. Rulings cited here
are recorded in `docs/DECISIONS_RESOLVED.md` under the same date.*

---

## JACK — build the creature (max 7 open, ranked)

### J1 — Unbundle `T1.08`: it blocks 45 specs on a bar set from one sample
**Why.** `T1.08` passed twice in August as "seed noise is small next to the
effect" (SNR ≥ 3; it measures 10.1 today). On Sep 13 a second claim was added —
held-out seed CV ≤ 7% — and that bar came from a single early reading. Two
honest repairs later the CV is 17%: the Stack-B recipe really is that noisy, and
that stays a red, visible fact. But five of its six direct dependents (`T2.01`,
`T2.02`, `T2.03`, `T2.14`, `D1.0`) run 3 seeds with their own sigma gate — they
quote no single-seed number, which is the only thing the 7% clause protects.
**Ruling (conduct):** the 7% clause and `T1.08`'s FAIL are untouched. The
original, twice-passed claim becomes its own spec and the multi-seed dependents
hang off that instead.
**Do.** (1) Register `T1.14` "Seed noise is measured and published": the August
gate verbatim — seeds change the outcome (control: a seed-ignoring arm shows zero
spread), SNR ≥ 3, `min_detectable_effect` recorded. Reuse `T1.08`'s job; one GPU
run serves both. (2) In the registry, change `depends_on` from `T1.08` to `T1.14`
for exactly those five. `T2.13` (seeds = 1) keeps `T1.08`. (3) Run `T1.14`.
**done when:** `T1.14` has a verdict, `T1.08` still reads FAIL with 7.0
byte-unmoved, and `run blocked` no longer lists `T1.08` as blocking `T2.01`,
`D1.0` or anything in Tier 5/6.
state: steps 1+2 DONE at 4818a2c — `T1.14` registered (imports T1.08's job, snr>=3 + MDE>0
+ spread control; cv reported not gated); T2.01/T2.02/T2.03/T2.14/D1.0 re-parented, T2.13
keeps T1.08; t1_08 file untouched. `run blocked`: T1.08 gone, T1.14 NOT_RUN blocks 45.
Step 3 IN FLIGHT: dispatch watcher pid 1698161, kaggle `jack-ladder-1794429036`, log
/data/tmp/dispatch_t1_14.log (~1 GPU-h). Next: confirm ledger row, render, mark DONE.

### J2 — Settle who controls the body (`D1.1`), then re-run "he can move" (`T2.01`)
**Why.** `D1.0` spent 17.6 GPU-hours and measured it: small plastic head **506**
(paired t = 20.4 against its own untrained twin), MLP 415, the two 57M-trunk
arms 344–350, SB3 reference 518. It was VOID because untrained twins beat the
*random-action* null at 3.9σ — in Humanoid an untrained net stands still and
out-scores flailing. The null was wrong, not the data. `T2.01` has been FAIL
since Aug 12 running the losing arm.
**Do.** Register `D1.1`: same four arms, same budget, learning gate = trained vs
**its own untrained twin**, paired, ≥ 3σ (a harder null — a strengthening);
margin rule unchanged. Dispatch on Kaggle via `scripts/dispatch.sh` (the week's
30 free hours are idle). Seat the winner in `docs/CHAMPIONS.md`'s control seat
BY VERDICT, losers kept as challengers. Then run `T2.01` under the seated
controller, thresholds unmoved.
**done when:** `D1.1` has a verdict, the control seat names its holder by
verdict, and `T2.01` has an attempt dated after it. Needs J1 step 2.
state: open — blocked on J1.

### J3 — One whole life with the learner in the loop (`LF.03`) — the creature gate
**Why.** Nothing has ever run end to end. The Tier-6 gates (`T6.01` "companion
session") are written against Stack B, the 57M brain that keeps losing; the
learning was shown on the survival stack (`cores.py` / `survival.py` / `w0.py`).
`LF.01`/`LF.02` already compose world + needs + diary + working memory around a
*scripted* forager. Put the learner where the script is.
**Do.** Design and register `LF.03`: the seated core (`wm-latent`) driving the
body in lethal W0 across ≥ 5 deaths — needs, every W0 sense live, diary written
and carried across death, save-and-resume once mid-life (`LF.02`'s mechanism),
no crash or NaN, weights measurably changing. Controls that must fail: a frozen
twin (weights do not change → flagged), a wiped diary (nothing crosses death →
flagged). It is an integration claim, not a performance claim — do not gate on
how well he does. Fix `LF.01` (VOID) on the way if it is in the path.
**done when:** `LF.03` is registered with its controls and has a verdict. From
then `LF.03`, not `T6.01`, is what "one whole life" means (D35 amended to match).
state: open — not started. CPU only; pilot one life first to size it.

### J4 — Run the one-brain contest (`UB.10`) on the idle GPU
**Why.** The fusion bakeoff — six arms including A5, experts + router, "if A5
wins, one brain is the wrong shape" — has sat VOID for 59 days. It is `gpu<2h`
and frees 4 specs. This is the owner's rule that architecture is raced.
**Do.** Read the VOID's reason in the ledger row; repair the apparatus (not the
bar); pilot; dispatch.
**done when:** `UB.10` has a WINNER/TIE verdict or a VOID with a different,
recorded cause and the next repair named.
state: open — not started.

### J5 — `XL.01` "death does not erase what he learned": give it the power to see
**Why.** Pooled, memory helps (food in 19.9 s with the diary vs 27.3 s wiped,
ratio 0.73). It fails because 1 seed of 3 clears at n_lives = 8. That is a
sample too small for the claim, not a refuted claim.
**Do.** Size n_lives / seeds from the measured spread so the pre-registered
effect is detectable (pilot, declare, then run). Threshold unmoved. ~12 min CPU
per attempt.
**done when:** a run at the declared power has a verdict, whichever way.
state: open — not started.

### J6 — `NE.01` fails on 2 of ~12 clauses while its claim holds; repair those two
**Why.** The scripted forager survives 18,000 decisions and random dies — the
needs ARE a real control problem. It is red on `ok_night_shelter` and
`ok_spread` and has been untouched 22 days. It frees 7 specs (the whole needs
family, incl. "life N+1 is faster because he remembers").
**Do.** Find out why each clause fails — world calibration (shelter thermal
balance, need-spread across deaths) or a clause that cannot fire. Fix the world
or the apparatus; never the bar.
**done when:** `NE.01` has a new attempt with both clauses explained in the
commit.
state: open — not started.

### J7 — Finish the deeper world (W1): `W1.00` FAIL, `W1.04` VOID
**Why.** Nine instruments agree W0 is too shallow: lookahead does not pay in it
(`DP.05`), a ridge regression beats the world model (`T2.05`). Fast/slow
(`BO.01`) cannot be raced until a world exists where thinking ahead is worth
something. `W1.01`, `W1.02` already pass.
**Do.** Repair `W1.00` and `W1.04`; then re-run `DP.05`'s question in W1.
**done when:** both have new verdicts and `DP.05`-in-W1 is registered or run.
state: open — not started.

**Next up (not open yet):** `ME.11.B–F` memory-retrieval bakeoff (lexical recall
on paraphrase is 0.0); `LC.07` at scale through save/resume on Kaggle CPU;
`HR.1` hearing corpus.

---

## SYSTEM — work on the machine (max 3 open)

### S1 — Measure the reorganisation after 48 hours
Write the four numbers into `docs/PROGRESS.md` and here: weekly-meter points per
worker run (was 0.66 on the old 116 KB prompt), slots lost to the budget, free
GPU-hours expiring this week, age of the oldest unrepaired blocker. If points
per run did not fall, say why.
**done when:** the four numbers are on the page with their commands.
state: open — earliest 2026-11-13.

### S2 — Make the megabyte files small without breaking their readers
`docs/LESSONS.md` 1.4 MB, `docs/LOOP_JOURNAL.md` 2.4 MB,
`docs/DECISIONS_NEEDED.md` 750 KB. Tools and specs reference these PATHS (64, 16
and many files) — check each real reader (`grep -rn`) before moving a byte. Move
history to `docs/archive/` and keep each live file ≤ 150 KB at its old path.
Resolved decisions leave `DECISIONS_NEEDED.md` entirely.
**done when:** the three live files are under the cap and `coverage`,
`decisions --check`, `champions --check` and `run status` still exit as before.
state: open — not started.

### S3 — `T0.13` says 13 gate clauses in passing specs can never fire
`LT.01` (3), `SO.02` (5), `T0.17` (3), `T0.33` (2). A passing spec with a clause
that cannot fire is a certificate for something untested. Arm each clause or
remove it with the reason, re-run the spec, and let `T0.13` re-judge.
**done when:** `T0.13`'s `disarmed_conjunct_keys` is 0 or each survivor is
explained in its spec.
state: open — not started.

---

## INBOX — one line each, workers add, the Director empties

- scripts/test_lib_usage.sh has 1 pre-existing failure: 'the skip line prints both meters' assumes the worker model has its own weekly meter (true for fable, not opus). Make the case set its model explicitly.

## DONE (kept one day)

(none yet)
