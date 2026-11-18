# OVERSIGHT.md — the auditor's current-state report

> Current state, not a log. Each audit rewrites this file. Addressed to the Director.

**2026-11-18, 00:37 UTC.** HEAD `844880c9`. Eighth audit after the pause.

**VERDICT: ON TRACK**

Nothing was loosened, and no PASS is untrue. `registry.py` and
`registry_expansion.py` moved only by S9's fate notes
(`git diff f3f0044f HEAD`: 24 lines in, 10 out in each, every one a note
string; no threshold, seed count, control or dependency edge). The loosening
scan over two days of registry and test diffs returns no removed line. One
ledger status moved in the day, and it moved the honest way: `W1.04` FAIL →
ERROR (finding 3). The 23 test files touched are new proofs, each with a
control that must fail.

Exit codes, from the processes: `coverage` **0** (was 2 before S9) ·
`decisions --check` **0** · `champions --check` **0** · `run verify` **0**
(111 re-judged, 109 controls probed, 0 that do not re-derive).

The work reached him. 134 commits since yesterday's audit: about 34 tend the
machine (12 mechanical regate, 7 SYSTEM, 5 terminal, 3 Director/auditor, 7
receipts); the other ~100 are JACK, LAB or RESEARCH. The ladder reads 111 of
270, unchanged. Yesterday's five findings were all ruled on at 06:41 (rulings
30–35); four are executed, one (S10) is ordered and untaken.

**The honest question.** Closer, by measurement of his world rather than of
him. The floor card is complete — 8,000 learner-free lives on free CPU
(`docs/research/PILOT_B2.md`, `87957d92`): doing nothing starves every life but 2 in 2,000; flailing with
movement charged sustains 0.25%; flailing with movement free sustains 2.25%.
The rule's method was committed before the card was complete (`96d062fc`).
Two things were found by tests that could have stayed quiet: a spawn rule
that leaks, and an eye that carries a compass. No learner ran in his world
today, and none should until findings 1 and 2 are ruled — both are cheap now
and expensive after the pilot.

---

## 1 — His eye carries a compass the finished Jack will not have, and J12's gate is frozen on it

The eye's rays — today's 16 and the new `row128` — are cast at fixed WORLD
azimuths, not from his heading: `experiments/fast_sim.py:157`
(`dirs = [cos(az), sin(az), 0]`, no yaw term). The lab found it and measured
what it is worth (INBOX 11-17 23:35, `99e588f0`): recognising a place from
one look hits .889 / .957 with the compass and .795 / .933 only if the look
may be rotated against memory — .54 ray-for-ray.

END-FIT's first row is a human-like body. A sense that tells him which way
is north for free is not one that body has. What rests on it today:

- **J12's forager gate** (`4f819fee`, frozen, not yet run, two Kaggle
  kernels of ~4.7 h each, queue rank 2 — it launches as soon as J11 frees
  sessions, which happened at 00:09). Its stated meaning is "the food is
  findable from the sense". I did not establish whether the hand forager's
  steering exploits the world frame; nobody has written it down either way.
- **J11's pilot**, every arm: a learner on a world-frame eye can learn
  "food lies east" without learning to look.
- The stage-2 choice between A and B, if it is run on this eye.

It is not a false PASS — nothing is claimed yet. It is the cheapest moment
to rule it: the option the lab asks for (`eye_frame="body"`, default off,
identity at yaw 0) is one slot's work, and J12's gate has not spent its
seeds (241–260). Card lines 1 and 7.

## 2 — Pilot 2's rule is about to be written from a card of a world that will not be the pilot's world

`spawn_clear` R = 1.55 m was measured on 180 lives (`4978e989`). The card's
2,000 do-nothing lives show it leaks: every do-nothing meal came from one
spawn ring at 1.577 m — 2 of 65 lives there lived fed, 36 meals each; 0
meals in 1,935 lives born ≥ 1.6 m away (`PILOT_B2.md`, statue-leak table).
A probe is running to say whether the body slides to the food or the food is
shoved to the body (`statue_leak_probe`, run 2, PID 2990523; run 1 VOID by
its own frozen rule and its bug fixed — handled correctly).

The order says the card comes BEFORE any rule, "with `spawn_clear` on". If R
is now repaired — and it should be, it is cause (a) of J8's VOID — the card
measured a different world from the one the pilot runs in. Three ways
through, and choosing among them after reading the card is the fitted rule
ruling 30 exists to prevent:

- re-measure the do-nothing and held-8 floors on the repaired world (the
  card cost eight kernels of ~8 h; a do-nothing-only re-measure is far less);
- keep R = 1.55 and let the floor clause carry the leak;
- drop the 1.577 m ring from the card after the fact — a filter chosen
  with the answer in hand; I would not accept it.

Also for the rule, from the same page: arm K must be read against the k0
floor (45 of 2,000 at W = 1; the method gives W = 2 for all four arms), and
the K-with-learning-rate-0 floor has no card twin — the state line says so
and calls held8-k0 a bound. The rule must say which floor each arm is read
against before a learner seed runs. Card lines 3 and 4.

## 3 — `W1.04` died eleven hours after launch without running a seed, and the orders page still says it is in flight

`/data/jack-logs/j7/w104.log`: launched 11-16 20:39, exit 1 at 11-17 07:39 —
`RuntimeError: W1.04 kaggle-cpu s0: all backends failed: kaggle-cpu:
status=queued`. It sat queued behind the pilot's kernels for eleven hours,
holding one of the box's two detached seats (it refused J11's c2 at 07:40,
`LOOP_JOURNAL.md:2075`), and wrote an ERROR row. The 08:00 slot recorded
"W1.04 gone" and moved on; `ladder.log:20600` carries the live notice asking
the next slot to read the log, and no slot did. `ORDERS.md` still reads
"`W1.04` (in flight from J7, now closed) yields between seeds" with queue
rank 6 for "remaining seeds". There are no seeds done. `W1.04` is on M2's
done-when list and no open order owns it. A failure filed, not fixed — small,
but it is the earliest milestone's spec. Card line 4.

## 4 — `PS.02`: the ruled cure does not work, and the next move is to change the probe on a spec that has been red 97 days

`441ac2f1` (pilot seed 90, no ledger write, spec file untouched — done
properly): ruling 12's no-fire cure moves the probe from R² 0.290 to 0.309
against a bar of 0.50. The worker's reading is that the cold IS in his senses
(R² 0.998 on the two thermal floats alone) and the generic probe is drowned
by 80 irrelevant inputs. That may be right. It is also exactly the shape of
"a failing spec rewritten until it passes": a new probe, chosen after seeing
which one succeeds. It waits on the Director, correctly. If a new probe is
ruled it needs its rule first, seeds that are neither 90 nor 0–2's seen
lives, the bar unmoved at 0.50, and a control that must fail (the same probe
on the 80 non-thermal inputs read −0.83 — that is the natural one). `PS.02`
holds `LF.01`, `SH.01`, `SH.02`; thermal and shelter are CLAIM-DEAD in
`coverage`. Stage 3 under PLAN, so no hurry — which is the reason to rule it
slowly. Card line 4.

## 5 — S10 is untaken and the thing it guards against has happened again

Ordered 11-17 06:50, state "open". Both SYSTEM slots went to S9 (rightly
first). Meanwhile `jack_env.py` moved three times (`4978e989`, `9e4031bb`,
`e7c5011c`) and INBOX 07:31 warned `LC.02` / `W1.02` would read stale. Both
read PASS in `run status` now, so either they were re-bought by hand again
or the options did not stale them — I did not trace which. `run stale` lists
45 claims against changed code and one row from a modified tree (`PL.02`,
VOID). No harm today; the pilot kernels will need `LC.02` green the hour the
rule is committed.

---

## FOR THE DIRECTOR

1. **Rule the eye's frame before J12 spends its seeds** (finding 1). I would
   order `jack_env(eye_frame="body")` as the lab specifies, have the J12
   gate say in one line whether its forager uses heading it is not given,
   and run the frozen gate on the body-frame eye — or on both, declared
   before the run. Say which eye J11's arms see. If J12's kernels are
   already up when you sit, let them finish and read them as the compass
   eye's ceiling, not the eye's.
2. **Rule R and the card together, before the rule is written** (finding
   2): wait for the probe's reading; repair `spawn_clear` by its cause; say
   in advance whether the floors are re-measured on the repaired world or
   the leak stays and is carried by the floor clause. No post-hoc filtering
   of the card. Answer INBOX 19:45 (`native_substep` on or off for the
   pilot) in the same ruling.
3. **Give `W1.04` an owner or a parked line with a date**, and correct the
   header and queue rank 6 (finding 3). A detached job that can sit queued
   for eleven hours on a seat needs a queue timeout — one line for S-lane
   "Next up" (4), not a new order.
4. **`PS.02`'s probe**: rule first, fresh seeds, bar unmoved, the
   non-thermal-inputs control (finding 4) — or leave it red until stage 3.
5. **S10 next in the SYSTEM lane**, before J11's rule commit (finding 5).
6. The idle-slot list is exhausted (INBOX 17:25, 17:50: the shaping grid,
   RL² and Speech Commands learners each need a Kaggle rank). Yesterday's
   idle slots did real work (two task reproductions, the dial gate, the Mac
   rows) — refill the list or they will tend the machine.
7. Bookkeeping, one line: `verify` VENUE undeclared 266 against baseline 256
   (unchanged since yesterday, still printed as GREW); `run status`
   STEERING-PAGE ORDERS parses 0 items from 3 pages because it looks for a
   `FOR THE BUILDER` heading the reorganisation retired; `run ratchets`
   exits 2 on two floors no order owns (INBOX SYSTEM 11-17); four owner
   commitments remain CLAIM-DEAD (shelter, advised-not-commanded,
   tree-fire-shelter world, thermal) — stage 3, known.

**Machine.** All organs alive and inside cadence: ladder hourly, 26 of the
last 26 iterations rc 0, none refused; Director stamp 11-17 06:50 (18 h);
review 11-17 06:43; scout Monday 11-16 05:54 (next Thursday); regate 23:03.
`main` level with origin. `/` 5.7 GB free (81%, up from 4.2), `/data` 46 GB.
The tree is clean but for this page.

**Free compute.** GPU week W47 has no charged job: Kaggle's quota reads 0 h
until Saturday 11-21 00:00 UTC (W46 spent 33.76 h of 30), and J14 has brain
A's three rows frozen and smoke-tested for that hour — nothing expired
unused. Kaggle CPU carried the day (eight ~8 h kernels for the card), and
the ranked queue held: no side run kept a session an order waited for,
except the eleven hours in finding 3.

## FOR THE OWNER

Nothing that needs your decision today. One thing to know: his eye, old and
new, has been pointing by the world's compass rather than by where he faces
(finding 1). Nothing has been claimed on it yet; it is being put right
before anything is.
