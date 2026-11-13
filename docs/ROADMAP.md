# ROADMAP — the route from today to the goal (≤ 6 KB, Director-maintained)

`GOAL.md` says what Jack must become. `docs/ORDERS.md` says what the workers do
this week. This page is the bridge: the milestones in between, so the Director
always knows what comes next when the current orders are finished — in month
two, month six, with nobody watching.

**Rules.** A milestone is DONE only when its named specs PASS in the ledger.
The Director updates `status:` every Sunday, adds spec ids as registered, and
keeps an open order on the earliest unfinished milestone; later ones may run
in parallel when a compute lane is free. What a milestone IS comes from
`GOAL.md` and is the owner's; how it is tested may only be strengthened.

**Which Jack.** Two model stacks exist. The learning this project has actually
demonstrated is on the survival stack (`experiments/cores.py`, `survival.py`,
`w0.py`); the older 57M-parameter brain (`UnifiedBrain.py`) has lost every
control race so far. Milestones are judged on whatever holds the seats in
`docs/CHAMPIONS.md` by verdict — today that is the survival stack — and the old
brain's parts re-enter only as challengers in a cook-off, never by default.

---

**M1 — One whole life.** The learning brain lives a life end to end: world,
needs, the senses he has, diary, death and retry.
done when: `LF.03` PASS. · status (11-12): **MET** — `LF.03` PASS at `6a92fd4`
(3 seeds × 5 deaths, frozen-twin and wiped-diary controls both fired). He lives
but does not yet improve or use his diary: `LF.04` (order J1) asks that next.

**M2 — A world and a curriculum that can teach.** A world deep enough that
passivity dies, outcomes differ, and thinking ahead pays; the order he meets it
in decided by cook-off ("no curriculum we write" is the incumbent arm).
done when: `W1.00`–`W1.04` and the world-rule gates `W.1`–`W.8` PASS; `DP.05`'s
question passes in W1; `NE.01` PASS; the curriculum cook-off has a winner; then
W2 (owner 11-13): tree → wood → fire → cooked food and a buildable shelter exist
as specs and PASS. · status (11-13): `W1.02` passes; `W1.00` FAIL, `W1.04`
re-running; `W.1`, `W.2`, `NE.01` FAIL (one clock ruled, k = 72); curriculum
arms built, unraced; W2 unregistered (orders J6, J7, J5).

**M3 — The shape of his brain, by verdict.** Every architecture seat in
`docs/CHAMPIONS.md` held BY VERDICT, none by default or decree: the control path
(`D1.1`), the learning core at scale (`LC.07`), how the senses fuse and how
unified the brain is (`UB.10`, `BO.01`), fast and slow (`DP.01`–`DP.03`), and
its size (a size cook-off, to be registered).
done when: those seats read BY VERDICT and the seated brain passes END-FIT's
gates (`CHAMPIONS.md`). · status (11-13): working memory BY VERDICT (`WM.02`);
fusion raced, UNDECIDED (`UB.10` FAIL, no arm separates); control VACANT
(`D1.0` running); learning core's verdict is a VOID (orders J2, J3).

**M4 — One body that moves.** He moves well, in the same body he survives in —
today movement is tested on a humanoid and survival on a wheeled climber.
done when: `T2.01` PASS under the seated controller; the survival world runs
with the humanoid body (`LT.08`); hands that hold, carry and strike, the grip
rule won in a cook-off (unregistered). · status (11-13): `T2.01` FAIL since
Aug 12, behind `D1.0` (order J2).

**M5 — Every sense, in the one brain, each load-bearing.** Sight from his own
eye, hearing with his own learned ear, touch, balance, smell, taste, pain,
temperature, hunger and thirst, voice — each wired into the main brain, each
passing the hide-it, fake-sense and combination tests; nothing ready-made
inside him. Includes the cook-off of a sense added later vs present from birth.
done when: `experiments.coverage` shows a passing CLAIM spec for every sense
and the unison gates pass under the seated brain. · status: 5 senses in the
brain as short summaries; smell and taste tested in separate rigs; no learned
ear (`HR.1` FAIL, bar held by D54's default); language slot empty.

**M6 — Words.** The parent voice teaches him; he ties words to his own life,
says true things about himself, is nobody's puppet, and can be talked to.
done when: the word-grounding cook-off (`LG.03`–`LG.06`) has a winner; `LG.00`,
`LG.02`, `SO.02`, `NE.09` PASS with his own ear and voice in the loop; `HR.*`
claim specs PASS without a ready-made model inside him; advice in words makes
him try a step sooner and a lying adviser loses him. · status (11-11): no
learned part of Jack speaks or hears yet. `LG.00`, `LG.02`, `SO.02` pass in
rigs whose speaker is the diary plus a ready-made model — parent-rig results,
not his words (Director ruling 11-11); nothing connected in his world.

**M7 — He keeps learning.** The thesis: he learns without forgetting
(`T5.03`), stays plastic (`T5.04`), consolidates in sleep (`T5.05`, `NE.05`,
`ME.7`), explores because he wants to (`CU.1`–`CU.7`, `T5.06`), does not
saturate (`T5.08`), and death does not erase what he learned (`XL.01`, `NE.08`).
The ladder and the apple: he climbs it unprompted (`LT.03`+).
Handed to another machine he keeps what he knew and learns more (order J8).
done when: those PASS. · status (11-13): none run; `XL.01` close but failing.

**M8 — A living Jack.** Hours-long lives with everything at once; he remembers
across sessions and deaths; he can be watched, talked to and left things.
done when: an integration spec on the survival stack for a multi-hour life
with all of M1–M7 live PASSES (to be registered; replaces the Stack-B `T6.*`).
· status: not started.

**M9 — Beyond the jungle.** More worlds and transfer between them (`GEN.06`),
other minds (`GEN.02`, `GEN.03`, `GEN.09`), the told world (`LG.11`).
done when: those PASS. · status: not started.
