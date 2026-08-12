"""The sensory-inventory audit: the one check that reads from OUTSIDE the repo.

WHY THIS EXISTS (the scar). On 2026-08-10 the overseer grepped all 137
registered specs for `smell|olfact|taste|gustat|voice|vocal|pain|thermo|
temperature` and got **one** hit — the word "voiced" describing a struck geom in
PG.5's audio spec. Five of the senses GOAL.md calls constitutional had ZERO
specs. They were not blocked and not failing: they were ABSENT, and a capability
that was never registered is invisible to `run next`, `run blocked`,
`run status` and the Review alike — it reads as completeness in every organ this
system has. `docs/LESSONS.md` had named that failure 30 hours earlier
("AMBITION blindness — the map is the thing with the hole") and prescribed the
guard: *at least one recurring audit must measure against a reference from
OUTSIDE the project's own documents.* The lesson was not the guard. This is.

WHAT MAKES IT AN OUTSIDE REFERENCE. `INVENTORY` below is the human sensory
inventory as biology hands it to us, not as our registry describes itself. It
does not derive from `LADDER`, from `GOAL.md`'s wording, or from any file this
project can edit while chasing a green ladder. Adding a spec cannot make an
entry disappear; the only way to change the standard is to change biology's
list, deliberately, in this file, in a commit that says so.

WHY IT DOES NOT GREP. The failure that motivated it was itself a grep artifact
in miniature: "voiced" matched `voice` and voice does not exist. Coverage is
therefore claimed by an EXPLICIT DECLARATION — each sense names the spec ids
that carry it, and the audit asserts those ids resolve in the live registry. A
stray word can never buy coverage; a deleted or renamed spec always loses it.
The keyword scan survives only as an ADVISORY (`unmapped_mentions`), which
reports specs that talk about a sense without claiming it — the opposite error,
and the cheap one.

Read it with `python -m experiments.run senses`. Guarded by spec T0.20.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass(frozen=True)
class Sense:
    key: str
    name: str
    why: str
    """What this channel buys that no other channel can. Biology's argument, not ours."""
    specs: tuple[str, ...] = ()
    """Spec ids that CLAIM this channel. Declared, never inferred."""
    load_bearing: tuple[str, ...] = ()
    """Spec ids that would prove this channel is LOAD-BEARING, not merely present.

    GOAL.md's own standard, quoted: *"we PROVE each one is — ablate a sense,
    something measurable must degrade"*. A spec belongs here only if removing
    THIS channel is the manipulation and a measured quantity is required to
    degrade against a matched placebo. A fixture certificate never belongs
    here, however good its numbers: `SM.01` proves the odour field obeys its
    declared rules, `PG.6` proves the eye resolves — neither says the brain
    uses the channel for anything.

    Deliberately NOT required to be a subset of `specs`: `UB.11` is the
    standing ablation matrix and gives EVERY sense a row, but it is not a
    spec about smell any more than it is a spec about hearing, and putting it
    in `specs` would let a sense keep its coverage after its own family was
    deleted — the exact failure `T0.20` P3 exists to catch. The tiers compose
    instead: LOAD-BEARING requires the sense to already be SENSOR *and* to
    have a passing entry here.
    """
    mentions: str = ""
    """Advisory regex — for the unmapped-mention scan only. NEVER grants coverage."""
    effector: bool = False


# ── THE OUTSIDE REFERENCE ────────────────────────────────────────────────
# The human sensory inventory. Sourced from human biology and restated by the
# owner on 2026-08-09 (GOAL.md:41-43); it is the standard this project is
# measured against, and it does not shrink because a sense is inconvenient.
# VOICE is on the list although it is an effector, not a sense: GOAL.md puts it
# there deliberately — "he must be able to make sound, not only receive it".
INVENTORY: tuple[Sense, ...] = (
    Sense("sight", "sight (vision)",
          "the dominant human channel; spatial layout at a distance",
          specs=("PG.6", "T2.03", "T3.01"), load_bearing=("T3.01", "UB.11"),
          mentions=r"vision|visual|camera|\beye\b|pixel|rendered frame"),
    Sense("hearing", "hearing (audition)",
          "works around corners and in the dark; carries events sight misses",
          specs=("PG.5", "PG.7", "UB.4"), load_bearing=("UB.4", "UB.11"),
          mentions=r"audio|acoustic|hearing|stereo|spectrogram"),
    Sense("touch", "touch (mechanoreception)",
          "contact is the only sense that confirms a manipulation happened",
          specs=("UB.5",), load_bearing=("UB.5", "UB.11"),
          mentions=r"touch|tactile|mechanorecept"),
    Sense("proprioception", "proprioception & balance (vestibular)",
          "where his body is without looking; the substrate of every motor skill",
          specs=("T3.02", "UB.16"), load_bearing=("T3.02", "UB.11"),
          mentions=r"propriocept|vestibul|body schema"),
    Sense("smell", "smell (olfaction)",
          "finds food, fire and decay at a distance AND THROUGH OCCLUSION — "
          "the sense that works when sight fails",
          specs=("SM.01", "SM.02"), load_bearing=("SM.02", "UB.11"),
          mentions=r"smell|olfact|odour|odor"),
    Sense("taste", "taste (gustation)",
          "conditioned taste aversion: one-trial learning with long delay "
          "tolerance, the fastest learning in biology and a capability nothing "
          "else in his design has",
          specs=("TA.01", "TA.02", "TA.03"), load_bearing=("TA.03", "UB.11"),
          mentions=r"taste|gustat|flavour|flavor"),
    Sense("pain", "pain (nociception)",
          "a fast unconditioned cost signal that sensitises rather than "
          "habituating; the only thing that makes failing expensive",
          specs=("PS.03",),
          mentions=r"\bpain\b|nocicept"),
    Sense("temperature", "temperature (thermoception)",
          "cold nights are the only pressure that teaches shelter-building",
          specs=("PS.02",), load_bearing=("SH.01",),
          mentions=r"thermo|temperature|\bcold\b|hypothermi"),
    Sense("interoception", "interoception (hunger, thirst, fatigue)",
          "the needs ARE the curriculum — curiosity is the explorer, needs are "
          "the reason",
          specs=("PS.01",), load_bearing=("UB.11",),
          mentions=r"interocept|hunger|thirst|fatigue|homeostat"),
    Sense("voice", "voice (vocalisation) — an EFFECTOR, listed by the owner",
          "how a creature acts on other creatures; gates emergent language and "
          "the other-minds expansion",
          specs=("VO.01", "VO.02"), effector=True,
          # Voice is the one entry whose load-bearing route is NOT `UB.11`:
          # that matrix ablates INPUT modalities, and muting a mouth degrades
          # nothing a lone agent does. The only test that can cost him
          # something for losing his voice needs someone to talk to — VO.02,
          # blocked on a second Jack. Declared anyway, and deliberately: an
          # empty tuple would have made voice unable to reach LOAD-BEARING by
          # construction, which reads identically to "no route exists".
          load_bearing=("VO.02",),
          mentions=r"\bvoice\b|vocalis|vocaliz|utterance"),
)

# ── THE FOUR TIERS ───────────────────────────────────────────────────────
# Asked for by the overseer at audits 4 through 8 and correct: the file used to
# have three tiers, and its top one — DEMONSTRATED, "some declared spec is PASS"
# — said something GOAL.md never asks for. `PG.6` is a ridge probe whose own
# docstring says it certifies THE SENSOR, NOT THE NET, and it made sight read
# `[PASS]`. `SM.01` (2026-08-11) is the same shape and would have done the same
# for smell within minutes of being written.
#
# GOAL.md's standard is one sentence and it is not "a spec passed": *"we PROVE
# each one is load-bearing — ablate a sense, something measurable must
# degrade"*. So the tiers now say which of the two things happened:
#
#   ABSENT        nothing in this repo claims the channel
#   REGISTERED    a spec claims it; none has run green
#   SENSOR        the channel EXISTS and behaves — a fixture certificate passes.
#                 This is real work and it is where smell, taste, sight and
#                 hearing genuinely stand today.
#   LOAD-BEARING  removing the channel measurably COSTS him something, against a
#                 matched placebo. This is the claim GOAL.md actually makes, and
#                 as of 2026-08-11 nothing in the inventory has reached it.
#
# The point of the split is that the gap is now VISIBLE rather than flattering:
# a report that says SENSOR for ten senses and LOAD-BEARING for none is telling
# the truth about a project that has built sensors and not yet earned them.
ABSENT = "ABSENT"
REGISTERED = "REGISTERED"
SENSOR = "SENSOR"
LOAD_BEARING = "LOAD-BEARING"

TIERS = (ABSENT, REGISTERED, SENSOR, LOAD_BEARING)


@dataclass
class Coverage:
    sense: Sense
    declared: List[str]
    """Spec ids this sense claims."""
    missing: List[str] = field(default_factory=list)
    """Declared ids that do NOT resolve in the live registry — a broken claim."""
    passing: List[str] = field(default_factory=list)
    lb_declared: List[str] = field(default_factory=list)
    """Load-bearing spec ids, as declared."""
    lb_missing: List[str] = field(default_factory=list)
    """...that do NOT resolve in the live registry."""
    lb_passing: List[str] = field(default_factory=list)
    """...that are PASS. Non-empty is the ONLY route to LOAD-BEARING."""
    unmapped_mentions: List[str] = field(default_factory=list)

    @property
    def registered(self) -> List[str]:
        return [s for s in self.declared if s not in self.missing]

    @property
    def status(self) -> str:
        if not self.registered:
            return ABSENT
        if not self.passing:
            return REGISTERED
        # A passing ablation is necessary AND it is not sufficient on its own:
        # the sense must also have a passing spec of its own, or `UB.11` alone
        # would promote a channel this repo has never built.
        return LOAD_BEARING if self.lb_passing else SENSOR


def audit(by_id: Optional[Dict[str, object]] = None,
          ledger=None,
          inventory: tuple[Sense, ...] = INVENTORY) -> List[Coverage]:
    """Coverage of the human sensory inventory by the LIVE registry.

    `by_id` and `ledger` are injectable so the T0.20 control can hand this a
    registry with a family removed and a registry carrying a decoy — a check
    that cannot be shown the bad case is not a check.
    """
    if by_id is None:
        from .registry import BY_ID as by_id  # noqa: N813
    specs = list(by_id.values())
    mapped = {sid for s in inventory for sid in s.specs}

    out: List[Coverage] = []
    for sense in inventory:
        cov = Coverage(sense=sense, declared=list(sense.specs),
                       lb_declared=list(sense.load_bearing))
        cov.missing = [sid for sid in sense.specs if sid not in by_id]
        cov.lb_missing = [sid for sid in sense.load_bearing if sid not in by_id]
        if ledger is not None:
            from .protocol import Status
            # `status is Status.PASS` and nothing else, DELIBERATELY. This is
            # the syntax `T0.22` retired on the borrow path and 2026-08-11
            # retired on the dependency path, so it is the third hit of the
            # grep that lesson prescribes — and here the weaker rule is the
            # right one. Coverage asks "was this capability ever demonstrated",
            # which a stale PASS still answers; a re-run pending is a fact about
            # freshness and `run stale` is the organ that reports it. Counting
            # coverage as zero because a file was edited would make this
            # instrument swing on edits rather than on evidence.
            cov.passing = [sid for sid in cov.registered
                           if ledger.status(sid) is Status.PASS]
            cov.lb_passing = [sid for sid in sense.load_bearing
                              if sid not in cov.lb_missing
                              and ledger.status(sid) is Status.PASS]
        if sense.mentions:
            pat = re.compile(sense.mentions, re.I)
            for sp in specs:
                if sp.id in mapped:
                    continue
                text = " ".join(str(x) for x in
                                (sp.title, sp.hypothesis, sp.notes or ""))
                if pat.search(text):
                    cov.unmapped_mentions.append(sp.id)
        out.append(cov)
    return out


def absent(covs: List[Coverage]) -> List[str]:
    return [c.sense.key for c in covs if c.status == ABSENT]


def load_bearing(covs: List[Coverage]) -> List[str]:
    """The senses that meet GOAL.md's actual standard. Today: none."""
    return [c.sense.key for c in covs if c.status == LOAD_BEARING]


def render(covs: List[Coverage]) -> str:
    lines = ["", "  THE HUMAN SENSORY INVENTORY — the reference is biology,",
             "  not this repository. A sense with no spec is ABSENT, and",
             "  ABSENT is invisible to every other report this system makes.",
             "",
             "  SENSOR = the channel exists and behaves (a fixture certificate).",
             "  LOAD-BEARING = ablating it measurably costs him something. That",
             "  is what GOAL.md asks for; a passing sensor spec is not it.",
             ""]
    width = max(len(c.sense.name) for c in covs)
    for c in covs:
        mark = {ABSENT: "[ABSENT ]", REGISTERED: "[spec'd ]",
                SENSOR: "[SENSOR ]", LOAD_BEARING: "[LOADBRG]"}[c.status]
        ids = ", ".join(c.registered) or "—"
        lines.append(f"    {mark} {c.sense.name:<{width}}  {ids}")
        if c.passing:
            lines.append(f"              {'':<{width}}  sensor: "
                         f"{', '.join(c.passing)}")
        if c.lb_passing:
            lines.append(f"              {'':<{width}}  load-bearing: "
                         f"{', '.join(c.lb_passing)}")
        elif c.lb_declared:
            lines.append(f"              {'':<{width}}  load-bearing awaits: "
                         f"{', '.join(c.lb_declared)}")
        else:
            lines.append(f"              {'':<{width}}  load-bearing: NO SPEC "
                         f"would prove it")
        if c.lb_missing:
            lines.append(f"              {'':<{width}}  !! load-bearing id NOT "
                         f"in the registry: {', '.join(c.lb_missing)}")
        if c.missing:
            lines.append(f"              {'':<{width}}  !! declared but NOT in "
                         f"the registry: {', '.join(c.missing)}")
        if c.unmapped_mentions:
            lines.append(f"              {'':<{width}}  (mentioned, not claimed,"
                         f" by {len(c.unmapped_mentions)}: "
                         f"{', '.join(c.unmapped_mentions[:6])})")
    gone = absent(covs)
    lb = load_bearing(covs)
    lines += ["", f"  {len(covs) - len(gone)}/{len(covs)} of the inventory has a "
                  f"registered spec.",
              f"  {len(lb)}/{len(covs)} are LOAD-BEARING — the standard GOAL.md "
              f"sets: {', '.join(lb) if lb else 'none'}."]
    if gone:
        lines += [f"  ABSENT: {', '.join(gone)} — not blocked, not failing. "
                  f"Nothing in this repo claims them.", ""]
    else:
        lines += ["  Nothing in the inventory is unrepresented.", ""]
    return "\n".join(lines)


if __name__ == "__main__":  # pragma: no cover
    from .protocol import Ledger
    print(render(audit(ledger=Ledger())))
