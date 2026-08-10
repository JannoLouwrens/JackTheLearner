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
          specs=("PG.6", "T2.03", "T3.01"),
          mentions=r"vision|visual|camera|\beye\b|pixel|rendered frame"),
    Sense("hearing", "hearing (audition)",
          "works around corners and in the dark; carries events sight misses",
          specs=("PG.5", "PG.7", "UB.4"),
          mentions=r"audio|acoustic|hearing|stereo|spectrogram"),
    Sense("touch", "touch (mechanoreception)",
          "contact is the only sense that confirms a manipulation happened",
          specs=("UB.5",),
          mentions=r"touch|tactile|mechanorecept"),
    Sense("proprioception", "proprioception & balance (vestibular)",
          "where his body is without looking; the substrate of every motor skill",
          specs=("T3.02", "UB.16"),
          mentions=r"propriocept|vestibul|body schema"),
    Sense("smell", "smell (olfaction)",
          "finds food, fire and decay at a distance AND THROUGH OCCLUSION — "
          "the sense that works when sight fails",
          specs=("SM.01", "SM.02"),
          mentions=r"smell|olfact|odour|odor"),
    Sense("taste", "taste (gustation)",
          "conditioned taste aversion: one-trial learning with long delay "
          "tolerance, the fastest learning in biology and a capability nothing "
          "else in his design has",
          specs=("TA.01", "TA.02", "TA.03"),
          mentions=r"taste|gustat|flavour|flavor"),
    Sense("pain", "pain (nociception)",
          "a fast unconditioned cost signal that sensitises rather than "
          "habituating; the only thing that makes failing expensive",
          specs=(),
          mentions=r"\bpain\b|nocicept"),
    Sense("temperature", "temperature (thermoception)",
          "cold nights are the only pressure that teaches shelter-building",
          specs=(),
          mentions=r"thermo|temperature|\bcold\b|hypothermi"),
    Sense("interoception", "interoception (hunger, thirst, fatigue)",
          "the needs ARE the curriculum — curiosity is the explorer, needs are "
          "the reason",
          specs=("PS.01",),
          mentions=r"interocept|hunger|thirst|fatigue|homeostat"),
    Sense("voice", "voice (vocalisation) — an EFFECTOR, listed by the owner",
          "how a creature acts on other creatures; gates emergent language and "
          "the other-minds expansion",
          specs=("VO.01", "VO.02"), effector=True,
          mentions=r"\bvoice\b|vocalis|vocaliz|utterance"),
)

ABSENT = "ABSENT"
REGISTERED = "REGISTERED"
DEMONSTRATED = "DEMONSTRATED"


@dataclass
class Coverage:
    sense: Sense
    declared: List[str]
    """Spec ids this sense claims."""
    missing: List[str] = field(default_factory=list)
    """Declared ids that do NOT resolve in the live registry — a broken claim."""
    passing: List[str] = field(default_factory=list)
    unmapped_mentions: List[str] = field(default_factory=list)

    @property
    def registered(self) -> List[str]:
        return [s for s in self.declared if s not in self.missing]

    @property
    def status(self) -> str:
        if not self.registered:
            return ABSENT
        return DEMONSTRATED if self.passing else REGISTERED


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
        cov = Coverage(sense=sense, declared=list(sense.specs))
        cov.missing = [sid for sid in sense.specs if sid not in by_id]
        if ledger is not None:
            from .protocol import Status
            cov.passing = [sid for sid in cov.registered
                           if ledger.status(sid) is Status.PASS]
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


def render(covs: List[Coverage]) -> str:
    lines = ["", "  THE HUMAN SENSORY INVENTORY — the reference is biology,",
             "  not this repository. A sense with no spec is ABSENT, and",
             "  ABSENT is invisible to every other report this system makes.",
             ""]
    width = max(len(c.sense.name) for c in covs)
    for c in covs:
        mark = {ABSENT: "[ABSENT ]", REGISTERED: "[spec'd ]",
                DEMONSTRATED: "[PASS   ]"}[c.status]
        ids = ", ".join(c.registered) or "—"
        lines.append(f"    {mark} {c.sense.name:<{width}}  {ids}")
        if c.passing:
            lines.append(f"              {'':<{width}}  demonstrated: "
                         f"{', '.join(c.passing)}")
        if c.missing:
            lines.append(f"              {'':<{width}}  !! declared but NOT in "
                         f"the registry: {', '.join(c.missing)}")
        if c.unmapped_mentions:
            lines.append(f"              {'':<{width}}  (mentioned, not claimed,"
                         f" by {len(c.unmapped_mentions)}: "
                         f"{', '.join(c.unmapped_mentions[:6])})")
    gone = absent(covs)
    lines += ["", f"  {len(covs) - len(gone)}/{len(covs)} of the inventory has a "
                  f"registered spec."]
    if gone:
        lines += [f"  ABSENT: {', '.join(gone)} — not blocked, not failing. "
                  f"Nothing in this repo claims them.", ""]
    else:
        lines += ["  Nothing in the inventory is unrepresented.", ""]
    return "\n".join(lines)


if __name__ == "__main__":  # pragma: no cover
    from .protocol import Ledger
    print(render(audit(ledger=Ledger())))
