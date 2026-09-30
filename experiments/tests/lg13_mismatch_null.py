"""The mismatched-meaning null on the seated Language-routing champion.

NOT A REGISTERED SPEC. This is the STANDING CONDITION the 2026-09-29 paired
ruling (`lg13-champion-makes-lg10s-invariance-conjuncts-structural` +
`so10-tie-break-hands-the-seat-to-an-ineligible-arm`, `e41c932`) placed on the
`meaning-mass` champion: its `unanimity`/`swap_agree`/`variety` readings are
DISCHARGED-BY-CONSTRUCTION, not passed, and may not be quoted as evidence
anywhere until the champion clears a null its construction cannot pass — *"the
same mechanism run with its structure drawn from a DIFFERENT trial's intent.
If the meaning is chosen deterministically from the wrong trial and `match`
still reads high, the construction was carrying the number."*

WHAT RUNS. For every seed and every report trial i, the champion's own code
object (`lg_13._sel_meaning_mass`, imported, not copied) chooses a meaning
from trial i's OWN candidate pool — but scored under the ARM PROMPT OF TRIAL
j = (i+1) mod N_REPORT, i.e. with the intent conditioning drawn from the
wrong trial. Nothing else moves: same pools, same scaffold, same two frozen
mouths at the store's pinned revisions, same TEMP. The intent enters this rig
ONLY through the prompt (`ARM_ASK.format(intent=...)`), so swapping the
prompt's intent is the whole and only mismatch.

THE BAR, pre-registered here before any number existed (this file is committed
before its first run; the bar is the rig's own, not invented for this null):

    CLEARS   iff  mismatched match <= NULL_MATCH_MAX (0.35) for EVERY seed
                  and BOTH mouths — the bar LG.10 registered for its state-free
                  null on 2026-09-02, applied unchanged.
    INDICTED iff  any (seed, mouth) reads above 0.35: the pool/scaffold
                  construction, not the mouth's intent-conditioned scores, was
                  carrying the champion's 1.0000, and the struck readings STAY
                  struck. No bar moves in either direction on any branch.

TWO READINGS CARRIED BESIDE THE VERDICT, both expected and both load-bearing:

  * the MATCHED anchor (weights from trial i's own prompt, pure cache read)
    must reproduce the champion's recorded 1.0000 on every cell — if it does
    not, this diagnostic's plumbing is wrong and the run is VOID, not a
    reading.
  * unanimity-under-mismatch is printed and is expected to read 1.0 HOWEVER
    the verdict falls, because the meaning choice is deterministic — that is
    the demonstration that the structural conjuncts cannot see this sabotage,
    which is why the ruling struck them.

THE CERTIFIED ARTIFACT IS NEVER WRITTEN. New (prompt, candidate) scores go to
STORE_MM (`/data/lg13_mismatch_verdicts.json`); reads fall through to the
certified `/data/lg10_llm_verdicts.json` first, so shared pairs cost nothing
and LG.10/LG.13's content-hashed evidence cannot be disturbed by this file.

Run (scores what is missing, then analyses; ~10 min first time, seconds after):

    /data/venvs/jackthelearner/bin/python -m experiments.tests.lg13_mismatch_null
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from .lg_10_jack_chooses_what_to_say import (ARM_ASK, ARTIFACT, MODELS,
                                             N_REPORT, NULL_MATCH_MAX,
                                             SCAFFOLD_SHA, S_DRAWS,
                                             _build_trials, _canonical, _key,
                                             _pool, _prompt)
from .lg_13_chooser_seat_bakeoff import SEEDS, _sel_meaning_mass

STORE_MM = "/data/lg13_mismatch_verdicts.json"
OUT = "/data/lg13_mismatch_null.json"


def _mm_prompt(trials: list, i: int) -> str:
    """Trial i's arm prompt with trial j=(i+1)%N_REPORT's intent in it —
    byte-identical to trial j's own arm prompt, which is what makes the
    certified store a partial cache for this pass."""
    j = (i + 1) % N_REPORT
    return _prompt(ARM_ASK.format(intent=_canonical(*trials[j]["intent"])))


def _pairs_for(seed: int) -> list:
    """Every (prompt, candidate) this null needs for one seed."""
    mem, trials = _build_trials(seed)
    pairs = []
    for i in range(N_REPORT):
        mm_p = _mm_prompt(trials, i)
        for u, _m in _pool(mem, trials[i]):
            pairs.append((mm_p, u))
    return pairs


def _load(path: str) -> dict:
    return json.loads(Path(path).read_text()) if Path(path).exists() else {}


def llm_pass() -> None:
    """Score the missing pairs, `lg_10.llm_pass` line for line, into STORE_MM.
    The certified store is READ for cache hits and never written."""
    import gc
    import os

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    os.environ.setdefault("HF_HOME", "/data/caches/huggingface")
    cert = _load(ARTIFACT)
    revs = cert.get("_meta", {}).get("models", {})
    store = _load(STORE_MM) or {"_meta": {"models": dict(revs),
                                          "scaffold_sha": SCAFFOLD_SHA}}
    all_pairs = []
    for seed in SEEDS:
        all_pairs.extend(_pairs_for(seed))

    for model_id in MODELS:
        tok = AutoTokenizer.from_pretrained(model_id)
        tok.padding_side = "right"
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=torch.float32, low_cpu_mem_usage=True).eval()
        revision = getattr(model.config, "_commit_hash", None) or "local"
        # The mouths are pinned by the certified store; a moved revision would
        # make this null a reading about different mouths — refuse loudly.
        assert revs.get(model_id) in (None, revision), \
            f"{model_id} revision {revision} != certified {revs.get(model_id)}"

        todo = {}
        for prompt, option in all_pairs:
            k = _key(model_id, revision, prompt, option)
            if k not in cert and k not in store:
                todo[k] = (prompt, option)
        todo = list(todo.items())
        print(f"[lg13-mm] {model_id}: {len(todo)} pairs to score", flush=True)

        B = 8
        with torch.no_grad():
            for s in range(0, len(todo), B):
                chunk = todo[s:s + B]
                texts = [p + o for _, (p, o) in chunk]
                n_pre = [len(tok(p).input_ids) for _, (p, _o) in chunk]
                enc = tok(texts, return_tensors="pt", padding=True)
                logp = torch.log_softmax(model(**enc).logits, dim=-1)
                for j, (k, _) in enumerate(chunk):
                    ids = enc.input_ids[j]
                    n = int(enc.attention_mask[j].sum())
                    span = range(n_pre[j], n)
                    tot = sum(float(logp[j, t - 1, ids[t]]) for t in span)
                    store[k] = round(tot / max(1, len(span)), 5)
                if s % (B * 20) == 0:
                    print(f"[lg13-mm] {model_id} {s + len(chunk)}/{len(todo)}",
                          flush=True)
        del model, tok
        gc.collect()
        Path(STORE_MM).write_text(json.dumps(store))
    print(f"[lg13-mm] wrote {STORE_MM}", flush=True)


def analyse() -> dict:
    """The null and its anchor, off the two stores. No model is loaded."""
    cert = _load(ARTIFACT)
    mm = _load(STORE_MM)
    revs = cert.get("_meta", {}).get("models", {})

    def _score(model_id, prompt, option):
        k = _key(model_id, revs.get(model_id, "local"), prompt, option)
        return cert.get(k) if k in cert else mm.get(k)

    out = {"bar_null_match_max": NULL_MATCH_MAX, "cells": {}, "missing": 0}
    worst_mm = 0.0
    anchor_min = 1.0
    for seed in SEEDS:
        mem, trials = _build_trials(seed)
        for model_id in MODELS:
            mm_hits, anchor_hits, unan = [], [], []
            for i in range(N_REPORT):
                pool = _pool(mem, trials[i])
                arm_p = _prompt(ARM_ASK.format(
                    intent=_canonical(*trials[i]["intent"])))
                mm_p = _mm_prompt(trials, i)
                sc_mm, sc_ok = [], []
                for u, m in pool:
                    s_mm = _score(model_id, mm_p, u)
                    s_ok = _score(model_id, arm_p, u)
                    if s_mm is None or s_ok is None:
                        out["missing"] += 1
                        continue
                    sc_mm.append((s_mm, u, m))
                    sc_ok.append((s_ok, u, m))
                if not sc_mm or len(sc_mm) != len(pool):
                    continue
                # The champion's own code object; the meaning it returns is
                # deterministic, the rng only draws the wording.
                meanings = [_sel_meaning_mass(
                    sc_mm, random.Random(f"lg13mm:{seed}:{i}:{model_id}:{d}"))[1]
                    for d in range(S_DRAWS)]
                mm_hits.append(float(meanings[0] == trials[i]["intent"]))
                unan.append(float(len(set(meanings)) == 1))
                anchor = _sel_meaning_mass(
                    sc_ok, random.Random(f"lg13ok:{seed}:{i}:{model_id}"))[1]
                anchor_hits.append(float(anchor == trials[i]["intent"]))
            cell = {
                "match_mismatched": round(sum(mm_hits) / max(1, len(mm_hits)), 4),
                "match_anchor": round(
                    sum(anchor_hits) / max(1, len(anchor_hits)), 4),
                "unanimity_mismatched": round(sum(unan) / max(1, len(unan)), 4),
                "n_trials": len(mm_hits),
            }
            out["cells"][f"seed{seed}:{model_id.split('/')[-1]}"] = cell
            worst_mm = max(worst_mm, cell["match_mismatched"])
            anchor_min = min(anchor_min, cell["match_anchor"])

    out["worst_match_mismatched"] = round(worst_mm, 4)
    out["anchor_min"] = round(anchor_min, 4)
    out["anchor_ok"] = anchor_min == 1.0 and out["missing"] == 0
    out["clears"] = bool(out["anchor_ok"] and worst_mm <= NULL_MATCH_MAX)
    out["verdict"] = ("VOID: anchor or store incomplete — a reading about the "
                      "plumbing, not the champion" if not out["anchor_ok"]
                      else "CLEARS" if out["clears"] else
                      "INDICTED: the construction was carrying the number")
    Path(OUT).write_text(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    llm_pass()
    res = analyse()
    print(json.dumps(res, indent=1))
    print(f"[lg13-mm] verdict: {res['verdict']} "
          f"(worst mismatched match {res['worst_match_mismatched']} vs bar "
          f"{res['bar_null_match_max']}; anchor min {res['anchor_min']})")
