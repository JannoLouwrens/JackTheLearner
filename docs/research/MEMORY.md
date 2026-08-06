# Memory Systems for Jack — Mechanism Catalogue (researched 2026-08-06)

Serves GOAL.md ("Memory makes it him"). Stance: memory = plain files/SQLite on
the ARM box, inspectable, restart-proof; GPUs only for ephemeral consolidation
jobs. Every mechanism ships with H (hypothesis), Kill, Null, and a Control that
must fail.

## 1. Episodic — "what happened, retrievable by cue"

**1.1 Key-value episodic store / episodic control** (MFEC 1606.04460, NEC
1703.01988, AEC 2506.01442). Store (state-embedding → outcome); act by k-NN
over embeddings. ARM: SQLite `(ts, embedding, action, outcome, meta)` + FAISS
flat/hnswlib, fine at <1M rows. Test: seen-task episodes-to-criterion drops vs
no-lookup; Null: shuffled outcome labels; Control: random-embedding retrieval
must NOT help.

**1.2 Retrieval-augmented RL** (RA-RL 2202.08417; 2306.10698; 2603.07110;
2603.18272). Policy cross-attends over top-k retrieved trajectories, trained
end-to-end. ARM: store+index on box; T4 job pulls snapshots. Test: beats same
arch fed k RANDOM chunks; Null: recency-only; Control: chunks from a different
world must drop to null (else the retrieval slot is ignored — documented
failure mode, always run).

**1.3 Event-log memory stream** (Generative Agents 2304.03442; MrSteve
2411.06736; AriGraph 2407.04363). Timestamped NL records; score = recency ×
importance × similarity. ARM: ONE SQLite FTS5 table — the day-one item,
near-zero cost. Test: cued QA over own history ≥80% at 1k events; Null:
recency-only; Control: query a FABRICATED event — must abstain; confabulation
= broken threshold.

**1.4 Verbal lessons from failure** (Reflexion 2303.11366). After failure,
write a one-line lesson; prepend retrieved lessons on retry. Test: retry
success rises; Null: pure resampling (the known confound); Control: unrelated
lessons must not help.

## 2. Semantic — consolidated knowledge

**2.1 SIESTA wake/sleep** (2303.10725, code github.com/yousuf907/SIESTA;
REMIND 1910.02509; stability gap 2306.01904). Wake: frozen backbone,
backprop-free prototype updates, PQ-quantized latents to disk. Sleep: budgeted
GPU rehearsal. **Best architectural fit for Jack**: wake on ARM, latent buffer
a file, sleep = ephemeral-GPU job. Test: after sleep, old-concept accuracy
holds (≤2pt drop) while new absorbs; Null: wake-only AND naive fine-tune;
Control: sleep with emptied buffer MUST forget.

**2.2 Complementary Learning Systems as the contract** (McClelland 1995;
CLS-ER 2201.12604; surveys 2512.13564, 2603.07670). Fast instance store + slow
semantic store + consolidation between. Rule for Jack: EVERY episodic table
must have a consolidation consumer. Test of consolidation itself: consolidate
N episodes into a rule, DELETE the episodes, performance holds; Control:
consolidating shuffled episodes must yield a worse rule.

**2.3 Reflection trees** (2304.03442). Periodic "what follows from the last
~100 memories?" → stored, source-linked beliefs. Test: aggregation questions
("what does my owner usually do on Sundays?") beat top-k raw events at equal
tokens; Control: another agent's reflections must hurt.

**2.4 Graph memory, bitemporal** (HippoRAG 2405.14831 / 2502.14802;
Zep/Graphiti 2501.13956; AriGraph). Triples + Personalized-PageRank retrieval;
`t_valid/t_ingested/t_invalidated` so facts supersede, never vanish. ARM: NO
Neo4j — SQLite edges + in-process networkx PPR. Test: 2-hop questions beat
vector-only; Control: 1-hop questions must show NO graph advantage.

## 3. Working memory in the policy

**3.1 Context-window** (DT 2106.01345, GTrXL 1910.06764): K≤64 on ARM. Test:
delayed-cue success falls exactly past K — if not, the env leaks; fix env first.

**3.2 GRU recurrence — PRIORITISE**: O(1)/step, microseconds on ARM, and the
hidden state checkpoints to disk (`wm.state`) → the only working memory that
natively SURVIVES RESTARTS, matching the owner's directive. Test: solves
delays where windowed transformer fails at ≤10% cost; Control: zeroing hidden
state mid-episode must drop to memoryless.

**3.3 State-space models** (S5-for-RL 2303.03982 resettable states = in-context
RL; Decision Mamba 2403.19925/2406.00079; Drama 2410.08893; Mamba 2312.00752).
Linear-time long-range; adapts within one episode where GRU needs retraining.
Test: zero-shot task-perturbation adaptation beats GRU; Control: resetting SSM
state at the perturbation must erase the adaptation.

**3.4 MemGPT/Letta paging** (2310.08560): context = RAM, stores = swap; agent
calls memory_search/append/core_memory_edit; small always-in-context owner
block. Test: answers whose evidence exceeds context beat truncate-to-fit;
Control: tools over an EMPTY DB must fail (answers come from store, not
weights).

## 4. Scaling, forgetting, compression

**4.1 Degradation facts** (LongMemEval 2410.10813; LoCoMo 2402.17753; Mem0
2504.19413): assistants drop ~30% on sustained memory; full-context reading
drops 30-60% vs oracle retrieval AT ONLY ~50 SESSIONS. Full-context is not a
memory system. Standing test at every store decade (10²→10⁵): precision@1 vs
recency-null; gap-to-oracle is the degradation curve. Rubric = LongMemEval's
five abilities incl. ABSTENTION.

**4.2 Forgetting policy** (MemoryBank 2305.10250 Ebbinghaus decay +
reinforce-on-recall; A-Mem 2502.12110; Zep supersede). Nightly cron: decay,
supersede (never delete contradicted facts — invalidate), hard-delete only
never-recalled+superseded. Test: beats FIFO on frequently-referenced old
facts; Control: knowledge-UPDATE questions must FAIL in the no-supersede
variant (stale answers) — else the questions don't conflict.

**4.3 Compression** (ReadAgent gists 2402.09727; Compressive Transformer
1911.05507; SIESTA PQ). Replace aging raw with gists + pointers. Test: 10×
smaller, ≤5pt QA loss; Null: truncate to same bytes; Control: fine-detail
questions inside compressed spans MUST degrade (else you weren't compressing).

## 5. Cross-session persistence & the owner (companion core)

Three artifacts, all plain files (MemoryBank/SiliconFriend 2305.10250; Mem0
2504.19413 — 67% LoCoMo at ~1.8k tokens vs 26k full-context; Letta blocks; Zep):
1. `profile.json` — small always-in-context owner block, self-edited
2. `facts.db` — bitemporal preferences/facts with provenance; session-end
   extract→dedupe→ADD/UPDATE/SUPERSEDE/NOOP pass (Mem0 pattern)
3. the §1.3 episodic stream — "remember when we…" texture; companionship
   lives in specifics

Test: session-N+1 spontaneously honours a preference stated once in session ≤N;
Null: fresh no-memory Jack's base rate; **Control 1**: contradicting
preferences planted at N and N+3 — must follow newer; **Control 2 (the
inspectability guarantee)**: wipe the file, restart — adherence must drop to
base rate, proving memory lives ON DISK, not in weights or cache.

## 6. What SOTA agents do (steal-list)

| System | One line | Steal |
|---|---|---|
| Voyager 2305.16291 | verified executable skills as files, NL-indexed | HIGHEST-LEVERAGE: skills as git-able files, added only after passing verification — maps 1:1 onto the ledger discipline |
| MemGPT/Letta 2310.08560 | self-managed paging + memory blocks | the tool interface + owner block |
| Generative Agents 2304.03442 | stream + recency·importance·relevance + reflections | scoring + consolidation |
| Mem0 2504.19413 | session-end extract→compare→update pipeline | the extraction pass; LoCoMo numbers are the bar |
| Zep 2501.13956 | bitemporal KG, supersede-not-delete | the schema |
| A-Mem 2502.12110 | notes that link and supersede each other | link-on-write |
| 2026 skill line: SkillOS 2605.06614, PMD 2607.01480, 2605.23899, 2604.08224, survey 2512.13564 | abstract trajectories into editable, auditable, composable skills; working/episodic/semantic/procedural is the standard taxonomy | validates the whole design; skill CURATION is the frontier |

Voyager-pattern test: composite task (2 learned skills) much faster than
scratch; Null: no library, random-skill retrieval; Control: corrupt a
retrieved skill's body — the composite MUST fail (execution really uses it).

## 7. The composite for Jack (all files, all inspectable)

```
/data/jack/memory/
  events.db     episodic stream + lessons (§1.3, §1.4)   ← DAY ONE
  facts.db      bitemporal facts/preferences + edges (§2.4, §5)
  profile.json  owner block, always in context (§5)
  skills/       verified skill files + index (§6)
  latents.bin   SIESTA wake buffer (§2.1)
  wm.state      checkpointed GRU/SSM state (§3.2/3.3)
```
Nightly sleep cron: reflections → fact extraction → decay/supersede → gists →
(GPU when available) SIESTA rehearsal + retrieval-augmented retraining.

**Build order on this box:** events.db → profile+facts → lessons → reflections
→ skill library → forgetting → SIESTA loop → RA-policy + SSM (GPU-gated).
Skip until proven necessary: graph servers, in-weights episodic schemes,
anything needing a resident GPU.

**Standing tests forever:** §4.1 degradation curve at every store decade;
§5 Control 2 wipe-the-file.
