# HEARING_BAKEOFF.md — how Jack hears

> Researched 2026-08-09. Serves GOAL.md: *"One brain, all senses in unison."*
> He must hear the ladder creak, the splash, the thud of his own fall, and learn
> from it.
>
> Read `GOAL.md`, `SYSTEM.md`, `docs/LESSONS.md` first. This is the **hearing
> arm** of `docs/research/UNIFIED_BRAIN.md` §4. Its trunk → readout → percept
> contract and its **placebo-modality** methodology come from
> `docs/research/UNIFIED_BRAIN_BAKEOFF.md` (§2a, §5) and are not re-litigated
> here — they are adopted. Nothing in this document is a claim until it is in
> `experiments/ledger.json`.

---

## 0. Three different jobs that people call "hearing"

Conflating them is how audio work goes wrong. They have different inputs,
different failure modes, different homes in the architecture, and different
places on the ladder.

| | job | input | where it lives | why |
|---|---|---|---|---|
| **(a)** | **speech → text** | mic, human voice | **frozen pre-processor, OUTSIDE the brain** | same status as the LLM. Nothing is learned; it is a text source. |
| **(b)** | **who is speaking** | mic, human voice | **frozen embedder + a tiny enrolment store, OUTSIDE the brain** | Whisper cannot do this at all. It writes `EpisodicMemory.speaker`, which the whole memory pillar is built on. |
| **(c)** | **sounds of the world** | sim contact audio (and, later, ambient mic) | **INSIDE the unified brain, a raw modality trained jointly** | this is the "all senses in unison" part. It has to be co-trained or it is a bolt-on. |

**(a) and (b) are the same waveform and different questions.** Whisper discards
speaker identity by design — it is trained to be invariant to it. Running
Whisper harder will never yield (b).

**(c) is not the same waveform.** It is 16 kHz stereo synthesised by
`ContactAudio.py` from MuJoCo contacts, with exact ground-truth provenance. It
shares no distribution with human speech and should share no encoder with it.

### 0.1 Why (b) is the biggest hole in the memory pillar

`ME.9` (attributed recall) **PASSES at 1.0** — and `experiments/tests/me_9_attributed_recall.py:66,73,78`
hands `mem.record(...)` the speaker string as a literal. `EpisodicMemory.record`
(`EpisodicMemory.py:124`) takes `speaker: str` and never questions it. The whole
attribution apparatus — `what_did_they_tell_me(speaker, ...)`, the
swapped-provenance control, the `speaker` filter in `recall()`
(`EpisodicMemory.py:175`) — is downstream of a field that, in the live system,
**nothing produces**.

`AudioListener.py` is the live path and it produces text only
(`_transcribe` → `on_transcription(text)`, lines 324-345). There is no speaker
anywhere in it. So today, either every heard event is recorded under one
hard-coded name, or under a name a human typed. The first makes
`what_did_they_tell_me` a no-op; the second makes ME.9's claim untestable in
deployment.

This is exactly the disease `SYSTEM.md` was written against: a component whose
test supplies the input the real system cannot. ME.9 is *not* fraudulent — it
tests the retrieval contract, which is a real thing — but the sentence "Jack
remembers who told him what" is currently unearned end-to-end, and §2 below is
the spec that would earn it.

### 0.2 What the repo already has (verified 2026-08-09, not assumed)

| artifact | state |
|---|---|
| `AudioListener.py` | mic capture + circular buffer + **energy-threshold VAD** + Whisper wiring. Has an OpenAI-API transcription path (`_transcribe_api`, line 372) — **must be deleted**: paid API, `SYSTEM.md` forbids it. |
| `ContactAudio.py` | modal-resonance synth on MuJoCo contact **onsets**; constant-power stereo pan `p = -sin(azimuth)`; per-event ground-truth azimuth/lateral/elevation/distance. |
| `PG.5` | **PASS.** Bearing decodes from stereo at ≥0.9 within 10°; labels honest within 5°; spectral match ≥0.8; mono and shuffled-pan controls both fail. This is the certified fixture everything in §4 stands on. |
| `UnifiedBrain.AudioEncoder` (line 980) | two modes: frozen Whisper-tiny + wav2vec2-base (768→512 projector), else an 80-mel Conv2d CNN fallback. **Never received a gradient in any ledger-recorded run.** |
| `UNIFIED_BRAIN.md` §4 | already concluded *"Skip pretrained audio towers — sim audio is narrow"*, and specified `16 kHz → 64-mel × 2ch → conv stem → 4 tokens`. §3 tests that conclusion instead of inheriting it. |
| `UNIFIED_BRAIN_BAKEOFF.md` | contributes the placebo-modality null, the four-perturbation ablation, the synergy gap vs the unimodal late ensemble, and the SWAP-FLIP causal control. All reused verbatim here. |

**The AudioEncoder as written is wrong for both jobs it claims.** It puts
Whisper *inside* an `nn.Module` in the brain (job (a) belongs outside), and it
uses wav2vec2 — a model trained on 960 h of read English speech — as the encoder
for modal-resonator impact rings (job (c), a distribution wav2vec2 has never
seen). §4's bakeoff decides what replaces it; §3 says why the pretrained tower
is the arm expected to lose.

---

## 1. Survey, with measured cost on the hardware we actually have

> **Citation hygiene — read this before quoting any number below.** Following
> `UNIFIED_BRAIN_BAKEOFF.md` §1, but with a fourth tier this document had to
> invent:
>
> - **[M]** — measured **on this box** during this pass. Reproducible; the
>   command is in the text. These are the only numbers I stand behind fully.
> - **[V]** — fetched from a primary source (paper, repo, HF/PyPI API) during
>   this pass **and corroborated by at least two independent reports**.
> - **[f]** — extracted from a *figure* rather than a table, with the extraction
>   validated. Sound method, wide error bars.
> - **[u]** — **reported once and NOT corroborated.** The research pass that
>   produced these later reported it had been *unable* to obtain them, so the
>   two accounts disagree about whether the retrieval happened at all. **Treat
>   every [u] as a lead to verify, never as a number to cite.** They are kept
>   because they are useful hypotheses and deleting them would lose the pointer;
>   they are marked because `SYSTEM.md` law 1 does not have an exception for
>   research documents.
>
> Nothing in this document's *conclusions* rests on a [u] alone. Where an
> argument would have leaned on one, §1.3.2 and §1.4.2 say so explicitly and
> fall back to a [V] or [M].

### 1.0 The hardware, measured today, not quoted from a spec sheet

All numbers below were measured on this box on 2026-08-09 at `nice 19` with
`OMP_NUM_THREADS=2`, the setting `scripts/ladder_loop.sh` runs under, while the
box was carrying its normal tenant load (`load average: 1.99`).

```
CPU              Ampere Neoverse-N1 (aarch64), 4 shared cores, 22 GB RAM
fp32 dense GEMM  1 thread  38.1 GFLOP/s
                 2 threads 74.4 GFLOP/s
                 4 threads 73.8 GFLOP/s     <-- NO GAIN from the 3rd/4th core
```

**The 4th core is not ours.** Two tenant-carrying cores are already busy, so
every cost estimate in this document assumes **two usable threads and ~74
GFLOP/s peak**, and real transformer inference lands well under that.

Two constraints that bound everything else, and which no survey paper will tell
you:

- **`/data` is nearly full, and how full is not a constant.** Two readings
  forty minutes apart on 2026-08-09: **725 MB free (100 %)** and then **4.8 GB
  free (96 %)**. `HF_HOME=/data/caches/huggingface` (`scripts/ladder_loop.sh:81`)
  holds 8 GB; `/data/history` holds 73 GB and belongs to other tenants. So the
  budget is not a number to plan against, it is a **shared, volatile resource
  that can drop to ~700 MB without warning.** Any spec that downloads weights
  must (i) size for the worst observed case, (ii) verify free space before
  fetching, and (iii) not assume the cache survives to the next run. This is a
  *stronger* constraint than a fixed small quota, because a mid-download ENOSPC
  is how you get a corrupted cache rather than a clean failure. → **escalate to
  `docs/DECISIONS_NEEDED.md`**: pin a hearing model cache with a hard size cap,
  or move `HF_HOME` to `/` (6.2 GB free, and `ladder_loop.sh` refuses to start
  below 3 GB there).
- **Nothing audio-related is installed.** `/data/venvs/jackthelearner` has
  torch 2.8.0+cpu, transformers 4.57.6, numpy, scipy, mujoco. It has **no**
  `faster_whisper`, `ctranslate2`, `onnxruntime`, `torchaudio`, `soundfile`,
  `librosa`, `sklearn`, `speechbrain`. Every arm below carries an install cost
  as well as a runtime cost, and aarch64 wheel availability is a **go/no-go**
  that must be checked before a spec is written, not after it fails. It is
  checked below.

#### Installability go/no-go, queried against PyPI 2026-08-09

The venv is **Python 3.9.25 / aarch64 / glibc 2.34**. Python 3.9 is old enough
that upstream projects are actively dropping it, and this silently deletes arms:

| package | cp39-aarch64 wheel? | size | verdict |
|---|---|---|---|
| `ctranslate2` 4.8.1 | **yes** (`manylinux_2_27_aarch64`) | 16.5 MB | **GO** — faster-whisper's backend runs here |
| `faster-whisper` 1.2.1 | pure Python | 1.1 MB | **GO** |
| `onnxruntime` **1.28.0** (latest) | **NO** — cp311+ only | — | **must pin** |
| `onnxruntime` **1.19.2** | yes | 11.5 MB | **GO, pinned** |
| `vosk` 0.3.45 | yes (`py3-none-manylinux2014_aarch64`) | — | **GO** |
| `silero-vad`, `speechbrain`, `resemblyzer` | pure-Python sdists | small | **GO** (deps permitting) |
| `librosa` 0.11 | sdist; needs `numba`/`llvmlite` | — | **NO-GO on 3.9** — current `numba`/`llvmlite` publish no cp39 wheel at all |

Two consequences that change how the specs must be written:

- **Pin `onnxruntime<=1.19.2`.** Any spec that reaches for the WeSpeaker-ONNX
  embedder, or silero-vad's ONNX path, on the unpinned latest will fail to
  install — and it will fail *at implementation time*, wasting a ladder slot.
- **Do not depend on `librosa`.** Every measurement in this document used
  `torch.stft` + a mel matrix (§1.0) and `scipy`. That is deliberate and it must
  stay that way: `librosa` would drag in a `numba` that does not exist for this
  interpreter. Feature extraction stays in torch.

#### Measured: what audio already costs us

```
ContactAudio.render(), 2 ch @ 16 kHz         4.5 ms per second of audio  (RTF 0.0045)
ContactAudio.step()   over mj_step           +55.6 %  (150 us -> 233 us per step)
   => physics + audio runs at 21x real time at dt = 5 ms
log-mel, 2 ch, 64 bins, 25 ms/10 ms, 1.0 s      4.7 ms  (RTF 0.0047)
mel -> Conv2d stem (167 K params) -> 4 tokens,
   0.5 s window                                  5.6 ms  (RTF 0.011)
RAW waveform, wav2vec2-style 7-layer strided
   conv stem (4.21 M params), 0.5 s window      65.5 ms  (RTF 0.131)
   same, 1.0 s window                          156.5 ms  (RTF 0.157)
```

Two things fall straight out and they pre-empt a lot of argument:

1. **Synthesising and rendering the world's audio is free** — 0.45 % of real
   time. The `+55.6 %` on `mj_step` is the Python contact loop in
   `ContactAudioSynth.step` (`mj_contactForce` per contact per step), not the
   synthesis; it is the only audio cost worth optimising and it is still only
   21× real time for physics + audio together.
2. **A raw-waveform stem costs 12–25× a mel stem for the same window.** That is
   a *measured* number on the exact hardware, and it is the cost column that
   decides §4's tie-break. Any raw-waveform arm has to buy that back in accuracy.

### 1.1 ASR — job (a)

*(pending: the ASR survey agent's citations and published ARM RTF / WER numbers.
The measured PyTorch-shape numbers below are ours and stand regardless.)*

#### Measured here: Whisper-shaped encoder cost on this box

I could not download Whisper weights (`/data` was at 725 MB free, and a `pip install` of
`faster-whisper` pulls CTranslate2 + onnxruntime). So instead I **built the
architecture and timed it** — same shapes, same dtype, same threads. This is an
architecture-level measurement, not a proxy: Whisper's encoder is exactly two
Conv1d stems plus N pre-LN transformer blocks over a fixed 1500-frame sequence.

```
                                              30 s window     encoder-only RTF
whisper-tiny  shape (d=384, L=4,  7.6 M enc)     1.18 s            0.039
whisper-base  shape (d=512, L=6, 19.8 M enc)     2.43 s            0.081
whisper-small shape (d=768, L=12, 87.0 M enc)    5.92 s            0.197
              (fp32, torch 2.8 CPU, 2 threads, nice 19)

decoder step, NO KV cache (torch reference path):
  tiny  67.6 ms/step @ prefix 1 ... 87.8 ms @ prefix 32
  base 140.3 ms/step @ prefix 1 ... 188.6 ms @ prefix 32
```

**The load-bearing fact Whisper users keep missing: Whisper always pads to 30
seconds.** A 3-second "come here, Jack" costs a *full* 30-second encoder pass.
So for a typical short command:

| | encoder | ~15 decoded tokens | total | RTF on a 3 s utterance |
|---|---|---|---|---|
| tiny, torch fp32, no KV cache | 1.18 s | ~1.05 s | **~2.2 s** | **~0.75** (unusable live) |
| tiny, torch fp32, KV-cached (est.) | 1.18 s | ~0.1 s | ~1.3 s | ~0.43 |
| tiny, CTranslate2 int8 (est., 3–4× fp32) | ~0.3 s | ~0.05 s | **~0.35 s** | **~0.12** |
| base, CTranslate2 int8 (est.) | ~0.7 s | ~0.1 s | ~0.8 s | ~0.27 |
| small, CTranslate2 int8 (est.) | ~1.7 s | ~0.2 s | ~1.9 s | ~0.63 |

MEASURED: the encoder rows and the uncached decoder rows. ESTIMATED: the
int8/CTranslate2 rows, extrapolated from the measured fp32 shapes by the
commonly-reported 3–4× CTranslate2-int8-over-torch-fp32 factor — **this factor
is the single most load-bearing unverified number in this section and the
bakeoff in §3.1 exists partly to measure it.**

The engineering conclusions are independent of that factor, though:

- **The decoder, not the encoder, is where a naive implementation dies.** A
  67 ms/token uncached step is 20× what a KV-cached step should be. Whatever
  arm wins, it must be a runtime that caches cross-attention K/V
  (faster-whisper, whisper.cpp) — the `transformers` path in
  `AudioListener._transcribe_local:360-366` calls bare `.generate()` and is the
  slowest possible way to run this model.
- **A real VAD is not an optimisation, it is a correctness requirement.**
  `AudioListener._process_chunk` (lines 276-279) uses `rms >
  silence_threshold`, a fixed energy threshold defaulting to 0.01 (line 47). Under it, a door slam, Jack's own footfalls and the fan all open a
  "speech" segment and get sent to Whisper, which hallucinates fluent text on
  non-speech. The fix is a learned VAD gate (silero-vad, ~1–2 MB, sub-millisecond
  per 30 ms frame) **plus** `no_speech_prob` rejection **plus**
  `condition_on_previous_text=False`. Cheap; each is separately falsifiable; the
  spec in §3.1 makes the silence-hallucination test a hard control.
- **Use the `.en` variants.** Jack is spoken to in English; the multilingual
  models spend capacity on 98 other languages and are measurably worse at
  English at the same size.

### 1.2 Speaker identification — job (b)

*(pending: the speaker-ID survey agent's EER / params / license table.)*

The distinction that decides the whole design:

| | question | output | Jack needs it? |
|---|---|---|---|
| **diarization** | who spoke *when*, speakers unknown | `spk_0`, `spk_1`, … + time spans | **no** — anonymous cluster labels cannot fill `EpisodicMemory.speaker`, which needs *"ada"* |
| **verification** | is this the claimed speaker? (1:1) | score + threshold, measured as **EER** | as the *scoring primitive* |
| **identification, open-set** | which of N enrolled, or none? | one of N+1 labels | **yes — this is the job** |

Jack's requirement is the third and it is the least-served by off-the-shelf
pipelines, which are overwhelmingly built for the first. `pyannote.audio`'s
headline product is diarization; the useful part for us is the *embedding model
inside it*, not the pipeline around it.

The architecture that follows:

```
speech segment (from VAD)
   -> speaker embedding model (frozen, ~192-256 dim)
   -> cosine against N enrolled centroids
   -> argmax; if max cosine < tau  ->  "unknown"
   -> EpisodicMemory.record(channel="heard", speaker=<name or "unknown">, ...)
```

Four design commitments, each of which becomes a gate in §3.2:

1. **Enrolment is a few tens of seconds per person, on disk, human-inspectable.**
   Same discipline as `OwnerProfile.py` / the JSONL diary. A speaker profile is
   `{name, centroid[192], n_enrol_utts, enrol_seconds, created_at}`.
2. **"Unknown" is a first-class answer**, exactly as abstention is in
   `EpisodicMemory` (`abstain_below`, and ME.1's fabricated-event rejection). A
   misattribution is strictly worse than an abstention: an event filed under the
   wrong name **poisons the diary permanently** and every later recall repeats
   the error with confidence. `docs/LESSONS.md` already carries the matching
   lesson — *"abstention is a feature; abstaining on everything is a bug"* — so
   the spec must gate on **both** `misattribution_rate` (low) and
   `abstain_rate` (not saturated).
3. **The threshold `tau` is calibrated on a held-out split, never on test.**
   Otherwise the open-set number is fitted, and the "unknown" reject is theatre.
4. **Channel is the confound that will fake a pass.** If a speaker is enrolled
   and tested from the *same recording session*, a classifier can win by
   matching microphone, room and noise floor rather than voice. Every
   speaker-ID number in this project must be **cross-session** or it is
   worthless. This is the direct analogue of PG.5's circularity guard, and §3.2
   makes it a control that must fail.

### 1.3 Sound events — job (c), and the null hypothesis that it is not needed

#### 1.3.1 The candidates, ranked honestly

Citation hygiene follows `UNIFIED_BRAIN_BAKEOFF.md`: **[V]** = the number was
fetched from the paper, repo or HF API during this research pass; **[e]** =
estimated or extrapolated, with the basis named.

| model | AudioSet mAP | params | disk | tokens / 10 s | license (code / weights) |
|---|---|---|---|---|---|
| YAMNet | 0.306 [V] | 3.7 M | **4.1 MB** [V] | 10 patches | Apache-2.0 / Apache-2.0 |
| **EfficientAT `mn04_as`** | 0.432 [V] | **0.98 M** | **4.1 MB** [V] | 1000 fr | MIT / MIT |
| **EfficientAT `mn05_as`** | 0.443 [V] | 1.43 M | **5.9 MB** [V] | 1000 fr | MIT / MIT |
| EfficientAT `mn10_as` | 0.471 [V] | 4.88 M | 19.7 MB [V] | 1000 fr | MIT / MIT |
| **CED-tiny** | **0.481** [V] | **5.5 M** | **22.0 MB** [V] | **248** | Apache-2.0 / Apache-2.0 |
| PANNs CNN14 | 0.431 [V] | 80.8 M | 327 MB [V] | 1000 fr | MIT / **CC-BY-4.0** |
| AST | 0.459 [V] | 87 M | 346 MB [V] | 1188–1212 | **BSD-3** / BSD-3 |
| PaSST-S | 0.476 [V] | ~86 M | ~330 MB | ~1200 | Apache-2.0 |
| BEATs iter3+ | 0.486 [V] | 90 M | ~350 MB | 496 | MIT repo / **weights unlicensed** |
| LAION-CLAP | — (zero-shot) | 153.6 M | **615 MB** [V] | — | CC0 / Apache-2.0 |
| MS-CLAP 2023 | — (zero-shot) | ~172 M | 690 MB [V] | — | MIT / MS-PL |

Two corrections to the brief's premise, both worth having: **CED-tiny and
`mn04_as` dominate YAMNet on every axis** (CED-tiny: +0.175 mAP for 1.6× the
size, and 5× fewer tokens than AST), and **BEATs' checkpoints are not covered by
the repo's MIT licence** — they live outside the source tree.

CPU cost, MEASURED where it exists. The closest published analogue to our
hardware is a Raspberry Pi 4B (4× Cortex-A72); our Neoverse-N1 is meaningfully
faster per core, so these are conservative:

```
mn05_as / mn10_as, ONNX      < 0.25 s per 10 s clip   Pi 4B      [V] arXiv 2509.14049
PANNs CNN6 / CNN9 / CNN13     ~1 s   per 10 s clip    Pi 4B      [V] same
ConvNeXt / Wavegram / ResNet54 2-3 s per 10 s clip    Pi 4B      [V] same
PANNs CNN14, LiteRT           220 ms (99 ms of which is the log-mel front end!)
                                                      Pixel 8a   [V]
AST / PaSST / BEATs on CPU    no published wall-clock anywhere              [e]
CLAP encoders on CPU          no published wall-clock anywhere              [e]
```

Note the 45 % figure: **the log-mel front end was 99 ms of PANNs' 220 ms.** In
the SELD literature it reaches 43.4 % of the real-time budget on a Pi 3. Feature
extraction is a first-class cost, not a rounding error — which is consistent
with our own measurement in §1.0 (mel 4.7 ms vs conv stem 6.2 ms for a 1 s
window: the front end is *comparable to the whole encoder*).

#### 1.3.2 The decisive finding: AudioSet is starved and noisy at exactly Jack's sounds

**The corroborated core, [V].** Clip counts were computed directly from the
official 2,041,789-row `unbalanced_train_segments.csv`, and reproduced
independently across three research passes. Jack's sounds are the *rarest*
classes in AudioSet:

| class Jack needs | unbalanced train clips | rank / 527 |
|---|---|---|
| **Creak** | **29** | **525** |
| Crushing | 56 | 519 |
| Squeak | 72 | 513 |
| Bouncing | 94 | 506 |
| Crack | 142 | 499 |
| Knock | 202 | 486 |
| Bang | 240 | 474 |
| Shatter | 247 | 470 |
| Scrape | 297 | 454 |
| Thunk | 314 | 448 |
| Breaking | 346 | 439 |
| Slam | 724 | 364 |
| Splash, splatter | 818 | 344 |
| Walk, footsteps | 1,563 | 263 |
| Thump, thud | 1,680 | 243 |
| Roll | 1,928 | 216 |
| Wood | 3,128 | 141 |
| *(reference)* Music / Speech | 999,366 / 999,421 | 2 / 1 |

Median class: 1,558 clips. **`Creak` — the sound GOAL.md names first — has 29
training clips in a two-million-clip corpus, rank 525 of 527.** Six of Jack's
target classes sit in the bottom 25. And every one of them has exactly the
**60-clip floor in the eval split**, so any per-class AP for them is estimated
from ~60 weakly-labelled YouTube clips and is statistically noisy before it is
anything else.

Two structural problems compound the scarcity, both [V]:

- **Weak labels.** An AudioSet label means "this class occurs somewhere in this
  10 s clip." `ContactAudio`'s entire signal is a **0.30 s ring**
  (`VOICE_SECONDS = 0.30`), so the label-to-event ratio is ~30:1 and the pretext
  task never required temporal localisation. This is the motivation for
  **AudioSet-Strong** (Hershey et al., ICASSP 2021): re-annotating 67 k clips at
  ~0.1 s resolution moved ResNet-50 d′ from **1.13 → 1.41** on strong eval.
- **Ontology granularity mismatch.** AudioSet has `Wood`, but no
  *wood-on-wood impact* vs *wood-on-stone scrape*. A physics sim's natural label
  space is material-pair × interaction-type × energy. That space does not exist
  in the ontology, at any AP.

**The uncorroborated extension, [u] — do not cite.** One research pass reported
per-class AP values and a label-quality audit that, if real, would be a much
sharper version of the same argument: `Scrape` AP 0.057 (rank 519/527), `Crack`
0.053 (521st), `Creak` 0.074 with **11 %** label accuracy, `Roll` 0.190 with
**0 %** label accuracy, against `Splash` 0.432 / 90 % and `Thump, thud` 0.297 /
80 %; aggregate physics-subset mAP 0.332 vs 0.431 overall; and
Spearman(AP, label quality) = 0.48 on the physics subset vs 0.04 across all 527,
while Spearman(AP, log clip count) = −0.05. **Two later passes from the same
research reported being unable to obtain any per-class AP table at all**, noting
that PANNs publishes it only as a figure and AST/PaSST/BEATs publish none. The
two accounts cannot both be right. **The argument above does not need them** —
29 clips at rank 525 is sufficient on its own — so they are recorded as a lead
and excluded from every conclusion. Anyone who wants these numbers should
generate them: run a pretrained CNN14 or CED over the AudioSet eval segments and
compute AP per class. That is a few GPU-hours on Kaggle and it would settle it.

**And one finding that condemns code we already have, [V].** Self-supervised
*speech* models transfer badly to environmental sound: on ESC-50, wav2vec2
scores far below AudioSet-supervised models, and PANNs' own transfer table shows
frozen CNN14 + one linear layer at **0.918 vs 0.833 for a from-scratch CNN**,
while the same frozen embeddings *collapse* on acoustic-scene (0.589 vs 0.691)
and emotion (0.397 vs 0.692) tasks. `UnifiedBrain.AudioEncoder` (line 1020)
loads `facebook/wav2vec2-base-960h` as its ambient-sound encoder — the wrong
family for this job, and the transfer table also warns that frozen AudioSet
features only help when the target task is *"which object made this sound"*,
which is at least the right shape for Jack.

One operational note that only shows up in the ARM measurements and matters for
an always-on shared box, [V]: on a Pi 4B, PANNs models **stabilise at ~79 °C
after 8 minutes and their latency drifts 0.5 s → 0.6 s purely from clock
throttling**, while MobileNetV2/CNN6-class models stay under 65 °C. A tower
resident in Jack's loop is a sustained thermal load on hardware shared with
paying tenants, not a burst cost.

#### 1.3.3 The null hypothesis, now with evidence behind it

**No pretrained AudioSet
tower earns its parameters in Jack's world.**

`SYSTEM.md` law 3 says decisions are made by bakeoff, never by argument — so
this is **not** settled below. It is settled by **arm A6 of HR.6** (§4.2), a
frozen CED-tiny tower wired in at the same token count as every other arm. That
costs one extra arm in a bakeoff that is running anyway, instead of a separate
bakeoff, and it means the null hypothesis can lose. What follows is the *prior*,
and why A6 is expected to be the arm that dies:

- `ContactAudio` emits sound from a **four-mode free-bar resonator bank**
  (`MODE_RATIOS = (1.0, 2.76, 5.40, 8.93)`, `ContactAudio.py:49`) with
  `f0 = clip(180/char_size, 80, 4000)` and a constant-power pan. That is a
  *three-parameter* family: (f0, amplitude, pan). Every sound in Jack's world
  today is a point in that space.
- The ground-truth label is **already exact and free** — `AudioEvent` carries
  the voiced geom, the force, the azimuth, the elevation and the distance
  (`ContactAudio.py:57-71`). There is nothing for a classifier to *discover*.
- YAMNet / PANNs / AST / BEATs / PaSST were trained on YouTube. They have never
  heard a synthetic 2571 Hz four-partial exponential ring.
- **§1.3.2 makes it quantitative.** The classes GOAL.md names are the *worst*
  classes in AudioSet: `Scrape` 519th of 527, `Crack` 521st, `Creak` with 89
  clips and 11 % label accuracy, `Roll` with 0 % label accuracy. Transfer from
  a tower is transfer from labels that are mostly wrong.
- **And the disk says no.** PANNs CNN14 and BEATs are ~330–350 MB each, against
  a `/data` free space that was observed as low as 725 MB.

The honest counter-case, which the spec must leave room for: the moment Jack
hears a **real microphone** (`AudioListener`) rather than the synth, the
distribution is real-world audio and a pretrained tower becomes the obvious
choice — and the *frozen-embeddings-plus-small-head* route is well supported
there (PANNs CNN14 frozen + one linear layer reaches **0.918 on ESC-50 vs 0.833
trained from scratch** [V]). That is not on the ladder today, and the same
source shows the direction reverses under domain mismatch (on DCASE19-T1 and
RAVDESS, frozen PANNs is *much worse* than scratch: 0.589 vs 0.691, 0.397 vs
0.692 [V]) — which is precisely Jack's situation. The decision is therefore
**"no tower now; revisit when a real-microphone, no-ground-truth task exists,
and prefer CED-tiny or `mn04_as` over YAMNet when that day comes"**, recorded so
it can be re-opened rather than silently reinvented (`bakeoff.py`'s third
property).

Two specific notes worth carrying forward:

- **CLAP zero-shot: do not ship it.** Its one impact-sound datapoint is flat
  (§1.3.2), template choice swings it 5.5–8.0 points, and it costs 615–690 MB.
  What *is* interesting is that acoustic properties (RT60, LUFS, relative pitch)
  are **linearly recoverable from frozen CLAP embeddings** [V, arXiv 2607.03806]
  — the physics is in there, it is just unreachable by text prompting. That
  argues for frozen-features-plus-head, never for zero-shot.
- **The honest ceiling.** Across all 33 models on HEAR's *Vocal Imitations*
  task, the best score is **0.227** [V]. When the sound→label mapping is
  non-lexical, every frozen embedding fails. Do not expect a tower to name
  Jack's materials.

### 1.4 How audio enters the brain — representation, and the bearing problem

| representation | tokens / s | measured cost, 0.5 s window, 2 threads | preserves bearing? |
|---|---|---|---|
| raw waveform, wav2vec2-style strided conv stem (7 layers, 4.21 M) | ~50 | **65.5 ms** | yes in principle (ITD *and* ILD survive) |
| **2-channel log-mel (64 bins, 10 ms hop) + conv stem (167 K)** | 100 frames → pooled to **4 tokens** | **5.6 ms** | **yes, via ILD — measured below** |
| discrete codec tokens (EnCodec 24k / WavTokenizer-40 / Mimi) | 300 / 40 / 100 | encoder resident; no published ARM RTF | **predicted no — see below** |
| hand-crafted event vector `(t_onset, f0, level, pan)` | 1 per event | ~0 | exactly, by construction |

#### 1.4.1 What the literature settles

- **Fixed log-mel wins; learnable front-ends failed replication.** EfficientLEAF
  (arXiv 2207.05508 [V]) evaluated LEAF and EfficientLEAF on six audio
  classification tasks and found *"both fail to consistently outperform a fixed
  mel filterbank"* — a direct replication failure of LEAF's original claim.
  SincNet and Wavegram are in the same family. **Do not learn the front end.**
- **64 mel at a 10 ms hop is the efficient corner, and the number is measured.**
  EfficientAT's `mn10_as` release encodes its own ablation: 128→64 mel costs
  **1.0 mAP**, 10→20 ms hop costs **1.5 mAP**, and going the other way buys
  almost nothing (256 mel = +0.3) [V]. `UNIFIED_BRAIN.md` §4 already specified
  64-mel × 2 ch; this is the number that justifies it rather than the taste.
- **At Jack's data scale, use a CNN stem, not a transformer.** On ESC-50 with
  ~40 samples per class, the best tiny transformer reached **67.71 %** against a
  CNN's **88.50 %** [V, arXiv 2103.12157]. Separately, AST from scratch on
  balanced AudioSet (~20 k clips) scores **0.148 mAP** vs 0.347 pretrained [V].
  Transformers need either data or a pretrained init; Jack's audio stem has
  neither. The trunk can be attention; the *stem* should be convolutional.
- **Discrete-vs-continuous is genuinely contested, and mostly settled against
  discrete.** DASB (arXiv 2406.14294 [V]): *"Across all domains and tasks,
  continuous representations outperform discrete tokens"* — but it is
  **speech-only** by its own admission. The counter-evidence (arXiv 2309.10922
  [V]) finds EnCodec tokens *"within 1 % of mel-spectrogram features"* on
  average. Nobody has trained a small from-scratch transformer on
  *environmental* codec tokens with < 1000 h; that gap is real and it is not
  Jack's job to close it.
- **If tokens are used at all, use a single-quantiser codec.** MusicGen's
  Table 4 [V] measures the RVQ interleaving problem directly: **parallel is
  catastrophic** (FAD 2.58 vs flattening's 0.86), delay recovers ~90 % of
  flattening at 1/K the sequence length, flattening multiplies context by K.
  **WavTokenizer-40** (arXiv 2408.16532 [V]) makes the question disappear: one
  quantiser, V=4096, 40 tokens/s, MIT. Its caveat is measured and serious —
  PESQ collapses **2.17 → 1.14** out of domain (speech→music) [V], and Jack's
  impacts are further out of domain than music.
- **EnCodec is the only codec with a published CPU RTF**: 9.8× real time
  encoding, single thread, on a 2019 i7 — **but 1.6× with entropy coding on**
  [V]. We want raw RVQ indices, not a bitstream; that path must be off. A
  third-party Mimi measurement swings **~40× on runtime choice alone** (0.6× RT
  in PyTorch eager on an Android CPU vs ~26× with ONNX).

#### 1.4.2 The bearing problem, and why 2-channel stereo is the right scope

The strongest single citation for `ContactAudio`'s design is one that reads at
first like a criticism of it. Wilkins et al. (arXiv 2309.13343 [V]) run the
identical 604.5 K-param SELD baseline on FOA, binaural and plain stereo:

| input | localisation error | front→front | **front→back** | right→right |
|---|---|---|---|---|
| FOA (4 ch) | **16.9°** | 0.67 | **0.00** | 0.91 |
| binaural (2 ch) | 30.1° | 0.53 | 0.22 | 0.90 |
| **stereo (2 ch)** | 42.9° | 0.31 | **0.48** | **0.93** |

Read the last column against the fourth. **Stereo's lateral accuracy matches
FOA (0.93 vs 0.91) while 48 % of front sources land in the back.** DCASE has
formally conceded the point: **DCASE 2025 Task 3 moved to stereo and
azimuth-only**, citing front-back and top-bottom ambiguity.

That is exactly the scope `ContactAudio` documented for itself in 2026-08:
*"Panning encodes left/right ONLY: front-back disambiguation needs ITD/spectral
cues (future work)."* PG.5 correspondingly tests **folded** azimuth in
[−90°, 90°]. The fixture is not under-ambitious; it is at the same operating
point the field has ratified for two-channel audio, and the peer-reviewed
numbers say lateral accuracy is the part that survives.

The consequences for the stem:

- **ILD is fully retained by log-magnitude; ITD/IPD is destroyed by it.** The
  DCASE 2025 stereo baseline uses **log-mel and nothing else** — no IPD, no
  GCC — and reaches DOAE 24.5° [V]. For Jack this is not even a trade-off:
  `ContactAudio` synthesises **no interaural time difference at all** (both
  channels receive the identical `sig`, scaled by `gL`/`gR`;
  `ContactAudio.py:188-195`), so a phase-preserving front end has *nothing extra
  to preserve*. **GCC-PHAT, SALSA-Lite's IPD channels and every ITD feature are
  identically zero on this fixture** — they would be four channels of noise.
  (SALSA-Lite is otherwise excellent — 9× faster than log-mel+GCC-PHAT *and*
  more accurate, 0.30 s per 60 s clip [V, arXiv 2111.08192] — and becomes
  relevant only if `ContactAudio` ever grows a propagation-delay model.)
- **Summing to mono destroys bearing irrecoverably**, which is why PG.5's
  `mode="mono"` control fails at ≤ 0.30. A stem whose first operation averages
  the channel dimension *is* the mono control. One line, silently deleting
  Jack's only directional sense — HR.7 (§4.1) is the guard.

#### 1.4.3 Measured here: mel keeps the bearing, but the naive probe throws it away

I ran a miniature HR.7 on this box rather than assert the above. 108 PG.5-style
drops across 12 seeds, 80 ms decode window, 64-mel 2-channel log spectrogram,
bearing recovered from the interaural level difference and scored against PG.5's
own ≤ 10° gate:

```
                                              stereo            mono (control)
PG.5's raw energy decode (the incumbent)      1.00               0.10
naive MEAN over per-bin log-mel ILD           0.69  (med  6.4°)  0.10  (med 40.2°)
pooled mel ENERGY ILD                         0.99  (med  0.0°)  0.10  (med 40.2°)
energy-WEIGHTED per-bin log-mel ILD           1.00  (med  0.0°)  0.10  (med 40.2°)
```

So the headline is confirmed — **a 2-channel log-mel front end preserves bearing
to the full PG.5 gate, and mono destroys it** — but the *way* it nearly went
wrong is the more useful result, and it is a trap HR.7 must be written around.

**The link function is `atanh`, not linear.** Constant-power panning gives
`gL = √((1−p)/2)`, `gR = √((1+p)/2)`, so the log-domain ILD is

```
   log(gR) − log(gL) = ½·log((1+p)/(1−p)) = atanh(p)          exact
```

verified to machine precision at p = ±0.9, ±0.99. Two consequences:

1. **A *linear* probe on log-mel scores 0.40 where the analytic `tanh` link
   scores 1.00.** `atanh` saturates hard near the lateral extremes, so a linear
   readout is systematically wrong exactly where bearing matters most. A
   linear-probe HR.7 would have reported a **false negative on the correct
   representation** and killed the winning arm. The probe must be non-linear, or
   must predict `atanh(p)` and invert.
2. **Averaging per-bin log ILD is the wrong pooling.** The naive mean scored
   0.69 against the energy-weighted 1.00, because the `log(x + 1e-6)` floor
   pins near-silent bins to ~0 ILD and drags the mean toward centre —
   attenuating extreme pans. Pool in the **energy** domain, or weight per-bin
   ILD by bin energy. This is a real and easily-repeated implementation bug:
   the epsilon that keeps the log finite is also what eats the bearing.

Both are instances of `docs/LESSONS.md`'s *"measure the quantity you are
claiming, not a proxy that correlates with it"* — and neither would have been
visible from the architecture diagram.

---

## 2. THE END-TO-END CHAIN: voice → speaker ID → `EpisodicMemory.speaker` → attributed recall

This is the spec that closes §0.1. It is written as a chain because **each link
is individually passing and the composition is untested** — the classic place
for a project to believe something it has not shown.

```
  microphone / corpus utterance
        │
        ▼
  [1] VAD gate            silero-vad; energy VAD is the null baseline
        │  speech segment (>= 1.0 s)
        ├──────────────────────────────┐
        ▼                              ▼
  [2] ASR (frozen)              [3] speaker embedder (frozen)
      faster-whisper .en             ECAPA / WeSpeaker, 192-d
        │  text                       │  e
        │                             ▼
        │                        cosine vs N enrolled centroids
        │                        argmax; < tau -> "unknown"
        │                             │  name
        └──────────────┬──────────────┘
                       ▼
  [4] EpisodicMemory.record(channel="heard", speaker=name, text=text)
                       │
                       ▼
  [5] what_did_they_tell_me("ada", "the ladder")   <-- ME.9's own battery
```

Links [1]–[3] are **outside** the brain and frozen. Link [4] is
`EpisodicMemory.py:124` unchanged. Link [5] is `ME.9` unchanged. **The only new
code is [3] and the profile store**; the value of the spec is that it measures
the composition rather than the parts.

### 2.1 What makes it falsifiable

**The metric is ME.9's own metric, recomputed with the speaker field inferred
instead of handed over.** Nothing else changes — same speakers, same overlapping
topics, same three question forms, same 0.80 bar. ME.9 is the ceiling; the gap
between them is exactly the cost of not knowing who is talking.

Three quantities, all pre-registered:

- `attributed_acc_from_voice` — top-1 retrieval correct on channel **and**
  speaker **and** topic, per channel. Gate: **≥ 0.80 on all three channels**,
  the same bar ME.9 already clears.
- `misattribution_rate` — fraction of heard events filed under a *wrong named*
  speaker. Gate: **≤ 0.02**. This is the number that matters most and it is not
  the complement of accuracy: filing under `"unknown"` is a miss but not a
  poisoning, filing Bruno's words under Ada is a permanent false memory that
  every later recall repeats.
- `unknown_reject_rate` — fraction of *out-of-set* speakers correctly filed as
  `"unknown"`. Gate: **≥ 0.90**, with the *complementary* gate
  `abstain_rate_on_enrolled ≤ 0.15` so that "call everyone unknown" cannot pass.
  (`docs/LESSONS.md`: abstaining on everything is a bug. Both directions must be
  gated or the test is one-sided.)

### 2.2 The nulls

1. **Single-speaker null.** Every heard event recorded under the most frequent
   speaker. With 3 enrolled speakers this scores ~1/3 on `heard` by luck and
   **1.0 on `said` and `did`** (those are always "jack"). This null is why the
   gate must be *per channel* and why the headline number must never be a
   three-channel average — the same stratum-hiding failure `docs/LESSONS.md`
   records for ME.11's registers.
2. **Random-speaker null.** Speaker drawn uniformly from the enrolled set.
   Scores ~1/N on `heard`.
3. **Text-only null (the interesting one).** Infer the speaker from the
   *transcript* with a lexical classifier. ME.9's corpus is built so every topic
   is discussed by at least two parties — but a real conversation is not, and
   real people have idiolects. If the text-only null matches the voice pipeline,
   **the voice channel is decorative** and Jack should just read the words. This
   is the placebo-modality idea from `UNIFIED_BRAIN_BAKEOFF.md` §2(a) applied to
   attribution, and it is the null most likely to embarrass the voice arm.

### 2.3 The control that must invert (mirroring ME.9's swapped provenance)

`me_9_attributed_recall.py:65,78` swaps the *labels* — Jack's lines relabelled
as the speaker's and vice versa — and demands accuracy fall **below half** the
true accuracy. The audio analogue swaps one level lower down:

> **SWAP-THE-VOICES.** Take the enrolled profiles for speakers A and B and
> exchange their *enrolment audio* (A's profile is built from B's voice and vice
> versa). Everything else — transcripts, topics, questions, gold answers, the
> memory store's code, the retrieval weights — is byte-identical.
>
> The pipeline must now file A's utterances under "B". Therefore:
> **every question of the form "what did A tell me about X" must retrieve B's
> event**, and attributed accuracy on the A/B subset must **invert**: measured
> `swapped_acc_AB ≤ 0.10` while `swapped_inverted_acc_AB ≥ 0.80`.

Why the *inverted* accuracy is required and not just a collapse: a collapse is
also what you get if the embedder is simply broken, or if the audio failed to
load, or if everything got filed as "unknown". **Only the inversion shows the
system read the voice and reached the wrong-but-predicted conclusion.** That is
`UNIFIED_BRAIN_BAKEOFF.md`'s SWAP-FLIP argument — *the output moved in the
direction the intervention predicted* — imported into the memory pillar. Under
`docs/LESSONS.md`'s "a control that fails alongside the experiment is a gift":
if both `swapped_acc` and `swapped_inverted_acc` are near zero, the fault is in
the shared audio plumbing, and you have been handed its location.

A second control, cheaper and equally necessary:

> **SESSION-ONLY control (must fail).** Enrol from the same recording session as
> the test utterances. Accuracy will go *up*. That is the point: if
> cross-session and same-session accuracy are indistinguishable, the corpus has
> no session variation and the headline number is measuring a microphone.

### 2.4 The specs

New prefix **`HR`** (hearing), tiers 2–4, in the exact `Spec(...)` form of
`experiments/registry_expansion.py`. `HR.*` is chosen over extending `UB.*`
because `UNIFIED_BRAIN_BAKEOFF.md` has already committed `UB.9`–`UB.16` to the
registry and a collision would be silent. `experiments/run.py:_module_for` globs
`hr_1_*.py` from the spec id, so no runner change is needed; only appending to
`EXPANSION` is.

```python
    # ── HEARING: the corpus fixture ─────────────────────────────────────

    Spec("HR.1", 2, "The voice corpus is honest before anyone is scored",
         hypothesis="A speaker corpus exists on this box with >=8 enrolled and "
                    ">=8 held-out UNKNOWN speakers, disjoint enrolment/test "
                    "utterances, and CROSS-SESSION test material, such that no "
                    "non-vocal channel cue can identify a speaker.",
         falsified_by="A probe on non-vocal features alone (silence-segment "
                      "spectrum, DC offset, noise floor, clip loudness) "
                      "identifies the speaker above chance+5% — then every "
                      "speaker-ID number downstream is a microphone "
                      "measurement, not a voice measurement.",
         null_baseline="Chance = 1/n_enrolled for the channel probe.",
         metric="min_channel_leak_margin", budget=Budget.CPU, seeds=3,
         depends_on=[],
         control="A DELIBERATELY LEAKY variant — enrolment and test drawn from "
                 "the same session — must be identified WELL above chance by "
                 "the same probe. A leak detector that cannot see a planted "
                 "leak has measured nothing (docs/LESSONS.md, T0.13).",
         kills="HR.2, HR.3, HR.4. A speaker experiment on a leaky corpus "
               "measures the leak.",
         notes="Corpus: LibriSpeech dev-clean, CC-BY-4.0, 40 speakers "
               "(20M/20F), ~5.4 h, split 20 enrolled / 20 impostor, PLUS a "
               "handful of owner-recorded utterances for the deployment case. "
               "VERIFIED REACHABLE 2026-08-09: "
               "https://www.openslr.org/resources/12/dev-clean.tar.gz returns "
               "HTTP 200, Content-Length 337,926,286 (338 MB). test-clean is a "
               "further 347 MB and 40 DISJOINT speakers if a larger impostor "
               "set is wanted, but 685 MB exceeds the worst-case free disk — "
               "prefer the 20/20 split of dev-clean alone. REJECTED: VCTK "
               "(110 speakers, multi-session, ideal on paper) is an 11.7 GB "
               "download — verified Content-Length 11,749,118,645 — and does "
               "not fit on this box at any observed free-space level. NOTE THE "
               "DISK: /data free space was observed swinging between 725 MB and "
               "4.8 GB within an hour (shared with other tenants), so the "
               "corpus and the models may not both fit. Check free space "
               "BEFORE fetching and size for the worst case — "
               "cannot both live there — see the escalation in "
               "docs/research/HEARING_BAKEOFF.md section 1.0. LibriSpeech "
               "speakers are single-session per chapter, so cross-session "
               "means cross-CHAPTER at minimum and the control above is what "
               "certifies that it is enough."),

    # ── HEARING: the end-to-end chain ───────────────────────────────────

    Spec("HR.4", 2, "He knows who told him, from the voice alone",
         hypothesis="With the speaker field produced by a voice embedder "
                    "instead of handed to the test, ME.9's attributed-recall "
                    "battery still clears 0.80 on ALL THREE channels "
                    "(heard/said/did), with misattribution <=0.02 and "
                    "unknown-speaker rejection >=0.90.",
         falsified_by="Any channel below 0.80, OR misattribution above 0.02, OR "
                      "unknown-rejection below 0.90, OR abstention on enrolled "
                      "speakers above 0.15 (calling everyone 'unknown' is not a "
                      "pass). Also falsified if the TEXT-ONLY null matches the "
                      "voice pipeline — then the voice channel is decorative "
                      "and Jack should just read the words.",
         null_baseline="THREE nulls, reported per channel, never averaged: "
                       "(i) single-speaker (everything filed under the most "
                       "frequent name — scores 1.0 on said/did by "
                       "construction, which is exactly why the gate is per "
                       "channel); (ii) random speaker from the enrolled set; "
                       "(iii) TEXT-ONLY attribution by a lexical classifier on "
                       "the transcript — the placebo channel.",
         metric="min_channel_attributed_acc_from_voice",
         budget=Budget.CPU_LONG, seeds=3,
         depends_on=["HR.1", "HR.2", "HR.3", "ME.9"],
         control="SWAP-THE-VOICES, mirroring ME.9's swapped-provenance control "
                 "one level lower. Exchange the ENROLMENT AUDIO of speakers A "
                 "and B; leave transcripts, topics, questions and gold answers "
                 "byte-identical. Attribution must INVERT, not merely collapse: "
                 "swapped_acc_AB <= 0.10 AND swapped_INVERTED_acc_AB >= 0.80. A "
                 "collapse alone is also what a broken loader produces; only "
                 "the inversion shows the voice was read and the predicted "
                 "wrong conclusion reached. Second control: same-session "
                 "enrolment must score HIGHER — if it does not, the corpus has "
                 "no session variation and HR.1 passed wrongly.",
         kills="The sentence 'Jack remembers who told him what' as an "
               "end-to-end claim. ME.9 keeps its PASS — it tests the retrieval "
               "contract, and that contract is real — but until HR.4 passes, "
               "the speaker field is supplied by the test harness and by "
               "nothing in the live system (AudioListener.py produces text "
               "only; EpisodicMemory.record takes `speaker` on trust).",
         notes="This is the biggest hole in the memory pillar and it is a "
               "COMPOSITION failure, not a component failure: ME.9 PASSES at "
               "1.0 and every link is individually fine. Ceiling analysis: "
               "end-to-end accuracy ~ ME.9_acc x speaker_id_acc x asr_acc, so "
               "the 0.80 bar needs speaker-ID at ~0.85 cross-session with ASR "
               "near-perfect on the command register. RUN IT TWICE and report "
               "both, so a failure localises to a LINK rather than to 'the "
               "chain': (i) GOLD transcripts + inferred speaker, isolating "
               "HR.3's contribution; (ii) HR.2's transcripts + inferred "
               "speaker, the real system. The gap between them is the ASR tax, "
               "and if it is large the fix belongs in HR.2, not here. "
               "MISATTRIBUTION IS NOT THE COMPLEMENT OF ACCURACY: 'unknown' is "
               "a miss, a wrong NAME is a permanent false memory that every "
               "later recall repeats with confidence. Gate them separately."),
```

---

## 3. The bakeoffs for jobs (a) and (b)

`experiments/bakeoff.py` resolves a TIE by `Arm.cost`, and **returns VOID if any
tied arm left cost undeclared** (`bakeoff.py:161-170`; `docs/LESSONS.md`, *"a
default of zero is not unknown"*). Both bakeoffs below therefore name their cost
unit explicitly, and every arm declares one.

### 3.1 ASR bakeoff — HR.2

**Cost unit: RTF measured on this box at `nice 19`, `OMP_NUM_THREADS=2`, on a
3-second utterance including the 30-second pad.** Lower is better; the tie-break
takes the cheaper arm. Resident MB is reported alongside but is not the
tie-breaker, because free disk (observed as low as 725 MB) is a hard
admission gate rather than a
gradient (an arm whose weights do not fit is not slow, it is impossible).

```python
    Spec("HR.2", 2, "ASR bakeoff: the cheapest transcriber that gets Jack's words right",
         hypothesis="At least one open-weight, locally-runnable ASR arm "
                    "transcribes Jack's command register with word accuracy "
                    ">= 0.90 at RTF <= 0.30 on 2 ARM threads, and beats the "
                    "no-ASR null by >= 3 sigma.",
         falsified_by="Every arm that clears 0.90 accuracy has RTF > 0.30 (no "
                      "live transcription on this box — escalate: batch "
                      "transcription only, or a smaller command grammar), OR "
                      "no arm clears 0.90 (Jack's vocabulary is the problem, "
                      "not the model).",
         null_baseline="A no-ASR transcriber that emits the most frequent "
                       "command string regardless of the audio. Word accuracy "
                       "= the majority-class rate; the learning gate is 3 "
                       "sigma over it.",
         metric="word_accuracy_at_rtf_budget", budget=Budget.CPU_LONG, seeds=3,
         depends_on=["HR.1"],
         control="TWO controls that must fail. (a) SILENCE HALLUCINATION: 60 s "
                 "of room tone and 60 s of ContactAudio impacts, containing no "
                 "speech, must yield ZERO transcribed words. Whisper "
                 "hallucinates fluent text on non-speech and the current "
                 "energy-VAD (AudioListener.py:276, rms > 0.01) opens a speech "
                 "segment for a door slam (AudioListener.py:276-279, "
                 "silence_threshold 0.01). An arm that fails this is "
                 "disqualified whatever its WER. (b) SHUFFLED AUDIO: "
                 "phase-scrambled speech must transcribe to near-nothing — an "
                 "arm that still emits plausible commands is decoding its "
                 "language-model prior, not the audio.",
         kills="The transformers .generate() path in "
               "AudioListener._transcribe_local (lines 360-366) and the entire "
               "_transcribe_api path (line 372) — the latter unconditionally, "
               "because it calls a PAID OpenAI endpoint and SYSTEM.md forbids "
               "paid compute. Deleting it is not contingent on this bakeoff.",
         notes="ARMS, cost = MEASURED RTF on a 3 s utterance at nice 19, "
               "OMP_NUM_THREADS=2 (state MB resident alongside): "
               "A0 faster-whisper tiny.en int8 (cost ~0.12 est, ~75 MB). "
               "A1 faster-whisper base.en int8 (cost ~0.27 est, ~145 MB). "
               "A2 faster-whisper small.en int8 (cost ~0.63 est, ~480 MB - "
               "EXCEEDS the worst-case observed free-disk budget of 725 MB "
               "alongside anything else; "
               "admit only if /data is freed first). "
               "A3 distil-whisper distil-small.en via CTranslate2. "
               "A4 whisper.cpp base.en Q5_0 (no Python deps, NEON). "
               "INSTALLABILITY CHECKED 2026-08-09: ctranslate2 4.8.1 DOES ship "
               "a cp39-aarch64 wheel (16.5 MB) and faster-whisper 1.2.1 is "
               "pure Python, so A0-A3 are runnable on this interpreter. "
               "onnxruntime's latest (1.28.0) has DROPPED cp39 — pin <=1.19.2. "
               "librosa is a NO-GO on Python 3.9 (numba/llvmlite publish no "
               "cp39 wheel); do feature extraction in torch. "
               "A5 vosk-model-small-en-us (~40 MB, streaming, the cheap "
               "reference arm whose FAILURE would indict the task per "
               "docs/LESSONS.md). "
               "Every arm runs behind the SAME silero-vad gate, or the "
               "comparison is a comparison of VADs. "
               "MEASURED HERE 2026-08-09 (torch fp32, 2 threads, nice 19): "
               "whisper-shaped encoders over a 30 s window cost 1.18 s (tiny) "
               "/ 2.43 s (base) / 5.92 s (small), and an UNCACHED decoder step "
               "costs 68 ms (tiny) / 140 ms (base). Whisper always pads to "
               "30 s, so a 3 s command pays the full encoder. The int8 "
               "speedups above are EXTRAPOLATED, not measured; measuring them "
               "is half the point of this spec. "
               "TEST SET: two registers, reported SEPARATELY and gated on the "
               "MINIMUM (docs/LESSONS.md, ME.11's deleted register): "
               "(R1) short imperatives from Jack's actual command grammar "
               "('climb the ladder', 'come here', 'what did ada tell you'); "
               "(R2) PROPER NOUNS - the enrolled speakers' names - which small "
               "Whisper models mangle, and which HR.4 depends on because a "
               "question is addressed to a NAME."),
```

### 3.2 Speaker-ID bakeoff — HR.3

**Cost unit: MB resident (model weights + runtime), measured with the model
loaded and one embedding computed.** Secondary: ms per second of audio. MB is
the tie-breaker because `/data`'s free space — observed as low as 725 MB — is
the binding constraint.

```python
    Spec("HR.3", 2, "Speaker-ID bakeoff: which of the enrolled few, or nobody",
         hypothesis="At least one open-weight speaker embedder gives >= 0.85 "
                    "balanced open-set identification accuracy over "
                    "(N enrolled + unknown) on CROSS-SESSION audio, from "
                    "<= 30 s of enrolment per speaker, with the decision "
                    "threshold calibrated on a held-out split.",
         falsified_by="No arm reaches 0.85 cross-session at <= 30 s enrolment "
                      "— then HR.4's 0.80 end-to-end bar is unreachable and "
                      "the honest options are (i) more enrolment audio, "
                      "(ii) fewer enrolled people, or (iii) Jack ASKS who is "
                      "speaking. Record which, do not quietly lower the bar.",
         null_baseline="Chance = 1/(N+1) with balanced classes. PLUS a "
                       "REFERENCE ARM simple enough that its failure indicts "
                       "the task: nearest-centroid on mean MFCCs. If the "
                       "reference arm also fails, the corpus or the protocol "
                       "is broken, not the models (docs/LESSONS.md, T1.02).",
         metric="open_set_balanced_accuracy", budget=Budget.CPU_LONG, seeds=3,
         depends_on=["HR.1"],
         control="THREE controls. (a) SAME-SESSION enrolment must score "
                 "HIGHER; if it does not, HR.1's corpus has no session "
                 "variation. (b) SILENCE segments must be rejected as "
                 "'unknown' >= 0.95 of the time — an embedder that confidently "
                 "names a speaker from room tone is scoring the channel. "
                 "(c) THRESHOLD SENSITIVITY: sweep tau and report the full "
                 "curve. An arm whose accuracy is flat in tau has an "
                 "'unknown' class that is not doing anything, and its "
                 "open-set number is a closed-set number wearing a hat.",
         kills="Five of six embedders. The survivor is what writes "
               "EpisodicMemory.speaker; the rest are deleted, not kept.",
         notes="ARMS, cost = MB RESIDENT (weights + runtime), measured with "
               "one embedding computed; ms/s of audio reported alongside. "
               "MB is the tie-breaker because /data free space was observed as "
               "low as 725 MB. "
               "A0 mean-MFCC nearest centroid (~0 MB, scipy only) - the "
               "reference arm. "
               "A1 SpeechBrain ECAPA-TDNN spkrec-ecapa-voxceleb (192-d). "
               "A2 WeSpeaker ResNet34 via ONNX Runtime (the embedder inside "
               "pyannote 3.1; ONNX avoids the speechbrain dependency tree). "
               "PIN onnxruntime<=1.19.2 — checked 2026-08-09, the latest "
               "(1.28.0) publishes no cp39-aarch64 wheel and this venv is "
               "Python 3.9.25. "
               "A3 x-vector spkrec-xvect-voxceleb - the cheap deep arm. "
               "A4 Resemblyzer / GE2E d-vector (256-d, ~17 MB) - cheapest deep "
               "arm; expected worse EER, may still clear 0.85 on N<=8 speakers, "
               "which is the ONLY question that matters here. "
               "A5 CAM++ / ERes2NetV2 (3D-Speaker) if an ONNX export exists. "
               "DIARIZATION IS NOT AN ARM. pyannote's pipeline answers 'who "
               "spoke when' with anonymous cluster ids; EpisodicMemory.speaker "
               "needs a NAME. Only the embedder inside it is a candidate. "
               "ENROLMENT LENGTH IS AN AXIS, NOT A CONSTANT: report accuracy "
               "at 5 s / 15 s / 30 s / 60 s per speaker and gate on 30 s. What "
               "a person will actually sit still for is the real constraint. "
               "N IS AN AXIS TOO: report at N = 2, 4, 8 enrolled. Jack needs a "
               "household, not VoxCeleb."),
```

---

## 4. The bakeoff for job (c): how contact audio enters the brain

This is the arm of `UB.10` that decides the audio stem, run **before** `UB.10`
so the fusion bakeoff is not simultaneously a representation bakeoff. It obeys
`UNIFIED_BRAIN_BAKEOFF.md`'s constraints exactly: **token count is equalised
across arms** (or the comparison is a comparison of token budgets, per
2601.16667), and the empirical null for "decorative" is the **placebo modality**
— a matched-noise channel with identical token count, encoder capacity and
dropout rate — not an assumed zero.

Two specs, deliberately separated: **HR.7 is a cheap fixture that can kill an
arm before it trains** (does the stem still know left from right?), and **HR.6
is the bakeoff**.

### 4.1 HR.7 — bearing survives the encoder

```python
    Spec("HR.7", 2, "The audio stem does not deafen him to direction",
         hypothesis="A probe on the audio STEM's output tokens recovers the "
                    "source lateral angle to within 10 degrees on >= 0.9 of "
                    "PG.5's drop events — the same gate PG.5 applies to the raw "
                    "stereo signal. THE PROBE MUST NOT BE LINEAR IN THE "
                    "LOG-MEL: the constant-power pan law makes the log-domain "
                    "interaural level difference exactly atanh(p), so a linear "
                    "readout saturates at the lateral extremes. MEASURED HERE "
                    "2026-08-09 on 108 PG.5-style drops: linear probe 0.40, "
                    "analytic tanh link 1.00, mono control 0.10. A linear-probe "
                    "version of this spec would report a FALSE NEGATIVE on the "
                    "correct representation and kill the winning arm.",
         falsified_by="Any candidate stem whose tokens lose bearing. Directional "
                      "hearing is the ONLY thing PG.5 certifies, and it is what "
                      "makes audio useful for ACTION (turn toward the sound); a "
                      "stem that discards it reduces audio to an event detector.",
         null_baseline="PG.5's own mono render fed through the same stem — "
                       "bearing must be undecodable (<= 0.30), which is the "
                       "same bar PG.5's mono control clears.",
         metric="stem_bearing_probe_accuracy", budget=Budget.CPU, seeds=3,
         depends_on=["PG.5"],
         control="CHANNEL-SWAPPED input (L and R exchanged) must INVERT the "
                 "probe's sign on >= 0.9 of events, not merely degrade it. A "
                 "degradation is also what a broken probe produces; only the "
                 "inversion shows the stem read the interaural difference.",
         kills="Any stem in HR.6 that fails, before it is ever trained. Named "
               "prediction, pre-registered: the DISCRETE-TOKEN arm fails this. "
               "ContactAudio encodes bearing purely as interaural LEVEL "
               "difference (ContactAudio.py:188-195 applies gains gL/gR to the "
               "IDENTICAL signal — there is no interaural TIME difference at "
               "all), and RVQ codecs quantise a few-dB level offset inside a "
               "single codebook cell. If it passes, that prediction was wrong "
               "and this document is wrong with it.",
         notes="Log-mel preserves bearing provably: the pan law is a "
               "per-channel GAIN, so log(gR) - log(gL) = atanh(p) exactly "
               "(verified to machine precision at p = +-0.9, +-0.99), a "
               "constant offset between the two channels' log-mel planes, "
               "independent of mel bin. TWO IMPLEMENTATION TRAPS, both "
               "MEASURED here 2026-08-09 and both of which silently destroy "
               "the number: (1) a LINEAR probe scores 0.40 where the analytic "
               "tanh link scores 1.00, because atanh saturates exactly where "
               "bearing matters most; (2) the naive MEAN of per-bin log ILD "
               "scores 0.69 against 1.00 for energy-weighted pooling, because "
               "the log(x + 1e-6) floor pins near-silent bins to zero ILD and "
               "drags the mean toward centre. Pool in the ENERGY domain. The "
               "architectural failure mode is separate and worse: a stem whose "
               "first op averages over the CHANNEL dimension IS the mono "
               "control, silently. One line, deleting Jack's only directional "
               "sense, and nothing else in the ladder would notice. This spec "
               "is that guard. Scope note: two-channel stereo loses front/back "
               "(48% of front sources land in the back, arXiv:2309.13343) but "
               "its LATERAL accuracy matches 4-channel FOA (0.93 vs 0.91) - "
               "DCASE 2025 Task 3 moved to stereo, azimuth-only for exactly "
               "this reason, so PG.5's folded-azimuth scope is the field's "
               "ratified operating point, not a shortcut."),
```

### 4.2 HR.6 — the representation bakeoff, with the no-audio null

**Cost unit: milliseconds per 0.5 s audio window, measured on this box at
`nice 19`, `OMP_NUM_THREADS=2`** (the numbers in §1.0 are the first three arms'
costs, already measured). Params reported alongside.

```python
    Spec("HR.6", 4, "How contact audio enters the brain: mel vs raw vs tokens vs nothing",
         # ARMS: A0 no-audio | A0b placebo | A1 raw | A2 mel | A3 mel+ILD
         #       A4 discrete tokens | A5 hand-crafted event vector
         #       A6 frozen CED-tiny tower   (see notes for costs)
         hypothesis="At matched tokens-per-modality, matched trainable "
                    "parameters (+-5%), matched steps and matched data order, "
                    "at least one audio representation beats BOTH the "
                    "NO-AUDIO ablation and the PLACEBO-AUDIO channel by >= 3 "
                    "sigma on the audio-dependent battery, and the ranking is "
                    "stable across 3 paired seeds.",
         falsified_by="Every audio arm ties the PLACEBO channel — hearing is "
                      "decorative at this scale and does not earn its "
                      "parameters (the Tier-3 rule; report it, do not re-run "
                      "until it looks better). OR: the hand-crafted EVENT-VECTOR "
                      "arm ties every learned encoder — which indicts the "
                      "FIXTURE, not the brain, and sends the work to section 5 "
                      "of docs/research/HEARING_BAKEOFF.md rather than to a "
                      "bigger model.",
         null_baseline="TWO nulls, and the second is the load-bearing one. "
                       "(i) NO-AUDIO: the audio stem removed entirely, "
                       "parameters returned to the trunk so total capacity "
                       "matches. If the brain performs identically without "
                       "hearing, audio has not earned its parameters. "
                       "(ii) PLACEBO AUDIO: a matched-noise channel with the "
                       "SAME token count, encoder capacity and dropout rate, "
                       "wired in exactly like the real one "
                       "(UNIFIED_BRAIN_BAKEOFF.md section 2a). Its spread ACROSS "
                       "SEEDS is the empirical null distribution for "
                       "'decorative', re-estimated every run rather than "
                       "assumed to be zero.",
         metric="audio_margin_over_placebo", budget=Budget.GPU, seeds=3,
         depends_on=["HR.7", "PG.5", "PG.7"],
         control="Every surviving arm must FAIL the cross-episode SWAP "
                 "ablation: swapping the audio stream between episodes, "
                 "preserving both marginals and the temporal statistics, must "
                 "hurt. Swap is the only perturbation that isolates "
                 "CORRESPONDENCE, which is what binding means. An arm "
                 "invariant to swapping has learned the audio MARGINAL and its "
                 "score is uninterpretable. Second, opposite-direction "
                 "control: the PLACEBO column must be SMALL. A large placebo "
                 "delta means the ablation procedure is measuring "
                 "off-manifold shock rather than information, and every other "
                 "column in the matrix is void.",
         kills="Four of six audio front-ends, and possibly the audio modality "
               "itself. Also kills UnifiedBrain.AudioEncoder's wav2vec2 path "
               "if A4 loses: wav2vec2 is trained on 960 h of read English "
               "speech and Jack's audio is four-partial exponential rings.",
         notes="ARMS, cost = MEASURED ms per 0.5 s window at nice 19, "
               "OMP_NUM_THREADS=2 (params alongside). TOKEN COUNT EQUALISED AT "
               "4 for every arm (arXiv:2601.16667 - unequal token budgets make "
               "this a comparison of token budgets). "
               "A0 NO-AUDIO ablation - the null. cost 0. "
               "A0b PLACEBO AUDIO - matched noise, 4 tokens, same capacity. "
               "cost = A2's. "
               "A1 RAW WAVEFORM, wav2vec2-style 7-layer strided conv stem. "
               "MEASURED 65.5 ms / 0.5 s window, 4.21 M params - 12x A2. "
               "A2 2-CHANNEL LOG-MEL (64 bins, 25 ms/10 ms) -> Conv2d stem -> "
               "4 tokens. MEASURED 5.6 ms / 0.5 s window, 167 K params. The "
               "incumbent recommendation from UNIFIED_BRAIN.md section 4. "
               "A3 A2 + EXPLICIT BEARING FEATURES: per-band interaural level "
               "difference appended as extra channels. Cheap; tests whether "
               "the stem needs help finding what PG.5 proved is there. "
               "A4 DISCRETE TOKENS (EnCodec / WavTokenizer / Mimi encoder, "
               "frozen). Predicted to fail HR.7 before it gets here, and its "
               "encoder weights may not fit in the worst-case 725 MB of free "
               "/data. "
               "A6 FROZEN PRETRAINED TOWER: CED-tiny (5.5 M, 22 MB, 0.481 "
               "AudioSet mAP, Apache-2.0) embeddings -> 4 tokens. This is the "
               "arm that decides section 1.3's null hypothesis BY BAKEOFF "
               "rather than by argument (SYSTEM.md law 3). CED-tiny, not "
               "YAMNet: it is +0.175 mAP at 1.6x the size with 5x fewer "
               "tokens than AST. Predicted to lose, and the prediction is "
               "specific enough to be wrong: the AudioSet classes Jack needs "
               "are the WORST in the dataset (Scrape rank 519/527, Crack "
               "521st, Creak 29 training clips and 11% label accuracy, Roll 0% "
               "label accuracy), impacts occupy a ~3% duty cycle inside 10 s "
               "weakly-labelled clips so the pretext task never taught "
               "temporal localisation, and ContactAudio's output is a "
               "three-parameter synthetic family no YouTube tower has heard. "
               "If A6 WINS, section 1.3 of "
               "docs/research/HEARING_BAKEOFF.md is wrong and the tower ships. "
               "NOT AN ARM: wav2vec2, which UnifiedBrain.AudioEncoder:1020 "
               "currently loads. On HEAR it scores 0.561 on ESC-50 against "
               "PANNs' 0.909 and CED-base's 0.967 — self-supervised SPEECH "
               "features are 20-40 points worse than AudioSet features on "
               "environmental sound. It is excluded on measured evidence, not "
               "taste. "
               "A5 HAND-CRAFTED EVENT VECTOR (t_onset, f0, level, pan) from "
               "ContactAudio's own labels, projected to 4 tokens. cost ~0. "
               "A5 IS THE MOST INFORMATIVE ARM AND THE ONE NOBODY WANTS TO "
               "RUN: it is docs/LESSONS.md's reference-arm rule inverted. Its "
               "FAILURE would be reassuring. Its SUCCESS - matching every "
               "learned encoder - would mean the sim's audio is a "
               "3-parameter family (f0, amplitude, pan) that a lookup table "
               "captures, so no representation experiment run on it can "
               "distinguish anything, and the fixture must grow (section 5) "
               "before the question is even well-posed. "
               "STAGING: A0, A0b, A2, A5 are pure CPU and cost minutes. Run "
               "them FIRST; if A2 does not beat A0b on CPU, the GPU arms are "
               "cancelled and hearing goes back to the drawing board for free."),
```

---

## 5. The sounds Jack must actually distinguish — and what the fixture cannot make

GOAL.md's sentence is a specification: *"he must hear the **ladder creak**, the
**splash**, and the **thud of his own fall**."* Measured against
`ContactAudio.py` as it stands on 2026-08-09, **two of those three sounds do not
exist and the third cannot happen.** These are not opinions; each row below was
run.

### 5.1 What the fixture makes today (verified by running it)

```
drop obj0 into the pool (2.6, -2.4) : 4 events, all IMPACT rings
                                      the water entry itself produces NOTHING;
                                      the only event is the basin-floor contact
                                      at t=0.46 s, force 816 N
drop obj0 onto dry floor (0.0, 2.0) : 3 events, same kind, force 898 N
                                      => a SPLASH is acoustically just a
                                         slightly quieter thud
slide/roll a sphere for 3.0 s       : 3 events total (onsets only)
                                      => no rolling, no scraping, no sustained
                                         sound of any kind
bodies present in the playground    : ['world', 'apple', 'obj0', 'obj1',
                                       'obj2', 'seesaw']
                                      => THE HUMANOID IS NOT IN THE PLAYGROUND
                                         (playground.build_mjcf's
                                          with_humanoid defaults to False)
```

Why, from the code:

- **No splash.** `Water` (`playground.py:246`) is a *force field* applied through
  `data.xfrc_applied`, deliberately, because MuJoCo's global `density`/`viscosity`
  would make Jack swim through air. A force field generates no MuJoCo contact,
  and `ContactAudioSynth.step` fires only on newly-appearing **contact pairs**
  (`ContactAudio.py:103-130`). Water entry is therefore acoustically silent.
- **No creak, no scrape, no roll.** The synth voices contact **onsets** only —
  *"Resting/rolling contact persists as an active pair and does not retrigger"*
  (`ContactAudio.py:105-106`). A creak is stick-slip friction under sustained
  load; there is no friction-driven noise source in the module. Its own
  docstring says so: *"noise for scraping"* is listed in `UNIFIED_BRAIN.md` §4
  as intended, and is absent from v1.
- **No thud of his own fall.** Jack is not in the playground. When he is, the
  synth will voice his contacts — but `_make_event` voices the **smaller geom of
  the pair** (`ContactAudio.py:135`), so a foot hitting the floor rings at the
  *foot's* fundamental, and nothing marks the event as *self* rather than
  *world*. Self/other is the distinction that makes "learn from your own fall"
  possible and it is not represented.

### 5.2 The inventory

The concrete list, with what each requires and where the label comes from.
"Have" = producible today. Ordered by how load-bearing it is for GOAL.md.

| # | sound | GOAL.md needs it for | have? | what it takes | label source |
|---|---|---|---|---|---|
| 1 | **object impact** (thud) | the baseline event; PG.5's whole basis | **yes** | — | `AudioEvent.voiced_geom`, `force` |
| 2 | **bearing of any event** | *turn toward what you heard* — audio's only action-relevant channel | **yes** | — | `azimuth` / `lateral`, PG.5-certified |
| 3 | **material / size identity** (a small ball vs a big box) | binding audio identity to visual position (UB.9's XOR bit) | **yes**, via `f0 = 180/char_size` | — | `char_size`, deterministic |
| 4 | **self vs world** (*my* fall vs *a* fall) | *"the thud of his own fall, and learn from it"* | **no** — Jack isn't in the scene | humanoid in the playground; a `self` flag on events whose voiced geom belongs to Jack's body | `model.geom_bodyid` ∈ Jack's bodies — free and exact |
| 5 | **water entry (splash)** | *"the splash"*; the swim curriculum's only non-visual cue | **no** | a surface-crossing detector in `Water.apply` emitting a broadband, fast-decaying noise burst scaled by entry velocity | crossing time + `v_z` — free and exact |
| 6 | **creak under load** (stick-slip) | *"the ladder creak"*; the "about to slip" warning that makes a ladder learnable by ear | **no** | a friction-noise voice driven by tangential velocity × normal force on a persisting contact | tangential force / slip velocity from the contact — free and exact |
| 7 | **rolling / sliding** (sustained) | *is it still moving after I pushed it?* — occluded object permanence | **no** | same sustained-contact machinery as #6 | contact persistence + relative velocity |
| 8 | **impact hardness / surface** (wood vs water vs metal) | *what did it land on* | **partly** — `MODE_RATIOS` are global, so every material rings the same | per-geom material → mode ratios + decay τ | MJCF material, free |
| 9 | **distance** | *how far away was that* | **yes**, as 1/d attenuation | — | `distance` |
| 10 | **front/back** | full localisation | **no, by design** | ITD and/or spectral shaping; `ContactAudio` synthesises **no** interaural time difference at all | `azimuth` (already labelled) |

Rows 5, 6 and 7 are the same piece of missing machinery: **a sustained,
noise-based voice driven by a persisting contact**, versus the impulsive modal
voice that exists. That is one focused addition to `ContactAudio.py`, not three.
Row 4 is a one-line flag once the humanoid is in the scene.

**Row 3 is the reason §4's arm A5 matters.** With only rows 1–3 and 9 available,
Jack's entire auditory world is `(onset, f0, level, pan)` — four numbers — and a
representation bakeoff run on it is measuring how well each encoder recovers four
numbers. Rows 5–8 are what make the question *"which representation?"* a real
question. **Growing the fixture is a prerequisite for HR.6 being informative,
not a nice-to-have afterwards.**

### 5.3 The task that would prove GOAL.md's sentence

**BLIND PLAYGROUND AUDIT (BPA).** Vision occluded. Jack must classify each
audio event, out of view, into one of four classes and report its bearing:

```
  {object impact, water entry, creak-under-load, rolling}   chance = 25%
  + bearing to within 15 degrees                            chance ~ 17% at that tolerance
```

Constructed so the other senses are at chance **by construction**, on the HNS
discipline from `UNIFIED_BRAIN_BAKEOFF.md` §3:

- **Proprioception is at chance** because every event happens away from Jack and
  he is not in contact with anything. This deliberately removes his dominant
  modality — the one that otherwise absorbs everything (`UNIFIED_BRAIN_BAKEOFF.md`
  §1.4: 348 clean proprio dims against a 2-channel audio stream).
- **Vision is at chance** because it is occluded, and the events are out of the
  frame.
- **Touch is at chance** for the same reason as proprioception.
- **The unimodal late ensemble is therefore at chance**, so every point above
  25 % is attributable to hearing alone. (Note the asymmetry with UB.9: HNS needs
  *fusion* and is at chance for every single modality; BPA needs *audio* and is
  at chance for every modality except audio. They test different things and both
  are needed — BPA certifies that hearing carries content, UB.9 certifies that
  the content gets bound to the other senses.)

Its controls, each of which must fail:

1. **Mono render** → bearing must collapse to chance, class accuracy must
   survive. The two halves failing *differently* is itself the evidence that
   bearing and identity are separate channels (PG.5's mono control, one level up).
2. **L/R swap** → the reported bearing must **invert**, not degrade. Same
   SWAP-FLIP logic as everywhere else in this document.
3. **Spectrum-flattened audio** (one fixed f0, fixed amplitude) → class accuracy
   must collapse to chance while bearing survives. If class accuracy survives, a
   non-spectral cue is leaking.
4. **No-audio arm** → chance on both. If it is above chance, the episode
   sampler is leaking the class (e.g. water events only ever happen at the pool's
   fixed location, so *position* predicts *class*). **This is the confound most
   likely to be missed**: the pool is at a fixed `(2.6, −2.4)` in every
   playground, so "water entry" is perfectly predicted by bearing alone unless
   the pool is relocated per episode. `PlaygroundParams` does not currently
   randomise pool position — it randomises `pool_size` and `pool_depth` only.
   Either randomise the location or the task is broken before it starts.

And the *learning* half of GOAL.md's sentence — *"and learn from it"* — is one
step further: with **self-vs-world** (row 4) available, does adding audio to the
state change what Jack *explores*? That is a curiosity claim (`CU.*`), it depends
on the humanoid being in the playground, and it should not be smuggled into a
perception spec.

```python
    Spec("HR.5", 2, "The playground makes the sounds GOAL.md names",
         hypothesis="ContactAudio emits distinguishable, correctly-labelled "
                    "events for water entry, creak-under-load and "
                    "rolling/sliding, in addition to impacts; and events "
                    "caused by Jack's own body carry a SELF flag.",
         falsified_by="Any of the four is absent, or a linear probe on band "
                       "energies cannot separate the four classes above "
                       "chance+20% — a sound Jack cannot distinguish is not a "
                       "sound he can learn from.",
         null_baseline="Chance = 0.25 over the four classes; plus the CURRENT "
                       "impact-only synth, on which water entry, creak and "
                       "rolling are all literally the same event type "
                       "(MEASURED 2026-08-09: dropping an object into the pool "
                       "and onto dry floor both produce ONLY impact rings, "
                       "force 816 N vs 898 N; 3 s of sliding produces 3 onset "
                       "events and no sustained sound).",
         metric="four_class_audio_separability", budget=Budget.CPU, seeds=3,
         depends_on=["PG.5", "PG.2"],
         control="POSITION-ONLY probe must be at chance. The pool sits at a "
                 "FIXED (2.6, -2.4) in every playground — PlaygroundParams "
                 "randomises pool_size and pool_depth but NOT location — so "
                 "'water entry' is perfectly predicted by bearing alone unless "
                 "the pool is relocated per episode. If the position-only "
                 "probe succeeds, the class labels are geography and every "
                 "later audio-classification number is void.",
         kills="The GOAL.md sentence 'he must hear the ladder creak, the "
               "splash, the thud of his own fall' as anything but aspiration. "
               "Two of those three sounds do not exist in the fixture today "
               "and the third cannot occur, because the humanoid is not in the "
               "playground (playground.build_mjcf(with_humanoid=False); bodies "
               "are world/apple/obj0-2/seesaw).",
         notes="Rows 5, 6 and 7 of the inventory in "
               "docs/research/HEARING_BAKEOFF.md section 5.2 are ONE piece of "
               "missing machinery: a sustained NOISE voice driven by a "
               "persisting contact (tangential velocity x normal force), "
               "versus the impulsive MODAL voice that exists. Water entry is a "
               "surface-crossing detector inside Water.apply emitting a "
               "broadband burst scaled by entry velocity - Water is a FORCE "
               "FIELD (playground.py:246) and generates no MuJoCo contact, "
               "which is exactly why it is currently silent. Self/other is one "
               "flag: geom_bodyid in Jack's body set. All three labels are "
               "free and exact, which is the whole reason sim audio is worth "
               "having (docs/research/UNIFIED_BRAIN.md section 4). "
               "PREREQUISITE FOR HR.6 BEING INFORMATIVE: with only impacts, "
               "Jack's entire auditory world is (onset, f0, level, pan) - four "
               "numbers - and a representation bakeoff on it measures how well "
               "each encoder recovers four numbers."),

    Spec("HR.8", 4, "Blind playground audit: hearing carries content, not just parameters",
         hypothesis="With vision occluded and every event out of contact with "
                    "Jack, the model classifies audio events 4-way (impact / "
                    "water entry / creak / rolling) and reports bearing to "
                    "within 15 degrees, well above chance (>= 0.70 class "
                    "accuracy, lower bootstrap CI > 0.25).",
         falsified_by="Class accuracy at chance, OR indistinguishable from the "
                      "no-audio arm — hearing carries no content in Jack's "
                      "world.",
         null_baseline="Chance 0.25 on class, ~0.17 on bearing at 15 degrees. "
                       "PLUS the NO-AUDIO arm, which must sit at chance: if it "
                       "does not, the episode sampler leaks the class through "
                       "position and the task is broken.",
         metric="bpa_class_x_bearing", budget=Budget.CPU_LONG, seeds=3,
         depends_on=["HR.5", "HR.6"],
         control="FOUR controls, and they must fail DIFFERENTLY, which is the "
                 "evidence that bearing and identity are separate channels. "
                 "(a) MONO render: bearing collapses to chance, class accuracy "
                 "SURVIVES. (b) L/R SWAP: reported bearing INVERTS rather than "
                 "degrades. (c) SPECTRUM-FLATTENED: class accuracy collapses, "
                 "bearing SURVIVES. (d) NO-AUDIO: both at chance. A control "
                 "that takes down both halves at once indicts the shared "
                 "plumbing, not the mechanism (docs/LESSONS.md).",
         kills="'Hearing is load-bearing' (UB.4) as a claim. BPA is the "
               "cheapest experiment that could establish it and it needs no "
               "controller and no GPU.",
         notes="Proprioception, touch and vision are at chance BY "
               "CONSTRUCTION - every event happens away from Jack, out of "
               "frame - so the unimodal late ensemble is at chance and every "
               "point above 0.25 is hearing. Complementary to UB.9 (Heard, Not "
               "Seen), not a substitute: BPA certifies that audio carries "
               "CONTENT; UB.9 certifies that the content gets BOUND to the "
               "other senses. Both are needed and they can fail "
               "independently."),
```

---

## 6. Cost — free compute only

Free compute only, per `SYSTEM.md`. No paid API, no OpenAI Whisper endpoint;
every arm above is open-weight and locally runnable. **`AudioListener._transcribe_api`
(line 372) calls a paid OpenAI endpoint and must be deleted unconditionally** —
that is not contingent on any bakeoff.

### 6.1 The binding constraint is disk, not compute

| | budget | state |
|---|---|---|
| `/data` (holds `HF_HOME`) | 100 GB | **725 MB → 4.8 GB free**, swinging within the hour. Shared with tenants; `/data/history` alone is 73 GB. Size for 725 MB. |
| `/` | 30 GB | 6.2 GB free; `ladder_loop.sh` refuses to start below 3 GB, so ~3 GB is usable. |
| CPU | 4 shared ARM cores | **2 effective threads** (measured: 4 threads = 74 GFLOP/s, same as 2). |
| Kaggle | 30 h/week | ~23 h left this week |
| Colab T4 | elastic | |

Rough weight footprints, against a worst-case 725 MB:

```
silero-vad                            ~2 MB      fits
faster-whisper tiny.en int8          ~75 MB      fits
faster-whisper base.en int8         ~145 MB      fits
faster-whisper small.en int8        ~480 MB      fits ALONE, and nothing else
ECAPA-TDNN (speechbrain)             ~80 MB      fits
WeSpeaker ResNet34 ONNX             ~100 MB      fits
Resemblyzer                          ~17 MB      fits
LibriSpeech dev-clean corpus        ~337 MB      fits, but not beside small.en
ctranslate2 + faster-whisper wheels  ~18 MB      fits
onnxruntime 1.19.2 wheel (pinned)    ~12 MB      fits
CED-tiny (HR.6 arm A6)               ~22 MB      fits — cheapest credible tower
EfficientAT mn05_as                   ~6 MB      fits
PANNs CNN14 / BEATs / AST         ~330 MB ea.    two of them IS the whole budget
EnCodec 24k (HR.6 arm A4)            ~93 MB      fits
DAC 44.1k                           ~307 MB      fits alone
WavTokenizer ckpt                    ~1.8 GB     DOES NOT FIT (optimizer state
                                                 included; model is ~285 MB fp32,
                                                 so it needs re-exporting first)
LAION-CLAP                          ~615 MB      does not fit with anything
```

Note the shape of that list. **The two jobs that genuinely cannot be solved from
the simulator's own ground truth — ASR and speaker ID — are also the cheap
ones** (~75–145 MB + ~80–100 MB). The expensive entries are all towers for a job
the simulator already labels exactly. Skipping them is what makes the rest fit.

**Read down that column and §1.3's null hypothesis stops being a research
opinion and becomes an accounting fact.** The pretrained sound-event towers are
the single most expensive thing on the list and the least justified: the sim
already emits exact labels. Skipping them is what buys room for the ASR model,
the speaker embedder and the corpus — the two jobs that genuinely cannot be
solved from the sim's own ground truth.

→ **Escalate to `docs/DECISIONS_NEEDED.md`:** free `/data`, or relocate `HF_HOME`
to `/`, before HR.1 is implemented. HR.1 through HR.4 are blocked on disk, not on
science.

### 6.2 What dies on 4 ARM cores before anything costs a GPU-hour

`UNIFIED_BRAIN_BAKEOFF.md` §6's staging principle: **give every arm a chance to
die for free.** For hearing, almost all of it can.

| # | falsifier | cost | kills |
|---|---|---|---|
| 1 | HR.5 four-class separability + the position-only control | ~15 min CPU | HR.8, and the GOAL.md sentence, if the fixture cannot make the sounds |
| 2 | HR.7 bearing-survives-the-stem probe (per candidate stem) | ~5 min CPU each | any stem that deafens him to direction, **before it trains** |
| 3 | HR.6 arms A0 / A0b / A2 / A5 (no-audio, placebo, mel, event-vector) | ~30 min CPU | the whole audio modality, if mel does not beat placebo; or the FIXTURE, if the event vector ties everything |
| 4 | HR.1 corpus leak probe + planted-leak control | ~10 min CPU | HR.2, HR.3, HR.4 on a leaky corpus |
| 5 | HR.2 silence-hallucination control | ~10 min CPU | any ASR arm that invents words from room tone, whatever its WER |
| 6 | HR.3 full bakeoff (embedding extraction is the only cost) | ~1–2 h CPU | five of six embedders |
| 7 | HR.4 end-to-end chain (no training at all — frozen models + `EpisodicMemory`) | ~30 min CPU | the sentence "Jack remembers who told him what" |

**Total: under 4 hours of CPU for the entire hearing programme except HR.6's
GPU arms and HR.8.** Nothing in §2 or §3 needs a GPU at any point: ASR and
speaker embedding are frozen forward passes, and `EpisodicMemory` is lexical with
no torch dependency at all (`EpisodicMemory.py:36-39`).

| item | where | estimate |
|---|---|---|
| HR.1–HR.5, HR.7, and HR.6's CPU arms | 4 ARM cores | **< 4 h total** |
| HR.6 GPU arms (A1 raw, A3 ILD, A4 tokens × 3 seeds) | T4 / P100 | 3–6 GPU-h |
| HR.8 blind playground audit, 3 seeds | CPU_LONG, or 1–2 GPU-h if scaled | ~2 h CPU |
| **GPU total for the whole hearing programme** | | **3–6 GPU-h** |

That is a quarter of one Kaggle week for all of hearing — because the expensive
half (speech) is frozen inference and the cheap half (world sound) is generated
by the simulator at 0.45 % of real time.

### 6.3 Ordering

```
[disk escalation]  ->  HR.5  ->  HR.7  ->  HR.6(CPU arms)  ->  HR.1  ->  HR.2
                                                                  |
                                                                  v
                                                       HR.3  ->  HR.4
                                                                  |
                            [GPU]  HR.6(full)  ->  HR.8  ->  UB.9 / UB.10
```

HR.5 goes first because it can invalidate everything downstream for 15 minutes
of CPU, and because it is the only spec on the list that tests GOAL.md's own
sentence.

---

## 7. What we refuse to claim

- **That Whisper knows who is speaking.** It is trained to be invariant to
  speaker identity. Job (b) needs a different model and it is not optional.
- **That ME.9 shows Jack remembers who told him what.** ME.9 shows the
  *retrieval contract* is sound, which is real and worth having. The speaker
  field is handed to it by the test
  (`me_9_attributed_recall.py:66,73,78`); nothing in the live system produces
  it. HR.4 is the spec that would close that, and until it passes the
  end-to-end sentence stays out of every capability list.
- **That the playground makes the sounds GOAL.md names.** Verified by running
  it on 2026-08-09: water entry is silent, nothing creaks, nothing rolls, and
  the humanoid is not in the scene so *"the thud of his own fall"* cannot occur.
  Impacts and bearing are real and PG.5-certified. That is the whole inventory.
- **That any pretrained audio tower is needed — but we do not claim the
  converse by argument either.** §1.3's null hypothesis is that none earns its
  parameters against a simulator emitting exact labels, and the evidence is
  strong: the AudioSet classes GOAL.md names are the worst in the dataset
  (`Roll` 0 % label accuracy, `Creak` 11 % from 29 training clips), impacts are
  a ~3 % duty cycle under weak labels, and the disk accounting in §6.1 nearly
  moots the question. But `SYSTEM.md` law 3 forbids settling this by argument,
  so it is **arm A6 of HR.6** — frozen CED-tiny at matched token count — and it
  can win. §1.3.3 records the condition for re-opening (a real-microphone,
  no-ground-truth task) so the decision is not silently reinvented.
- **That the wav2vec2 encoder currently in `UnifiedBrain.AudioEncoder`
  (line 1020) is defensible.** On HEAR it scores 0.561 on ESC-50 against PANNs'
  0.909 and CED-base's 0.967. Self-supervised *speech* features are the wrong
  family for environmental sound by a measured 20–40 points, and it is excluded
  from HR.6 on that basis rather than left in as a courtesy arm.
- **That contact audio has earned its parameters in the brain.** Until an audio
  arm beats the PLACEBO channel — not zero, the matched-noise channel — hearing
  is decorative at this scale and loses its parameters under the Tier-3 rule.
  This document carves no exception for the sense it is about.
- **That a bearing-preserving stem is a foregone conclusion.** Log-mel provably
  preserves interaural level difference; discrete codecs are predicted to
  destroy it. HR.7 is where that prediction gets tested, and a document that
  predicts and is wrong is worth more than one that hedges.
