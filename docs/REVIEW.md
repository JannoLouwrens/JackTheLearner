# Jack — pipeline review and the path to a living, learning companion

---

## 1. The verdict

Jack is a 117.9M-parameter randomly-initialised network wired into a viewer that cannot start on this box, running a control loop that has never received a gradient, driven by an action module that no training phase in the repo touches — and on the one mode that *does* start (`--text-only`), there is no tick loop at all, so the "brain" never sees an observation. The real problem is not that he is untrained; it is that **five independent breaks sit between training and behaviour** (a checkpoint format the runtime rejects, an observation space the runtime doesn't build, an action head no loss reaches, a `self.apply(_init_weights)` that re-randomises the frozen LLM, and three of three training phases that either crash or optimise a self-referential target), so even a perfect GPU run tonight would produce zero observable change. A curious Jack who explores unprompted and keeps learning is entirely achievable on 30 free GPU-hours a week — but not by training a bespoke 105M brain from scratch, and not by adding gradients to the live loop; it is achievable by freezing a pretrained trunk, learning one small policy, and moving "learning" into an experience log, a memory store, and an overnight consolidation job.

---

## 2. Is everything connected?

No. The honest summary: **the live path is real but hollow, and ~38.6% of the model's parameters have no call site on any live path.**

### 2.1 The curiosity stack — lead finding

`AutonomousMind` **is constructed** (`UnifiedBrain.py:4008`, `enable_intrinsic_motivation=True` at `:196`) — the "never instantiated" hypothesis is refuted. It owns `IntrinsicCuriosityModule` (`:3672`), `SkillDiscovery` (`:3673`), `Empowerment` (`:3674`), `Metacognition` (`:3675`), `AutotelicGoalGenerator` (`:3676`), and measures **4,987,769 parameters** exactly.

Its entire public API is dead outside a demo block. The six entry points — `compute_intrinsic_reward` (`:4794`), `explore_autonomously` (`:4850`), `should_ask_for_help` (`:4911`), `discover_skills` (`:4933`), `get_empowerment` (`:4970`), `get_curiosity_reward` (`:4984`) — have **exactly four call sites in the entire repo**: `UnifiedBrain.py:5859, 5863, 5867, 5872`, all inside `if __name__ == "__main__":` (`:5789`), under `print("[TEST 6] Intrinsic Motivation")`. `AutonomousMind.get_training_loss` (`:3747`) has **zero call sites anywhere**, including the demo. `VirtualWorld.py`, `TaskManager.py`, `TrainingPipeline.py` and `tests/` contain zero references. The one test file that mentions the subsystem sets `enable_intrinsic_motivation=False` (`tests/test_all_fixes.py:462`).

The wiring existed once: `archive/RobustTrainer.py:8614-8766` called it. Commit `ba012b6` — "replaces 8970-line RobustTrainer, zero dead code" — deleted the caller and kept the parameters.

What stands in its place on the live path is `VirtualWorld._update_autonomous` (`:807`), whose entire exploration branch is:

```python
if idle_time > self.config.idle_threshold:
    if self.frame_count % (self.config.target_fps * 30) == 0:
        self._log_chat("Jack", "Hmm, I think I'll explore a bit...")
        self.current_task = "Exploring"
```

A hardcoded string on a frame counter. No goal, no action, no module consulted.

### 2.2 Wiring inventory

**WIRED (executes on the live path):**
- `VirtualWorld.run():476` → `_update_brain():628` → `TaskManager.tick():290` → `brain.act_with_mood():4614` → `act_dual_system():4544` → `forward()`
- Encoders (proprio/touch), 8-layer backbone (`layers`, 36.7M), `cross_modal_fusion` (9.5M), `action_head`, `physics_head`, `value_head`, `task_completion_head`
- `ActionExpert` (4.6M) — the module whose output actually reaches `mj_data.ctrl` (`VirtualWorld.py:890`)
- `EmotionalState` (mood token at `UnifiedBrain.py:4193`), `MovementMoodCoupling` (`:4620`)
- `ObjectDetector` (1.30M) and `NavigationPlanner` (0.16M) — reachable, but only through `chat():5347` → `_execute_command():5160`, a regex verb matcher, and only when `state is None`, i.e. the TextOnlyWorld path (`VirtualWorld.py:1603`). *This corrects the audit's "test-only callers" claim.*
- `CompanionMemory`, `InnerMonologue`, `Personality` (as prompt text only), `Persistence`

**ORPHANED (constructed, zero live call sites) — 45,538,295 params, 38.6% of the 117,888,028-param brain:**

| Module | Params | Why dead |
|---|---|---|
| `hierarchical_planner` | 37,166,972 | `forward()` gates on `use_hierarchy` (`:4292`), default `False` (`:4117`); only caller passing `True` is `plan_with_hierarchy():4316`, which has no callers |
| `autonomous_mind` | 4,987,769 | §2.1 |
| `world_model` | 2,974,977 | `forward()` gates on `action is not None` (`:4273`); `act_dual_system` never passes `action` |
| `semantic_anchors` | 408,577 | Only read by `compute_language_grounding_loss():5715`, which has zero live callers |

Also orphaned: `TemporalMemory` (12,635,136 params, `:3872`) — `forward()` touches it only under `if memory is not None:` (`:4219`) and **no caller in the repo passes `memory=`**. `System0Controller` (`:2494`) — `system0_enabled=False` (`:164`), `self.system0` has zero call sites. `MoCapLoader.py` (935 lines) — no non-archive import. `AMPDiscriminator` (`:388`) — zero live construction. `TextToSpeech` (`:3978`) — every runtime call passes `speak=False` (`VirtualWorld.py:1279, 1283, 1603`), and no TTS backend is installed.

**STUBBED (called, but structurally a no-op):**
- `AlphaGeometryLoop` — `TaskManager._try_creative_reasoning():642` feeds it `torch.randn(256)` for both state and goal (`:658-659`, with the comments "Would use actual state from last tick" still in place), and the caller discards the return (`:623-635`). **Worse than filed:** `TaskManager.py:665` gates on `metadata.get('solved')`, but `AlphaGeometryLoop.solve()` returns `'solved'` on exactly one path — the *timeout* path (`:345`, `solved: False`). Its two success returns (`:333` mode `direct`, `:390` mode `creative`) omit the key entirely. So `_try_creative_reasoning` returns `None` **unconditionally**, including on success, and the "I had a creative idea!" monologue line is unreachable. A second, entirely unread `AlphaGeometryLoop` is constructed at `UnifiedBrain.py:4075`. Commit `a67b482` "Restore AlphaGeometryLoop and wire creative reasoning into system" wired nothing.
- `Persistence._collect_autonomous_mind` (`:530-549`) probes `curiosity_scores`, `skill_library`, `metacognition_log` — none of which exist on `AutonomousMind`. Returns `None` on every save. Same for `brain.obs_projection` (`:353`), which does not exist on `UnifiedBrain`. Both advertised as v1.2 features (`:21`, `:53`); the self-test (`:1079-1080`) passes vacuously because `_migrate_save` inserts both keys as literal `None` (`:847-850`).
- `Personality.get_behavior_bias` (`:465-513`) — docstring says "used by the action selector"; zero callers. Personality changes what Jack *says*, never what he does.

### 2.3 How much of the 16,643 lines is live

Best accounting: roughly **4,000 lines execute at all** in a normal session; roughly **1,200 execute per frame** in the GUI loop — which cannot start on this box, because `mujoco` and `pygame` are not installed in `/data/venvs/jackthelearner` and `DISPLAY` is unset, so `create_world():1714` returns `TextOnlyWorld` regardless of `--text-only`. In that mode `TextOnlyWorld.run()` (`:1508-1576`) is a blocking `input()` REPL: no `_update_brain`, no `_step_physics`, no `task_manager.tick`. `task_manager.tick` appears **once** in the entire repo, at `VirtualWorld.py:628`. So on the machine Jack actually lives on, the executing surface is a few hundred lines of chat routing, and the 117.9M-parameter network never receives an observation tensor at all (`_chat` calls `brain.chat(message, speak=False)` with no `state=`, so `UnifiedBrain.chat():5340` takes the keyword-matching branch).

**And zero lines anywhere update a weight while Jack is alive.** `VirtualWorld.py` has exactly two grad-related lines in 1,920: `torch.no_grad()` (`:626`) and `brain.eval()` (`:1882`). This understates it — `UnifiedBrain.py` contains **zero** occurrences of `torch.optim`, `.backward()`, `.zero_grad()` across 5,897 lines. Deleting the `no_grad` would change nothing; there is no optimizer in the runtime import graph. The single parameter-mutating line (`update_target_encoder`, `:1868`) has zero callers.

---

## 3. The blocking defects

Ranked by what stops progress soonest. All CONFIRMED; refuted items are excluded (notably: the "3.4 GB download onto an 85%-full root disk" is **wrong** — `~/.cache/huggingface` symlinks to `/data` with 35 GB free, `/` is at 59%, and SmolLM2-1.7B is already fully and validly cached, 218 tensors verified).

### D1 — `self.apply(self._init_weights)` re-randomises the frozen LLM *(fatal)*

`UnifiedBrain.__init__` ends with `self.apply(self._init_weights)` at `UnifiedBrain.py:4088`. `LLMEncoder` assigns `self.llm = AutoModelForCausalLM.from_pretrained(...)` (`:1231`) inside an `nn.Module`, so SmolLM2-1.7B is a registered grandchild. `_init_weights` (`:4096-4102`) does `nn.init.normal_(module.weight, std=0.02)` on every `nn.Linear` and `nn.Embedding` with no name check and no exclusion list — and `nn.init.*` runs under `no_grad`, so `requires_grad=False` is no protection. Measured on this box: a frozen nested module's weights went from std 0.210 → 0.018 under this exact function.

Reachability is not hypothetical: `llm_enabled=True` (`:122`), `VirtualWorld.main():1874-1881` overrides only vision/audio, transformers 4.57.6 is installed, and the weights are cached — so `[LLM] Loaded!` prints, then every q/k/v/o/gate/up/down projection and `embed_tokens` is overwritten with noise. `.generate()` does not raise; it returns fluent-looking garbage, so the template fallback never fires. Command understanding, `ResponseGenerator._generate_with_llm` (`:2685`), `answer_question` (`:2763`) and `CompanionMemory` embedding recall (`:2825`) all run on a randomly-initialised 1.7B transformer. **"Hold a conversation" cannot work until this is fixed.** (`AudioListener.py:362` loads whisper-tiny outside the brain, so speech-to-text is unaffected.)

**Fix:** collect the ids of pretrained subtrees before init and skip them in a `named_modules()` walk, or re-load the pretrained `state_dict` immediately after `apply()`. ~15 lines.

### D2 — The module that produces Jack's torques receives gradient from no loss *(fatal)*

Two independent action pathways exist. `ActionHead` feeds `output['actions']` (`:4261`) — this is what PPO optimises (`TrainingPipeline.py:455`). `ActionExpert` flow-matching (`:4557-4577`) is what the runtime executes, because `dual_system_enabled` and `action_expert_enabled` both default `True` (`:160`, `:167`), so `act_with_mood` → `act_dual_system` takes the cached-features branch.

Measured, per parameter tensor: after `compute_physics_loss().backward()`, `action_expert` gets **0/45**; after a PPO-shaped loss on `output['actions']`+`output['value']`, `action_head` gets 5/11 and `action_expert` still gets **0/45**. Phase 8 touches only `emotional_state` and `movement_mood.speed_net`. Run every implemented phase to convergence and the module producing Jack's joint torques is still at random init, integrating `torch.randn` through an untrained velocity field.

**Fix:** pick one pathway. If `ActionExpert` is the runtime path (it is), add `compute_flow_matching_loss` (`:5545`) to a behaviour-cloning phase. Otherwise set `action_expert_enabled=False` and let `act_dual_system` fall through to `output['actions']` at `:4581`.

### D3 — There is no path from a trained checkpoint into the runtime *(fatal)*

`TrainingPipeline.save():304-319` writes `{'model','obs_proj','log_std','obs_mean','obs_var','obs_count','epoch','global_step'}` — no `'version'`, no `'model_weights'`. `CompanionPersistence.load_all():213-216` raises `RuntimeError('Invalid save file format ... missing version field)'` before any migration path. `_apply_state():565` reads `data.get('model_weights')`. `VirtualWorld.py:1889-1899` catches the exception, prints `[Main] Failed to load save: {exc}` at `:1897`, and continues with random weights.

Reproduced: a dict with TrainingPipeline's exact key set fed to `load_all` raises immediately. The classes match perfectly (`TrainingPipeline.py:232` constructs the same `UnifiedBrain`), so this is a pure envelope-format defect — cheap to fix, and it means the training→runtime bridge **has never once been exercised**.

**Fix:** `export_for_runtime(path)` writing the Persistence schema, and `raise SystemExit` on load failure when `--load` was explicitly passed.

### D4 — Train/serve observation skew *(fatal)*

`VirtualWorld._get_observation_tensor():1348-1356` builds `np.concatenate([qpos, qvel])`, truncated/zero-padded to 256, **no normalisation, no projection** — its own docstring admits "for now we pad here". `TrainingPipeline` normalises with running mean/var (`:269-284`) and projects through `obs_proj` (`:235-241`), which lives on the *pipeline*, not the brain: an AST walk of `UnifiedBrain` finds no `obs_projection` attribute, and `grep -c obs_projection UnifiedBrain.py` = 0. Persistence's slot for it (`:353-361`, `:574-581`) is permanently dead.

Layouts do not correspond at any index. Additional latent mismatch: `mujoco_obs_dim=376` (`:67`) is the **Humanoid-v4** width; Humanoid-v5 is 348, so `project_obs` silently zero-pads 348→376 and 28 input dims are permanently zero.

Also fatal for the same reason at the actuator end: `action_dim=17` (`:83`) vs 59 motors in `humanoid_full.xml`, 22 in `jack_room.xml`/`humanoid_terrain.xml`. `VirtualWorld.py:888-890` does `n_act = min(len(action), nu)` — silent truncation, no warning. And with no `--scene`, the inline `_DEFAULT_SCENE_XML` (`:147-198`) contains a floor, four walls, a ball and a cube and **no humanoid** — `nu == 0`, so the ctrl write is skipped entirely. The default launch renders an empty room.

**Fix:** one canonical `build_observation()` used by both sides; move `obs_proj` + running stats onto `UnifiedBrain` as registered buffers; add `assets/humanoid_v5.xml` (Gymnasium's exact 17-actuator body) as the single train-and-live scene; assert `nu == action_dim` at init and refuse to start otherwise.

### D5 — All three training phases are broken *(fatal ×3)*

**Phase 2 crashes on minibatch 2 of 40.** `collect_rollout` computes `state = self.project_obs(obs_tensor)` at `:533`, one line *above* the `with torch.no_grad():` at `:535`. Every stored state keeps a graph through `obs_proj`; `torch.stack(states)` (`:561`) makes one `StackBackward0` fanning into all 512 branches; the first `loss.backward()` (`:494`) frees them all. Reproduced verbatim: `minibatch 0: OK / RuntimeError on minibatch 1: Trying to backward through the graph a second time`. This fires after 512 steps of a 2,000,000-step run — **Phase 2 has never executed**. Fix: `.detach()` at `:533`. One line.

**Phase 8 crashes on its first iteration.** `:818` sets `pad_vector.requires_grad_(True)` on a leaf buffer (`EmotionalState.py:365`); `update()` then rebinds it to non-leaf tensors (`:593, 614, 620, 623`); `:843` calls `requires_grad_(False)` on the non-leaf → `RuntimeError: you can only change requires_grad flags of leaf variables`. Reproduced. Its objective is content-free anyway: `-0.05 * stack.var(dim=0).sum()` (`:834`) is unbounded below, so the optimum is saturating PAD at ±1.

**Phase 0 rewards destroying the observation encoder.** `:632` sets `state = self.project_obs(raw_s)`; `:652-656` hands *that 256-d learned latent* to `calc.predict_robot_state`, which reads `s_np[:3]` as position and `s_np[2]` as height. Those are coordinates of a randomly-initialised MLP that is **in the same optimizer** (`:378`) and trained by the same loss, with targets recomputed every batch. `compute_physics_loss` (`UnifiedBrain.py:5529-5530`) likewise compares `output['next_state']` against `project_obs(raw_next)` — both sides depend on `obs_proj`. The reachable global optimum is `obs_proj` emitting a constant: both MSEs → 0, total bottoms at the −0.046 rule-entropy floor. **A loss falling by orders of magnitude in the first epochs is the failure signature, not success.**

### D6 — What happens if he runs Phase 0 on a T4 tonight

Honestly, step by step:

1. `TRAIN_ON_COLAB.ipynb` cell 2 is `!git clone https://github.com/YOUR_USERNAME/JackTheWalker.git` — an unedited placeholder, wrong project name. It fails immediately. Has never been run.
2. After fixing that and `pip install gymnasium[mujoco] mujoco sympy`: the brain builds (117.9M params, ~470 MB fp32), prints a wall of "ENABLED" banners, prints `[WARN] Checkpoint not found`.
3. The loop starts. tqdm shows loss ≈ 47,000 (measured `compute_physics_loss` at init with realistic targets) or ≈ 85,757 (measured, raw-joule targets), falling fast. That fall is the collapse described in D5, not learning.
4. The physics head **never sees the action** — `forward()` passes only `state, goal, noisy_actions` into the tokenizer (`:4227`); `action` is consumed only inside the world-model block (`:4273`). Measured: `max|physics(a=0) − physics(a=1000·randn)| = 0.0` exactly. But targets 4, 5, 8, 9 are all functions of `force_magnitude` alone, and actions come from `env.action_space.sample()` (`:624`), i.i.d. and independent of state. **Those four outputs can never beat a constant.** "Internalise F=ma" is structurally unlearnable under this objective at any compute budget.
5. Only 5 of the 10 "physics quantities" are independent — verified: `col2==col0+col1`, `col5==0.3*col4`, `col6==0.5*col3`, `col8==0.02*col4`, `col9==col4`.
6. The SymPy teacher **has no gravity.** `predict_robot_state` (`SymbolicCalculator.py:286-390`) computes `acceleration = force/mass` with no gravity term. Measured: a 50 kg body at 1.3 m under zero torque stays at 1.3 m forever. No mass matrix, no contact, no ground reaction, hard-coded `joint_inertias = 0.1` for all 17 joints. It is a strictly worse dynamics model than the MuJoCo already running at `VirtualWorld.py:875`.
7. Cost: 64 GPU→CPU syncs per batch from the per-sample `.cpu()` loops (`:652`, `:688-694`) around SymPy that costs 0.327 ms/call — 500K targets is 2.7 minutes of actual compute wrapped in hours of stalls. Every epoch writes `phase0_latest.pt` **and** `phase0_best.pt` at 860 MB each. At the end: a ~930 MB `ewc.pt` whose "Fisher" is `output['physics'].pow(2).mean() + output['actions'].pow(2).mean()` (`:170`) — the gradient of output *magnitude*, which is not Fisher information.
8. It prints `[DONE] Phase 0. Best loss: <small>` and exits, having written ~4 GB that **nothing in the repo can read** (D3).

Net: 0.5 GPU-hours spent, a collapsed encoder saved as "best", and zero change to Jack.

### D7 — Failures are invisible by design *(major)*

`_update_brain():601-661` wraps observation building, all five senses, `task_manager.tick` and action extraction in `except Exception as exc: logger.debug(...)`, while `main()` configures `level=logging.INFO` (`:1842`). If the brain raises on **every frame**, nothing prints; `self.current_action` stays `None`, so `_step_physics` skips the ctrl write but still calls `mj_step` and the humanoid ragdolls limp — reading as a balance bug, not a dead brain. Meanwhile `_update_emotional` runs *outside* the try, so the mood bars keep animating and the HUD looks healthy. Same pattern at `:698` (blinds vision silently), `:758`, `:861`, `:896`, `:931`, `:1322`.

### D8 — The task system cannot complete a task, and lies when it fails *(major)*

Subtask advancement requires `task_done_prob > 0.8` (`TaskManager.py:70, 332`). Measured over 40 ticks of the untrained head: min 0.5050, max 0.5228, mean 0.5136, std 0.0041. It will never cross. The stuck path needs 500 *consecutive* ticks with |Δ| < 0.01; measured P(|Δ|>0.01) = 0.077, so P ≈ 0.923^500 ≈ **2e-18** — which also makes the AlphaGeometry and LLM-replan branches behind it unreachable. The only reachable exit is `frames_running > 3000`, three times. At the measured 6.8 Hz that is 22 minutes per subtask; "make tea" has 9 subtasks → 3.3 hours.

Then `_on_subtask_timeout` (`:587-598`) sets `FAILURE` and calls `_advance()`, which at the end of the list calls `_on_task_complete()` (`:531-533`), which **unconditionally** increments `tasks_completed`, fires `GOAL_ACHIEVED` with `reward=1.0`, writes memory `"Completed task: {task}!"` at importance 0.9, records `"I did it! ... That feels good."`, and appends `{'success': True}`. `tasks_failed` is declared at `:202` and **never incremented anywhere**. After 3.3 hours of a humanoid twitching on the floor, Jack's persistent memory records that he made tea and his mood improves. Every downstream consumer — mood, memory, self-narrative, and any future learning signal — is poisoned with false reward. This is a ~10-line fix and it is the highest-value correctness fix in the repo.

### D9 — Persistence loses the one thing the product is about *(major)*

`_collect_emotional_state:407` does `state['history'] = list(emo.history)`. `MoodHistory` (`EmotionalState.py:493`) defines `__len__` but not `__iter__` — measured: `TypeError: 'MoodHistory' object is not iterable`. The blanket `except Exception` at `:418` catches it and returns `None`, **discarding `pad_vector`, `baseline` and `gru_hidden` which were already collected successfully at `:400-416`**. Verified end-to-end: `emotional_state` field in save = `None`. For a persistent emotional companion, the emotional state is the one thing never persisted. Fix: `emo.history.to_json()` (`EmotionalState.py:93`) — the serialiser already exists.

Also: auto-save fires once on frame 1 then never again (`Persistence.py:114/165` writes `time.time()`, `VirtualWorld.py:471` compares `time.monotonic()` — reproduced: first call `True`, second 400 s later `False`). `_peek_save` (`:861-888`) fully `torch.load`s 472 MB per file despite its own comment claiming a meta trick, and runs on every save via `_prune_old_saves` (`:168`) × up to 10 saves. Weights load `strict=False` with the `_IncompatibleKeys` result discarded (`:565-571`), and no architecture fingerprint is stored — a save from before a model change *appears* to load and silently produces a random-weight brain. `weights_only=False` (`:206, 872`) makes `--load <any .pt>` an arbitrary-code-execution vector.

### D10 — The test suite cannot distinguish trained from untrained *(major)*

The counts reproduce: 45/45, 21/21, 10/10 — all on random weights, with `checkpoints/` empty and no `.pt` anywhere. `tests/test_integration.py:46-53` is eight consecutive `assert x is not None`. Six of ten integration tests contain **zero assert statements** and `return True` unconditionally. `test_make_coffee_scenario:204` passes because `make_coffee()` (`UnifiedBrain.py:5283-5305`) returns a hardcoded nine-element list literal with `status: "planned"` — the network is never invoked. `tests/test_pipeline_audit.py` has **zero `def test_` functions** (it defines its own `test(name, condition)` helper at `:9`), so the commit's "45 pipeline tests" is not a pytest count. And `tests/test_all_fixes.py:263` / `test_pipeline_audit.py:83` feed `rl_update` a synthetic `torch.randn(N,256)` leaf tensor, which is exactly why the fatal Phase 2 crash has no coverage.

### D11 — README.md reports results for experiments that were never run *(major, and the highest personal risk)*

`README.md:224-236` presents: "Phase 0: Physics Accuracy 94.2%", "Energy Conservation Error < 2.1%", "Walking Speed 1.4 m/s", "Episodes Before Falling 850+", "Push Recovery 73%", "System 2 Activation Rate 8.7%", "Training is performed on a single RTX 3090", and — the sole empirical support for the entire thesis — "Agents trained WITH Phase 0 recover from perturbations 31% more often." `checkpoints/` is empty, `datasets/` is 0 bytes, no weights exist on this box, gymnasium and mujoco aren't installed. `README.md:348` marks "MathReasoner + SymbolicCalculator: Working"; `MathReasoner.py` exists only in `archive/` and is imported by no live file.

This is a public MIT-licensed repo under your name feeding a masters submission. **Delete or relabel today.** It costs twenty minutes and it is the single largest liability in the project.

---

## 4. Can Jack keep learning?

**Not today — not partially, not "unconfigured". It is unimplementable without new code, because the world he lives in records nothing.** `grep buffer|trajector|transition|record|episode` over `VirtualWorld.py` returns two hits, both render buffers (`:267`, `:912`). Every method in the 2026 literature — replay, generative replay, sleep consolidation, offline RL, LoRA consolidation — consumes a recorded stream. There is no stream.

**But yes, it is achievable, and on your budget.** The answer is not to add gradient steps to the 50 Hz loop. Keeping `brain.eval()` and `torch.no_grad()` in the actor is *correct by design* — an actor must not mutate weights mid-episode on a CPU box shared with paying tenants. The pattern is a **three-timescale actor/sleeper split**.

### T0 — Non-parametric memory (every interaction, zero gradients)

Most of what a user perceives as "Jack learning" belongs here: remembering preferences, not repeating mistakes, improving at seen tasks. JitRL (arXiv:2601.18510, ICML 2026 Spotlight) achieves RL-like continual adaptation on a *frozen* LLM purely by retrieving past experience and modulating logits — outperforming weight-update methods at >30× lower cost, with a closed-form justification as the KL-constrained policy-optimisation solution. This runs on CPU today, with the already-frozen SmolLM2.

Concretely: replace `CompanionMemory` (`UnifiedBrain.py:2791`, currently a Python list of dicts with keyword-match fallback at `:2811`) with SQLite — one row per event `(ts, text, importance, embedding)` — and Generative-Agents retrieval scoring, `recency*0.5 + relevance*3 + importance*2` (arXiv:2304.03442), plus a nightly reflection pass. 2026 long-memory benchmarks (LoCoMo, LongMemEval) consistently show flat stores without temporal metadata fail exactly the questions a companion gets asked ("what did we do yesterday").

### T1 — A tiny online head (50 Hz, CPU, real gradients)

If you want genuine within-session adaptation, use **stream-x** (Elsayed, Vasan & Mahmood, arXiv:2410.14606): eligibility traces + ObGD/AdaptiveObGD + SparseInit at 90% + LayerNorm + reward scaling. **No replay buffer, no target networks, no batch updates** — memory is O(parameters), the reference implementation is ~150 lines, and it works with a single untuned hyperparameter set (α=1, κ=3 policy / κ=2 value). It is the first method to break the "stream barrier" while matching batch-RL sample efficiency on MuJoCo/DM Control/Atari.

Constraint: apply it to a **≤2M-parameter head only**. Never the trunk.

### T2 — Sleep consolidation on the ephemeral GPU

This is the published wake/sleep pattern — SIESTA (arXiv:2303.10725) formalises a wake online phase alternating with a compute-restricted sleep offline phase; sleep-time compute is formally equivalent to offline policy improvement. Your ephemeral T4 is not a limitation to work around; **it is the sleep phase**, and there is literature to justify it in a thesis. Jack is awake on CPU and dreams on a T4.

The job: pull the newest experience shards → LoRA consolidation over a reservoir-sampled replay buffer (CLEAR, arXiv:1811.11682) → run the frozen regression gate → upload only the adapter delta (a few MB, not 860 MB). Promotion is a separate gated step; the sleep job never writes to the live actor.

### What changes in the code

1. **Transition logger** in `VirtualWorld._update_brain`: append `(obs, action, reward components, next_obs, done, ts, task_id, mood)` to rolling shards under `/data` (33 GB free), not root. ~50-100 lines. **This is the highest-leverage change in the entire repo** and everything else depends on it.
2. **`export_for_runtime` / `--brain-weights`** (D3).
3. **One canonical observation builder** (D4).
4. **Frozen SymPy regression gate** — versioned probes, deterministic, <60 s on CPU, run before and after every consolidation. No adapter reaches the live actor without passing it.
5. `AutonomousMind` wired as an **offline reward labeller** over logged transitions during sleep — not as a live actuator (see §5).

### Plasticity and stability — the parts the repo has zero awareness of

`grep plasticity|continual|forget|lifelong|dormant|streaming|LoRA|adapter|EWC|replay` over `RESEARCH_PAPERS.md` (324 lines) returns **zero hits**.

**Plasticity loss is what actually ends indefinite learning.** Dohare, Hernandez-Garcia, Lan, Rahman & Sutton, *Loss of plasticity in deep continual learning*, Nature 632:768-774 (2024): standard deep learning silently loses the ability to learn under continual training — ImageNet binary-classification accuracy fell from 89% to ~77% (linear-model level) by task 2000. Mechanisms: dormant/saturated units, parameter-norm growth, reduced effective rank. **Without instrumentation you cannot distinguish "Jack has converged" from "Jack can no longer learn"**, and that ambiguity is fatal to the word "indefinitely."

Ranked interventions (arXiv:2405.19153, NeurIPS 2024): soft shrink-and-perturb gives the best generalisation and composes with LayerNorm; L2 and regenerative regularisation also work; **ReDo performed worst** because it cannot exploit positive transfer. arXiv:2508.00212 (CoLLAs 2025) finds reinitialising *weights* beats reinitialising *units*, with the advantage largest for narrow, LayerNorm-heavy networks — which describes Jack exactly (`d_model=512`, `n_layers=8`). Log dormant-unit fraction, parameter-norm growth and per-layer effective rank on every consolidation, from day one.

**On forgetting: stop investing in EWC.** It exists only in `TrainingPipeline.py` (`ReplayBuffer:97`, `EWC:149`, Fisher `:711-717`, penalty `:472-490`), zero hits in the runtime, and its "Fisher" is not Fisher (D6.7). The 2026 Continual RL survey (arXiv:2506.21872) rates regularisation methods *low* on both task-scaling and compute efficiency, and notes they degrade past ~20 tasks — and compute is precisely your bottleneck.

More decisively: **the from-scratch 105M trunk is the worst possible starting point for continual learning.** arXiv:2603.03818 shows pretrained VLAs are "remarkably resistant to forgetting" — simple Experience Replay sometimes achieves *zero* forgetting with a tiny buffer — while small from-scratch policies forget catastrophically, concluding "large-scale pretraining fundamentally changes the dynamics of continual learning." Jack is the bad case. arXiv:2603.11653 (RLC 2026, best paper at the ICRA 2026 RL4IL workshop) then shows that with a pretrained backbone, plain sequential fine-tuning with LoRA under on-policy RL is "remarkably strong" and frequently beats sophisticated continual-RL methods with "little to no forgetting."

**Freeze a pretrained trunk and learn adapters, and continual learning stops being a research problem and becomes a build task.** That is the whole answer, and it makes §6 unavoidable.

---

## 5. Can Jack be genuinely curious?

Yes — but almost nothing in the current stack contributes, and two modules actively reward standing still.

### Delete outright

**`Empowerment` (`UnifiedBrain.py:3269-3363`)** — three independent disqualifiers. (a) `:3358` returns `E[log q(a|s,s')]` alone; the variational bound (Mohamed & Rezende, arXiv:1509.08731, cited in its own docstring at `:3280`) is `I(A;S'|S) ≥ E[log q] − E[log p(a|s)]`. Without the source-entropy term the objective is maximised by shrinking `action_std` to its `exp(-5)` clamp floor — measured `log_q` = −34.07 at random init versus **+69.38** with a perfect inverse model and std pinned at the floor. **The argmax of this reward function is paralysis.** (b) `inverse_dynamics`/`forward_dynamics` appear in no loss anywhere, so it is random-network noise fed into the reward mix at ~20% weight. (c) 16-sample MI estimation over a 17-dim continuous action space is not tractable regardless; no 2024-2026 humanoid result uses empowerment as a primary objective. Nothing to repair.

**`AutotelicGoalGenerator` (`:3525-3649`)** — `update_goal_statistics` (`:3624`) has zero callers, so `goal_ptr` is permanently 0, so `generate_goal` always takes the `strategy == "random" or goal_ptr == 0` branch at `:3587`. The learning-progress, curiosity and competence branches (`:3595, 3602, 3610`) are unreachable dead code — and `explore_autonomously:4889` explicitly requests `learning_progress`, the one branch that can never run. Every call returns a Gaussian sample from an untrained prior. (`goal_ptr` is also a plain int, not a buffer, so even the goal bank is unreachable after a reload.)

**`explore_autonomously` (`:4850`)** — the action is read from `forward()` at `:4879`; the skill is sampled at `:4884` and the goal at `:4888`, *after*. The returned action at `:4898` is the unconditioned policy output. The author's own comment at `:4897` says "could integrate skill conditioning here". Any demo showing "Jack chose skill 37 and did X" is not showing causation.

### Fix, but deprioritise

**ICM/RND** — three real bugs worth fixing only if you keep them: (a) `:3144` returns `(reward − mean)/std`; RND (Burda et al., arXiv:1810.12894) divides by the running std of intrinsic *returns* and explicitly does **not** subtract the mean. Mean-centring makes ~half of all intrinsic rewards negative, and for a humanoid with fall-termination the return-maximising policy is **to fall over on purpose**. One-line fix. (b) `compute_rnd_reward:3082` feeds the frozen RND target the output of `feature_encoder`, which is itself trained by the ICM losses (`:3153-3158`) — the target's input drifts every step, so novelty never decays and the collapse optimum is a constant encoder. Give RND its own frozen encoder or raw normalised observations. (c) `_normalize_reward:3139-3141` does `self.reward_mean.copy_(...)` on a graph-carrying tensor with no `no_grad`/`self.training` guard — measured `reward_count 2.0 → 4.0` under `model.eval()` + `torch.no_grad()`, and in grad mode the buffer becomes non-leaf (`grad_fn=CopyBackwards`) so the **second** `backward()` raises a version-counter error. The first trainer that wires this in dies on step 2.

**`SkillDiscovery` (DIAYN)** — cut from 50 skills to ~8 and reframe. Currently the policy is never conditioned on the skill and `z` is resampled every call (`:4884, 4953`) instead of held per episode, so `I(S;Z)` is identically zero and the discriminator cannot beat chance. `num_skills=50` (`:3673`) also mismatches `Metacognition.learning_priority`'s 20 (`:3411`). But even fixed, DIAYN (arXiv:1802.06070) at humanoid dimensionality is documented to converge to distinguishable **static postures**, not behaviours.

### Keep and repurpose — this is the real answer to twitching

**`Metacognition`'s 5-head ensemble (`:3389-3396`)** is the best asset in the stack and it is pointed the wrong way. Today the heads map `d_model → action_dim`, i.e. they measure *policy* disagreement, and `uncertainty = sigmoid(variance)` (`:3441`) has a hard floor at 0.5 so Jack can never report confidence.

**Re-point them at next-latent prediction from (state, action) and use the ensemble variance as the curiosity reward.** This is exactly the published fix: Pathak et al., *Self-Supervised Exploration via Disagreement* (arXiv:1906.04161) and Plan2Explore (arXiv:2005.05960). On a purely stochastic transition every ensemble member converges to the same predictive mean, so **disagreement goes to zero while ICM prediction error stays high** — the signal is structurally immune to sensor noise. If that is too expensive, LPM (arXiv:2509.25438, ICLR 2026) is lighter: one extra network predicting the dynamics model's expected error, reward = the drop between iterations.

### The twitching-in-place failure mode, addressed directly

ICM's reward is forward-model prediction error (arXiv:1705.05363, reproduced at `:3062-3066`). Prediction error cannot separate epistemic uncertainty (reducible, worth exploring) from aleatoric noise (irreducible). For a humanoid this is not a toy concern: joint-encoder noise, contact jitter and actuator noise are high-entropy and irreducible, so **a policy that shivers in place farms unbounded ICM reward forever.** You have three structural preventions, and you should use all three:

1. **Epistemic signal, not prediction error** — the ensemble-disagreement swap above.
2. **A motion prior from mocap.** `MoCapLoader.py` (935 lines) is orphaned and `datasets/` is empty. RGSD (arXiv:2510.06203) documents that unsupervised skill discovery on a 69-dim-action humanoid produces "joints moving randomly and highly unstructured motions", and fixes it by contrastively grounding the skill latent in reference motion; BFM-Zero (arXiv:2511.04131), ExBody2 (arXiv:2412.13196), GMT and ASAP all build on mocap retargeting. If the policy is regularised toward a human-motion manifold, high-frequency jitter is off-manifold and penalised **regardless of how novel it looks**.
3. **Goal commitment.** Once a goal is selected, hold it for N seconds regardless of novelty spikes. This is Go-Explore's "derailment" (arXiv:1901.10995), which no module in Jack addresses — and it costs a timer. It is also what makes exploration *look* purposeful to a human observer rather than distractible, which is directly your product goal.

### What 2026 actually recommends instead

**LLM goal proposal over a structured world state**, with learning-progress prioritisation. MAGELLAN (arXiv:2502.07709) has an LLM predict its own competence and learning progress across a goal space, generalising over *semantic* relationships instead of per-goal counters — precisely the weakness of Jack's goal bank, which stores 1000 vectors with independent counters matched by raw L2 at a hardcoded 0.5 threshold (`:3636`). OMNI-EPIC (arXiv:2405.15568) proposes tasks that are both learnable and interesting given an archive of what's been learned. SENSEI (arXiv:2503.01584, ICML 2025) distills a VLM "interestingness" signal into a world-model reward and jointly maximises semantic reward and uncertainty — the closest published analogue to what Jack should be, from image observations and low-level actions. CurricuLLM (arXiv:2409.18382, ICRA 2025) validated LLM-emitted subtask curricula on a real humanoid.

The counterweight, so you don't over-reach: Voyager (arXiv:2305.16291) skill libraries do not transfer to continuous control — "inapplicable to continuous control problems". The dividing line is control granularity: LLMs for goal/skill selection, learned policies for torques.

**The critical asymmetry to exploit:** `InnerMonologue` is already invoked in the live frame loop (`VirtualWorld.py:821-857`) and already receives a `current_goal` field (`InnerMonologue.py:261, 342`) from an `AutonomousMind` that never supplies one. The plumbing for genuine autonomy is built and pointed at a dead source. Point it at a small set of **named goals with explicit success predicates**, feed it a *diffed* world-state block ("object X moved", "Y is new since last session"), and Jack becomes autotelic in the sense the literature means — **with no training at all.** One caveat: `InnerMonologue._use_llm` is set `True` only in `__init__` when a local HF causal-LM is present (`:225`); `UnifiedBrain` attaches the API provider afterwards (`:4065`) and never flips the flag, so the Anthropic/OpenAI path is dead. One line fixes it, and it is the difference between template noise and a real inner voice on this hardware.

---

## 6. The architecture decision

**Compose a frozen pretrained trunk with a small learned policy. Stop training the bespoke 105M brain from scratch. Freeze it in a branch.**

I am not hedging on this. Six reasons, in order of force:

1. **There is no data.** `datasets/` is 0 bytes. `MoCapLoader` finds zero `.bvh` files; its CMU download URLs 404 (verified: `una-dinosauria/cmu-mocap` ships ASF/AMC, not BVH), the `HTTPError` is swallowed, and `MoCapDataset.__len__` returns `max(1, len(index))` while `__getitem__` returns **randomised sinusoids with a random language label** (`:690, 701-706, 748-772`). A future Phase 1 would train on fake sine waves and report a falling loss. Twenty heads, each needing a supervision signal that does not exist.

2. **Every comparable 2026 system does the opposite.** SmolVLA (arXiv:2506.01844) is 450M on a *frozen* SmolVLM-2 and trains on one GPU. X-VLA adapts to a new robot by tuning 9M of 900M (~1%). openpi's π0.5 LoRA path trains 441M of 3.6B with 3.2B frozen. GR00T N1.5 puts a frozen Eagle/Cosmos-Reason VLM on a DiT action head.

3. **It solves continual learning for free** (§4): pretrained trunks barely forget with trivial replay (arXiv:2603.03818); LoRA + sequential fine-tuning then beats every sophisticated CL method (arXiv:2603.11653).

4. **The parameter budget is already indefensible.** Measured: `hierarchical_planner` 37.17M > `layers` 36.71M — the untrained planner is *larger than the backbone it sits on*, and it has no live call site. `temporal_memory` 12.64M, never invoked. `vision_encoder` **0.24M**, a from-scratch toy conv net (`PrismaticVisionEncoder:581`), because `use_pretrained_vision=False`. A 0.24M from-scratch encoder will never see. `ARCHITECTURE.md` worries the model is too small and proposes 256M-500M. **It is not too small — it is 38.6% inert and 100% untrained.** Do not scale a model whose parameters have zero gradient history.

5. **The compute arithmetic.** Measured on this box: 6,095 ATen dispatches per B=1 forward, so the rollout loop is CPU-dispatch-bound at ~27 env-steps/s — a T4 would sit ~95% idle. RL-Zoo3's Humanoid PPO recipe is 1e7 steps (the repo defaults to 2e6, 5× short, with `batch_size=64` instead of 256), which at current throughput is 84-139 hours: **more than your entire weekly quota, and impossible in a 12-hour session.** And RL-Zoo3's `benchmark.md` publishes **no PPO row for Humanoid at all** — only a2c 388, sac 6232±280, td3 5567, tqc 7239, all at 2M steps with a ~140K-param MLP. PPO-Humanoid is weak enough it wasn't published; running it with a 116M transformer is strictly worse.

6. **The goal is a virtual companion first.** For that, the binding capability is conversation grounded in world state plus persistent memory — both of which a frozen LLM gives you today. `UnifiedBrain.chat():5327` currently classifies a question by `startswith` over `['what','who',...,'can','do']` and a command by `startswith` over `['go','walk',...]`; "can you go get the ball" is misrouted to `ask()` because "can" is a question word, and "the ball, could you get it" matches neither and returns a canned template. No amount of adding keywords fixes that class of failure.

### The target architecture

- **S2 (0.2-2 Hz):** frozen LLM. Reads a rendered *structured world-state block* + top-k retrieved memories + PAD mood; returns **either speech or a structured tool call** from a fixed verb set — `goto(x,y)`, `pickup(obj)`, `look_at(obj)`, `say(text)`, `wait(s)`. This is the SIMA 2 (arXiv:2512.04797) / Voyager shape.
- **S1 (20-50 Hz):** one small learned policy — **the only thing trained from scratch**, 5-30M params, trained in MuJoCo on the ephemeral GPU, distilled into `ActionExpert`.
- **S0:** MuJoCo PD/motor actuators, **not learned**. Note `System0Controller` is genuinely optional here: every actuator in every scene is a direct-force `<motor>` with a `gear` (`humanoid_full.xml:323-338`, gear=100/200/300), so `ctrl` *is* a normalised torque command and MuJoCo clamps it to `ctrlrange` (`-1..1` or `-0.4..0.4`). The S0 tier's job — converting position targets to torques — only matters on real hardware, exactly as its own config comment says (`:164`). **Delete the three-timescale claim from the docs, not the design.**
- **Vision:** swap `PrismaticVisionEncoder` for a frozen DINOv2-S or SigLIP-B (~22-90M, downloadable) with a small trained projection into `d_model`. `use_pretrained_vision` (`:105`) is already the flag.

Jack's own config declares `llm_freeze: bool = True` with SmolLM2-1.7B at `UnifiedBrain.py:122-126`. **You already half-made this decision, then built a second from-scratch brain beside it and wired neither into the world.** Finish the decision.

**What "keeps learning" then means, concretely:** the SQLite memory grows; a skill library grows (name, precondition, parameterised controller call, success predicate — this is where continual learning actually lives, and Voyager's result is that it compounds *without* catastrophic forgetting, no EWC required); and S1 is periodically re-fine-tuned offline on logged episodes.

### Realistic scope at ~30 GPU-hrs/week, solo

- **3 months:** a humanoid that walks in MuJoCo, plus an LLM that remembers you between sessions.
- **6 months:** walks to named objects, a small skill library, LLM-driven idle behaviour that reads as curiosity.
- **12 months:** simplified pick-and-place and a defensible thesis.

"Make me coffee and bring it here" is not a 12-month target. Neither is hardware.

---

## 7. Does the physics-first thesis survive?

**No. Drop Phase 0 for the virtual companion goal.** Do not port it, do not fix it — delete it from the roadmap. Three independent grounds, any one sufficient.

**The premise is falsified.** The closest published experiment to Jack's claim is Vafa et al., *What Has a Foundation Model Found? Using Inductive Bias to Probe for World Models* (arXiv:2507.06952, ICML 2025): transformers trained on orbital trajectories until prediction error was tiny — exactly Phase 0's setup with a far better teacher — then probed with symbolic regression. The recovered force law was **nonsense**, and the models failed on adjacent physics tasks; they formed task-specific heuristics, not Newtonian mechanics. `README.md:109` asserts the opposite as fact ("it understands F=ma... applies physics principles, not pattern matching") with no evidence. Separately, Botev et al. (arXiv:2111.05458, NeurIPS 2021 D&B) benchmarked Hamiltonian/Lagrangian models against unstructured baselines across 17 physical-system datasets: physics-inspired priors "fail to significantly improve upon standard techniques" — and those had *true* structural constraints, whereas Jack has only an MSE term.

**Jack uses the weakest rung and calls it neuro-symbolic.** The PIRL taxonomy (arXiv:2309.01909) separates observational bias, learning bias, and *inductive* bias — architecture that structurally cannot violate the law. MSE against SymPy outputs is observational+learning bias, the forms with no guarantee. `PhysicsRuleBank` (`UnifiedBrain.py:1669-1683`) is `nn.Parameter(torch.randn(100, 256))` with a Python list of 25 hardcoded strings ("F=ma", "torque", …) plus 75 named `learned_i`. Nothing binds row 0 to Newton's second law. `README.md:208` sells `get_active_rules` as "shows which rules it's using — so you can see WHY". It reports which random vector got attention.

**The implementation is worse than no teacher.** No gravity, no mass matrix, no contact, no ground reaction, hardcoded `I=0.1` for all 17 joints, joint limits by `np.clip`; 10 targets of rank ~4 with `col9 == col4` identically; targets computed from a trainable latent with a reachable collapse optimum; the physics head never sees the action. **MuJoCo — already running at `VirtualWorld.py:875` — is a strictly better teacher, and free.**

**And the payoff is for a phase that doesn't exist.** "MathReasoner detects the F=ma violation and infers the motor is weaker" is not how anyone does this. A scalar prediction error does not identify *which* parameter changed; that is system identification, and the field's answer is RMA (Kumar et al., arXiv:2107.04034, RSS 2021): a privileged policy conditioned on ground-truth extrinsics in sim, plus an adaptation module regressing those extrinsics from ~50 steps of state-action history at ~10 Hz. That literally *is* "the motor is weaker, adapt" — as a learned regressor, trainable on a single T4. 2026 descendants: SplitAdapter (arXiv:2606.03297), Phys2Real (arXiv:2510.11689). Ranked for Jack, best to worst: RMA-style adaptation > strong domain randomisation > residual learning on a nominal controller > symbolic violation detection, which appears in **no deployed system**.

Symbolic runtime verification *is* real in 2026 — as CBF-QP / MPC safety filters on actions in real units (arXiv:2410.11157, arXiv:2405.13863, arXiv:2511.06385, CBF-RL arXiv:2510.14959). None require the network to have "understood" physics. Jack's `verify_action_safe` (`SymbolicCalculator.py:400-424`) compares against 500 N / 100 Nm ceilings while Humanoid-v5 actions live in `[-0.4, 0.4]` zero-padded 17→57 — **it returns "safe" for every action the system can ever produce.**

The AlphaGeometry analogy also breaks on the property that made AlphaGeometry work. Euclidean geometry has a closed formal axiomatisation, so DD+AR can *verify a proof exactly and cheaply*; the LM only proposes auxiliary constructions (Nature 625:476-482, 2024; AG2 arXiv:2502.03544). Humanoid dynamics has no proof object, its axioms are themselves approximate, and verification is the hard part. `SymbolicVerifier.verify_idea` (`AlphaGeometryLoop.py:158-198`) is a box check plus one Euler step of the gravity-free fake dynamics plus `if next_distance < current_distance: return True` — greedy one-step hill-climbing with a clamp. Remove the framing before a viva.

### What to keep

- **`SymbolicCalculator` as a frozen regression gate**, not a training teacher. Versioned probes with SymPy ground truth, deterministic, <60 s on CPU, mandatory before any adapter promotion (§4). This is a genuine differentiator — an incorruptible, non-drifting supervision anchor is exactly what stops multi-week intrinsic-reward drift into noise-chasing — but it is a *gate*, not a curriculum.
- **A unit-correct action limiter** in the live loop, against actual `ctrlrange`. ~30 lines, genuinely useful for a long-running companion. Call it a safety limiter.
- **The 0.1-weighted `dynamics_loss`** inside `compute_physics_loss` (`:5530`) — invert the weights, delete the physics head, index the *raw* observation, and Phase 0 becomes an honest self-supervised dynamics pre-train against MuJoCo. That is defensible.

### The thesis reframe

Move the differentiator from *physics-first* to **continual learning in a persistent embodied companion**. Physics-first is not defensible against 2026 literature and you would argue uphill for a year. Continual learning in a live loop is genuinely under-explored, is what you actually want, and is the repo's biggest measured hole — `brain.eval()` + `torch.no_grad()` + no optimizer means Jack literally cannot learn while living. Fixing that is a bigger and more novel contribution than Phase 0 ever was.

---

## 8. The plan

Assumptions: solo, infrequent multi-hour bursts, Kaggle 30 GPU-hrs/week with a 12 h session cap, Colab T4 ephemeral. **Do not buy GPU time.** After the fixes below, the biggest job on this roadmap is ~4 GPU-hours.

### Stage 0 — Make it runnable and make it fail loudly *(one evening, 0 GPU)*

**Goal:** stop the silent failures and the reputational bleeding.
**Why now:** every hour of GPU time before this produces a traceback or an invisible no-op.

| Task | Files |
|---|---|
| Delete or relabel the fabricated results table + "Working" status column | `README.md:224-236`, `:348` |
| Install `mujoco`, `pygame`, `gymnasium[mujoco]` into `/data/venvs/jackthelearner`; run `nice -n 19`, `OMP_NUM_THREADS=2` | env |
| Default scene → a real humanoid scene; assert `mj_model.nu == config.action_dim` and refuse to start | `VirtualWorld.py:139, 207-211, 888-890` |
| `logger.exception` (rate-limited) on brain-tick failure; abort after 10 consecutive; zero `current_action` instead of reapplying stale | `VirtualWorld.py:660-661, 888-890` |
| `raise SystemExit` on `--load` failure | `VirtualWorld.py:1897` |
| Split `_init_mujoco` so a Renderer failure doesn't null `mj_model`/`mj_data`; `os.environ.setdefault('MUJOCO_GL','egl')` when `DISPLAY` unset | `VirtualWorld.py:398-411` |
| Fix `_on_task_complete` to branch on outcome; increment `tasks_failed` | `TaskManager.py:531-533, 587-598, 707-740` |
| Fix emotional-state save (`history.to_json()`), per-field try/except, save assertion | `Persistence.py:406-420` |
| Fix auto-save clock; sidecar `.meta.json` for `list_saves`; `max_saves=3`; absolute `/data` save dir; `weights_only=True` | `Persistence.py:114/165, 861-888, 168` |
| `make_env` returning `None` → hard abort in Phase 0; delete the `torch.randn` fallback | `TrainingPipeline.py:585-588, 640-647` |
| `.detach()` at `TrainingPipeline.py:533`; leaf fix at `:843` | `TrainingPipeline.py` |
| `InnerMonologue._use_llm = True` when an API provider is attached | `InnerMonologue.py:216-227`, `UnifiedBrain.py:4065` |

**Acceptance:** `python VirtualWorld.py` opens a scene containing a humanoid with `nu == 17`; killing the brain mid-run produces an ERROR traceback and a clean exit; a save/reload round-trip restores PAD mood and history; `python TrainingPipeline.py --phase 2 --timesteps 20000` runs to completion on CPU (~75 min at the measured 10 steps/s) and mean episode reward rises off the ~60-80 random baseline.
**Unblocks:** everything.

### Stage 1 — The smallest run that proves the pipeline and visibly changes Jack *(2-3 sessions, ~4 GPU-hrs)*

**Goal:** a gradient computed on a GPU changes what the humanoid does in the room.
**Why now:** this is the one thing that has never happened, and until it happens no other work is verifiable.

**1a — prove the pipe with water (no GPU, ~2 hours).** Before training anything, prove export→load→act end to end with a *deliberately trivial* checkpoint: train `ActionExpert` for 200 steps to output constant zeros via `compute_flow_matching_loss`, export, load, observe the humanoid go limp on command. If that doesn't work, no amount of SAC will help.

- Add `TrainingPipeline.export_for_runtime()` writing `{'version', 'model_weights', 'obs_projection_weights', 'obs_mean', 'obs_var', 'arch': asdict(config), 'param_count'}` — `TrainingPipeline.py:304-319`, `Persistence.py:340-349`
- Add `VirtualWorld --brain-weights`; refuse on arch-fingerprint mismatch; capture and log `_IncompatibleKeys` — `VirtualWorld.py:1807-1899`, `Persistence.py:565-571`
- Move `obs_proj` + running stats onto `UnifiedBrain` as `self.obs_projection` and registered buffers — `TrainingPipeline.py:235-246` → `UnifiedBrain.py`
- One `build_observation(mj_model, mj_data)` emitting the Humanoid-v5 348-d vector, used by both — `VirtualWorld.py:1348-1356`, `TrainingPipeline.py:531`
- Add `assets/humanoid_v5.xml` as the canonical body
- Wire `compute_flow_matching_loss` into a new `train_phase1` — `UnifiedBrain.py:5545`, `TrainingPipeline.py`

**1b — make him walk (~4 GPU-hrs).** Step 1: SB3 **SAC**, `MlpPolicy [256,256]`, Humanoid-v5, 2M steps, `normalize=True`, on a Kaggle P100 — 2-4 h, published target **6232 ± 280**. Step 2: roll out 500K `(obs, action)` pairs and behaviour-clone into `ActionExpert` with `compute_flow_matching_loss` — B=256, ~2000 steps, ~15 min on a T4. Step 3: `export_for_runtime` → `VirtualWorld --brain-weights`.

*Why SAC-then-distill and not PPO-from-scratch:* SAC at 2M is the only configuration in this space with a published, independently reproduced number; PPO-Humanoid isn't in the RL-Zoo3 benchmark at all, and PPO-ing a 116M transformer would need 1e7 steps to maybe match. Distillation converts a proven 3-hour result into Jack's own weights in 15 minutes of stable supervised learning.

Also in this stage, because they're cheap and gate throughput: replace the single env with `SyncVectorEnv(num_envs=64)` and `batch_size` 64→256 (amortises the 6,095 dispatches; ~27 → ~800-1500 env-steps/s, worth ~50×, more than any GPU upgrade); only run the flow-matching loop when `dual_system.needs_new_chunk()` (`UnifiedBrain.py:2477-2484` currently discards 15 of every 16 chunks — measured 3 adoptions in 40 calls, ~2.8 wasted CPU-seconds per wall-second); time-based checkpointing with fp16 `*_best` and no optimizer state (860 MB + 930 MB EWC per set today); rsync checkpoints to `/data` over an SSH deploy key stored as a Kaggle Secret, Drive as backup only.

**Acceptance:** a Humanoid-v5 rollout under the loaded policy survives **>500 steps** without falling; the same checkpoint in `VirtualWorld` produces visually identical gait and matches the gym episode return within 20%; a behavioural test asserting the 500-step survival is added to `tests/` and **fails on random weights**. Change the README status column from "Working" to "Trained" only for what passes.
**Unblocks:** every claim about behaviour; the thesis's first real figure.

### Stage 2 — The learning loop *(3-4 sessions, ~2 GPU-hrs/week ongoing)*

**Goal:** Jack measurably improves between sessions.

- **Transition logger** in `_update_brain` → rolling shards on `/data` (~50-100 lines) — `VirtualWorld.py:601-661`
- **SQLite memory** replacing `CompanionMemory`, Generative-Agents scoring, nightly reflection — `UnifiedBrain.py:2791-2830`
- **Frozen SymPy regression gate** (§4/§7) — new module from `SymbolicCalculator.py`
- **Sleep job notebook**: shards → LoRA consolidation + reservoir replay + physics-gate batches → gate → upload adapter delta (few MB). Fix the clone URL (`TRAIN_ON_COLAB.ipynb` cell 2 still says `YOUR_USERNAME/JackTheWalker`) and replace every time estimate with a measured number.
- **Plasticity instrumentation**: dormant-unit fraction, param-norm growth, per-layer effective rank, logged every consolidation; soft shrink-and-perturb on trainable heads
- `TaskManager.get_state()/load_state()` (`:746-781`, currently zero callers) into `_collect_world_state` (`VirtualWorld.py:1402-1418`)

**Acceptance:** a session's experience appears on disk; a sleep run produces an adapter that passes the gate and measurably improves a held-out task success rate; effective rank and dormant fraction are plotted across ≥5 consolidations. **Reload a save and Jack's mood, memories and in-flight task survive.**
**Unblocks:** the thesis's actual contribution.

### Stage 3 — Curiosity that means something *(2-3 sessions, 0 GPU)*

**Goal:** Jack does things unprompted that a viewer reads as curiosity, and the behaviour is causally produced by a real signal.

- Delete `Empowerment`, `AutotelicGoalGenerator`, `explore_autonomously` (§5)
- Named goal set (10-20) with explicit success predicates + LLM proposer conditioned on a **diffed** world-state block; per-goal learning-progress tracker with a real outcome-reporting call site; target ~50% success rate (the competence idea at `:3610` was correct)
- Goal commitment timer (derailment); episodic novelty or a returnable-state archive (detachment)
- Re-point `Metacognition`'s ensemble at next-latent prediction; disagreement variance as the curiosity reward; fix the `sigmoid` floor at `:3441`
- Replace `VirtualWorld.py:866-869`'s hardcoded string with an actual goal selection
- Symbolic completion predicates per skill type read from `mj_data`, with the neural head as a tiebreaker — `TaskManager.py:326-342`; scale `max_frames` by the *measured* loop rate

**Acceptance:** with no user input for 10 minutes, Jack selects ≥3 distinct goals, commits to each for ≥15 s, narrates them through `InnerMonologue`, and the selection changes when the room changes. A twitch-in-place policy scores *lower* curiosity reward than a locomoting one.

### Stage 4 — Grounded conversation *(2 sessions, 0 GPU)*

Delete the substring router (`VirtualWorld.py:1262-1265` and its verbatim duplicate at `:1594-1597` — measured to route "What did you make yesterday?", "I made a big mistake today", "remove that please" all to TaskManager) and the `startswith` classifier (`UnifiedBrain.py:5327, 5337`). One LLM call per utterance: world-state block + retrieved memories + PAD mood → speech or a structured tool call. Make `TextOnlyWorld` non-blocking (stdin on a daemon thread + a real fixed-rate loop calling `task_manager.tick`) or delete it in favour of a single headless loop implementation.

**Acceptance:** "make me coffee and bring it here" decomposes correctly; "the ball, could you get it" is understood; "can you go get the ball" is not misrouted to Q&A. Jack ticks and acts in headless mode.

### Stage 5 — Thesis *(writing, ongoing)*

Reframe around continual learning (§7). Report the plasticity curves, the gate-pass rate, and the cross-session improvement. Include the inductive-bias probe as an ablation if any physics claim survives — and expect it negative.

---

## 9. What to delete

Deleting is progress. Roughly 47M inert parameters and ~600 KB of unreferenced source.

**Code — delete now:**
- `archive/AlphaGeometryLoop.py` — byte-identical to the root copy (md5 `99806de674f7afbecfd6e8b1128f84ef`). The March 29 "restore" copied it up and left the original.
- `UnifiedBrain.py:4072-4080` — the second, never-read `self.creative_loop`.
- `TaskManager._try_creative_reasoning` (`:642-670`) and its caller block (`:621-635`) — fed `torch.randn`, output discarded, and unreachable anyway via the `'solved'` key mismatch. Or fix all three; do not leave it.
- `Empowerment` (`UnifiedBrain.py:3269-3363`), its construction (`:3674`), its reward-mix slot (`:3721-3733`), `get_empowerment` (`:4970`) — ~95 lines, 3 untrained networks, one objective whose argmax is paralysis.
- `AutotelicGoalGenerator` (`:3525-3649`) and `progress_estimator` (`:3554`) — unreachable by construction.
- `explore_autonomously` (`:4850`) — returns a skill and goal that provably cannot influence its returned action.
- `Persistence._collect_autonomous_mind` / `_apply_autonomous_mind` (`:530-549`, `:624`, `:755-760`) and the `obs_projection` branches (`:353-361`, `:574-581`) — permanent no-ops with a vacuous self-test.
- `Personality.get_behavior_bias` (`:465-513`) — or wire it; the docstring is a false wiring claim.
- `SemanticActionAnchors` (`:3865`) + `compute_language_grounding_loss` (`:5652`) — 408,577 params, zero forward-pass reads. Delete together, or add the loss to a phase.
- `TextToSpeech` construction, or set `enable_tts=False` (`:142`) — the startup banner advertises a capability switched off three call sites downstream.
- `MoCapLoader`'s synthetic fallback (`:690, 701-706, 748-772`) — must raise on empty dataset, never fabricate sinusoids. Also its unread `use_ik`/`scale_factor` (`:53-54`) and the "GMR-style recursive IK" docstring claim (there is no IK code).
- `compute_hierarchical_loss` (`:5759`) — zero callers anywhere including archive.
- `RUN_ON_COLAB.ipynb` — invokes `RobustTrainer.py --phase 0..7`, a file that no longer exists at root.

**Gate off by default until wired** (so GPU-hours and checkpoints stop carrying them): `hierarchical_planner` (37.17M), `temporal_memory` (12.64M), `world_model` (2.97M), `autonomous_mind` (4.99M). Also `find_checkpoint('phase1_best')` (`:737`) and `('phase7_best')` (`:803`) — dead references to phases with no implementation.

**Archive:** nothing live imports it — `grep` for every module name returns zero hits outside comments at `UnifiedBrain.py:1694-1700`. `archive/RobustTrainer.py` alone is 386 KB. It's in git history; removing it makes every future grep and audit cheaper. Note it still *imports from live UnifiedBrain* (`:61-62`) and calls three of the orphaned losses, which is the only reason those functions look used.

**Docs — resolve the contradictions:**
- `TRAINING_PIPELINE_PLAN.md` is titled **"JackTheWalker"** with 8 phases (0-7); `README.md` describes a 5-phase roadmap for **JackTheLearner**; the code implements `{0, 2, 8}` (`TrainingPipeline.py:591, 727, 797`, CLI choices at `:897`). Three schemes, none corresponding. **Pick the code's numbers and rewrite both documents.**
- `README.md:224-236` results table → delete or mark "target metrics, not measured" (§3 D11).
- `README.md:348` "Working" → "Constructs, untrained" or "Implemented, unwired", per module.
- `README.md:451-462` MathReasoner sim-to-real narrative → delete; replace with RMA if hardware ever happens.
- `ARCHITECTURE.md` "CURRENT GAPS" 256M-500M scaling suggestion → delete. The model is not too small; it is 38.6% inert and 100% untrained.
- `RESEARCH_PAPERS.md` → add the continual-learning literature it has zero hits for.
- `requirements.txt` → add `pygame` and the audio/TTS deps, or mark the GUI as an optional extra.

---

## 10. Open questions for the owner

Each changes the work. My recommendation is given so silence defaults sensibly.

**1. Is the masters thesis contribution physics-first, or continual learning?**
→ **Continual learning in a persistent embodied companion.** Physics-first is refuted by arXiv:2507.06952 and arXiv:2111.05458 and you'd argue uphill for a year; continual learning is under-explored, is what you actually want, and is the repo's biggest real hole. *Silence defaults to: reframe.*

**2. Do you accept freezing the 105M trunk?**
→ **Yes.** This is the load-bearing decision (§6) and it must be made before the logging format and adapter plumbing are designed, or you'll build them twice. *Silence defaults to: freeze; keep the trunk in a branch for the thesis's "what we tried" chapter.*

**3. Is the humanoid body negotiable in the short term?**
→ **Yes — standardise on Gymnasium's 17-actuator Humanoid-v5 body for both training and living.** `jack_room.xml` (22 motors, and the *only* asset with an `eye` camera, at `:272`, while not being in `SCENE_CATALOG` at all) becomes the room, with a v5-compatible humanoid in it. Defer the 59-DOF `humanoid_full.xml` until a 17-DOF gait exists. *Silence defaults to: v5 body.*

**4. Local LLM or API for S2?**
→ **API** (`llm_api_enabled`, `UnifiedBrain.py:131`, currently `False` so `brain.api_llm` is always `None` and `TaskManager._decompose_with_llm:428` / `_replan:676` are dead). SmolLM2-1.7B in fp16 on a CPU-only ARM box shared with paying tenants is a latency and RAM problem even after D1 is fixed. Keep the local path as an offline fallback behind an explicit flag. *Silence defaults to: API primary, local opt-in.*

**5. Keep `SymbolicCalculator` at all?**
→ **Yes, but only as a frozen regression gate and a unit-correct action limiter.** Fix the missing gravity or replace the dynamics with MuJoCo; either way it stops being a training teacher. *Silence defaults to: keep as gate, delete Phase 0.*

**6. Is "make me coffee and bring it here" a thesis deliverable or a north star?**
→ **North star.** Committing to it as a deliverable forces the 8-phase pipeline, which is the worst possible fit for burst compute — Phase 7 is unreachable if Phase 0 alone eats a session. *Silence defaults to: north star; the deliverable is walk + converse + remember + one grounded task.*

**7. Do you want `--text-only` to remain a supported mode?**
→ **No — collapse to one loop.** Make the headless path the same fixed-rate loop rendering to an offscreen buffer with an optional stdin channel. Two loop implementations, one of which has no loop, is how this defect survived. *Silence defaults to: one loop, `TextOnlyWorld` deleted.*

**8. Buy GPU time?**
→ **No, not yet.** After the vectorisation fix, the full RL-Zoo3 recipe is ~3 h and Stage 1 is ~4 h — both inside Kaggle's free 30 h/week with 80%+ headroom, for months. Your bottleneck is a one-line autograd bug and five silent plumbing mismatches; money fixes none of them. If unattended multi-day runs ever become the constraint, skip Colab Pro and go straight to RunPod Community RTX 4090 (~$0.34/hr) or Vast.ai interruptible — $50/mo buys ~145 4090-hours, roughly an order of magnitude more compute than Colab Pro. *Silence defaults to: free tiers only.*

**9. Adopt MJX / JAX?**
→ **No.** MuJoCo's own docs put CPU humanoid at 650K steps/s (M3 Max) to 1.8M (64-core), i.e. ~15-27K per core; Colab's 2 vCPUs supply ~30-50K steps/s, and your policy can consume at most ~1500 even after vectorising. **The simulator is already 20-30× faster than the brain.** MJX solves a bottleneck you don't have, at the cost of rewriting all 16,643 lines in JAX. Revisit only if you ever run a <1M-param policy above 50K steps/s. Same answer for Isaac Lab and Genesis. *Silence defaults to: MuJoCo CPU + PyTorch.*