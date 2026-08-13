"""T4.02 — No modality collapse.

HYPOTHESIS (registry). Per-modality gradient norms stay within an order of
magnitude. Falsified by: one modality's gradient dominates by >10x — the
others are ignored. Null: balanced contribution. Metric:
max_modality_grad_ratio. The 10x gate is the registry's own sentence, written
2026-08-04 — exogenous, never touched by any pilot (the T2.08 lesson about
absolute gates calibrated from their own pilot does not arise).

WHAT IS MEASURED, SAID PLAINLY. UnifiedBrain fuses five senses — proprio,
vision, touch, audio, language — each as exactly one [B,1,d_model] token into
CrossModalFusion (UnifiedBrain.forward, "ENCODE ALL MODALITIES"). This spec
trains the SHIPPED brain under its SHIPPED objective (action_training_loss,
the one loss the robot actually uses, all five senses supplied — the T1.11
runtime-parity path) and reads |d loss / d token_m| at that boundary every
step. Same shape, same d_model space, directly comparable. The claim gated is
about the SENSORY FUSION boundary: proprio also enters the backbone directly
through JointTokenizer's joint tokens, so a weak proprio FUSION gradient with
a healthy joint-token path would honestly read as imbalance here — that is
the reading T4.02 wants, because a sense whose fusion token is ignored is a
sense the other senses cannot teach (GOAL.md: what he hears must be able to
teach what he sees).

THE FIXTURE — matched information, or the ratio measures nothing. If targets
are independent of inputs, the optimum ignores every modality, all boundary
gradients decay toward noise, and a ratio of noises is not a measurement
("assert at a point where the state can still tell the two outcomes apart").
So each modality m carries an independent k=8-dim latent z_m embedded in its
raw signal (proprio: fixed random subspace of the 256-d state; vision: 8
smooth random patterns, amplitude 0.18 around 0.5 grey; touch: subspace of
the 10-d contact vector; audio: 8 sinusoid amplitudes, 180..1440 Hz;
language: 1-of-8 token template, the class one-hot is the latent), and
target_actions = sum over m of a per-dim STANDARDISED linear readout of z_m,
divided by sqrt(5), plus 0.1 noise. Standardisation constants come from a
4096-draw reference sample, so every modality contributes an equal ~1/5
variance share BY CONSTRUCTION — the registry's "balanced contribution" null
made physical. Realised shares are measured on that reference draw and gated
as rig health (each in [0.10, 0.30] -> else VOID): a task that rewards the
senses equally is the licence to call an unbalanced gradient ARCHITECTURE.
Fresh batches every step (no memorisation). Latent dims, template ids,
readouts, and stream all derive from a per-seed RandomState (seeded
(seed*9973+17) mod 2^32 — small, no numpy overflow).

AGGREGATION. Per step, per modality: sqrt of the summed squared boundary
gradients over that step's backward (the loss forwards twice — flow term and
aux term — so each hook must fire exactly 2x/step, asserted as fired_ok ->
else VOID: attribution that did not attach is not a zero reading). Per-seed
statistic: mean norm over the LAST HALF of 300 steps (time-average what
oscillates; let the init transient pass), ratio = max_m / min_m, sentinel
1e12 if a modality reads exactly zero (JSON has no inf; a hard zero is
genuine collapse and must FAIL, not error). First-quarter ratio is reported
beside it so drift is visible. GATED: the WORST seed (report per partition,
gate the minimum — here max ratio over seeds <= 10).

LEARNING GATE (VOID, not FAIL). Mean loss over the last quarter must be
below the first quarter on every seed. A run that learned nothing cannot
arbitrate gradient routing (T2.02's principle); a NaN anywhere is VOID.

CONTROL (must fail). A plant the detector must catch: vision's fusion token
is wrapped so forward values are IDENTICAL and the backward gradient is
scaled x100 (y = c*x - (c-1)*x.detach()). Why not scale the values:
CrossModalFusion is pre-LN residual (x + attn(LN(x))), so a value-scale
plant's measured factor is model-dependent — LN eats it on the attention
path and the residual keeps it — and "this cannot leak, we normalised" is
exactly the argument PG.7 forbids; the grad-scale plant makes the TRUE
gradient of the modified computation 100x larger at the measured tensor, so
what is certified is the whole live detector (hook attachment, attribution,
accumulation, ratio) at a real operating point. Measured at a FROZEN
snapshot (no optimizer steps, 50 backwards): stepping x100 gradients at
lr 3e-4 would just be measuring divergence. Gate: in the control run the
planted modality must dominate the others' minimum by > 10x on every seed,
else the detector cannot see dominance and the run is VOID. CONDITION on
this control (LESSONS: write the condition into the guard note): it
certifies detection whenever honest vision sits within 10x of the others'
minimum at init. If the true world has vision >10x BELOW the other senses'
minimum, the control can read under 10x and VOID a run whose experiment
metrics already show a decisive FAIL — the verdict goes conservative, the
metrics stay legible, and no PASS can be fabricated by this hole.

GPU. One submission for the whole spec (module cache — run_spec calls
_experiment once per seed; the T2.01/T2.03 pattern). Kaggle first: expiring
W32 hours are assigned to this spec (OVERSIGHT 14th audit B3), a Kaggle
kernel computes server-side if the watcher dies, and JACK_REUSE_KERNEL
reattaches at zero quota. No mujoco, no transformers, no downloads: torch +
numpy + the cloned repo. Science lives in THIS module and the JOB string
only imports it (T0.16 lesson).

COVERS: one brain / unison (rule)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..protocol import Ledger, Status, run_spec
from ..registry import BY_ID
from ..gpu import build_job, submit

# The claim is about the shipped brain's fusion routing; its file hashes into
# the certificate.
IMPL_DEPS = ["UnifiedBrain.py"]

SEEDS = [0, 1, 2]
SMOKE_SEED = 90

MODALITIES = ("proprio", "vision", "touch", "audio", "language")
K_LATENT = 8                   # per-modality latent dims (touch caps this at 10)
N_CLASSES = 8                  # language classes (one-hot latent)
TOKENS_LEN = 12
IMG_RES = 224                  # what T1.03 feeds the shipped vision encoder
N_REF = 4096                   # reference draw for standardisation + shares
TARGET_NOISE = 0.1

BATCH = 16
STEPS = 300                    # experiment: train, measure over the last half
CTRL_STEPS = 50                # control: frozen snapshot, no optimizer steps
LR = 3e-4
FWD_PER_STEP = 2               # action_training_loss forwards twice per loss

# Gates — all exogenous (registry sentence / construction ideals), none from
# a pilot:
RATIO_MAX = 10.0               # the hypothesis's own "order of magnitude"
SHARE_BAND = (0.10, 0.30)      # ideal 0.20 for 5 matched modalities
PLANT_GAIN = 100.0             # control's backward scale on the vision token
ZERO_SENTINEL = 1e12           # recorded ratio when a modality reads exactly 0


# ── the matched-information fixture ──────────────────────────────────────
class _Fixture:
    """Per-seed generative world: each modality carries an equal, independent,
    standardised share of the target. Dims come from the LIVE config, not a
    transcription (LESSONS: when you can reference, reference)."""

    def __init__(self, seed: int, cfg):
        self.rng = np.random.RandomState((seed * 9973 + 17) % 2**32)
        rng, K = self.rng, K_LATENT
        self.obs_dim = cfg.obs_dim
        self.audio_n = cfg.audio_sample_rate
        self.tgt_dim = cfg.action_chunk_size * cfg.action_dim
        self.chunk = (cfg.action_chunk_size, cfg.action_dim)

        self.E_p = (rng.randn(K, self.obs_dim) / np.sqrt(K)).astype(np.float32)
        low = rng.randn(K, 3, 7, 7)                      # smooth: 7x7 -> 224
        self.B_v = (np.kron(low, np.ones((1, 1, IMG_RES // 7, IMG_RES // 7))
                            ).reshape(K, -1) / np.sqrt(K)).astype(np.float32)
        self.E_t = (rng.randn(K, 10) / np.sqrt(K)).astype(np.float32)
        t = np.arange(self.audio_n, dtype=np.float64) / float(self.audio_n)
        freqs = 180.0 * np.arange(1, K + 1)
        self.A_a = (np.sin(2 * np.pi * freqs[:, None] * t[None, :])
                    / np.sqrt(K)).astype(np.float32)
        self.templates = rng.randint(1, cfg.vocab_size,
                                     size=(N_CLASSES, TOKENS_LEN))
        self.W = {m: (rng.randn(K, self.tgt_dim) / np.sqrt(K)).astype(np.float32)
                  for m in MODALITIES}

        # Standardisation constants + realised shares, from one reference draw.
        zs, _ = self._draw_z(N_REF)
        f = {m: zs[m] @ self.W[m] for m in MODALITIES}
        self.mu = {m: f[m].mean(0) for m in MODALITIES}
        self.sd = {}
        for m in MODALITIES:
            sd = f[m].std(0)
            sd[sd < 1e-6] = 1e-6
            self.sd[m] = sd
        fhat = {m: (f[m] - self.mu[m]) / self.sd[m] for m in MODALITIES}
        tgt = sum(fhat.values()) / np.sqrt(len(MODALITIES))
        denom = tgt.var(0) + TARGET_NOISE ** 2
        self.shares = {m: round(float(np.mean(
            fhat[m].var(0) / (len(MODALITIES) * denom))), 4) for m in MODALITIES}

    def _draw_z(self, n: int):
        z = {m: self.rng.randn(n, K_LATENT).astype(np.float32)
             for m in ("proprio", "vision", "touch", "audio")}
        cls = self.rng.randint(0, N_CLASSES, size=n)
        z["language"] = np.eye(N_CLASSES, dtype=np.float32)[cls]
        return z, cls

    def draw(self, n: int):
        """One fresh batch: raw inputs per modality + target actions."""
        rng = self.rng
        z, cls = self._draw_z(n)
        state = z["proprio"] @ self.E_p + 0.5 * rng.randn(n, self.obs_dim
                                                          ).astype(np.float32)
        vis = 0.5 + 0.18 * (z["vision"] @ self.B_v) \
            + 0.08 * rng.randn(n, 3 * IMG_RES * IMG_RES).astype(np.float32)
        vision = np.clip(vis, 0.0, 1.0).reshape(n, 3, IMG_RES, IMG_RES)
        touch = z["touch"] @ self.E_t + 0.5 * rng.randn(n, 10).astype(np.float32)
        audio = z["audio"] @ self.A_a + 0.3 * rng.randn(n, self.audio_n
                                                        ).astype(np.float32)
        tokens = self.templates[cls]
        fhat = [((z[m] @ self.W[m]) - self.mu[m]) / self.sd[m]
                for m in MODALITIES]
        target = (sum(fhat) / np.sqrt(len(MODALITIES))
                  + TARGET_NOISE * rng.randn(n, self.tgt_dim).astype(np.float32))
        return {"state": state.astype(np.float32), "vision": vision,
                "touch": touch.astype(np.float32),
                "audio": audio.astype(np.float32), "tokens": tokens,
                "target": target.reshape(n, *self.chunk).astype(np.float32)}


# ── the detector ─────────────────────────────────────────────────────────
class _GradMeter:
    """Tensor-gradient hooks on the five fusion-boundary modules. Every
    forward's output registers a backward hook; per step the squared norms
    accumulate and pop() returns (l2 norm, fired count) per modality."""

    def __init__(self, modules: dict):
        import torch
        self._torch = torch
        self.acc = {m: 0.0 for m in MODALITIES}
        self.fired = {m: 0 for m in MODALITIES}
        self.handles = [mod.register_forward_hook(self._fwd(name))
                        for name, mod in modules.items()]

    def _fwd(self, name):
        def hook(_module, _args, out):
            if self._torch.is_grad_enabled() and out.requires_grad:
                out.register_hook(lambda g, n=name: self._add(n, g))
        return hook

    def _add(self, name, g):
        self.fired[name] += 1
        self.acc[name] += float(g.detach().double().pow(2).sum().item())

    def pop(self):
        norms = {m: self.acc[m] ** 0.5 for m in MODALITIES}
        fired = dict(self.fired)
        self.acc = {m: 0.0 for m in MODALITIES}
        self.fired = {m: 0 for m in MODALITIES}
        return norms, fired


class _GradScale:
    """Forward identity, backward x`gain` — the control's plant. Built as a
    module wrapper so the planted projection still answers to the same
    attribute the shipped forward reads."""

    def __new__(cls, inner, gain: float):
        import torch

        class _Wrap(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.inner = inner
                self.gain = gain

            def forward(self, x):
                y = self.inner(x)
                return self.gain * y - (self.gain - 1.0) * y.detach()

        return _Wrap()


def _train_measure(seed: int, steps: int, batch: int, device: str,
                   plant_gain: float | None = None) -> dict:
    """Build the shipped brain, run `steps` of the shipped objective on the
    matched fixture, read the boundary gradients. plant_gain None = the
    experiment (Adam steps); a float = the control (frozen snapshot)."""
    import torch
    from UnifiedBrain import UnifiedBrain, UnifiedBrainConfig

    torch.manual_seed(seed)
    # llm_enabled=False -> the trainable LSTM fallback tower + language_proj:
    # the PLASTIC text tower, download-free on any backend. Everything else is
    # the shipped default (T1.03's build).
    cfg = UnifiedBrainConfig(llm_enabled=False)
    brain = UnifiedBrain(cfg).to(device).train()
    fx = _Fixture(seed, cfg)

    inner_vision = brain.vision_proj
    if plant_gain is not None:
        brain.vision_proj = _GradScale(inner_vision, plant_gain).to(device)
    meter = _GradMeter({
        "proprio": brain.proprio_encoder,
        "vision": inner_vision,          # pre-plant: where the x100 lands
        "touch": brain.touch_proj,
        "audio": brain.audio_proj,
        "language": brain.language_proj,
    })

    opt = (torch.optim.Adam(brain.parameters(), lr=LR)
           if plant_gain is None else None)

    losses, norm_rows, fired_ok = [], [], True
    finite = True
    for _ in range(steps):
        b = fx.draw(batch)
        out = brain.action_training_loss(
            torch.from_numpy(b["state"]).to(device),
            torch.from_numpy(b["target"]).to(device),
            language=torch.from_numpy(b["tokens"]).long().to(device),
            vision=torch.from_numpy(b["vision"]).to(device),
            touch=torch.from_numpy(b["touch"]).to(device),
            audio=torch.from_numpy(b["audio"]).to(device))
        loss = out["loss"]
        if not bool(torch.isfinite(loss)):
            finite = False
            break
        if opt is not None:
            opt.zero_grad(set_to_none=True)
        else:
            brain.zero_grad(set_to_none=True)
        loss.backward()
        norms, fired = meter.pop()
        fired_ok = fired_ok and all(fired[m] == FWD_PER_STEP for m in MODALITIES)
        norm_rows.append(norms)
        losses.append(float(loss))
        if opt is not None:
            opt.step()

    for h in meter.handles:
        h.remove()

    def window_mean(rows):
        return {m: float(np.mean([r[m] for r in rows])) for m in MODALITIES}

    def ratio_of(mean_norms):
        lo, hi = min(mean_norms.values()), max(mean_norms.values())
        return ZERO_SENTINEL if lo == 0.0 else hi / lo

    n = len(norm_rows)
    late = window_mean(norm_rows[n // 2:]) if n else {m: 0.0 for m in MODALITIES}
    early = window_mean(norm_rows[:max(1, n // 4)]) if n else dict(late)
    q = max(1, len(losses) // 4)
    others_min = min(v for m, v in late.items() if m != "vision") if n else 0.0
    return {
        "n_params": int(sum(p.numel() for p in brain.parameters())),
        "finite": finite, "fired_ok": bool(fired_ok), "steps_run": n,
        "shares": fx.shares,
        "norms": {m: round(v, 6) for m, v in late.items()},
        "norms_first": {m: round(v, 6) for m, v in early.items()},
        "ratio": round(ratio_of(late), 4),
        "ratio_first": round(ratio_of(early), 4),
        "vision_dominance": round(
            ZERO_SENTINEL if others_min == 0.0 else late["vision"] / others_min, 4),
        "loss_first": round(float(np.mean(losses[:q])), 4) if losses else float("nan"),
        "loss_last": round(float(np.mean(losses[-q:])), 4) if losses else float("nan"),
    }


# ── remote entry point ───────────────────────────────────────────────────
def remote_run(seeds: list, steps: int = STEPS, ctrl_steps: int = CTRL_STEPS,
               batch: int = BATCH) -> dict:
    """Everything for the given seeds; runs on the GPU VM (or locally at
    reduced steps/batch for the smoke — argument SHAPES stay production)."""
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = {"gpu": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
           "seeds": []}
    for seed in seeds:
        row = {"seed": seed,
               "experiment": _train_measure(seed, steps, batch, device),
               "control": _train_measure(seed, ctrl_steps, batch, device,
                                         plant_gain=PLANT_GAIN)}
        out["seeds"].append(row)
        print("SEED_DONE", json.dumps(row), flush=True)
    return out


# ── GPU submission (one per spec — module cache, T2.01 pattern) ──────────
JOB = r'''
import json, os as _o
from experiments.tests.t4_02_no_modality_collapse import remote_run
out = remote_run(__SEEDS__)
json.dump(out, open(_o.path.join(_o.environ["JACK_OUT"], "t402.json"), "w"),
          indent=1)
print("DONE", out["gpu"], flush=True)
'''

_CACHE: dict = {}


def _submit(seeds: list) -> dict:
    body = JOB.replace("__SEEDS__", repr(list(seeds)))
    job = build_job(body)
    # Timeout scales with seeds: one submission runs every seed's experiment
    # AND control serially (LESSONS: multiply by seeds and arms before sizing
    # any budget or timeout).
    res = submit(job, prefer="kaggle",
                 est_hours=round(0.1 + 0.12 * len(seeds), 2),
                 timeout_s=1800 + 900 * len(seeds),
                 fetch=["t402.json"])
    if not res.ok:
        raise RuntimeError(f"T4.02 job failed on {res.backend}: {res.message}")
    out = json.loads(Path(res.artifacts["t402.json"]).read_text())
    out["backend"] = res.backend
    return out


def _experiment(seed: int) -> dict:
    if not _CACHE:
        _CACHE.update(_submit(SEEDS))
    rows = [r["experiment"] for r in _CACHE["seeds"]]
    all_shares = [s for r in rows for s in r["shares"].values()]
    return {
        "gpu": _CACHE["gpu"], "backend": _CACHE["backend"],
        "max_modality_grad_ratio": max(r["ratio"] for r in rows),
        "ratio_per_seed": [r["ratio"] for r in rows],
        "ratio_first_per_seed": [r["ratio_first"] for r in rows],
        "norms_per_seed": [r["norms"] for r in rows],
        "share_min": min(all_shares), "share_max": max(all_shares),
        "loss_first_per_seed": [r["loss_first"] for r in rows],
        "loss_last_per_seed": [r["loss_last"] for r in rows],
        "loss_decreased_all": float(all(
            r["loss_last"] < r["loss_first"] for r in rows)),
        "finite_all": float(all(r["finite"] for r in rows)),
        "fired_ok_all": float(all(r["fired_ok"] for r in rows)),
        "n_params": rows[0]["n_params"],
    }


def _control(seed: int) -> dict:
    rows = [r["control"] for r in _CACHE["seeds"]]
    return {
        "ctrl_vision_dominance_min": min(r["vision_dominance"] for r in rows),
        "ctrl_ratio_per_seed": [r["ratio"] for r in rows],
        "ctrl_norms_per_seed": [r["norms"] for r in rows],
        "ctrl_finite_all": float(all(r["finite"] for r in rows)),
        "ctrl_fired_ok_all": float(all(r["fired_ok"] for r in rows)),
    }


def _check(m: dict, c: dict):
    # Rig first: an invalid run is VOID, not evidence about the hypothesis.
    if not (m["finite_all"] and c["ctrl_finite_all"]):
        return Status.VOID          # NaN somewhere — measured nothing
    if not (m["fired_ok_all"] and c["ctrl_fired_ok_all"]):
        return Status.VOID          # a hook did not attach; 0 is not a reading
    if not (SHARE_BAND[0] <= m["share_min"]
            and m["share_max"] <= SHARE_BAND[1]):
        return Status.VOID          # fixture shares unbalanced — task, not brain
    if not m["loss_decreased_all"]:
        return Status.VOID          # a non-learner cannot arbitrate routing
    if c["ctrl_vision_dominance_min"] <= RATIO_MAX:
        return Status.VOID          # detector cannot see imposed dominance
    # The claim, on the worst seed.
    return m["max_modality_grad_ratio"] <= RATIO_MAX


def run(ledger: Ledger | None = None):
    return run_spec(BY_ID["T4.02"], _experiment, _check, control_fn=_control,
                    ledger=ledger)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "smoke":
        # Local, CPU, minutes: production shapes (full config, 224 vision,
        # 16 kHz audio, real objective), reduced ONLY in steps/batch. Proves
        # hooks attach, fixture shares balance, the plant inflates, and the
        # whole remote path runs — before any quota is spent.
        out = remote_run([SMOKE_SEED], steps=8, ctrl_steps=4, batch=2)
        row = out["seeds"][0]
        exp, ctl = row["experiment"], row["control"]
        assert exp["fired_ok"] and ctl["fired_ok"], "hooks did not all fire"
        assert all(v > 0 for v in exp["norms"].values()), "a dead modality?"
        assert ctl["vision_dominance"] > exp["vision_dominance"], \
            "plant did not inflate vision"
        print(json.dumps(out, indent=1))
        print("SMOKE OK")
    else:
        run()
