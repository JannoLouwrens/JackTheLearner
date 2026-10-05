"""T1.08 Step 1 — decompose heldout_cv_pct 40.006: how much is the EVAL SAMPLER?

THE DESIGN THIS EXECUTES: `docs/REVIEW_QUEUE.md`,
`t108-pipeline-repair-has-no-design` (DISPOSITIONED 2026-10-04, Review FULL),
STEP 1. Steps 0 and 1 are DUE 2026-10-08; Step 0 verified and committed alone
(1efd54f, 2026-10-05). This probe is the Step 1 measurement — A DIAGNOSTIC,
NOT A REPAIR. Step 2a/2b are deliberately NOT implemented in the same slot;
the Step 2 row gets routed off this probe's number by whichever desk sits
after it lands.

THE MEASUREMENT (the design's own terms): the seed-dependent inputs of
t1_08_seed_variance are a CLOSED set — task pinned at Generator(900), batch
order deterministic — leaving weight init, stochastic forward during training,
and the eval sampler draw (`generate_actions_flow_matching` starts from
`torch.randn(...)` UNSEEDED off the global RNG, UnifiedBrain.py:4523). So:
train ONE arm exactly as the spec does (seed 0, same task, same recipe via
make_action_optimizer, same STEPS/BS), then re-evaluate the SAME checkpoint
K=16 times under K distinct sampler seeds, and report `eval_cv_pct` beside
the recorded `heldout_cv_pct`.

PRE-REGISTERED, BEFORE THE RUN — all three outcomes declared so none can be
chosen after the fact (CV convention mirrors the spec's `_stats`: sample std,
ddof=1, CV = 100*std/|mean|):

  EVAL-DOMINANT      eval_cv_pct >= 28.289  (= 40.006/sqrt(2): under additive
                     variances, eval-draw noise accounts for >= half the
                     recorded across-seed variance)  -> the defect is in the
                     METRIC, repair is Step 2a (spec-local, zero certificates
                     outside T1.08).
  TRAINING-DOMINANT  eval_cv_pct <= 7.0  (eval noise alone cannot reach even
                     the bar)  -> the spread is training-borne, repair is
                     Step 2b (recipe; 19-certificate mechanical bill, 4
                     semantic — priced on the row).
  MIXED              7.0 < eval_cv_pct < 28.289  -> both mechanisms real;
                     routed off the number, no branch fires from this probe.

DECLARED LIMITS, also pre-registered: (i) variance additivity and
(ii) one-checkpoint representativeness (eval-draw dispersion at seed 0's
checkpoint stands in for all seeds) are the design's own assumptions; the
MIXED lane exists because of them. `heldout_natural` records the draw under
the RNG state exactly as the spec leaves it after training, so the probe's
checkpoint is tied to the spec's recorded metric.

WHAT DOES NOT MOVE: MAX_HELDOUT_CV_PCT stays 7.0 byte-unmoved under every
branch. This probe writes NO ledger row — it is not a run of T1.08 and must
not be recorded as one.

Cost: one GPU job, ~0.3 h (the est_hours already declared at t1_08:190 covers
three arms; this trains one arm + 17 forward passes). Backend: kaggle
preferred — 2026-W40 has 0.0 of 30 free hours drawn, expiring Sat 2026-10-10.
Artifact: /data/t108_step1_evalcv.json (plus JACKRESULT stdout-carry per the
t108 §9d lesson: the whole payload rides one delimited line).
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, "/home/opc/jackthelearner")
from experiments.gpu import build_job, submit  # noqa: E402

ART = Path("/data/t108_step1_evalcv.json")
ATTEMPT3_CV = 40.006           # ledger T1.08 attempt 3 (deda088, 2026-09-13)
EVAL_DOMINANT_MIN = round(ATTEMPT3_CV / 2 ** 0.5, 3)   # 28.289
BAR = 7.0                      # MAX_HELDOUT_CV_PCT — quoted, never moved here

JOB = r'''
import json, torch, torch.nn.functional as F
from UnifiedBrain import UnifiedBrain, UnifiedBrainConfig

DEV = "cuda" if torch.cuda.is_available() else "cpu"
# Identical to t1_08_seed_variance.JOB: same task generator, same constants,
# same recipe. A decomposition measured under a different configuration would
# not decompose the number it is supposed to decompose.
N_TRAIN, N_TEST, STEPS, BS, RANK = 2048, 512, 1500, 64, 8
K_DRAWS, SAMPLER_SEED_BASE, SEED = 16, 1000000, 0

def make_task(cfg):
    g = torch.Generator().manual_seed(900)
    n = N_TRAIN + N_TEST
    obs = torch.randn(n, cfg.obs_dim, generator=g)
    A = torch.randn(cfg.obs_dim, RANK, generator=g) / (cfg.obs_dim ** 0.5)
    B = torch.randn(RANK, cfg.action_chunk_size * cfg.action_dim, generator=g)
    tgt = (torch.tanh(obs @ A) @ B).view(n, cfg.action_chunk_size, cfg.action_dim) * 0.3
    return obs.to(DEV), tgt.to(DEV)

torch.manual_seed(SEED)
cfg = UnifiedBrainConfig()
cfg.llm_enabled = False
cfg.enable_intrinsic_motivation = False
brain = UnifiedBrain(cfg).to(DEV).train()
obs, tgt = make_task(cfg)
tr_o, tr_t, te_o, te_t = obs[:N_TRAIN], tgt[:N_TRAIN], obs[N_TRAIN:], tgt[N_TRAIN:]
opt, step_fn = brain.make_action_optimizer(lr=3e-4, warmup_steps=100,
                                           max_grad_norm=2.0)
for step in range(STEPS):
    i = (step * BS) % (N_TRAIN - BS)
    loss = brain.action_training_loss(tr_o[i:i+BS], tr_t[i:i+BS])["loss"]
    opt.zero_grad(); loss.backward(); step_fn()
brain.eval()

def heldout():
    with torch.no_grad():
        pred = brain.generate_actions_flow_matching(te_o)
        return float(F.mse_loss(pred.float(), te_t.float()))

# The canonical draw first: global RNG state exactly as the spec leaves it
# after training — this is the number attempt 3 would have recorded for this
# checkpoint, tying the probe to the recorded metric.
heldout_natural = heldout()
draws = []
for k in range(K_DRAWS):
    torch.manual_seed(SAMPLER_SEED_BASE + k)
    draws.append(heldout())
n = len(draws)
mean = sum(draws) / n
std = (sum((v - mean) ** 2 for v in draws) / max(n - 1, 1)) ** 0.5
out = {"gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
       "train_seed": SEED, "k_draws": K_DRAWS,
       "sampler_seed_base": SAMPLER_SEED_BASE,
       "heldout_natural": heldout_natural, "draws": draws,
       "eval_mean": mean, "eval_std": std,
       "eval_cv_pct": round(100 * std / max(abs(mean), 1e-9), 3)}
import os as _o
json.dump(out, open(_o.path.join(_o.environ["JACK_OUT"], "t108_step1.json"), "w"),
          indent=1)
print("JACKRESULT", json.dumps(out), flush=True)
print("DONE", flush=True)
'''


def main():
    job = build_job(JOB)
    res = submit(job, prefer="kaggle", est_hours=0.3, timeout_s=3000,
                 fetch=["t108_step1.json"])
    d = None
    path = res.artifacts.get("t108_step1.json") if res.ok else None
    if path:
        d = json.loads(Path(path).read_text())
    else:
        # stdout-carry survives a lost artifact (t108 ruling §9d).
        for line in (res.stdout or "").splitlines():
            if line.startswith("JACKRESULT "):
                d = json.loads(line[len("JACKRESULT "):])
                break
    if d is None:
        raise RuntimeError(
            f"no artifact and no JACKRESULT line from {res.backend}: "
            f"ok={res.ok} message={res.message!r} "
            f"stdout_tail={(res.stdout or '')[-400:]!r}")
    d["backend"] = res.backend
    d["attempt3_heldout_cv_pct"] = ATTEMPT3_CV
    cv = d["eval_cv_pct"]
    if cv >= EVAL_DOMINANT_MIN:
        d["reading"] = (f"EVAL-DOMINANT (eval_cv_pct {cv} >= {EVAL_DOMINANT_MIN})"
                        " -> Step 2a: the defect is in the metric, spec-local")
    elif cv <= BAR:
        d["reading"] = (f"TRAINING-DOMINANT (eval_cv_pct {cv} <= {BAR})"
                        " -> Step 2b: the spread is training-borne, recipe")
    else:
        d["reading"] = (f"MIXED ({BAR} < eval_cv_pct {cv} < {EVAL_DOMINANT_MIN})"
                        " -> routed off the number; no branch fires here")
    ART.write_text(json.dumps(d, indent=1))
    print(json.dumps(d, indent=1), flush=True)
    print(f"ARTIFACT {ART}", flush=True)


if __name__ == "__main__":
    main()
