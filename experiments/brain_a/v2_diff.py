"""Rows A1-A3 (size 1m, v2 agent): ours beside the official recipe, setting by setting (J14 (b), 11-21).

INBOX 11-21 19:15 asked it: A2-v1S was found NOT at the published size (official v1 `dmc_proprio`
patterns never matched `mlp_layers`), and "the 1m rows A1-A3 were never diffed this way against
THEIR official config". The 11-16 line diff (`README.md` §"Line diff") was read by eye from the yaml;
this one resolves both sides by their own code:
  theirs = danijar/dreamerv3 @ e01491f `configs.yaml`, `defaults` updated by `dmc_proprio` (which
           itself carries the size-1m block), resolved by the OFFICIAL `elements.Config` (3.19.1 =
           the commit's `requirements.txt` floor; sdist extracted at /data/tools/elements_3191) -
           its `.*\\.units` / `.*\\.depth` patterns are `re.match`ed against flat keys;
  ours   = sheeprl 0.8.5 hydra-composed with `rows.overrides("A2", 0)` (what the A2 kernel ran;
           A1/A3 differ only in domain/task/total steps).
Prints SAME / DIFFERS per setting, and names the DIFFERS rows that `rows.py`'s rule already
discloses ("ours beside theirs"). Configs only: no training, no score, no band.

CONTROL (must change the answer): `--planted` resolves theirs from `defaults` alone (the official
size-200m-class defaults, no `dmc_proprio`): the size rows must flip to DIFFERS; if they do not, the
table is not reading the resolved config.

Run: /data/venvs/jack-a-probe/bin/python -m experiments.brain_a.v2_diff [--planted]"""
import subprocess, sys, types

CLONE, COMMIT = "/data/src/dreamerv3_e01491f", "e01491f"
ELEMENTS_CONFIG = "/data/tools/elements_3191/elements-3.19.1/elements/config.py"


def theirs(planted: bool = False) -> dict:
    import yaml
    src = open(ELEMENTS_CONFIG).read().replace("from . import path", "path = None")
    mod = types.ModuleType("off_elements_config")
    exec(compile(src, "elements_config.py", "exec"), mod.__dict__)
    cfgs = yaml.safe_load(subprocess.run(["git", "-C", CLONE, "show", f"{COMMIT}:dreamerv3/configs.yaml"],
                                         check=True, capture_output=True, text=True).stdout)
    cfg = mod.Config(cfgs["defaults"])
    if not planted:
        cfg = cfg.update(cfgs["dmc_proprio"])
    return dict(cfg.flat)


def ours() -> dict:
    from experiments.brain_a import rows, v1s_diff
    return v1s_diff.ours(base=rows.overrides("A2", 0))


A, W = "agent.", "algo.world_model."
TABLE = [
    # size (the rows the control must flip)
    ("rssm deter", A + "dyn.rssm.deter", W + "recurrent_model.recurrent_state_size"),
    ("rssm hidden (gru in)", A + "dyn.rssm.hidden", W + "recurrent_model.hidden_size"),
    ("rssm hidden (prior)", A + "dyn.rssm.hidden", W + "transition_model.hidden_size"),
    ("rssm hidden (posterior)", A + "dyn.rssm.hidden", W + "representation_model.hidden_size"),
    ("classes", A + "dyn.rssm.classes", W + "discrete_size"),
    ("ENCODER mlp units", A + "enc.simple.units", W + "encoder.dense_units"),
    ("DECODER mlp units", A + "dec.simple.units", W + "observation_model.dense_units"),
    ("reward head units", A + "rewhead.units", W + "reward_model.dense_units"),
    ("cont head units", A + "conhead.units", W + "discount_model.dense_units"),
    ("actor units", A + "policy.units", "algo.actor.dense_units"),
    ("critic units", A + "value.units", "algo.critic.dense_units"),
    # structure
    ("stoch", A + "dyn.rssm.stoch", W + "stochastic_size"),
    ("gru blocks", A + "dyn.rssm.blocks", W + "recurrent_model.blocks"),
    ("dyn layers", A + "dyn.rssm.dynlayers", W + "recurrent_model.dynamics_layers"),
    ("img (prior) layers", A + "dyn.rssm.imglayers", W + "transition_model.layers"),
    ("obs (posterior) layers", A + "dyn.rssm.obslayers", W + "representation_model.layers"),
    ("latent unimix", A + "dyn.rssm.unimix", W + "unimix"),
    ("kl free nats", A + "dyn.rssm.free_nats", W + "kl_free_nats"),
    ("ENCODER mlp layers", A + "enc.simple.layers", W + "encoder.mlp_layers"),
    ("encoder symlog", A + "enc.simple.symlog", W + "encoder.symlog_inputs"),
    ("DECODER mlp layers", A + "dec.simple.layers", W + "observation_model.mlp_layers"),
    ("decoder bspace", A + "dec.simple.bspace", W + "observation_model.block_space"),
    ("reward head layers", A + "rewhead.layers", W + "reward_model.mlp_layers"),
    ("reward bins", A + "rewhead.bins", W + "reward_model.bins"),
    ("cont head layers", A + "conhead.layers", W + "discount_model.mlp_layers"),
    ("actor layers", A + "policy.layers", "algo.actor.mlp_layers"),
    ("actor min std", A + "policy.minstd", "algo.actor.min_std"),
    ("actor max std", A + "policy.maxstd", "algo.actor.max_std"),
    ("actor outscale", A + "policy.outscale", "algo.actor.outscale"),
    ("actor unimix", A + "policy.unimix", "algo.actor.unimix"),
    ("critic layers", A + "value.layers", "algo.critic.mlp_layers"),
    ("critic bins", A + "value.bins", "algo.critic.bins"),
    ("critic outscale", A + "value.outscale", "algo.critic.outscale"),
    # losses / returns
    ("loss rec", A + "loss_scales.rec", "algo.loss_scales.observation"),
    ("loss rew", A + "loss_scales.rew", "algo.loss_scales.reward"),
    ("loss con", A + "loss_scales.con", "algo.loss_scales.continue"),
    ("loss dyn", A + "loss_scales.dyn", "algo.loss_scales.dynamic"),
    ("loss rep", A + "loss_scales.rep", "algo.loss_scales.representation"),
    ("loss policy", A + "loss_scales.policy", "algo.loss_scales.policy"),
    ("loss value", A + "loss_scales.value", "algo.loss_scales.value"),
    ("loss repval", A + "loss_scales.repval", "algo.loss_scales.replay_value"),
    ("repval loss on", A + "repval_loss", "algo.critic.replay_loss"),
    ("repval grad on", A + "repval_grad", "algo.critic.replay_grad"),
    ("reward grad", A + "reward_grad", W + "reward_grad"),
    ("contdisc", A + "contdisc", W + "continue_discount"),
    ("actor entropy", A + "imag_loss.actent", "algo.actor.ent_coef"),
    ("lambda", A + "imag_loss.lam", "algo.lmbda"),
    ("slowreg", A + "imag_loss.slowreg", "algo.critic.slow_regularizer"),
    ("slow critic rate", A + "slowvalue.rate", "algo.critic.tau"),
    ("slow critic every", A + "slowvalue.every", "algo.critic.per_rank_target_network_update_freq"),
    ("imagination length", A + "imag_length", "algo.horizon"),
    ("discount", lambda t: 1 - 1 / t[A + "horizon"], "algo.gamma"),
    ("return norm decay", lambda t: 1 - t[A + "retnorm.rate"], "algo.actor.moments.decay"),
    ("return norm limit", A + "retnorm.limit", "algo.actor.moments.max"),
    ("return norm low pct", lambda t: t[A + "retnorm.perclo"] / 100, "algo.actor.moments.percentile.low"),
    ("return norm high pct", lambda t: t[A + "retnorm.perchi"] / 100, "algo.actor.moments.percentile.high"),
    # optimiser
    ("lr", A + "opt.lr", "algo.optimizer.lr"),
    ("opt eps", A + "opt.eps", "algo.optimizer.eps"),
    ("agc", A + "opt.agc", "algo.optimizer.agc"),
    ("warmup", A + "opt.warmup", "algo.optimizer.warmup"),
    ("weight decay", A + "opt.wd", "algo.optimizer.weight_decay"),
    ("beta1", A + "opt.beta1", lambda o: o["algo.optimizer.betas"][0]),
    ("beta2", A + "opt.beta2", lambda o: o["algo.optimizer.betas"][1]),
    # run recipe
    ("batch size", "batch_size", "algo.per_rank_batch_size"),
    ("batch length", "batch_length", "algo.per_rank_sequence_length"),
    ("replay context", "replay_context", "algo.replay_context"),
    ("online replay", "replay.online", "buffer.online"),
    ("replay size", "replay.size", "buffer.size"),
    ("prefill (random, before training)", lambda t: "none" if not any(
        k.startswith(("run.pretrain", "run.train_fill")) for k in t) else "KEY", "algo.learning_starts"),
    ("train ratio", "run.train_ratio",
     lambda o: o["algo.replay_ratio"] * o["algo.per_rank_batch_size"] * o["algo.per_rank_sequence_length"]),
    ("action repeat", "env.dmc.repeat", "env.action_repeat"),
    ("envs", "run.envs", "env.num_envs"),
    ("precision", "jax.compute_dtype", "fabric.precision"),
]
# DIFFERS rows `rows.py`'s rule names under "ours beside theirs" (the 2023 protocol: repeat 2, 4 envs,
# ratio 512; 32-true) - expected, disclosed. Any OTHER DIFFERS row is undisclosed.
DISCLOSED = {"train ratio", "action repeat", "envs", "precision"}
# DIFFERS rows only the 11-16 README line diff names (not the rule's text): replay 5e6 vs 500,000 (no
# run here exceeds 500k decisions), prefill none vs `learning_starts` 1024.
DISCLOSED_README = {"replay size", "prefill (random, before training)"}
SIZE_ROWS = {r[0] for r in TABLE[:11]}


def main() -> int:
    from experiments.brain_a.v1s_diff import same
    planted = "--planted" in sys.argv[1:]
    t, o = theirs(planted), ours()
    diffs = []
    for what, tk, ok in TABLE:
        tv = tk(t) if callable(tk) else t[tk]
        ov = ok(o) if callable(ok) else o[ok]
        s = same(tv, ov)
        diffs += [] if s else [what]
        tag = "SAME   " if s else ("DIFFERS (disclosed)" if what in DISCLOSED else
                                   "DIFFERS (README only)" if what in DISCLOSED_README else "DIFFERS")
        print(f"  {tag:21s} {what:34s} theirs {tv!s:10s} ours {ov!s}")
    undisclosed = [d for d in diffs if d not in DISCLOSED | DISCLOSED_README]
    print(f"{'PLANTED (defaults only, no dmc_proprio) ' if planted else ''}{len(TABLE) - len(diffs)}/"
          f"{len(TABLE)} SAME; disclosed DIFFERS: {', '.join(d for d in diffs if d in DISCLOSED) or 'none'};"
          f" README-only DIFFERS: {', '.join(d for d in diffs if d in DISCLOSED_README) or 'none'};"
          f" UNDISCLOSED DIFFERS: {', '.join(undisclosed) or 'none'}")
    if planted:
        flipped = SIZE_ROWS & set(diffs)
        print(f"CONTROL: {len(flipped)}/{len(SIZE_ROWS)} size rows flipped -> "
              f"{'FIRES (the table reads the resolved config)' if flipped == SIZE_ROWS else 'DID NOT FIRE'}")
        return 0 if flipped == SIZE_ROWS else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
