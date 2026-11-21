"""A2-v1S: ours beside the official recipe, setting by setting (J14 (b), 11-21).

The frozen reading of `docs/research/prereg/a2_size_s.md` orders, on a miss, "diff ours against
official `ec526b9` setting by setting". This does it from the two sources themselves, not from
memory of either:
  theirs = danijar/dreamerv3 @ ec526b9 `configs.yaml`, `defaults` + `dmc_proprio`, resolved by the
           OFFICIAL `embodied/core/config.py` of that commit (its regex-pattern update is the point:
           `.*\\.layers` / `.*\\.units` are `re.match`ed against flat keys);
  ours   = sheeprl 0.8.5 hydra-composed with `rows.BASE_V1S` (exactly what the kernel runs).
Prints one line per setting: SAME / DIFFERS. Reads configs only: no training, no score, no band.

CONTROL (must change the answer): `--planted` resolves theirs with the patterns loosened to
`.*layers` / `.*units` (what a reader ASSUMES they do) - the encoder/decoder rows must then flip to
SAME; if they do not flip, the table is not reading the resolved config.

`--matched-enc` (or `--extra k=v ...`) composes extra overrides on top of BASE_V1S, prints
every flat key they move (BLAST) and the table for the composed config.

Run (needs sheeprl + pyyaml: the probe venv; the official clone at /data/src/dreamerv3_e01491f):
  /data/venvs/jack-a-probe/bin/python -m experiments.brain_a.v1s_diff [--planted]"""
import subprocess, sys, types

CLONE, COMMIT = "/data/src/dreamerv3_e01491f", "ec526b9"


def _show(path: str) -> str:
    return subprocess.run(["git", "-C", CLONE, "show", f"{COMMIT}:{path}"], check=True,
                          capture_output=True, text=True).stdout


def theirs(planted: bool = False) -> dict:
    import yaml
    src = _show("dreamerv3/embodied/core/config.py").replace("from . import path", "path = None")
    mod = types.ModuleType("off_config")
    exec(compile(src, "off_config.py", "exec"), mod.__dict__)
    cfgs = yaml.safe_load(_show("dreamerv3/configs.yaml"))
    block = dict(cfgs["dmc_proprio"])
    if planted:
        block = {k.replace(r"\.layers", "layers").replace(r"\.units", "units"): v for k, v in block.items()}
    return dict(mod.Config(cfgs["defaults"]).update(block).flat)


def ours(extra: tuple = ()) -> dict:
    from hydra import compose, initialize_config_module
    from omegaconf import OmegaConf
    from experiments.brain_a import rows
    with initialize_config_module(config_module="sheeprl.configs", version_base="1.3"):
        cfg = compose(config_name="config", overrides=list(rows.BASE_V1S) + list(extra))
    OmegaConf.resolve(cfg)      # `${algo.dense_units}` etc. -> the value the kernel runs
    flat = {}

    def walk(node, pre):
        for k, v in node.items():
            if isinstance(v, dict):
                walk(v, pre + k + ".")
            else:
                flat[pre + k] = v
    for top in ("algo", "env", "buffer", "fabric"):
        walk(OmegaConf.to_container(cfg[top], resolve=True), top + ".")
    return flat


# (what, theirs: key or function of the flat dict, ours: key)
W = "algo.world_model."
TABLE = [
    ("batch size", "batch_size", "algo.per_rank_batch_size"),
    ("batch length", "batch_length", "algo.per_rank_sequence_length"),
    ("gradient steps per decision", lambda t: t["run.train_ratio"] / (t["batch_size"] * t["batch_length"]), "algo.replay_ratio"),
    ("prefill (random transitions)", lambda t: max(t["batch_size"] * t["batch_length"], int(t["run.train_fill"])), "algo.learning_starts"),
    ("envs", "envs.amount", "env.num_envs"),
    ("action repeat", "env.dmc.repeat", "env.action_repeat"),
    ("precision", "jax.precision", "fabric.precision"),
    ("replay size", "replay_size", "buffer.size"),
    ("pretrain steps", "run.pretrain", "algo.per_rank_pretrain_steps"),
    ("rssm deter", "rssm.deter", W + "recurrent_model.recurrent_state_size"),
    ("rssm units (gru in)", "rssm.units", W + "recurrent_model.dense_units"),
    ("rssm units (prior)", "rssm.units", W + "transition_model.hidden_size"),
    ("rssm units (posterior)", "rssm.units", W + "representation_model.hidden_size"),
    ("stoch", "rssm.stoch", W + "stochastic_size"),
    ("classes", "rssm.classes", W + "discrete_size"),
    ("rssm unimix", "rssm.unimix", "algo.unimix"),
    ("rssm action clip", "rssm.action_clip", "algo.actor.action_clip"),
    ("ENCODER mlp layers", "encoder.mlp_layers", W + "encoder.mlp_layers"),
    ("ENCODER mlp units", "encoder.mlp_units", W + "encoder.dense_units"),
    ("DECODER mlp layers", "decoder.mlp_layers", W + "observation_model.mlp_layers"),
    ("DECODER mlp units", "decoder.mlp_units", W + "observation_model.dense_units"),
    ("reward head layers", "reward_head.layers", W + "reward_model.mlp_layers"),
    ("reward head units", "reward_head.units", W + "reward_model.dense_units"),
    ("reward bins", "reward_head.bins", W + "reward_model.bins"),
    ("cont head layers", "cont_head.layers", W + "discount_model.mlp_layers"),
    ("cont head units", "cont_head.units", W + "discount_model.dense_units"),
    ("actor layers", "actor.layers", "algo.actor.mlp_layers"),
    ("actor units", "actor.units", "algo.actor.dense_units"),
    ("actor min std", "actor.minstd", "algo.actor.min_std"),
    ("actor max std", "actor.maxstd", "algo.actor.max_std"),
    ("actor unimix", "actor.unimix", "algo.actor.unimix"),
    ("actor entropy scale", "actent", "algo.actor.ent_coef"),
    ("critic layers", "critic.layers", "algo.critic.mlp_layers"),
    ("critic units", "critic.units", "algo.critic.dense_units"),
    ("critic bins", "critic.bins", "algo.critic.bins"),
    ("slow critic fraction", "slow_critic_fraction", "algo.critic.tau"),
    ("slow critic every", "slow_critic_update", "algo.critic.per_rank_target_network_update_freq"),
    ("model lr", "model_opt.lr", W + "optimizer.lr"),
    ("model adam eps", "model_opt.eps", W + "optimizer.eps"),
    ("model grad clip", "model_opt.clip", W + "clip_gradients"),
    ("actor lr", "actor_opt.lr", "algo.actor.optimizer.lr"),
    ("actor adam eps", "actor_opt.eps", "algo.actor.optimizer.eps"),
    ("actor grad clip", "actor_opt.clip", "algo.actor.clip_gradients"),
    ("critic lr", "critic_opt.lr", "algo.critic.optimizer.lr"),
    ("critic adam eps", "critic_opt.eps", "algo.critic.optimizer.eps"),
    ("critic grad clip", "critic_opt.clip", "algo.critic.clip_gradients"),
    ("kl dyn scale", "loss_scales.dyn", W + "kl_dynamic"),
    ("kl rep scale", "loss_scales.rep", W + "kl_representation"),
    ("kl free nats", "dyn_loss.free", W + "kl_free_nats"),
    ("imagination horizon", "imag_horizon", "algo.horizon"),
    ("discount", lambda t: 1 - 1 / t["horizon"], "algo.gamma"),
    ("lambda", "return_lambda", "algo.lmbda"),
    ("return norm decay", "retnorm.decay", "algo.actor.moments.decay"),
    ("return norm max", "retnorm.max", "algo.actor.moments.max"),
    ("return norm low pct", lambda t: t["retnorm.perclo"] / 100, "algo.actor.moments.percentile.low"),
    ("return norm high pct", lambda t: t["retnorm.perchi"] / 100, "algo.actor.moments.percentile.high"),
]
SAME_BY_NAME = {("float16", "16-mixed")}


def same(a, b) -> bool:
    if (a, b) in SAME_BY_NAME:
        return True
    try:
        return abs(float(a) - float(b)) <= 1e-9 * max(1.0, abs(float(a)))
    except (TypeError, ValueError):
        return str(a) == str(b)


# A matched-encoder arm (J14 (b) NEXT, 11-21): the four overrides that would give ours the official
# dmc_proprio encoder + decoder. NOT frozen, NOT launched - the Director rules whether it runs.
MATCHED_ENC = (W + "encoder.mlp_layers=5", W + "encoder.dense_units=1024",
               W + "observation_model.mlp_layers=5", W + "observation_model.dense_units=1024")


def blast(extra) -> list:
    """Every flat key of ours that the extra overrides move: [(key, before, after)]."""
    a, b = ours(), ours(tuple(extra))
    return [(k, a.get(k), b.get(k)) for k in sorted(set(a) | set(b)) if a.get(k) != b.get(k)]


def main() -> int:
    args = sys.argv[1:]
    planted = "--planted" in args
    extra = ()
    if "--matched-enc" in args:
        extra = MATCHED_ENC
    if "--extra" in args:          # --extra k=v [k=v ...] (to the end of the line)
        extra = tuple(args[args.index("--extra") + 1:])
    if extra:
        moved = blast(extra)
        for k, x, y in moved:
            print(f"  MOVED    {k}: {x!s} -> {y!s}")
        print(f"BLAST {len(moved)} key(s) moved by {len(extra)} override(s)")
    t, o = theirs(planted), ours(extra)
    diffs = []
    for what, tk, ok in TABLE:
        tv = tk(t) if callable(tk) else t[tk]
        ov = o[ok]
        ok_ = same(tv, ov)
        diffs += [] if ok_ else [what]
        print(f"  {'SAME   ' if ok_ else 'DIFFERS'}  {what:30s} theirs {tv!s:10s} ours {ov!s:10s}"
              f"  [{tk if isinstance(tk, str) else 'derived'} | {ok}]")
    print(f"{'PLANTED (patterns loosened) ' if planted else ''}{len(TABLE) - len(diffs)}/{len(TABLE)} SAME;"
          f" DIFFERS: {', '.join(diffs) or 'none'}")
    if any("$" in str(o[ok]) for _, _, ok in TABLE):
        print("UNREADABLE: an unresolved interpolation is in the table")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
