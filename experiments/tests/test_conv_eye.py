"""J22 (1) h4, tests committed with the arm (before any pilot run):
  * the shift moves rays (whole [d, t] pairs, a roll) and nothing else;
  * the actor loss gives the encoder zero gradient, and an actor update
    leaves the critic's params and the actor's encoder copy unchanged;
  * the actor's encoder copy is the critic's after sync.
Run: /data/venvs/jack-std/bin/python -m experiments.tests.test_conv_eye"""
import jax
import jax.numpy as jnp
import numpy as np

from experiments import conv_eye as C

REST = 42


def _obs(b=8, seed=0):
    return jnp.asarray(np.random.default_rng(seed).normal(size=(b, C.VISION_DIMS + REST)), jnp.float32)


def test_shift_moves_rays_and_nothing_else():
    o = _obs(64)
    r = C.roll_rays(o, jax.random.PRNGKey(3))
    assert np.array_equal(np.asarray(r[:, C.VISION_DIMS:]), np.asarray(o[:, C.VISION_DIMS:]))
    rays_o = np.asarray(o[:, :C.VISION_DIMS]).reshape(-1, C.N_RAYS, C.N_CH)
    rays_r = np.asarray(r[:, :C.VISION_DIMS]).reshape(-1, C.N_RAYS, C.N_CH)
    shifts = []
    for a, b in zip(rays_o, rays_r):
        hit = [s for s in range(-C.SHIFT, C.SHIFT + 1) if np.array_equal(np.roll(a, s, axis=0), b)]
        assert len(hit) == 1, "each sample is one roll of whole rays within +-SHIFT"
        shifts.append(hit[0])
    assert len(set(shifts)) > 3, "per-sample shifts, not one for the batch"


def _states():
    from flax.training.train_state import TrainState
    from sbx.common.type_aliases import RLTrainState
    import optax
    o, a = _obs(), jnp.zeros((8, 8))
    actor = C.ConvEyeActor(net_arch=[64, 64], action_dim=8)
    qf = C.ConvEyeVectorCritic(net_arch=[64, 64], use_layer_norm=True)
    ap = actor.init(jax.random.PRNGKey(0), o)
    qp = qf.init({"params": jax.random.PRNGKey(1), "dropout": jax.random.PRNGKey(2)}, o, a)
    act = TrainState.create(apply_fn=actor.apply, params=ap, tx=optax.adam(3e-4))
    q = RLTrainState.create(apply_fn=qf.apply, params=qp, target_params=qp, tx=optax.adam(3e-4))
    return act, q, o


def test_actor_loss_gives_encoder_zero_gradient():
    act, q, o = _states()
    act = C.sync_encoder(act, q)
    leaves = lambda t: [np.asarray(x) for x in jax.tree_util.tree_leaves(t)]
    assert all(np.array_equal(x, y) for x, y in zip(
        leaves(act.params["params"]["encoder"]), leaves(q.params["params"]["encoder"])))

    def loss(p):
        d = act.apply_fn(p, o)
        acts = d.sample(seed=jax.random.PRNGKey(5))
        return -q.apply_fn(q.params, o, acts).min(0).mean()
    g = jax.grad(loss)(act.params)
    assert all(np.all(x == 0) for x in leaves(g["params"]["encoder"]))
    assert any(np.any(x != 0) for x in leaves(g["params"]["head"]))
    # one real update through the arm's own path: critic untouched, copy unmoved
    from sbx import SAC
    from flax.training.train_state import TrainState
    import optax
    from sbx.sac.sac import EntropyCoef
    ent = EntropyCoef(0.1)
    es = TrainState.create(apply_fn=ent.apply, params=ent.init(jax.random.PRNGKey(0))["params"],
                           tx=optax.adam(3e-4))
    a2, q2, *_ = C.ConvEyeSAC.update_actor_and_temperature(act, q, es, o, -8.0, jax.random.PRNGKey(9))
    assert all(np.array_equal(x, y) for x, y in zip(leaves(q2.params), leaves(q.params)))
    assert all(np.array_equal(x, y) for x, y in zip(
        leaves(a2.params["params"]["encoder"]), leaves(q.params["params"]["encoder"])))
    assert any(not np.array_equal(x, y) for x, y in zip(
        leaves(a2.params["params"]["head"]), leaves(act.params["params"]["head"])))


def test_shuffled_eye_moves_whole_rays_and_keeps_the_world():
    """ShuffledEye: each step permutes whole rays, nothing after the eye moves,
    and the world underneath is the sighted arm's (same seed, same obs set)."""
    from experiments.blind_eye import ShuffledEye
    from experiments.pilot_b2 import pilot_world
    sighted = pilot_world("H-conv-row128", True)(923)
    shuf = pilot_world("H-conv-row128-shuffled", True)(923)
    assert isinstance(shuf, ShuffledEye)
    o1, _ = sighted.reset(seed=923)
    o2, _ = shuf.reset(seed=923)
    a = np.zeros(sighted.action_space.shape, np.float32)
    for _ in range(3):
        n = shuf.n_vision
        assert n == C.VISION_DIMS
        assert np.array_equal(o1[n:], o2[n:])
        r1, r2 = o1[:n].reshape(-1, 2), o2[:n].reshape(-1, 2)
        assert sorted(map(tuple, r1)) == sorted(map(tuple, r2)), "same rays, whole pairs"
        o1, *_ = sighted.step(a)
        o2, *_ = shuf.step(a)


def test_partial_file_keeps_the_trace(tmp=None):
    import json, tempfile
    from pathlib import Path
    from experiments.pilot_b import PilotLives, from_partial
    d = Path(tempfile.mkdtemp())
    pl = PilotLives.__new__(PilotLives)
    pl.__dict__.update(partial_path=d / "c.partial.json", total_decisions=7, lives=[{"x": 1}],
                       trace_ref=None)
    pl.write_partial()
    assert "trace" not in json.loads(pl.partial_path.read_text()), "trace off: file unchanged"
    tr = [{"t": 5000, "actor_loss": -1.0}]
    pl.trace_ref = tr
    pl.write_partial()
    assert json.loads(pl.partial_path.read_text())["trace"] == tr
    assert from_partial("a", 1, pl.partial_path)["trace"] == tr


def main():
    test_partial_file_keeps_the_trace()
    test_shuffled_eye_moves_whole_rays_and_keeps_the_world()
    test_shift_moves_rays_and_nothing_else()
    test_actor_loss_gives_encoder_zero_gradient()
    print("test_conv_eye PASS")


if __name__ == "__main__":
    main()
