"""The probe's comparator must see a one-ulp change, a shape change and a missing key — else a
'bit-continuous' PASS could be the comparator being blind."""
import numpy as np
import torch

from experiments.j14_grad_resume_probe import _flat, diff


def test_identical_is_zero_and_a_one_ulp_change_is_seen():
    a = _flat({"world_model": {"w": torch.ones(3)}, "opt": [{"step": 5}]})
    b = _flat({"world_model": {"w": torch.ones(3)}, "opt": [{"step": 5}]})
    assert diff(a, b)["differ"] == 0
    b["world_model.w"] = np.nextafter(np.ones(3, dtype=np.float32), 2, dtype=np.float32)
    d = diff(a, b)
    assert d["differ"] == 1 and d["max_abs"] > 0


def test_shape_change_int_change_and_missing_key_are_seen():
    a = _flat({"x": torch.zeros(2), "n": 3, "only_a": torch.zeros(1)})
    b = _flat({"x": torch.zeros(3), "n": 4})
    d = diff(a, b)
    assert d["differ"] == 2 and d["only_in_one"] == ["only_a"]


def test_prefix_filter():
    a = _flat({"world_model": {"w": torch.zeros(1)}, "actor": {"w": torch.zeros(1)}})
    b = _flat({"world_model": {"w": torch.zeros(1)}, "actor": {"w": torch.ones(1)}})
    assert diff(a, b, "world_model")["differ"] == 0 and diff(a, b, "actor")["differ"] == 1
