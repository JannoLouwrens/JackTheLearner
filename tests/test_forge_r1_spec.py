"""Round 1's frozen rule (F3, ORDERS J18, ruling 69): the committed spec says
what J18's FROZEN TERMS say, and every number the forge derived re-derives.
Each clause is also planted broken and must be caught."""
import copy
import json
from pathlib import Path

from experiments.forge import Refused, climb, validate_spec
from experiments.forge import r1
from experiments.forge.metric import metric_hash

SPEC_PATH = Path(__file__).resolve().parent.parent / "experiments/forge/rounds/r1/spec.json"
J18_IDS = ["K1", "K2", "K3", "K5", "K6", "N3", "C1", "C2"]


def spec():
    return json.loads(SPEC_PATH.read_text())


def j18(s):
    """J18's terms as assertions; raises AssertionError naming the term."""
    validate_spec(s)
    assert [c["id"] for c in s["candidates"]] == J18_IDS, "the 8 candidates of J18"
    src = {c["id"]: c["source"] for c in s["candidates"]}
    assert [i for i in J18_IDS if src[i] == "control"] == ["C1", "C2"], "2 plain-variation controls"
    assert s["baseline"] == "E-food" and s["baseline_settings"]["ent_coef"] == 0.1, "baseline fixed 0.1"
    assert len(s["twins"]) >= 1 and s["twin_settings"][s["twins"][0]]["lr"] == 0.0, "never-trains twin"
    assert s["rungs"]["S"]["decisions"] == 150_000, "S = 150k"
    assert s["rungs"]["M"]["decisions"] == 300_000, "M = 300k"
    assert len(s["rungs"]["S"]["seeds"]) == 3, "3 seeds a candidate at S"
    seeds = [x for r in s["rungs"].values() for x in r["seeds"]]
    assert not set(seeds) & set(range(801, 806)), "never seeds 801-805"
    assert len(set(seeds)) == len(seeds), "no seed reused across rungs"
    o = s["outcome"]
    assert o["metric"] == "sustained_rate" and o["late_frac"] == 0.5, "late-half sustained share"
    assert "late meals" in o["tiebreak"], "tiebreak late meals/life"
    w = s["world"]
    assert w["arm"].startswith("E-food exactly"), "E2c's E-food arm exactly"
    for k in ("food_floor_only", "native_substep", "lazy_contact_audio"):
        assert w[k] is True, k
    assert w["eye_frame"] == w["motor_frame"] == "body", "body frames"
    assert s["machine"]["kaggle_sessions_max"] <= 2, "<= 2 Kaggle sessions"
    assert "S13 (b)" in s["acceptance"], "no result before the tripwire"
    k1 = next(c for c in s["candidates"] if c["id"] == "K1")["settings"]["param_resets"]
    assert any(0 < p < 150_000 for p in k1["S"]), ">= 1 reset inside rung S"


def broken(mutate, word):
    s = spec()
    mutate(s)
    try:
        j18(s)
    except (AssertionError, Refused) as e:
        assert word in str(e), f"caught for the wrong reason: {e}"
        return
    raise AssertionError(f"a spec broken on '{word}' passed")


def test_the_committed_spec_meets_j18():
    j18(spec())


def test_metric_hash_is_the_code_on_disk():
    assert spec()["outcome"]["metric_sha256"] == metric_hash()


def test_control_settings_are_the_stated_draws():
    c = {x["id"]: x["settings"] for x in spec()["candidates"]}
    assert c["C1"] == r1.control_draw(1) and c["C2"] == r1.control_draw(2)
    assert c["C1"] != c["C2"]
    for d in (c["C1"], c["C2"]):
        assert r1.BOX["lr"][0] <= d["lr"] <= r1.BOX["lr"][1]
        assert r1.BOX["tau"][0] <= d["tau"] <= r1.BOX["tau"][1]
        assert d["batch_size"] in r1.BOX["batch_size"]
        assert d["learning_starts"] in r1.BOX["learning_starts"]


def test_k1_resets_rederive_and_sit_inside_each_rung():
    s = spec()
    k1 = next(c for c in s["candidates"] if c["id"] == "K1")["settings"]
    for rung, size in s["rungs"].items():
        assert k1["param_resets"][rung] == r1.reset_points(size["decisions"])
        assert all(0 < p < size["decisions"] for p in k1["param_resets"][rung])
    assert r1.reset_points(150_000) == [50_000, 100_000]


def test_every_candidate_changes_something_and_has_a_prior():
    for c in spec()["candidates"]:
        assert c["prior"].strip() and c["change"].strip()
        assert {k for k in c["settings"] if k != "note"}, c["id"]


def test_budget_covers_its_own_derivation():
    s = spec()
    arms_s = len(s["candidates"]) + 2
    worst = arms_s * 3 * 2.5 + 6 * 5 * 5.0
    assert s["budget"]["core_hours"] >= worst
    assert s["budget"]["kernel_hours"] >= s["budget"]["core_hours"] / 4


def test_no_result_yet_and_the_rung_reads_empty():
    """Frozen, unrun: there is no results file, and reading rung S refuses for
    the missing baseline rather than inventing a reading."""
    assert not (SPEC_PATH.parent / "results.jsonl").exists()
    try:
        climb(spec(), [], "S")
    except Refused as e:
        assert "baseline" in str(e)
        return
    raise AssertionError("an unrun round was read")


def test_planted_wrong_terms_are_caught():
    broken(lambda s: s["rungs"]["S"].update(decisions=50_000), "S = 150k")
    broken(lambda s: s["rungs"]["M"].update(decisions=150_001), "M = 300k")
    broken(lambda s: s["rungs"]["S"].update(seeds=[801, 1402, 1403]), "801-805")
    broken(lambda s: s["rungs"]["S"].update(seeds=[1401, 1402]), "3 seeds")
    broken(lambda s: s["rungs"]["M"]["seeds"].__setitem__(0, 1401), "reused across rungs")
    broken(lambda s: s["candidates"].pop(1), "8 candidates")
    broken(lambda s: s["candidates"][6].update(source="published"), "controls")
    broken(lambda s: s["baseline_settings"].update(ent_coef="auto"), "fixed 0.1")
    broken(lambda s: s["outcome"].update(late_frac=0.0), "late-half")
    broken(lambda s: s["outcome"].update(requires_eating=False), "require eating")
    broken(lambda s: s["world"].update(motor_frame="world"), "body frames")
    broken(lambda s: s["machine"].update(kaggle_sessions_max=3), "Kaggle sessions")
    broken(lambda s: s["candidates"][0]["settings"]["param_resets"].update(S=[200_000]),
           "reset inside rung S")
    broken(lambda s: s["candidates"][5].update(prior=" "), "no prior")


if __name__ == "__main__":
    import sys
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    bad = 0
    for n, f in tests:
        try:
            f()
            print("PASS", n)
        except Exception as e:   # noqa: BLE001
            bad += 1
            print("FAIL", n, repr(e))
    print(f"{len(tests) - bad}/{len(tests)} passed")
    sys.exit(1 if bad else 0)
