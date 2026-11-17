"""Speech Commands v2: is our corpus the published one? (learner-free task check)

PRIORS §7 "Reproduce first", row "ear": Speech Commands, 12 classes,
speaker-disjoint; DS-CNN 94.4% / BC-ResNet-1 96.6-96.9%. Before any ear
learner is trained on `/data/jack_corpora/speech_commands_v2`, the corpus
itself must be the published task — the same move as N6-rl2-task.

Source: Warden 2018, "Speech Commands: A Dataset for Limited-Vocabulary
Speech Recognition", arXiv 1804.03209v1, §6 (read from the PDF via pypdf,
11-17): "The final dataset consisted of 105,829 utterances of 35 words" and
"There are 2,618 speakers recorded, each with a unique eight-digit
hexadecimal identifier". Splits: the corpus's own `validation_list.txt` and
`testing_list.txt`; train = every other word clip.

RULE (ruling 33 form, committed BEFORE the first run; no extension):
  R1 utterances == 105,829 (wav files in word folders, `_background_noise_`
     excluded), exact.
  R2 words == 35, exact.
  R3 speakers == 2,618 (prefix before `_nohash_`), exact.
  R4 splits speaker-disjoint: no speaker in more than one of train /
     validation / testing.
  CONTROL (must FAIL R4): the same files re-split at random (seed 0) into
     sets of the same three sizes — must share speakers across splits.
  Reported only: agreement of the lists with the TF `which_set` SHA-1 rule
     (10% / 10%), per-split sizes.
REPRODUCES iff R1-R4 hold and the control fails R4; otherwise MISS (named).

Cost: pure file listing, < 1 CPU-min. Not a learner; it says the data is
the paper's data, nothing about Jack's ear.

    python -m experiments.speech_commands_task
"""
import hashlib
import json
import os
import random
import re
import sys

ROOT = "/data/jack_corpora/speech_commands_v2"
PUB = {"utterances": 105829, "words": 35, "speakers": 2618}


def which_set(fname, val_pct=10, test_pct=10):
    """TF speech_commands input_data.which_set, verbatim logic."""
    base = os.path.basename(fname)
    hash_name = re.sub(r"_nohash_.*$", "", base)
    h = hashlib.sha1(hash_name.encode()).hexdigest()
    pct = ((int(h, 16) % (2 ** 27 - 1 + 1)) * (100.0 / (2 ** 27 - 1)))
    if pct < val_pct:
        return "validation"
    if pct < val_pct + test_pct:
        return "testing"
    return "training"


def speaker(rel):
    return os.path.basename(rel).split("_nohash_")[0]


def disjoint(splits):
    sp = {k: {speaker(f) for f in v} for k, v in splits.items()}
    ks = list(sp)
    shared = {f"{a}&{b}": len(sp[a] & sp[b])
              for i, a in enumerate(ks) for b in ks[i + 1:]}
    return all(v == 0 for v in shared.values()), shared


def main(root=ROOT):
    words = sorted(d for d in os.listdir(root)
                   if os.path.isdir(os.path.join(root, d))
                   and not d.startswith("_"))
    files = sorted(f"{w}/{f}" for w in words
                   for f in os.listdir(os.path.join(root, w))
                   if f.endswith(".wav"))
    val = set(open(os.path.join(root, "validation_list.txt")).read().split())
    test = set(open(os.path.join(root, "testing_list.txt")).read().split())
    train = [f for f in files if f not in val and f not in test]
    splits = {"training": train, "validation": sorted(val),
              "testing": sorted(test)}
    ours = {"utterances": len(files), "words": len(words),
            "speakers": len({speaker(f) for f in files})}
    ok_disj, shared = disjoint(splits)
    rng = random.Random(0)
    shuf = files[:]
    rng.shuffle(shuf)
    n_v, n_t = len(val), len(test)
    ctrl = {"validation": shuf[:n_v], "testing": shuf[n_v:n_v + n_t],
            "training": shuf[n_v + n_t:]}
    ctrl_disj, ctrl_shared = disjoint(ctrl)
    lab = {f: ("validation" if f in val else "testing" if f in test
               else "training") for f in files}
    agree = sum(which_set(f) == lab[f] for f in files) / len(files)
    checks = {"R1": ours["utterances"] == PUB["utterances"],
              "R2": ours["words"] == PUB["words"],
              "R3": ours["speakers"] == PUB["speakers"],
              "R4": ok_disj, "control_fails_R4": not ctrl_disj}
    out = {"published": PUB, "ours": ours,
           "split_sizes": {k: len(v) for k, v in splits.items()},
           "shared_speakers": shared, "control_shared": ctrl_shared,
           "which_set_agreement": round(agree, 5), "checks": checks,
           "verdict": "REPRODUCES" if all(checks.values()) else "MISS"}
    print(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    main(*sys.argv[1:])
