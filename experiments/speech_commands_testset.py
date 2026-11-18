"""Speech Commands v0.02 official 12-class TEST SET: is it the one BC-ResNet scores on, and is it
speaker-disjoint from the split a model trains on? (learner-free; KNOWN_ANSWERS N7-sc-testset)

Why: N7-kws (BC-ResNet-1, 96.6-96.9% on GSC v2 12-class) is scored on Google's canonical test
archive, a SEPARATE download the 11-18 cost probe found missing from the box. Before any ear
learner is scored against it, the archive must be the one the authors' loader expects, and it
must not share speakers with the train split we build from `/data/jack_corpora/speech_commands_v2`
(otherwise a test number would partly measure speaker memory, not keyword hearing).

Source of the expectations (read from the authors' code, `Qualcomm-AI-research/bcresnet@656f947a`,
copy at /data/tmp/bcres_src): main.py downloads
`https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_test_set_v0.02.tar.gz`
into its own test_dir and scans it with utils.ScanAudioFiles; utils.label_dict names the 12
classes; utils.sample_per_cls_v2 = [3077, 371, 408] caps `_silence_`/`_unknown_` at 408 in a
directory whose name contains "test". Local copy: /data/jack_corpora/speech_commands_test_set_v0.02
(fetched 11-18 03:23 UTC, 112,563,277 bytes, sha256 cc2a00c1...acd51df).

RULE (ruling 33 form, committed BEFORE the first run; no extension):
  S1 the archive's class folders == the 12 names of the authors' label_dict, exact.
  S2 `_silence_` == 408 and `_unknown_` == 408 clips (the authors' test cap is then a no-op,
     i.e. their loader sees the whole archive).
  S3 speaker-disjoint from training: no speaker id (prefix before `_nohash_`) of any test clip
     (the 10 keywords and `_unknown_`) is a speaker of the corpus's TRAIN split (train = word clips
     in neither validation_list.txt nor testing_list.txt).
  CONTROL (must FAIL S3): the same number of clips drawn at random (seed 0) from the corpus's
     train split, put through the identical S3 function, must share speakers with train.
  Reported only: total clips; speakers shared with the validation split; for keyword clips, the
     fraction whose `<word>/<name>` exists in the corpus, is listed in testing_list.txt, and is
     byte-identical to the corpus file; for `_unknown_`, the same against `<word>/<name>` parsed
     from `<word>_<name>.wav.wav`.
REPRODUCES iff S1-S3 hold and the control fails S3; otherwise MISS (named).

Cost: file listing + hashing ~4,900 small files, < 1 CPU-min. Not a learner; it says the test
data is the authors' and is clean of train speakers, nothing about Jack's ear.

    python -m experiments.speech_commands_testset
"""
import hashlib
import json
import os
import random
import sys

ROOT = "/data/jack_corpora/speech_commands_v2"
TEST = "/data/jack_corpora/speech_commands_test_set_v0.02"
OUT = "/data/jack-logs/speech_commands/testset_check.json"
LABELS = ["_silence_", "_unknown_", "yes", "no", "up", "down", "left", "right", "on", "off",
          "stop", "go"]  # utils.label_dict order
CAP_TEST = 408  # utils.sample_per_cls_v2[2]


def speaker(name):
    base = os.path.basename(name)
    return base.split("_nohash_")[0].split("_")[-1] if "_nohash_" in base else None


def corpus_splits():
    val = set(open(os.path.join(ROOT, "validation_list.txt")).read().split())
    tst = set(open(os.path.join(ROOT, "testing_list.txt")).read().split())
    train = []
    for w in sorted(os.listdir(ROOT)):
        d = os.path.join(ROOT, w)
        if not os.path.isdir(d) or w == "_background_noise_":
            continue
        for f in os.listdir(d):
            if f.endswith(".wav"):
                rel = f"{w}/{f}"
                if rel not in val and rel not in tst:
                    train.append(rel)
    return sorted(train), val, tst


def shares_speakers(clips, train_speakers):
    spk = {speaker(c) for c in clips} - {None}
    return sorted(spk & train_speakers)


def md5(p):
    return hashlib.md5(open(p, "rb").read()).hexdigest()


def main():
    folders = sorted(d for d in os.listdir(TEST) if os.path.isdir(os.path.join(TEST, d)))
    counts = {d: len([f for f in os.listdir(os.path.join(TEST, d)) if f.endswith(".wav")])
              for d in folders}
    train, val, tst = corpus_splits()
    train_spk = {speaker(c) for c in train}
    val_spk = {speaker(c) for c in val}

    test_clips, prov = [], {"exists": 0, "in_testing_list": 0, "identical": 0, "n": 0}
    prov_unk = dict(prov)
    for d in folders:
        if d == "_silence_":
            continue
        for f in sorted(os.listdir(os.path.join(TEST, d))):
            if not f.endswith(".wav"):
                continue
            test_clips.append(f"{d}/{f}")
            if d == "_unknown_":
                word, rest = f.split("_", 1)
                rel, p = f"{word}/{rest[:-4]}", prov_unk
            else:
                rel, p = f"{d}/{f}", prov
            p["n"] += 1
            src = os.path.join(ROOT, rel)
            if os.path.exists(src):
                p["exists"] += 1
                p["in_testing_list"] += rel in tst
                p["identical"] += md5(src) == md5(os.path.join(TEST, d, f))

    s1 = folders == sorted(LABELS)
    s2 = counts.get("_silence_") == CAP_TEST and counts.get("_unknown_") == CAP_TEST
    shared = shares_speakers(test_clips, train_spk)
    s3 = not shared
    ctrl = random.Random(0).sample(train, len(test_clips))
    ctrl_shared = shares_speakers(ctrl, train_spk)
    ctrl_fails = bool(ctrl_shared)
    verdict = "REPRODUCES" if (s1 and s2 and s3 and ctrl_fails) else "MISS"
    res = {"verdict": verdict, "S1_folders": s1, "S2_cap408": s2, "S3_train_disjoint": s3,
           "control_fails_S3": ctrl_fails, "control_shared_speakers": len(ctrl_shared),
           "counts": counts, "total": sum(counts.values()),
           "test_speakers": len({speaker(c) for c in test_clips} - {None}),
           "shared_with_train": shared[:20], "n_shared_with_train": len(shared),
           "shared_with_validation": len(({speaker(c) for c in test_clips} - {None}) & val_spk),
           "keyword_provenance": prov, "unknown_provenance": prov_unk,
           "corpus_train_clips": len(train)}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(res, open(OUT, "w"), indent=1)
    print(json.dumps(res, indent=1))
    return 0 if verdict == "REPRODUCES" else 1


if __name__ == "__main__":
    sys.exit(main())
