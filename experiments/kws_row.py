"""The ear's known-answer row (J14 (a)): BC-ResNet-1 on Speech Commands v2, 12 classes, the authors' code UNMODIFIED.

Source: `Qualcomm-AI-research/bcresnet@656f947a2a757ab5f36f649ed303ea0191cf553e` (main = this commit, 11-18),
cloned at run time; `main.py`, `utils.py`, `bcresnet.py`, `subspectralnorm.py` must match the sha256 pins
below or the run refuses (SOURCE_MISMATCH). Code only, never weights: every run starts from zero.
Published: Kim et al. 2021 (arXiv:2106.04140), Table 1, BC-ResNet-1 on GSC v2 12-class:
**96.9 +- 0.30** over 5 seeds (KNOWN_ANSWERS N7-kws; read from the e-print 11-18).

## THE RULE — committed 2026-11-18 before any seed runs (ruling 33's form, as brain A's rows)

- Command, per run: `python main.py --ver 2 --tau 1` in the authors' clone (data prepared once per kernel
  by the authors' own `DownloadDataset` + `SplitDataset`, exactly main.py's `--download` branch).
  Recipe = theirs, untouched: 200 epochs, 5 warm-up, SGD lr 0.1 cosine to 0, momentum 0.9, wd 1e-3,
  batch 100, noise mixing p 0.8, SpecAugment off at tau 1, final-epoch model scored on the official
  v0.02 test set (no checkpoint selection — main.py has none).
- Score = the `test acc:` line main.py prints at the end (percent, official test set, 4,890 clips).
- Band = published mean +- 2 x published SD = **96.3 .. 97.5** (|our mean - 96.9| <= 0.60).
- Seeds 0, 1, 2. The row PASSES when the 3-seed mean is inside the band.
- The ONE extension, stated now: a miss at 3 seeds runs seeds 3 and 4 (fresh) and the row is judged once
  more at 5 seeds, same band. Nothing after that: a miss at 5 is a named miss, written with its numbers.
- Data check before any epoch counts: main.py prints `check num of data train/valid/test`; it must read
  **36923/4445/4890** (the paper's e-print split, = our corpus count in N7-kws). Anything else stops the
  run (DATA_MISMATCH) before it spends the session.
- Control — the shuffled-label twin, seed 0: the same command, but the TRAIN set's labels are permuted
  once (seeded, class counts kept) before training; valid and test labels are untouched. It MUST score
  **<= 15.0%** test accuracy, else the row is VOID. Why 15.0 and not 8.3: the best constant guess scores
  408/4890 = 8.34%, but the authors' augmentation is label-dependent (`Preprocess.__call__`: a label-0
  "silence" clip gets noise amplitude U(0,1), every other label U(0,0.1)), so a shuffled twin can still learn
  "loud noise -> whatever label 0 now names"; at test time silence clips are pure zeros. That channel is
  worth at most the 408 silence clips (8.3 points) on top of chance; 15.0 sits below 8.34 + 8.34 and far
  below the band. A twin above 15.0 means the pipeline leaks labels some other way — VOID, not explained away.
- A run that is cut, crashes, or never prints `test acc:` is not a zero: the row is NOT JUDGED and says so.
- Settings that differ from the published protocol (ours beside theirs):
  * seeding: main.py seeds nothing; our launcher seeds `random`, `numpy`, `torch` (+cuda) with the run's
    seed before importing it (`_LAUNCHER` below) — a wrapper, never an edit;
  * stack: the backend's own torch/torchaudio (Kaggle image, recorded in VERSIONS) vs the README's
    torch 1.7.1 / torchaudio 0.7.2; GPU model unknown in the paper;
  * the twin's label permutation exists only in the control arm.
- Reported, not judged: per-epoch valid accuracy (main.py's `valid acc:` lines) and wall time per epoch.
- Not run on this box (18 CPU-h a seed, N7-kws): Kaggle GPU through `gpu.py` (prefer="kaggle"), pushed
  first, after brain A's rows and A's smoke (ruling 47). Cost UNMEASURED on a T4: EST_HOURS below is a
  guess from the box's 785 CPU-ms step (the model is 9.2K weights; the bottleneck on a GPU is expected to
  be `torchaudio.load` + the per-clip noise loop, num_workers 0). The first kernel's per-epoch wall time
  replaces the guess.

    python -m experiments.kws_row smoke            # box, CPU, ~5 min: the real main.py end to end on a
                                                   # 60-clip symlinked layout (both arms), parse checks
    python -m experiments.kws_row kernel s0 s1 s2 twin   # Kaggle GPU; runs in the order given
    python -m experiments.kws_row kernel s3 s4           # the extension, only after a miss at 3
    python -m experiments.kws_row score            # judge every harvested file
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import re
import subprocess
import sys
import time

REPO = "https://github.com/Qualcomm-AI-research/bcresnet"
COMMIT = "656f947a2a757ab5f36f649ed303ea0191cf553e"
SOURCE_SHA256 = {
    "main.py": "e34a3d69ef8e4def5e52b024c8d9dba8a505198f6b3ca899763694da5baa3d6e",
    "utils.py": "7497bb506186a196d67568d6deba23d1d77123581a08e90630a7caf5219f887d",
    "bcresnet.py": "cfff20b217fae3130763cdb9eca5fddfc425179cdf66306a47ca2fc62f2bf732",
    "subspectralnorm.py": "2d970e7ffcce7de20609b89e8192e873dca6c2023d04df56ea88f34fc284c833",
}
PUBLISHED = (96.9, 0.30)
BAND = (round(PUBLISHED[0] - 2 * PUBLISHED[1], 2), round(PUBLISHED[0] + 2 * PUBLISHED[1], 2))  # 96.3, 97.5
SEEDS = (0, 1, 2)
EXTENSION_SEEDS = (3, 4)
TWIN_SEED = 0
TWIN_MAX = 15.0
EXPECT_COUNTS = (36923, 4445, 4890)
EPOCHS = 200
EST_HOURS_PER_RUN = 1.5            # GUESS, unmeasured on a T4 (see the docstring)
RUN_DEADLINE_S = 10.0 * 3600       # in-kernel: no run starts after this; a running one is cut
REMOTE_TIMEOUT_S = int(11.5 * 3600)
OUT = "/data/jack-logs/kws_row"

COUNTS_RE = re.compile(r"check num of data train/valid/test (\d+)/(\d+)/(\d+)")
VALID_RE = re.compile(r"valid acc: ([0-9.]+)")
TEST_RE = re.compile(r"test acc: ([0-9.]+)")
SHUF_RE = re.compile(r"KWS_SHUFFLED (\d+)/(\d+)")

# The launcher: seeds, optionally permutes the TRAIN labels (twin), then runs the authors' main.py as
# __main__ with its own argv. It is written next to main.py; main.py itself is never touched.
_LAUNCHER = r'''
import os, random, runpy, sys
import numpy as np, torch
seed = int(os.environ["KWS_SEED"])
if os.environ.get("KWS_THREADS"):
    torch.set_num_threads(int(os.environ["KWS_THREADS"]))
random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, here)
import utils
if os.environ.get("KWS_SHUFFLE") == "1":
    _orig = utils.SpeechCommand.__init__
    def _init(self, root_dir, ver, transform=None):
        _orig(self, root_dir, ver, transform)
        if "train_12class" in root_dir:
            old = list(self.labels)
            random.Random(seed + 10_000).shuffle(self.labels)
            n = sum(a != b for a, b in zip(old, self.labels))
            print("KWS_SHUFFLED %d/%d" % (n, len(old)), flush=True)
    utils.SpeechCommand.__init__ = _init
if os.environ.get("KWS_PLUMB") == "1":   # N7-kws-2 (ruling 106): decode once + noise on device, wrappers only
    import atexit, kws_plumbing
    _st = kws_plumbing.install(utils)
    print("KWS_PLUMB installed", flush=True)
    atexit.register(lambda: print("KWS_PLUMB hits %d misses %d" % (_st["hits"], _st["misses"]), flush=True))
sys.argv = [os.path.join(here, "main.py")] + sys.argv[1:]
runpy.run_path(os.path.join(here, "main.py"), run_name="__main__")
'''


def sha256(path: str) -> str:
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def check_source(src: str) -> list:
    """[] when every pinned file matches; else the names that differ or are missing."""
    bad = []
    for f, h in SOURCE_SHA256.items():
        p = os.path.join(src, f)
        if not os.path.isfile(p) or sha256(p) != h:
            bad.append(f)
    return bad


def fetch_source(dst: str) -> str:
    if not os.path.isdir(os.path.join(dst, ".git")):
        subprocess.run(["git", "clone", "-q", REPO, dst], check=True)
    subprocess.run(["git", "-C", dst, "checkout", "-q", COMMIT], check=True)
    open(os.path.join(dst, "kws_launch.py"), "w").write(_LAUNCHER)
    import shutil   # N7-kws-2's plumbing beside the launcher (imported only when KWS_PLUMB=1)
    shutil.copy(os.path.join(os.path.dirname(os.path.abspath(__file__)), "kws_plumbing.py"),
                os.path.join(dst, "kws_plumbing.py"))
    return dst


def parse(text: str) -> dict:
    c = COUNTS_RE.search(text)
    t = TEST_RE.findall(text)
    s = SHUF_RE.search(text)
    return {"counts": tuple(int(x) for x in c.groups()) if c else None,
            "valid": [float(x) for x in VALID_RE.findall(text)],
            "test": float(t[-1]) if t else None,
            "shuffled": (int(s.group(1)), int(s.group(2))) if s else None}


def run_name(seed: int, twin: bool) -> str:
    return "twin_s%d" % seed if twin else "s%d" % seed


def run_one(src: str, work: str, seed: int, twin: bool, deadline: float = None,
            expect_counts=EXPECT_COUNTS, threads: int = None, env_extra: dict = None) -> dict:
    """One unmodified main.py process (through the launcher), cwd = `work` (main.py reads ./data)."""
    name = run_name(seed, twin)
    logf = os.path.join(work, name + ".log")
    env = dict(os.environ, KWS_SEED=str(seed), KWS_SHUFFLE="1" if twin else "0", PYTHONUNBUFFERED="1")
    if threads:
        env.update(KWS_THREADS=str(threads), OMP_NUM_THREADS=str(threads))
    env.update(env_extra or {})
    cmd = [sys.executable, os.path.join(src, "kws_launch.py"), "--ver", "2", "--tau", "1"]
    print("RUN", name, " ".join(cmd), flush=True)
    rec = {"name": name, "seed": seed, "twin": twin, "cmd": cmd, "cut": False, "stopped": None,
           "epoch_wall_s": []}
    t0 = time.time()
    last_n, last_t = 0, t0
    with open(logf, "w") as log, open(os.path.join(work, name + ".err"), "w") as err:
        # tqdm writes to stderr: kept apart so the stdout log stays parseable and small
        p = subprocess.Popen(cmd, cwd=work, stdout=log, stderr=err, env=env)
        while p.poll() is None:
            time.sleep(2)
            r = parse(open(logf, errors="replace").read())
            if r["counts"] is not None and expect_counts and tuple(r["counts"]) != tuple(expect_counts):
                rec["stopped"] = "DATA_MISMATCH %s != %s" % (r["counts"], expect_counts)
                p.kill()
                break
            if len(r["valid"]) > last_n:
                now = time.time()
                rec["epoch_wall_s"].append(round((now - last_t) / (len(r["valid"]) - last_n), 2))
                last_n, last_t = len(r["valid"]), now
                if last_n in (1, 5, 50, 100, 150, 200):
                    print("EPOCH", name, last_n, r["valid"][-1], rec["epoch_wall_s"][-1], flush=True)
            if deadline is not None and time.time() > deadline:
                rec["cut"] = True
                p.kill()
                break
        p.wait()
    text = open(logf, errors="replace").read()
    rec.update(parse(text))
    rec["rc"] = p.returncode
    rec["wall_s"] = round(time.time() - t0, 1)
    if rec["test"] is None:
        rec["log_tail"] = text[-2000:] + "\n--stderr--\n" + open(os.path.join(work, name + ".err"),
                                                               errors="replace").read()[-2000:]
    print("RUN_DONE", name, rec["rc"], rec["test"], rec["stopped"], rec["wall_s"], flush=True)
    return rec


def prepare_data(src: str, work: str) -> dict:
    """main.py's `--download` branch, by the authors' own functions, once per kernel (its logs are huge)."""
    t0 = time.time()
    code = ("import os,sys; sys.path.insert(0, %r); from utils import DownloadDataset, SplitDataset\n"
            "b='./data/speech_commands_v0.02'; u='https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz'\n"
            "t=b.replace('commands','commands_test_set'); ut=u.replace('commands_v','commands_test_set_v')\n"
            "os.makedirs('./data', exist_ok=True); os.mkdir(t); DownloadDataset(t, ut); os.mkdir(b); DownloadDataset(b, u); SplitDataset(b)\n"
            % src)
    with open(os.path.join(work, "prepare.log"), "w") as log:
        rc = subprocess.run([sys.executable, "-c", code], cwd=work, stdout=log, stderr=subprocess.STDOUT).returncode
    return {"rc": rc, "wall_s": round(time.time() - t0, 1)}


def plan_from_args(args) -> list:
    """['s0','s1','twin'] -> [(0, False), (1, False), (0, True)]."""
    out = []
    for a in args:
        if a == "twin":
            out.append((TWIN_SEED, True))
        elif re.fullmatch(r"s\d+", a):
            out.append((int(a[1:]), False))
        else:
            raise ValueError("unknown run %r" % a)
    return out


def run_kernel(plan, out_dir: str, deadline_s: float = RUN_DEADLINE_S, tag: str = "kws_row",
               work: str = "/tmp/kws_work") -> dict:
    """Kernel body: fetch + verify source, prepare data, then each run; the file is rewritten after each.
    Data and logs live in `work` (scratch: SplitDataset makes ~7 GB of copies, which must not become
    Kaggle output); only the json goes to `out_dir`."""
    t_end = time.time() + deadline_s
    os.makedirs(work, exist_ok=True)
    path = os.path.join(out_dir, tag + ".json")
    res = {"commit": COMMIT, "plan": [run_name(s, t) for s, t in plan], "runs": [], "stopped": None}
    src = fetch_source(os.path.join(work, "bcresnet_src"))
    bad = check_source(src)
    res["source_ok"] = not bad
    if bad:
        res["stopped"] = "SOURCE_MISMATCH %s" % bad
        json.dump(res, open(path, "w"))
        return res
    res["prepare"] = prepare_data(src, work)
    json.dump(res, open(path, "w"))
    if res["prepare"]["rc"] != 0:
        res["stopped"] = "PREPARE_FAILED"
        res["prepare_tail"] = open(os.path.join(work, "prepare.log"), errors="replace").read()[-3000:]
        json.dump(res, open(path, "w"))
        return res
    for seed, twin in plan:
        if time.time() > t_end:
            res["runs"].append({"name": run_name(seed, twin), "seed": seed, "twin": twin,
                                "not_started": "kernel deadline"})
            continue
        r = run_one(src, work, seed, twin, t_end)
        res["runs"].append(r)
        json.dump(res, open(path, "w"))
        if r["stopped"]:
            res["stopped"] = r["stopped"]
            break
    json.dump(res, open(path, "w"))
    return res


def complete(r: dict) -> bool:
    return (r.get("test") is not None and r.get("rc") == 0 and not r.get("cut") and not r.get("stopped")
            and tuple(r.get("counts") or ()) == EXPECT_COUNTS and len(r.get("valid", [])) == EPOCHS
            and (not r.get("twin") or (r.get("shuffled") or (0, 0))[0] > 0))


def judge(runs) -> dict:
    """runs: harvested run records (any number of files merged). Applies THE RULE above."""
    seeds, twin, notes = {}, None, []
    for r in runs:
        if not complete(r):
            notes.append("%s: not complete (rc=%s cut=%s stopped=%s test=%s %s)" % (
                r.get("name"), r.get("rc"), r.get("cut"), r.get("stopped"), r.get("test"),
                r.get("not_started", "")))
            continue
        if r["twin"]:
            twin = r["test"]
        else:
            seeds[r["seed"]] = r["test"]
    out = {"band": BAND, "seeds": seeds, "twin": twin, "notes": notes}
    want = set(SEEDS) | (set(EXTENSION_SEEDS) if set(EXTENSION_SEEDS) & set(seeds) else set())
    missing = sorted(want - set(seeds))
    if missing or twin is None:
        out["verdict"] = "NOT JUDGED"
        out["missing"] = missing + ([] if twin is not None else ["twin"])
        return out
    xs = [seeds[s] for s in sorted(want)]
    mean = sum(xs) / len(xs)
    sd = (sum((x - mean) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5 if len(xs) > 1 else 0.0
    out.update(mean=round(mean, 3), sd=round(sd, 3), n=len(xs))
    if twin > TWIN_MAX:
        out["verdict"] = "VOID (shuffled twin %.2f > %.1f)" % (twin, TWIN_MAX)
    elif BAND[0] <= mean <= BAND[1]:
        out["verdict"] = "PASS (in band)"
    else:
        out["verdict"] = "MISS"
    out["extension_due"] = out["verdict"] == "MISS" and len(xs) == len(SEEDS)
    return out


# ---------------------------------------------------------------------------------------------
# Box smoke: the REAL main.py, unmodified, end to end (200 epochs) on a tiny symlinked layout —
# 5 train / 2 valid / 2 test clips per class (silence = the official test set's zero clips).
# Proves: the source pin, the launcher, seeding, the twin's permutation, the parser, the counts stop.
# Says nothing about accuracy.
# ---------------------------------------------------------------------------------------------
CORPUS = "/data/jack_corpora/speech_commands_v2"
TESTSET = "/data/jack_corpora/speech_commands_test_set_v0.02"
KEYWORDS = ["down", "go", "left", "no", "off", "on", "right", "stop", "up", "yes"]


def build_smoke_layout(work: str, n_train=5, n_valid=2, n_test=2) -> tuple:
    base = os.path.join(work, "data", "speech_commands_v0.02")
    test = os.path.join(work, "data", "speech_commands_test_set_v0.02")
    os.makedirs(base, exist_ok=True)
    noise = os.path.join(base, "_background_noise_")
    if not os.path.exists(noise):
        os.symlink(os.path.join(CORPUS, "_background_noise_"), noise)
    silence = sorted(glob.glob(os.path.join(TESTSET, "_silence_", "*.wav")))
    unknown = sorted(glob.glob(os.path.join(CORPUS, "bed", "*.wav")))
    plan = [("train_12class", n_train, 0), ("valid_12class", n_valid, n_train), (None, n_test, n_train + n_valid)]
    for split, n, off in plan:
        root = test if split is None else os.path.join(base, split)
        for cls in KEYWORDS + ["_unknown_", "_silence_"]:
            d = os.path.join(root, cls)
            os.makedirs(d, exist_ok=True)
            pool = (silence if cls == "_silence_" else unknown if cls == "_unknown_"
                    else sorted(glob.glob(os.path.join(CORPUS, cls, "*.wav"))))
            for i, f in enumerate(pool[off:off + n]):
                dst = os.path.join(d, "%d.wav" % i)
                if not os.path.exists(dst):
                    os.symlink(f, dst)
    return 12 * n_train, 12 * n_valid, 12 * n_test


def smoke() -> int:
    work = os.path.join(OUT, "smoke")
    os.makedirs(work, exist_ok=True)
    src = fetch_source("/data/tmp/bcres_clone")
    bad = check_source(src)
    print("SOURCE", "ok" if not bad else "MISMATCH %s" % bad, flush=True)
    counts = build_smoke_layout(work)
    ta = {"PYTHONPATH": "/data/tools/torchaudio_lib"}   # torchaudio 2.8.0, --no-deps side dir (box only)
    out = {"source_ok": not bad, "counts_expected": counts, "runs": []}
    for seed, twin in ((0, False), (0, True)):
        r = run_one(src, work, seed, twin, time.time() + 900, expect_counts=counts, threads=2, env_extra=ta)
        out["runs"].append({k: v for k, v in r.items() if k != "valid"} | {"n_valid": len(r["valid"])})
    # the counts guard must fire on a wrong expectation (known-positive), within seconds
    r = run_one(src, work, 1, False, time.time() + 300, expect_counts=EXPECT_COUNTS, threads=2, env_extra=ta)
    out["guard"] = {"stopped": r["stopped"], "wall_s": r["wall_s"]}
    # N7-kws-2: the plumbed s0 must be the SAME run end to end (every valid acc and the test acc equal)
    plain0 = run_one(src, work, 0, False, time.time() + 900, expect_counts=counts, threads=2, env_extra=ta)
    pl = run_one(src, work, 0, False, time.time() + 900, expect_counts=counts, threads=2,
                 env_extra=dict(ta, KWS_PLUMB="1"))
    ptxt = open(os.path.join(work, "s0.log"), errors="replace").read()
    out["plumb"] = {"installed": "KWS_PLUMB installed" in ptxt, "rc": pl["rc"], "test": pl["test"],
                    "test_plain": plain0["test"], "valid_equal": pl["valid"] == plain0["valid"],
                    "n_valid": len(pl["valid"]), "hits_line": re.findall(r"KWS_PLUMB hits \d+ misses \d+", ptxt)}
    plumb_ok = (out["plumb"]["installed"] and pl["rc"] == 0 and pl["test"] is not None
                and pl["test"] == plain0["test"] and out["plumb"]["valid_equal"] and len(pl["valid"]) == EPOCHS)
    ok = (not bad and all(x["rc"] == 0 and x["test"] is not None and x["n_valid"] == EPOCHS
                          and tuple(x["counts"]) == counts for x in out["runs"])
          and out["runs"][1]["shuffled"] and out["runs"][1]["shuffled"][0] > 0
          and out["runs"][0]["shuffled"] is None
          and (r["stopped"] or "").startswith("DATA_MISMATCH") and plumb_ok)
    out["ok"] = bool(ok)
    json.dump(out, open(os.path.join(work, "smoke.json"), "w"), indent=1, default=str)
    print(json.dumps(out, default=str)[:3000])
    return 0 if ok else 1


_KERNEL_BODY = r'''
import json, os, subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "tqdm", "requests"], capture_output=True)
# gpu.py's preamble swaps the image's torch for 2.5.1+cu121 (P100 sm_60). The image's own
# torchaudio was built for the image's torch: its C++ lib then fails at import (OSError, not
# ImportError) or loads against the wrong ABI. Probe in a child process so a broken import
# cannot poison this one, and on any miss install torchaudio built for the torch that RUNS
# (same base version, same CUDA index, --no-deps so torch itself is never touched).
import importlib.metadata as _md
def _base(name):
    try:
        return _md.version(name).split("+")[0]
    except _md.PackageNotFoundError:
        return ""
_tv = _md.version("torch")
_probe = subprocess.run([sys.executable, "-c", "import torchaudio"], capture_output=True, text=True)
if _probe.returncode != 0 or _base("torchaudio") != _tv.split("+")[0]:
    _was = _base("torchaudio") or None
    _cmd = [sys.executable, "-m", "pip", "install", "-q", "--no-deps", "--force-reinstall",
            "torchaudio==" + _tv.split("+")[0]]
    if "+cu" in _tv:
        _cmd += ["--index-url", "https://download.pytorch.org/whl/" + _tv.split("+")[1]]
    r = subprocess.run(_cmd, capture_output=True, text=True)
    print("PIP_TORCHAUDIO", _tv, "was", _was, "probe_rc", _probe.returncode,
          "rc", r.returncode, r.stderr[-1500:], flush=True)
_probe = subprocess.run([sys.executable, "-c", "import torchaudio"], capture_output=True, text=True)
if _probe.returncode != 0 or _base("torchaudio") != _tv.split("+")[0]:
    # 11-21 02:30, kernel jack-ladder-1795228077: the Kaggle image is Python 3.13 and torchaudio
    # 2.5.1 has NO cp313 wheel ("from versions: none") -> TORCHAUDIO_MISMATCH, 0 epochs. Same cure as
    # brain A (speed._JOB): re-pin the trio to PyTorch's cu126 2.6.0 builds (cp313 wheels; still carry
    # sm_50..sm_90 for the P100), outside the preamble's PIP_CONSTRAINT, and refuse on an arch miss.
    import os as _os
    _env = {k: v for k, v in _os.environ.items() if k != "PIP_CONSTRAINT"}
    r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "torch==2.6.0", "torchvision==0.21.0",
                        "torchaudio==2.6.0", "--index-url", "https://download.pytorch.org/whl/cu126"],
                       capture_output=True, text=True, env=_env)
    open("/tmp/jack_torch_pin.txt", "w").write("torch==2.6.0\ntorchvision==0.21.0\ntorchaudio==2.6.0\n")
    _c = subprocess.run([sys.executable, "-c", "import torch; a = torch.cuda.get_arch_list(); "
                         "c = torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None; "
                         "print('EAR_TORCH', torch.__version__, a, c); "
                         "assert c is None or 'sm_%d%d' % c in a, 'NO KERNELS FOR THIS GPU'"],
                        capture_output=True, text=True)
    _tv = _md.version("torch")
    print("EAR_TORCH_REPIN", r.returncode, r.stderr[-800:], _tv, _c.returncode, _c.stdout, _c.stderr[-800:], flush=True)
    if r.returncode or _c.returncode:
        raise SystemExit("TORCH_REPIN_FAILED torch " + _tv + " " + r.stderr[-400:] + _c.stderr[-400:])
    _probe = subprocess.run([sys.executable, "-c", "import torchaudio"], capture_output=True, text=True)
if _probe.returncode != 0 or _base("torchaudio") != _tv.split("+")[0]:
    raise SystemExit("TORCHAUDIO_MISMATCH torch " + _tv + " torchaudio " + str(_base("torchaudio"))
                     + " " + _probe.stderr[-800:])
import importlib.metadata as md, platform, torch
print("VERSIONS", json.dumps({"python": platform.python_version(), "torch": torch.__version__,
      "torchaudio": md.version("torchaudio"), "cuda": torch.cuda.is_available(),
      "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}), flush=True)
if __PLUMB__:
    os.environ["KWS_PLUMB"] = "1"   # N7-kws-2 (ruling 106): run_one copies os.environ into each main.py
from experiments.kws_row import plan_from_args, run_kernel, RUN_DEADLINE_S
_r = run_kernel(plan_from_args(__PLAN__), JACK_OUT, tag=__TAG__, deadline_s=__DEADLINE__ or RUN_DEADLINE_S)
print("RESULT " + json.dumps({"stopped": _r["stopped"], "source_ok": _r.get("source_ok"),
      "prepare": _r.get("prepare"), "runs": [(x["name"], x.get("rc"), x.get("test"), x.get("cut"))
      for x in _r["runs"]]}), flush=True)
print("DONE", flush=True)
'''


def kernel_body(args, tag: str, plumb: bool = False, deadline_s: float = None) -> str:
    plan_from_args(args)  # refuse a bad plan on the box, not on the GPU
    return (_KERNEL_BODY.replace("__PLAN__", repr(list(args))).replace("__TAG__", repr(tag))
            .replace("__PLUMB__", repr(bool(plumb))).replace("__DEADLINE__", repr(deadline_s)))


# Row N7-kws-2 on Colab (ruling 106; fit rule frozen in KNOWN_ANSWERS: a seed fits Colab with 20% spare,
# <= 2.4 h). One run per Colab job — `kws_row` has no resume — and the runs in sequence, so this command,
# launched detached, IS the chain. The in-kernel run deadline is the fit rule's 2.4 h: a run over it is
# cut, and a cut run is NOT JUDGED (THE RULE), never a zero.
COLAB_RUN_DEADLINE_S = 2.4 * 3600
COLAB_TIMEOUT_S = int(3.0 * 3600)
OUT2 = "/data/jack-logs/kws_row2"


def colab_tag(run: str) -> str:
    return "kws2_" + run


def colab_chain(args, submit_fn=None, out_dir: str = OUT2) -> int:
    """Submit each run in `args` as its own Colab job, in order. A run whose json is home and
    `complete` is skipped (the chain can be relaunched); a run that comes home stopped/cut/failed
    stops the chain (a person reads why before another session is spent)."""
    plan_from_args(args)
    if submit_fn is None:
        from experiments.gpu import build_job, result_json, submit

        def submit_fn(body, tag):
            sub = submit(build_job(body), prefer="colab", est_hours=1.7, timeout_s=COLAB_TIMEOUT_S,
                         fetch=[tag + ".json"])
            print("SUBMIT", tag, sub.backend, sub.ok, sub.job_id, sub.message, flush=True)
            if not sub.ok:
                return None
            try:
                return result_json(sub, tag + ".json")
            except RuntimeError as e:   # no file and no RESULT line: stop, say why
                print("NO_RESULT", tag, e, flush=True)
                return None
    os.makedirs(out_dir, exist_ok=True)
    for run in args:
        tag = colab_tag(run)
        dst = os.path.join(out_dir, tag + ".json")
        if os.path.exists(dst):
            old = json.load(open(dst))
            if old.get("runs") and all(complete(r) for r in old["runs"]):
                print("SKIP", tag, "home and complete", flush=True)
                continue
        res = submit_fn(kernel_body([run], tag, plumb=True, deadline_s=COLAB_RUN_DEADLINE_S), tag)
        if res is None:
            print("CHAIN_STOP", tag, "submit failed", flush=True)
            return 1
        open(dst, "w").write(json.dumps(res))
        print("WROTE", dst, flush=True)
        runs = res.get("runs") or []
        if res.get("stopped") or not runs or not all(complete(r) for r in runs):
            print("CHAIN_STOP", tag, "not complete:", res.get("stopped"),
                  [(r.get("name"), r.get("rc"), r.get("cut"), r.get("test")) for r in runs], flush=True)
            return 1
    print("CHAIN_DONE", list(args), flush=True)
    return 0


def main(argv) -> int:
    os.makedirs(OUT, exist_ok=True)
    cmd = argv[1] if len(argv) > 1 else ""
    if cmd == "smoke":
        return smoke()
    if cmd == "score":
        runs = []
        for p in sorted(glob.glob(os.path.join(OUT, "kws_row_*.json"))):
            runs += json.load(open(p)).get("runs", [])
        print(json.dumps(judge(runs), indent=1))
        return 0
    if cmd == "colab" and len(argv) > 2:
        return colab_chain(argv[2:])
    if cmd == "kernel" and len(argv) > 2:
        from experiments.gpu import build_job, result_json, submit
        args = argv[2:]
        tag = "kws_row_" + "_".join(args)
        n = len(args)
        sub = submit(build_job(kernel_body(args, tag)), prefer="kaggle",
                     est_hours=0.3 + EST_HOURS_PER_RUN * n, timeout_s=REMOTE_TIMEOUT_S, fetch=[tag + ".json"])
        print("SUBMIT", sub.backend, sub.ok, sub.job_id, sub.message, flush=True)
        if not sub.ok:
            return 1
        dst = os.path.join(OUT, tag + ".json")
        open(dst, "w").write(json.dumps(result_json(sub, tag + ".json")))
        print("WROTE", dst, flush=True)
        return 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
