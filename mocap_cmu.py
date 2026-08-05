"""Real motion-language data: CMU ASF/AMC motion paired with HumanML3D text.

Why this module exists
----------------------
`MoCapLoader.py` parsed BVH and, when it found none, fabricated a sinusoid and
attached a label drawn by `np.random.randint` from a list of six strings. Ladder
spec T1.13 measured what that produced: dataset_len 1, real_f_ratio 0.0,
spectral_purity 0.9999, label_signal_advantage 0.0. The language was uncorrelated
with the motion BY CONSTRUCTION, so a grounding model trained on it could not
learn anything -- while showing a clean loss curve, because sinusoids are trivial
to fit. That is the worst kind of bug: one that hides its own failure.

The claim that no data was available turned out to be false. Two public sources,
neither needing an account:

  CMU Graphics Lab Motion Capture  http://mocap.cs.cmu.edu/allasfamc.zip  (1.0 GB)
      2514 AMC motion clips, 112 ASF skeletons. Free for all uses.
  HumanML3D texts                  github.com/EricGuo5513/HumanML3D       (12 MB)
      29232 files of human-written descriptions, ~3 per motion.

HumanML3D's index.csv names its source clips, and 2913 of them come from CMU --
which is the corpus above. Joining on that gives 2747 clips carrying 8196 real
descriptions over a 3635-word vocabulary, with per-clip frame ranges. Against the
ten hardcoded labels this replaces, that is not an incremental improvement.

Why ASF/AMC rather than BVH
---------------------------
AMC stores per-joint Euler rotations in degrees, one line per joint per frame,
which is precisely the `Dict[str, [x,y,z]]` that `SkeletonRetargeter.retarget_frame`
already consumes. BVH would require walking a hierarchy to recover the same
numbers. The one wrinkle is that AMC lists only a joint's ACTIVE degrees of
freedom, in an order declared by the ASF -- so the ASF must be parsed to know
whether "rfemur 1.2 3.4 5.6" means (rx, ry, rz) or (rz, ry, rx). It varies.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np

DATA_ROOT = Path("/data/jack-data")
PAIRS_JSON = DATA_ROOT / "cmu_humanml3d_pairs.json"

# CMU's ASF/AMC skeleton uses anatomical names; the retargeting table in
# MoCapLoader was written against BVH-converted names. Same joints, different
# vocabulary, so translate rather than duplicating the mapping.
AMC_TO_BVH_NAME: Dict[str, str] = {
    "lowerback": "Spine",
    "upperback": "Spine1",
    "thorax": "Spine2",
    "rfemur": "RightUpLeg",
    "rtibia": "RightLeg",
    "rfoot": "RightFoot",
    "lfemur": "LeftUpLeg",
    "ltibia": "LeftLeg",
    "lfoot": "LeftFoot",
    "rhumerus": "RightArm",
    "rradius": "RightForeArm",
    "rhand": "RightHand",
    "lhumerus": "LeftArm",
    "lradius": "LeftForeArm",
    "lhand": "LeftHand",
    "rclavicle": "RightShoulder",
    "lclavicle": "LeftShoulder",
    "head": "Head",
    "lowerneck": "Neck",
    "upperneck": "Neck1",
}

_AXIS_INDEX = {"rx": 0, "ry": 1, "rz": 2}


def parse_asf(path: Path) -> Dict[str, List[str]]:
    """Read a CMU skeleton and return each joint's rotational DOF order.

    Only the `dof` declarations matter here. A joint with `dof rx ry rz` writes
    three numbers per frame in that order; `dof rx` writes one. Reading AMC
    without this is how you silently get roll where you wanted pitch.
    """
    dofs: Dict[str, List[str]] = {"root": ["rx", "ry", "rz"]}
    current: Optional[str] = None
    in_bonedata = False

    for raw in path.read_text(errors="ignore").splitlines():
        line = raw.strip()
        if line.startswith(":bonedata"):
            in_bonedata = True
            continue
        if line.startswith(":hierarchy"):
            break
        if not in_bonedata:
            continue
        if line.startswith("name "):
            current = line.split()[1]
        elif line.startswith("dof ") and current:
            dofs[current] = [t for t in line.split()[1:] if t in _AXIS_INDEX]
    return dofs


def parse_amc(path: Path, dofs: Dict[str, List[str]]) -> List[Dict[str, np.ndarray]]:
    """Read an AMC motion file into per-frame {joint: [x, y, z]} degrees.

    Frames are delimited by a bare integer. The root line carries translation
    first and rotation after, so its rotation is taken from the LAST three values
    rather than the first three -- getting that backwards yields a skeleton that
    rotates when it should walk.
    """
    frames: List[Dict[str, np.ndarray]] = []
    current: Optional[Dict[str, np.ndarray]] = None

    for raw in path.read_text(errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith(("#", ":")):
            continue
        if line.isdigit():                       # frame delimiter
            if current:
                frames.append(current)
            current = {}
            continue
        if current is None:
            continue

        parts = line.split()
        joint, vals = parts[0], [float(v) for v in parts[1:]]
        rot = np.zeros(3, dtype=np.float64)

        if joint == "root":
            if len(vals) >= 6:
                rot[:] = vals[3:6]               # tx ty tz rx ry rz
        else:
            for axis, v in zip(dofs.get(joint, []), vals):
                rot[_AXIS_INDEX[axis]] = v

        current[AMC_TO_BVH_NAME.get(joint, joint)] = rot

    if current:
        frames.append(current)
    return frames


class CMUTextMotionCorpus:
    """The paired corpus: real CMU motion, real human-written descriptions.

    Deliberately does no retargeting itself -- `SkeletonRetargeter` in
    MoCapLoader already owns the CMU -> MuJoCo actuator mapping, and duplicating
    it would create exactly the kind of second implementation that the T0.07
    incident showed goes stale unnoticed.
    """

    def __init__(self, pairs_json: Path = PAIRS_JSON):
        if not pairs_json.exists():
            raise FileNotFoundError(
                f"{pairs_json} missing. Build it by joining HumanML3D's index.csv "
                "against the extracted CMU corpus -- see this module's docstring. "
                "Refusing to fall back to synthetic data (ladder spec T1.13)."
            )
        self.pairs = json.loads(pairs_json.read_text())
        self._asf_cache: Dict[str, Dict[str, List[str]]] = {}

    def __len__(self) -> int:
        return len(self.pairs)

    def _dofs_for(self, amc_path: Path) -> Dict[str, List[str]]:
        """Each CMU subject has its own ASF; cache per subject, not per clip."""
        subject = amc_path.parent.name
        if subject not in self._asf_cache:
            asf = next(amc_path.parent.glob("*.asf"), None)
            if asf is None:
                raise FileNotFoundError(f"no ASF skeleton beside {amc_path}")
            self._asf_cache[subject] = parse_asf(asf)
        return self._asf_cache[subject]

    def frames_for(self, entry: dict) -> List[Dict[str, np.ndarray]]:
        """Motion frames for one entry, clipped to HumanML3D's annotated range.

        The range matters: HumanML3D annotates a SEGMENT of a clip, so using the
        whole file would pair a description with motion it does not describe --
        reintroducing the exact decorrelation this module exists to remove.
        """
        amc = Path(entry["amc"])
        frames = parse_amc(amc, self._dofs_for(amc))
        s = max(0, int(entry["start_frame"]))
        e = int(entry["end_frame"])
        e = len(frames) if e < 0 else min(len(frames), e)
        return frames[s:e]

    def __iter__(self) -> Iterator[Tuple[List[Dict[str, np.ndarray]], List[str]]]:
        for entry in self.pairs:
            yield self.frames_for(entry), entry["descriptions"]

    def stats(self) -> dict:
        n_desc = sum(len(p["descriptions"]) for p in self.pairs)
        vocab = {w.lower().strip(".,!?") for p in self.pairs
                 for d in p["descriptions"] for w in d.split()}
        return {"clips": len(self.pairs), "descriptions": n_desc,
                "descriptions_per_clip": round(n_desc / max(len(self.pairs), 1), 2),
                "vocabulary": len(vocab),
                "distinct_source_clips": len({p["clip"] for p in self.pairs})}
