#!/usr/bin/env python3
"""Reap GPU job folders whose outputs are already safe under /data/jack-logs (S11).

`experiments/gpu.py` makes one scratch folder per job (job script, kernel push,
downloaded outputs). Until 11-24 they went to `/data/tmp????????` and nothing
removed them; now they go to `/data/tmp/gpu/`. A folder is removed only when:

  * nothing in it changed for AGE_H hours (an in-flight job is never touched), and
  * every file that is not a job INPUT has a byte-identical copy (size + sha256)
    under /data/jack-logs. A missing copy under 1 MB (a kernel log, a result
    json) is first COPIED to /data/jack-logs/gpu_jobs/<folder>/; a bigger one
    (checkpoints, npz) keeps the folder.

Anything else is kept and listed with the reason. Inputs are regenerable from the
pinned commit (`kernel_sha256` is in gpu_submissions.jsonl). Dry run by default;
`--apply` deletes. Called by scripts/regate.sh on its existing cron — no new cron.
"""
import hashlib, os, shutil, sys, time
from pathlib import Path

ROOTS = [Path("/data/tmp/gpu")] + sorted(Path("/data").glob("tmp????????"))
SAFE = Path("/data/jack-logs")
AGE_H = 24
RESCUE = SAFE / "gpu_jobs"
SMALL = 1 << 20
INPUTS = {"job.py", "kernel.py", "kernel-metadata.json", "job_carry.py",
          "budget.json", "budget.json.lock", "submissions.jsonl"}


def is_input(p: Path) -> bool:
    return p.name in INPUTS or p.name.startswith("gpu_pre_") or p.suffix == ".pyc"


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def folders():
    for r in ROOTS:
        if r == Path("/data/tmp/gpu"):
            yield from (d for d in sorted(r.glob("*")) if d.is_dir()) if r.is_dir() else ()
        elif r.is_dir():
            yield r


def main(apply: bool) -> int:
    now = time.time()
    plan = []                                    # (folder, outputs, reason-or-None)
    want_sizes = set()
    for d in folders():
        files = [p for p in d.rglob("*") if p.is_file() and not p.is_symlink()]
        newest = max((p.stat().st_mtime for p in files), default=d.stat().st_mtime)
        if now - newest < AGE_H * 3600:
            plan.append((d, [], f"touched {(now - newest) / 3600:.1f} h ago"))
            continue
        outs = [p for p in files if not is_input(p)]
        want_sizes.update(p.stat().st_size for p in outs)
        plan.append((d, outs, None))
    index: dict[int, list[Path]] = {}
    for p in SAFE.rglob("*"):
        try:
            if p.is_file() and not p.is_symlink() and p.stat().st_size in want_sizes:
                index.setdefault(p.stat().st_size, []).append(p)
        except OSError:
            pass
    hashed: dict[Path, str] = {}
    freed = kept = reaped = 0
    for d, outs, why in plan:
        rescue = []
        if why is None:
            for p in outs:
                cands = index.get(p.stat().st_size, [])
                h = sha(p)
                if any(hashed.setdefault(c, sha(c)) == h for c in cands):
                    continue
                if p.stat().st_size >= SMALL:
                    why = f"no copy of {p.relative_to(d)} ({p.stat().st_size >> 20} MB)"
                    break
                rescue.append(p)
        size = sum(p.stat().st_size for p in d.rglob("*") if p.is_file())
        if why:
            kept += 1
            print(f"KEEP  {d}  {size >> 20} MB  — {why}")
        else:
            reaped += 1
            freed += size
            print(f"REAP  {d}  {size >> 20} MB  — {len(outs)} output(s), "
                  f"{len(rescue)} small one(s) copied to {RESCUE.name}/{d.name}")
            if apply:
                for p in rescue:
                    dst = RESCUE / d.name / p.relative_to(d)
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(p, dst)
                    if sha(dst) != sha(p):
                        sys.exit(f"copy of {p} does not match — stopping before any delete")
                shutil.rmtree(d)
    print(f"{'reaped' if apply else 'would reap'} {reaped}, kept {kept}, "
          f"{freed / 1e9:.2f} GB {'freed' if apply else 'reclaimable'}")
    return 0


if __name__ == "__main__":
    sys.exit(main("--apply" in sys.argv))
