"""Freeze the scene-level held-out split and build the filtered ALFWorld data tree.

PRE-REGISTERED RULE (frozen 2026-07-12, BEFORE any S1.2 rollout sampling; do not
edit after the first pilot job has sampled a rollout):

- Universe: the scenes appearing in <src>/json_2.1.1/train task-dir names
  (scene = trailing integer of the task dir name). Families by scene number:
  kitchen < 200, living 2xx, bedroom 3xx, bath 4xx. Expected inventory:
  108 scenes, 27 per family (asserted; abort loudly if the data disagrees).
- Hold out 11 scenes: families in alphabetical order (bath, bedroom, kitchen,
  living) take k = 2, 3, 3, 3 respectively (alphabetically-first family takes
  the smaller share); within each family draw with random.Random(0).sample
  from the ascending-sorted scene list.
- Held-out scenes NEVER enter RL rollouts. They are the pure-E probe set:
  the theorem-protected "cannot internalize" residual is measured there.

Tree construction (--dst):
- json_2.1.1/train: REAL directories with FILE-level symlinks, containing only
  task dirs whose scene is NOT held out. Real dirs are required because
  alfred_tw_env.collect_game_files uses os.walk(followlinks=False), which does
  not descend directory symlinks.
- Everything else (json_2.1.1/valid_*, logic/, detectors/, ...) is a top-level
  directory symlink: those paths are only ever traversed THROUGH the link
  (path resolution), never enumerated as walk entries, so followlinks is moot.

Run ON RORQUAL (login node, stdlib only):
  python3 selfevolve/make_scene_holdout.py \
      --src /project/def-jbyu/$USER/boerz/alfworld_data \
      --dst /project/def-jbyu/$USER/boerz/alfworld_data_trainsplit \
      --manifest /project/def-jbyu/$USER/boerz/alfworld_data_trainsplit/scene_holdout_manifest.json
"""
import argparse
import json
import os
import random
import sys
from collections import defaultdict
from datetime import date

HOLDOUT_SEED = 0
FAMILY_K = {"bath": 2, "bedroom": 3, "kitchen": 3, "living": 3}
EXPECTED_SCENES_TOTAL = 108
EXPECTED_SCENES_PER_FAMILY = 27


def family_of(scene_num: int) -> str:
    if scene_num < 200:
        return "kitchen"
    if scene_num < 300:
        return "living"
    if scene_num < 400:
        return "bedroom"
    return "bath"


def scene_of_taskdir(name: str):
    tail = name.rsplit("-", 1)[-1]
    return int(tail) if tail.isdigit() else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--force", action="store_true",
                    help="rebuild even if --dst already exists")
    args = ap.parse_args()

    src = os.path.abspath(args.src)
    dst = os.path.abspath(args.dst)
    train_src = os.path.join(src, "json_2.1.1", "train")
    if not os.path.isdir(train_src):
        sys.exit(f"ERROR: {train_src} not found")
    if os.path.exists(dst):
        if not args.force:
            sys.exit(f"ERROR: {dst} already exists; the split is frozen. "
                     f"Use --force only if you know the pilot has NOT started.")
        import shutil
        shutil.rmtree(dst)

    # -- 1. scene inventory from task dir names (no JSON reads) --
    scenes_by_family = defaultdict(set)
    taskdirs = sorted(d for d in os.listdir(train_src)
                      if os.path.isdir(os.path.join(train_src, d)))
    unparsed = []
    for d in taskdirs:
        s = scene_of_taskdir(d)
        if s is None:
            unparsed.append(d)
            continue
        scenes_by_family[family_of(s)].add(s)
    if unparsed:
        sys.exit(f"ERROR: {len(unparsed)} task dirs with unparseable scene "
                 f"suffix, e.g. {unparsed[:3]} — rule assumptions violated.")
    n_total = sum(len(v) for v in scenes_by_family.values())
    if n_total != EXPECTED_SCENES_TOTAL or any(
            len(scenes_by_family[f]) != EXPECTED_SCENES_PER_FAMILY
            for f in FAMILY_K):
        sys.exit(f"ERROR: scene inventory mismatch: total={n_total}, "
                 f"per-family={ {f: len(scenes_by_family[f]) for f in sorted(FAMILY_K)} } "
                 f"(expected {EXPECTED_SCENES_TOTAL} / {EXPECTED_SCENES_PER_FAMILY} each). "
                 f"Pre-registered rule assumptions violated — investigate, do not patch.")

    # -- 2. deterministic held-out draw --
    rng = random.Random(HOLDOUT_SEED)
    holdout = {}
    for fam in sorted(FAMILY_K):  # alphabetical: bath, bedroom, kitchen, living
        holdout[fam] = sorted(rng.sample(sorted(scenes_by_family[fam]), FAMILY_K[fam]))
    holdout_flat = {s for ss in holdout.values() for s in ss}

    # -- 3. build the filtered tree --
    json_dst = os.path.join(dst, "json_2.1.1")
    train_dst = os.path.join(json_dst, "train")
    os.makedirs(train_dst)
    kept_dirs, dropped_dirs, files_linked, games_kept = 0, 0, 0, 0
    for d in taskdirs:
        if scene_of_taskdir(d) in holdout_flat:
            dropped_dirs += 1
            continue
        kept_dirs += 1
        src_task = os.path.join(train_src, d)
        for root, _dirs, files in os.walk(src_task):
            rel = os.path.relpath(root, train_src)
            os.makedirs(os.path.join(train_dst, rel), exist_ok=True)
            for fn in files:
                os.symlink(os.path.join(root, fn),
                           os.path.join(train_dst, rel, fn))
                files_linked += 1
                if fn == "game.tw-pddl":
                    games_kept += 1

    # other json_2.1.1 splits -> dir symlinks
    for entry in sorted(os.listdir(os.path.join(src, "json_2.1.1"))):
        if entry == "train":
            continue
        os.symlink(os.path.join(src, "json_2.1.1", entry),
                   os.path.join(json_dst, entry))
    # other top-level entries (logic, detectors, ...) -> dir symlinks
    for entry in sorted(os.listdir(src)):
        if entry == "json_2.1.1":
            continue
        os.symlink(os.path.join(src, entry), os.path.join(dst, entry))

    manifest = {
        "frozen_date": str(date.today()),
        "rule": ("11 held-out scenes; families alphabetical (bath,bedroom,kitchen,living) "
                 "take k=(2,3,3,3); random.Random(0).sample over the ascending-sorted "
                 "scene list per family; scene = trailing int of train task dir name"),
        "seed": HOLDOUT_SEED,
        "holdout_scenes": holdout,
        "n_scenes_total": n_total,
        "n_taskdirs_kept": kept_dirs,
        "n_taskdirs_dropped": dropped_dirs,
        "n_files_linked": files_linked,
        "n_games_kept": games_kept,
        "src": src,
        "dst": dst,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.manifest)), exist_ok=True)
    with open(args.manifest, "w") as f:
        json.dump(manifest, f, indent=2)
    print(json.dumps(manifest, indent=2))
    print(f"\nOK: {kept_dirs} task dirs kept / {dropped_dirs} dropped "
          f"({games_kept} game files), holdout scenes: {sorted(holdout_flat)}")


if __name__ == "__main__":
    main()
