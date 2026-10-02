"""Does lifting SIFT's keypoint cap hold up on real photos? Full-size Imagenette, cap 1000 vs uncapped.

The calibration and the full audit ran on imagenette2-160, where a photo has a few hundred SIFT
keypoints, so semantic.py's cap (SIFT_create(nfeatures=1000)) never binds. This reruns the
keypoint check on full-size imagenette2 (same file names), at the usual 640px working size:

1. how many photos have more than 1000 keypoints, i.e. how often the cap binds;
2. the 108 hand-verified audit pairs (full_audit/verified_pairs.csv), at cap 1000 and uncapped;
3. a seeded sample of random same-class train/val pairs outside the audit, as the false-alarm
   baseline (same class, so the embedding stage would usually pass them on).

Get the data: curl -LO https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz && tar xzf imagenette2.tgz
Run from the repo root: IMAGENETTE_FULL=path/to/imagenette2 python imagenette_exp/fullsize_cap.py
"""
import argparse
import csv
import glob
import multiprocessing as mp
import os
import random
import statistics
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
WINNOW = os.path.join(HERE, "..", "winnow")
sys.path.insert(0, WINNOW)
ROOT = os.environ.get("IMAGENETTE_FULL", os.path.join(HERE, "imagenette2"))
CAPS = (("cap_1000", 1000), ("uncapped", 0))
BARS = (25, 50, 68, 80, 100)
REAL = ("same_photo", "same_scene")


def init(nfeatures):
    os.chdir(WINNOW)
    global semantic
    import cv2
    import semantic
    cv2.setNumThreads(1)
    semantic._sift = cv2.SIFT_create(nfeatures=nfeatures)
    semantic._keypoints_for.cache_clear()


def keypoints(path):
    count = len(semantic._keypoints_for(path)[0])
    semantic._keypoints_for.cache_clear()    # one look per photo; keep worker memory flat
    return count


def inliers(pair):
    return semantic.keypoint_inliers(*pair)


def run(nfeatures, function, jobs, workers):
    with mp.Pool(workers, initializer=init, initargs=(nfeatures,)) as pool:
        return pool.map(function, jobs, chunksize=16)


def spread(values):
    values = sorted(values)
    return f"median {statistics.median(values):.0f} ({values[0]} to {values[-1]})"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--random", type=int, default=2000)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--out", default=os.path.join(HERE, "fullsize_cap.csv"))
    options = parser.parse_args()

    photos = sorted(glob.glob(f"{ROOT}/train/*/*.JPEG")) + sorted(glob.glob(f"{ROOT}/val/*/*.JPEG"))
    audit = list(csv.DictReader(open(f"{HERE}/full_audit/verified_pairs.csv")))
    missing = [f for r in audit for f in (r["train_file"], r["val_file"]) if not os.path.exists(f"{ROOT}/{f}")]
    if missing:
        sys.exit(f"{len(missing)} audit files not under {ROOT}, e.g. {missing[0]}")
    print(f"{len(photos)} photos under {ROOT}; {len(audit)} audit pairs", flush=True)

    # Random same-class pairs, never one of the audited pairs.
    by_class = defaultdict(lambda: {"train": [], "val": []})
    for path in photos:
        split, cls = path.split("/")[-3:-1]
        by_class[cls][split].append(os.path.relpath(path, ROOT))
    audited = {(r["train_file"], r["val_file"]) for r in audit}
    rng, unrelated = random.Random(1), set()
    classes = sorted(by_class)
    while len(unrelated) < options.random:
        cls = rng.choice(classes)
        pair = (rng.choice(by_class[cls]["train"]), rng.choice(by_class[cls]["val"]))
        if pair not in audited:
            unrelated.add(pair)
    unrelated = sorted(unrelated)

    groups = [((r["train_file"], r["val_file"]), r["verdict"]) for r in audit] + [(p, "random same-class") for p in unrelated]
    jobs = [(f"{ROOT}/{a}", f"{ROOT}/{b}") for (a, b), _ in groups]
    counts = {}
    for label, nfeatures in CAPS:
        counts[label] = run(nfeatures, inliers, jobs, options.workers)
        print(f"{label}: scored {len(jobs)} pairs", flush=True)
    per_photo = run(0, keypoints, photos, options.workers)

    with open(options.out, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["train_file", "val_file", "group", "inliers_cap_1000", "inliers_uncapped"])
        for i, ((a, b), group) in enumerate(groups):
            writer.writerow([a, b, group, counts["cap_1000"][i], counts["uncapped"][i]])
    print(f"wrote {len(groups)} rows to {options.out}")

    binds = sum(n > 1000 for n in per_photo)
    print(f"\nKeypoints per photo uncapped, at the 640px working size: {spread(per_photo)}, mean {sum(per_photo) / len(per_photo):.0f}")
    print(f"The 1000 cap binds on {binds} of {len(per_photo)} photos ({binds / len(per_photo):.0%})")

    names = ("same_photo", "same_scene", "different", "random same-class")
    print("\nInliers by group:")
    for name in names:
        rows = [i for i, (_, g) in enumerate(groups) if g == name]
        print(f"  {name:<18} n={len(rows):<5} cap 1000: {spread([counts['cap_1000'][i] for i in rows]):<28} "
              f"uncapped: {spread([counts['uncapped'][i] for i in rows])}")

    print(f"\nPairs at or over each bar (real = same_photo + same_scene):")
    print(f"  {'':<9} {'bar':>4}  {'real (of 76)':>12}  {'different (of 32)':>17}  {'random (of ' + str(len(unrelated)) + ')':>14}")
    for label, _ in CAPS:
        for bar in BARS:
            over = lambda name: sum(counts[label][i] >= bar for i, (_, g) in enumerate(groups)
                                    if (g in REAL if name == "real" else g == name))
            print(f"  {label:<9} {bar:>4}  {over('real'):>12}  {over('different'):>17}  {over('random same-class'):>14}")
