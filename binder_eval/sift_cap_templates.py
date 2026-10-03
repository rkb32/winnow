"""Winnow's flags on every train/val pair of the binder benchmark, at the default SIFT cap and uncapped,
before and after the shared-template filter.

    python binder_eval/sift_cap_templates.py DATASET_DIR [--workers 8] [--out sift_cap_templates.csv]

For each cap (nfeatures=1000, Winnow's default, and nfeatures=0, uncapped, both at the usual 640px
working size) it counts semantic._content_match keypoints on all 240 x 60 pairs, runs
semantic.set_aside_templates on every pair over the lowest bar, and prints how many planted pairs
and pairs outside the key stay flagged at keypoint bars 25, 50, 80 and 100. Pairs the perceptual hash
finds skip the filter, as in scan.py. The filter's own settings are left at their defaults, and then,
uncapped, its bar is scaled by how much the keypoint count per page grew. Writes one row per pair
that clears the lowest bar at either cap or is planted. About 15 minutes on 8 workers, most of it
uncapped. The filter's reference photos are drawn from a seed made of the full paths, so the
template-free counts differ when the benchmark is unzipped somewhere else.
"""
import argparse
import csv
import glob
import multiprocessing as mp
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
WINNOW = os.path.join(HERE, "..", "winnow")
sys.path.insert(0, WINNOW)
BARS = (25, 50, 80, 100)
CAPS = (("cap_1000", 1000), ("uncapped", 0))
KINDS = ("shared_1", "shared_2", "shared_3", "reshot", "exact_copy")


def page(path):
    return os.path.splitext(os.path.basename(path))[0]


def init(nfeatures, all_paths):
    os.chdir(WINNOW)
    global semantic, ALL_PATHS
    import cv2
    import semantic
    semantic._sift = cv2.SIFT_create(nfeatures=nfeatures)
    semantic._keypoints_for.cache_clear()
    ALL_PATHS = all_paths
    # set_aside_templates reports the distinct count only for the pairs it sets aside. With its bar
    # out of reach it sets every pair aside, so each comes back with its count.
    semantic.MIN_DISTINCT_INLIERS = 10 ** 9


def one_train_page(args):
    train_path, val_paths = args
    return [(page(train_path), page(val_path), len(semantic._content_match(train_path, val_path)[0]))
            for val_path in val_paths]


def keypoint_count(path):
    return len(semantic._keypoints_for(path)[0])


def distinct(args):
    train_path, val_path, inliers = args
    _, aside = semantic.set_aside_templates([(train_path, val_path, 0.0, inliers)], ALL_PATHS)
    return page(train_path), page(val_path), aside[0][4]


def run(nfeatures, all_paths, function, jobs, workers, chunksize):
    with mp.Pool(workers, initializer=init, initargs=(nfeatures, all_paths)) as pool:
        return list(pool.imap(function, jobs, chunksize))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--out", default="sift_cap_templates.csv")
    options = parser.parse_args()
    dataset, out_file = os.path.abspath(options.dataset), os.path.abspath(options.out)

    trains = sorted(glob.glob(f"{dataset}/images/train/*.jpg"))
    vals = sorted(glob.glob(f"{dataset}/images/val/*.jpg"))
    path_of = {page(p): p for p in trains + vals}
    key = {(r["train_page"], r["val_page"]): r["type"] for r in csv.DictReader(open(f"{dataset}/planted_pairs.csv"))}
    os.chdir(WINNOW)
    from duplicates import find_leaked_pairs
    from semantic import HASH_CONFIRM_INLIERS, MIN_DISTINCT_INLIERS
    by_hash = {(page(a), page(b)) for a, b, _ in find_leaked_pairs(trains, vals)}
    print(f"{len(trains)} train pages x {len(vals)} val pages, {options.workers} workers; "
          f"{len(by_hash)} pairs found by the perceptual hash", flush=True)

    inliers, distinct_count, keypoints = {}, {}, {}
    for label, nfeatures in CAPS:
        started = time.time()
        counts = run(nfeatures, trains + vals, keypoint_count, trains + vals, options.workers, 8)
        keypoints[label] = sum(counts) / len(counts)
        rows = run(nfeatures, trains + vals, one_train_page, [(t, vals) for t in trains], options.workers, 1)
        inliers[label] = {(t, v): n for page_rows in rows for t, v, n in page_rows}
        # scan.py confirms a hash match with 8 keypoints, or else by the embedding, which isn't loaded here.
        unconfirmed = [p for p in by_hash if inliers[label][p] < HASH_CONFIRM_INLIERS]
        if unconfirmed:
            sys.exit(f"{label}: hash pairs {unconfirmed} need the embedding to confirm them; this script doesn't load it")
        to_check = [(path_of[t], path_of[v], n) for (t, v), n in sorted(inliers[label].items())
                    if n >= BARS[0] and (t, v) not in by_hash]
        distinct_count[label] = {(t, v): d for t, v, d in run(nfeatures, trains + vals, distinct, to_check, options.workers, 16)}
        print(f"{label}: {keypoints[label]:.0f} keypoints per page, {len(inliers[label])} pairs, "
              f"{len(to_check)} through the template filter, {time.time() - started:.0f}s", flush=True)

    def flagged(label, bar, distinct_bar=None):
        over = {p for p, n in inliers[label].items() if n >= bar and p not in by_hash}
        if distinct_bar is not None:
            over = {p for p in over if distinct_count[label][p] >= distinct_bar}
        return over | by_hash

    def cell(pairs):
        hits = sum(p in key for p in pairs)
        return f"{hits:>3} / {len(pairs) - hits:<5}"

    rows = sorted({p for label, _ in CAPS for p in flagged(label, BARS[0])} | set(key))
    with open(out_file, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["train_page", "val_page", "group", "perceptual_hash",
                         "inliers_cap_1000", "distinct_cap_1000", "inliers_uncapped", "distinct_uncapped"])
        for p in rows:
            writer.writerow([*p, key.get(p, "outside the key"), int(p in by_hash),
                             inliers["cap_1000"][p], distinct_count["cap_1000"].get(p, ""),
                             inliers["uncapped"][p], distinct_count["uncapped"].get(p, "")])
    print(f"wrote {len(rows)} rows to {out_file}")

    outside_total = len(trains) * len(vals) - len(key)
    print(f"\nPlanted pairs found (of {len(key)}) / flags outside the key (of {outside_total}); "
          f"the template filter keeps a pair with {MIN_DISTINCT_INLIERS}+ distinct keypoints")
    print(f"  {'':<9} {'bar':>4}   {'no filter':<11}   {'template filter':<15}   precision after")
    for label, _ in CAPS:
        for bar in BARS:
            after = flagged(label, bar, MIN_DISTINCT_INLIERS)
            print(f"  {label:<9} {bar:>4}   {cell(flagged(label, bar))}   {cell(after):<15}   "
                  f"{sum(p in key for p in after) / len(after):>8.0%}")

    # The filter's bar was set at cap 1000. Scale it by how much the keypoint count grew uncapped,
    # a ratio of keypoint counts that never looks at the answer key.
    ratio = keypoints["uncapped"] / keypoints["cap_1000"]
    scaled = round(MIN_DISTINCT_INLIERS * ratio)
    print(f"\nUncapped, the filter's bar scaled by the keypoint ratio ({ratio:.2f}x): {MIN_DISTINCT_INLIERS} -> {scaled}")
    for bar in BARS:
        after = flagged("uncapped", bar, scaled)
        print(f"  {'uncapped':<9} {bar:>4}   {cell(after):<15}   {sum(p in key for p in after) / len(after):>8.0%}")

    print("\nPlanted pairs still flagged after the template filter, by type:")
    for label, _ in CAPS:
        for bar in BARS:
            after = flagged(label, bar, MIN_DISTINCT_INLIERS)
            kept = [f"{kind} {sum(p in after for p in key if key[p] == kind)}/{sum(key[p] == kind for p in key)}"
                    for kind in KINDS]
            missed = sorted(f"{t}/{v}" for t, v in key if (t, v) not in after)
            print(f"  {label:<9} {bar:>4}   {', '.join(kept)}   missed: {', '.join(missed) or 'none'}")
