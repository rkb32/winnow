"""Winnow's keypoint confirmation on every train/val page pair of a benchmark, in parallel.

    python binder_eval/pair_inliers.py DATASET_DIR [--workers 5] [--limit N] [--out pair_inliers.csv]

DATASET_DIR holds images/train/*.jpg and images/val/*.jpg (Jesse Diaz's binder-overlap-bench-v1).
Each row is one pair with the number of keypoint matches that agree on one geometric transform,
counted the way scan.py counts them (semantic._content_match). --limit N reads only the first N
train pages, for a quick check. The full 240 x 60 set took about 11 minutes on 5 workers.
"""
import argparse
import csv
import glob
import multiprocessing as mp
import os
import sys
import time

WINNOW = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "winnow")
sys.path.insert(0, WINNOW)


def page(path):
    return os.path.splitext(os.path.basename(path))[0]


def init():
    os.chdir(WINNOW)
    global content_match
    from semantic import _content_match as content_match


def one_train_page(args):
    train_path, val_paths = args
    rows = []
    for val_path in val_paths:
        points, _, _, _, fixed = content_match(train_path, val_path)
        rows.append((page(train_path), page(val_path), len(points), int(fixed)))
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--out", default="pair_inliers.csv")
    options = parser.parse_args()

    dataset = os.path.abspath(options.dataset)
    trains = sorted(glob.glob(f"{dataset}/images/train/*.jpg"))[:options.limit]
    vals = sorted(glob.glob(f"{dataset}/images/val/*.jpg"))
    print(f"{len(trains)} train pages x {len(vals)} val pages, {options.workers} workers", flush=True)
    started = time.time()
    with open(options.out, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["train_page", "val_page", "inliers", "fixed_background_used"])
        with mp.Pool(options.workers, initializer=init) as pool:
            for done, rows in enumerate(pool.imap_unordered(one_train_page, [(t, vals) for t in trains]), 1):
                writer.writerows(rows)
                handle.flush()
                if done % 10 == 0:
                    print(f"{done}/{len(trains)} train pages, {time.time() - started:.0f}s", flush=True)
    print(f"finished in {time.time() - started:.0f}s", flush=True)
