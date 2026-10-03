"""Winnow's keypoint count per pair at its default SIFT cap and uncapped, on the binder benchmark.

    python binder_eval/sift_cap.py DATASET_DIR [--random 400] [--workers 8] [--out sift_cap.csv]

Scores three groups of train/val pairs with semantic._content_match, once with SIFT capped at
1000 keypoints (Winnow's default, _sift = cv2.SIFT_create(nfeatures=1000)) and once uncapped
(nfeatures=0), both at the usual 640px working size: every planted pair in planted_pairs.csv,
every pair in winnow_flagged_pairs.csv outside the key, and a seeded random sample of the
remaining unrelated pairs. For each planted one-card pair it also counts the keypoints that land
on the shared card in each page, using the boxes in manifest.csv.
"""
import argparse
import csv
import multiprocessing as mp
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
WINNOW = os.path.join(HERE, "..", "winnow")
sys.path.insert(0, WINNOW)
BAR = 25


def init(nfeatures):
    os.chdir(WINNOW)
    global semantic
    import cv2
    import semantic
    semantic._sift = cv2.SIFT_create(nfeatures=nfeatures)
    semantic._keypoints_for.cache_clear()


def inliers(args):
    dataset, train, val = args
    points = semantic._content_match(f"{dataset}/images/train/{train}.jpg", f"{dataset}/images/val/{val}.jpg")[0]
    return train, val, len(points)


def keypoints_on_card(args):
    dataset, split, page, box = args
    keypoints, _, shape = semantic._keypoints_for(f"{dataset}/images/{split}/{page}.jpg")
    scale = shape[0] / 1280
    x0, y0, x1, y1 = (v * scale for v in box)
    return sum(1 for k in keypoints if x0 <= k.pt[0] <= x1 and y0 <= k.pt[1] <= y1)


def run(nfeatures, function, jobs, workers):
    with mp.Pool(workers, initializer=init, initargs=(nfeatures,)) as pool:
        return pool.map(function, jobs, chunksize=8)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("--random", type=int, default=400)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--out", default="sift_cap.csv")
    options = parser.parse_args()
    dataset = os.path.abspath(options.dataset)

    key = {(r["train_page"], r["val_page"]): r for r in csv.DictReader(open(f"{dataset}/planted_pairs.csv"))}
    boxes = {(r["page"], r["slot"]): [int(r[k]) for k in ("x0", "y0", "x1", "y1")]
             for r in csv.DictReader(open(f"{dataset}/manifest.csv")) if r["card_id"]}
    flagged = [(r["page_a"], r["page_b"]) for r in csv.DictReader(open(f"{HERE}/winnow_flagged_pairs.csv"))]
    false_flags = [p for p in flagged if p not in key]
    pages = sorted({page for page, _ in boxes})
    trains = [p for p in pages if p.startswith("train")]
    vals = [p for p in pages if p.startswith("val")]
    taken = set(key) | set(false_flags)
    unrelated = random.Random(1).sample([(t, v) for t in trains for v in vals if (t, v) not in taken], options.random)

    groups = [(p, key[p]["type"]) for p in key] + [(p, "false flag") for p in false_flags] + [(p, "random unrelated") for p in unrelated]
    jobs = [(dataset, *p) for p, _ in groups]
    one_card = [p for p in key if key[p]["type"] == "shared_1"]
    card_jobs = [(dataset, split, page, boxes[(page, key[p][f"{split}_slots"])])
                 for p in one_card for split, page in (("train", p[0]), ("val", p[1]))]

    counts, on_card = {}, {}
    for label, nfeatures in (("cap_1000", 1000), ("uncapped", 0)):
        for train, val, n in run(nfeatures, inliers, jobs, options.workers):
            counts.setdefault((train, val), {})[label] = n
        on_card[label] = run(nfeatures, keypoints_on_card, card_jobs, options.workers)
        print(f"{label}: done", flush=True)

    with open(options.out, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["train_page", "val_page", "group", "inliers_cap_1000", "inliers_uncapped"])
        for p, group in groups:
            writer.writerow([*p, group, counts[p]["cap_1000"], counts[p]["uncapped"]])

    def summary(label):
        planted = [counts[p][label] for p, g in groups if p in key]
        false = sorted(counts[p][label] for p, g in groups if g == "false flag")
        rand = sorted(counts[p][label] for p, g in groups if g == "random unrelated")
        one = sorted(counts[p][label] for p in one_card)
        print(f"\n{label}: one-card planted pairs {one[0]}-{one[-1]}; false flags median {false[len(false) // 2]} "
              f"(max {false[-1]}); random unrelated median {rand[len(rand) // 2]} (max {rand[-1]})")
        print(f"  bar  planted (of {len(planted)})  false flags (of {len(false)})  random unrelated (of {len(rand)})")
        for bar in (BAR, 50, 80, 100):
            print(f"  {bar:>3}  {sum(n >= bar for n in planted):>8}  {sum(n >= bar for n in false):>12}  {sum(n >= bar for n in rand):>12}")

    summary("cap_1000")
    summary("uncapped")
    print("\nOne-card pairs, keypoints on the shared card (train/val) at cap 1000, and inliers:")
    for i, p in enumerate(one_card):
        print(f"  {key[p]['pair']} same_slot={key[p]['same_slot']:<3} {on_card['cap_1000'][2 * i]:>3}/{on_card['cap_1000'][2 * i + 1]:<3}"
              f" inliers {counts[p]['cap_1000']:>3} -> {counts[p]['uncapped']:>3} uncapped")
