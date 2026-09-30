"""Score Winnow against the answer key of Jesse Diaz's binder-overlap-bench-v1 and write the list of
flagged pairs.

    python binder_eval/pair_inliers.py DATASET_DIR --out pair_inliers.csv
    python binder_eval/score.py DATASET_DIR pair_inliers.csv [--out winnow_flagged_pairs.csv]

A pair is flagged the way scan.py flags a train/test pair: by the perceptual hash (confirmed by 8
matching keypoints or a near-identical embedding), or by 25 matching keypoints. Every pair in this
set passes the 0.80 embedding step, so the keypoint count decides the second route.
"""
import argparse
import csv
import glob
import os
import sys
from collections import Counter, defaultdict

import numpy as np

START = os.getcwd()    # relative paths on the command line mean relative to here, not to winnow/
WINNOW = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "winnow")
sys.path.insert(0, WINNOW)
os.chdir(WINNOW)       # semantic.py loads its model relative to winnow/
from duplicates import find_leaked_pairs
from semantic import HASH_CONFIRM_INLIERS, HASH_CONFIRM_SIMILARITY, MIN_INLIERS, compute_embedding

CLAUDE_CALL = "not reviewed (more than 8 findings)"    # agent.REVIEW_MAX_FINDINGS: no image review above 8


def page(path):
    return os.path.splitext(os.path.basename(path))[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("inliers")
    parser.add_argument("--out", default="winnow_flagged_pairs.csv")
    options = parser.parse_args()
    data, inliers_file, out_file = (os.path.abspath(os.path.join(START, p)) for p in (options.dataset, options.inliers, options.out))

    train_paths = sorted(glob.glob(f"{data}/images/train/*.jpg"))
    val_paths = sorted(glob.glob(f"{data}/images/val/*.jpg"))
    path_of = {("train", page(p)): p for p in train_paths} | {("val", page(p)): p for p in val_paths}

    rows = list(csv.DictReader(open(inliers_file)))
    inliers = {(r["train_page"], r["val_page"]): int(r["inliers"]) for r in rows}
    fixed = {(r["train_page"], r["val_page"]): r["fixed_background_used"] == "1" for r in rows}
    key = list(csv.DictReader(open(f"{data}/planted_pairs.csv")))
    planted = {(r["train_page"], r["val_page"]) for r in key}

    def hash_confirmed(pair):
        if inliers[pair] >= HASH_CONFIRM_INLIERS:
            return True
        if fixed[pair]:
            return False
        a, b = (compute_embedding(path_of[side, name]) for side, name in (("train", pair[0]), ("val", pair[1])))
        return float(a @ b) >= HASH_CONFIRM_SIMILARITY

    hash_candidates = {(page(a), page(b)) for a, b, _ in find_leaked_pairs(train_paths, val_paths)}
    by_hash = {p for p in hash_candidates if p in inliers and hash_confirmed(p)}
    flagged = by_hash | {p for p, n in inliers.items() if n >= MIN_INLIERS}
    scored_pairs = len(inliers)
    print(f"{scored_pairs} pairs scored; flagged {len(flagged)} ({len(by_hash)} by the perceptual hash)")

    print("\nPlanted pairs flagged, by type:")
    for kind in ["shared_1", "shared_2", "shared_3", "reshot", "exact_copy"]:
        pairs = [(r["train_page"], r["val_page"]) for r in key if r["type"] == kind and (r["train_page"], r["val_page"]) in inliers]
        missed = sorted(inliers[p] for p in pairs if p not in flagged)
        print(f"  {kind:<11} {sum(p in flagged for p in pairs):>2}/{len(pairs):<2}  missed pairs' keypoints: {missed}")
    hits = sum(p in flagged for p in planted if p in inliers)
    outside = sorted((p for p in flagged if p not in planted), key=lambda p: -inliers[p])
    unplanted = scored_pairs - sum(p in inliers for p in planted)
    print(f"\nhits {hits}/{sum(p in inliers for p in planted)}; flagged outside the key: {len(outside)} of {unplanted} "
          f"unplanted pairs ({len(outside) / max(unplanted, 1):.1%})")

    print("\nKeypoint bar alone (planted pairs found / flags outside the key):")
    for cut in (25, 30, 40, 50, 60, 80):
        found = sum(inliers[p] >= cut for p in planted if p in inliers)
        extra = sum(n >= cut for p, n in inliers.items() if p not in planted)
        print(f"  >= {cut:<3} {found}/{len(planted)}   {extra}")

    cards = defaultdict(set)
    for m in csv.DictReader(open(f"{data}/manifest.csv")):
        if m["card_id"]:
            cards[m["split"], m["page"]].add(m["card_name"].strip().lower())
    same_name = sum(bool(cards["train", t] & cards["val", v]) for t, v in outside)
    print(f"\nFlags outside the key whose pages share a card name: {same_name} of {len(outside)}")
    print("Val pages with the most flags outside the key:", Counter(v for _, v in outside).most_common(3))

    with open(out_file, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["page_a", "page_b", "check", "matching_keypoints", "claude_call"])
        for t, v in sorted(flagged, key=lambda p: (-inliers[p], p)):
            writer.writerow([t, v, "perceptual hash" if (t, v) in by_hash else "embedding + keypoints", inliers[t, v], CLAUDE_CALL])
    print(f"\nwrote {len(flagged)} rows to {out_file}")


if __name__ == "__main__":
    main()
