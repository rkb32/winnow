"""Audit Imagenette's entire official train/val split for leaks with Winnow's own detectors.

Same thresholds and code as the live pipeline; only the pairwise loops are parallelized or
vectorized so 37 million train x val pairs finish in minutes. Every flagged pair is saved as a
side-by-side image so each one can be checked by eye before anything is claimed.
"""
import glob
import json
import os
import random
import sys
import time
from multiprocessing import Pool

import cv2
import numpy as np

sys.path.insert(0, "/app")
from duplicates import compute_hash, hash_distance
from semantic import CANDIDATE_THRESHOLD, MIN_INLIERS, compute_embedding, keypoint_inliers

ROOT = "/data/imagenette2-160"
OUT = "/data/full_audit"
HASH_THRESHOLD = 5
POPCOUNT = np.array([bin(i).count("1") for i in range(256)], np.uint8)


def single_thread():
    cv2.setNumThreads(1)


def hash_of(path):
    return compute_hash(path).flatten()


def inliers_of(pair):
    return keypoint_inliers(*pair)


def list_split(split):
    return sorted(glob.glob(f"{ROOT}/{split}/*/*.JPEG"))


def class_of(path):
    return path.split("/")[-2]


def contact_sheets(pairs, prefix, per_sheet=20, cols=4):
    tile = 150
    for start in range(0, len(pairs), per_sheet):
        tiles = []
        for p in pairs[start:start + per_sheet]:
            a = cv2.resize(cv2.imread(p["train"]), (tile, tile))
            b = cv2.resize(cv2.imread(p["val"]), (tile, tile))
            label = np.full((22, tile * 2, 3), 255, np.uint8)
            cv2.putText(label, f"#{p['id']} {p['method']} {p['score']}", (4, 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
            tiles.append(np.vstack([np.hstack([a, b]), label]))
        while len(tiles) % cols:
            tiles.append(np.full_like(tiles[0], 255))
        rows = [np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]
        cv2.imwrite(f"{OUT}/{prefix}_{start // per_sheet + 1:02d}.jpg", np.vstack(rows))


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    started = time.time()
    train, val = list_split("train"), list_split("val")
    print(f"{len(train)} train photos, {len(val)} val photos, {len(train) * len(val):,} pairs", flush=True)

    with Pool(initializer=single_thread) as pool:
        train_hashes = np.stack(pool.map(hash_of, train, chunksize=64))
        val_hashes = np.stack(pool.map(hash_of, val, chunksize=64))

        # BlockMeanHash compare() is Hamming distance; vectorize it, then prove it matches.
        random.seed(0)
        for _ in range(2000):
            i, j = random.randrange(len(train)), random.randrange(len(val))
            fast = int(POPCOUNT[np.bitwise_xor(train_hashes[i], val_hashes[j])].sum())
            assert fast == int(hash_distance(train_hashes[i][None], val_hashes[j][None])), (i, j)
        hash_pairs = []
        for start in range(0, len(val), 64):
            chunk = val_hashes[start:start + 64]
            dist = POPCOUNT[np.bitwise_xor(chunk[:, None, :], train_hashes[None, :, :])].sum(-1)
            for vi, ti in zip(*np.nonzero(dist <= HASH_THRESHOLD)):
                hash_pairs.append((int(ti), start + int(vi), int(dist[vi, ti])))
        print(f"hash stage: {len(hash_pairs)} pairs within distance {HASH_THRESHOLD} "
              f"({time.time() - started:.0f}s)", flush=True)

        train_emb = np.stack(pool.map(compute_embedding, train, chunksize=32))
        val_emb = np.stack(pool.map(compute_embedding, val, chunksize=32))
        similarity = train_emb @ val_emb.T
        hashed = {(t, v) for t, v, _ in hash_pairs}
        candidates = [(int(t), int(v)) for t, v in zip(*np.nonzero(similarity >= CANDIDATE_THRESHOLD))
                      if (t, v) not in hashed]
        print(f"embedding stage: {len(candidates):,} candidates at similarity >= {CANDIDATE_THRESHOLD} "
              f"({time.time() - started:.0f}s)", flush=True)

        inliers = pool.map(inliers_of, [(train[t], val[v]) for t, v in candidates], chunksize=256)

    flagged = [{"train": train[t], "val": val[v], "method": "hash", "score": d} for t, v, d in hash_pairs]
    flagged += [{"train": train[t], "val": val[v], "method": "keypoints", "score": n,
                 "similarity": round(float(similarity[t, v]), 3)}
                for (t, v), n in zip(candidates, inliers) if n >= MIN_INLIERS]
    flagged.sort(key=lambda p: (p["method"] != "hash", p["score"] if p["method"] == "hash" else -p["score"]))
    for k, p in enumerate(flagged):
        p["id"] = k
        p["same_class"] = class_of(p["train"]) == class_of(p["val"])

    with open(f"{OUT}/flagged.json", "w") as f:
        json.dump(flagged, f, indent=1)
    contact_sheets(flagged, "sheet")

    leaked_val = {p["val"] for p in flagged}
    print(f"\nflagged pairs: {len(flagged)} ({sum(p['method'] == 'hash' for p in flagged)} by hash, "
          f"{sum(p['method'] == 'keypoints' for p in flagged)} by embedding + keypoints)")
    print(f"distinct val photos with a match in train: {len(leaked_val)} of {len(val)} "
          f"({len(leaked_val) / len(val):.2%})")
    print(f"cross-class pairs (possible label conflicts): {sum(not p['same_class'] for p in flagged)}")
    print(f"done in {time.time() - started:.0f}s")
