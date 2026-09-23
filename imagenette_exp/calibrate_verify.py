"""Calibrates the keypoint check: RANSAC inliers for planted copies vs. innocent look-alike candidates."""
import glob
import random
import sys
import tempfile

import cv2
import numpy as np

sys.path.insert(0, "/app")
from semantic import CANDIDATE_THRESHOLD, compute_embeddings, keypoint_inliers

ROOT = "/data/imagenette2-160"


def sample(split, n):
    items = []
    for label, class_dir in enumerate(sorted(glob.glob(f"{ROOT}/{split}/*"))):
        items += [(p, label) for p in random.sample(sorted(glob.glob(f"{class_dir}/*")), n)]
    return items


def crop(img):
    h, w = img.shape[:2]
    return cv2.resize(img[h // 8: h - h // 8, w // 8: w - w // 8], (w, h))


def recompress(img):
    small = cv2.resize(img, None, fx=0.5, fy=0.5)
    return cv2.imdecode(cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 30])[1], cv2.IMREAD_COLOR)


def recolor(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.int16)
    hsv[..., 0] = (hsv[..., 0] + 12) % 180
    hsv[..., 2] = np.clip(hsv[..., 2] * 1.15, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


EDITS = [("resize+JPEG", recompress), ("crop", crop), ("mirror", lambda i: cv2.flip(i, 1)),
         ("crop+mirror+recolor", lambda i: recolor(cv2.flip(crop(i), 1)))]

random.seed(0)
train, test = sample("train", 100), sample("val", 50)
with tempfile.TemporaryDirectory() as tmp:
    source_of = {}
    for k, (path, _) in enumerate(random.sample(test, 80)):
        name, edit = EDITS[k % len(EDITS)]
        out = f"{tmp}/{name}_{k}.png"
        cv2.imwrite(out, edit(cv2.imread(path)))
        source_of[out] = (path, name)

    train_paths = [p for p, _ in train] + list(source_of)
    test_paths = [p for p, _ in test]
    tr, te = compute_embeddings(train_paths), compute_embeddings(test_paths)
    sim = np.stack([tr[p] for p in train_paths]) @ np.stack([te[p] for p in test_paths]).T

    positives = {name: [] for name, _ in EDITS}
    for copy, (src, name) in source_of.items():
        s = float(tr[copy] @ te[src])
        positives[name].append(keypoint_inliers(copy, src) if s >= CANDIDATE_THRESHOLD else -1)

    negatives = []
    for i, j in zip(*np.nonzero(sim >= CANDIDATE_THRESHOLD)):
        a, b = train_paths[i], test_paths[j]
        if source_of.get(a, (None,))[0] != b:
            n = keypoint_inliers(a, b)
            negatives.append(n)
            if n >= 20:
                pair = np.hstack([cv2.resize(cv2.imread(a), (200, 200)), cv2.resize(cv2.imread(b), (200, 200))])
                cv2.imwrite(f"/data/flag_check/inliers_{n}.jpg", pair)
                print("high-inlier innocent pair:", n, round(float(sim[i, j]), 3), a.split("/")[-1], b.split("/")[-1])

print(f"candidates (similarity >= {CANDIDATE_THRESHOLD}): {len(negatives)} innocent pairs\n")
print("planted copies, inliers (-1 = never became a candidate):")
for name, vals in positives.items():
    print(f"  {name:20} min {min(vals):4}  median {int(np.median(vals)):4}  {sorted(vals)[:6]}")
print(f"\ninnocent candidates, inliers: max {max(negatives)}, 99th pct {int(np.percentile(negatives, 99))}, "
      f"top 5 {sorted(negatives)[-5:]}")
for t in [10, 15, 20, 25, 30]:
    caught = sum(v >= t for vals in positives.values() for v in vals)
    print(f"  min_inliers {t:2}: catches {caught}/{sum(map(len, positives.values()))} planted, "
          f"{sum(v >= t for v in negatives)} false positives")
