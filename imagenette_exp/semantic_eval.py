"""How much does the embedding check add over the perceptual hash, and where should its threshold sit?

Positives: a real photo vs an edited copy of itself (the kind of copy that sneaks into
datasets). Negatives: pairs of genuinely different photos, including same-class pairs,
which are the hardest to tell apart.
"""
import glob
import itertools
import os
import random
import sys
import tempfile

import cv2
import numpy as np

sys.path.insert(0, "/app")
from duplicates import compute_hash, hash_distance, is_duplicate
from semantic import compute_embedding

random.seed(0)
VAL_DIR = "/data/imagenette2-160/val"
PER_CLASS = 10
EDITED = 30


def crop(img):
    h, w = img.shape[:2]
    return cv2.resize(img[h // 8: h - h // 8, w // 8: w - w // 8], (w, h))


def recolor(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.int16)
    hsv[..., 0] = (hsv[..., 0] + 12) % 180
    hsv[..., 2] = np.clip(hsv[..., 2] * 1.15, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def recompress(img):
    small = cv2.resize(img, None, fx=0.5, fy=0.5)
    return cv2.imdecode(cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 30])[1], cv2.IMREAD_COLOR)


EDITS = {
    "resize+JPEG q30": recompress,
    "crop 75%": crop,
    "recolor": recolor,
    "mirror": lambda img: cv2.flip(img, 1),
    "crop+mirror+recolor": lambda img: recolor(cv2.flip(crop(img), 1)),
}

originals = []
for class_dir in sorted(glob.glob(f"{VAL_DIR}/*")):
    originals += [(os.path.basename(class_dir), p) for p in random.sample(sorted(glob.glob(f"{class_dir}/*")), PER_CLASS)]

with tempfile.TemporaryDirectory() as tmp:
    hashes = {p: compute_hash(p) for _, p in originals}
    embeds = {p: compute_embedding(p) for _, p in originals}

    positives = {name: [] for name in EDITS}
    for _, path in random.sample(originals, EDITED):
        img = cv2.imread(path)
        for name, edit in EDITS.items():
            out = os.path.join(tmp, f"{len(os.listdir(tmp))}.png")
            cv2.imwrite(out, edit(img))
            positives[name].append((hash_distance(hashes[path], compute_hash(out)),
                                    float(embeds[path] @ compute_embedding(out))))

    negatives = []
    for (ca, a), (cb, b) in itertools.combinations(originals, 2):
        negatives.append((ca == cb, hash_distance(hashes[a], hashes[b]), float(embeds[a] @ embeds[b])))

thresholds = [0.85, 0.87, 0.88, 0.90]
print(f"{len(originals)} real photos, {EDITED} edited copies per edit type, {len(negatives)} different-photo pairs\n")
print("caught by hash alone, then by hash OR embedding at each threshold")
print(f"{'edit':22} {'hash':>6} " + " ".join(f"   >={t:.2f}" for t in thresholds))
for name, rows in positives.items():
    hash_recall = np.mean([is_duplicate(d) for d, _ in rows])
    both = " ".join(f"{np.mean([is_duplicate(d) or s >= t for d, s in rows]):>9.0%}" for t in thresholds)
    print(f"{name:22} {hash_recall:>6.0%} {both}")

print("\nfalse positives on different photos:")
print(f"{'':22} {sum(is_duplicate(d) for _, d, _ in negatives):>6} " +
      " ".join(f"{sum(s >= t for _, _, s in negatives):>9}" for t in thresholds))
same = [s for same_class, _, s in negatives if same_class]
print(f"\nhighest similarity between different photos: same class {max(same):.3f}, "
      f"any {max(s for _, _, s in negatives):.3f}")
