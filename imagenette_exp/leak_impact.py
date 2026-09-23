"""Does train/test leakage inflate measured accuracy, and does removing what Winnow flags undo it?

The model is a 1-nearest-neighbor classifier on 32x32 thumbnails: it memorizes its training
set outright, which exaggerates the mechanism by which leaks inflate scores in real models.
The direction of the effect carries over; the exact size won't.
"""
import glob
import random
import sys
import tempfile

import cv2
import numpy as np

sys.path.insert(0, "/app")
from duplicates import find_leaked_pairs
from semantic import compute_embeddings, confirm_hash_pairs, find_semantic_leaks

random.seed(0)
ROOT = "/data/imagenette2-160"
TRAIN_PER_CLASS, TEST_PER_CLASS, LEAKED = 100, 50, 75


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


EDITS = [("exact", lambda img: img), ("resize+JPEG", recompress), ("crop", crop), ("mirror", lambda img: cv2.flip(img, 1))]
_thumbs = {}


def thumb(path):
    if path not in _thumbs:
        _thumbs[path] = cv2.resize(cv2.imread(path), (32, 32)).astype(np.float32).ravel() / 255.0
    return _thumbs[path]


def accuracy(train, test):
    X, y = np.stack([thumb(p) for p, _ in train]), np.array([label for _, label in train])
    T, t = np.stack([thumb(p) for p, _ in test]), np.array([label for _, label in test])
    dist = (T ** 2).sum(1)[:, None] - 2 * T @ X.T + (X ** 2).sum(1)[None, :]
    return float((y[dist.argmin(1)] == t).mean())


train, test = sample("train", TRAIN_PER_CLASS), sample("val", TEST_PER_CLASS)
leaked = random.sample(test, LEAKED)

with tempfile.TemporaryDirectory() as tmp:
    planted = {}
    contaminated = list(train)
    for k, (path, label) in enumerate(leaked):
        name, edit = EDITS[k % len(EDITS)]
        out = f"{tmp}/leak_{k}.png"
        cv2.imwrite(out, edit(cv2.imread(path)))
        planted[out] = name
        contaminated.append((out, label))

    train_paths, test_paths = [p for p, _ in contaminated], [p for p, _ in test]
    train_emb, test_emb = compute_embeddings(train_paths), compute_embeddings(test_paths)
    hash_leaks = confirm_hash_pairs(find_leaked_pairs(train_paths, test_paths), {**train_emb, **test_emb})
    emb_leaks = find_semantic_leaks(train_emb, test_emb, hash_leaks)
    by_hash = {a for a, _, _ in hash_leaks}
    flagged = by_hash | {pair[0] for pair in emb_leaks}

    clean = accuracy(train, test)
    dirty = accuracy(contaminated, test)
    fixed = accuracy([(p, label) for p, label in contaminated if p not in flagged], test)

print(f"{len(train)} training photos, {len(test)} test photos, {LEAKED} test photos copied into training\n")
print(f"accuracy, clean training set:        {clean:.1%}")
print(f"accuracy, with leaked copies:         {dirty:.1%}   (+{(dirty - clean) * 100:.1f} points of fake accuracy)")
print(f"accuracy, after removing flagged:     {fixed:.1%}\n")
print(f"{'leak type':14} {'hash':>6} {'hash+emb':>9}")
for name, _ in EDITS:
    copies = [p for p, n in planted.items() if n == name]
    print(f"{name:14} {sum(p in by_hash for p in copies):>3}/{len(copies):<2} {sum(p in flagged for p in copies):>5}/{len(copies)}")
print(f"{'total':14} {len(by_hash & set(planted)):>3}/{LEAKED} {len(flagged & set(planted)):>5}/{LEAKED}")
print(f"\ninnocent training photos flagged: {len(flagged - set(planted))}")
