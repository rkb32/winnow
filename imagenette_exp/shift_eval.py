"""How often does the train/test shift check fire? On no shift (should be about 5%), and on shifts
of growing subtlety: a different class mix, then the same photos made darker, grayscale, blurred,
or low quality.

Run from the repo root: python imagenette_exp/shift_eval.py
"""
import glob
import os
import random
import sys
import tempfile

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "winnow"))
from semantic import compute_embeddings
from shift import check_shift

ROOT = os.environ.get("IMAGENETTE", os.path.join(HERE, "imagenette2-160"))
PER_CLASS = 40
TRIALS = 300
SIZES = [(15, 5), (10, 10), (100, 30)]

PERTURBATIONS = {
    "darker": lambda im: np.clip((im / 255.0) ** 2.5 * 255, 0, 255).astype(np.uint8),
    "grayscale": lambda im: cv2.cvtColor(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR),
    "blurred": lambda im: cv2.GaussianBlur(im, (0, 0), 3),
    "JPEG quality 8": lambda im: cv2.imdecode(cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, 8])[1], 1),
}


def embed_pool():
    random.seed(0)
    classes = sorted(glob.glob(f"{ROOT}/train/*"))
    paths = [(c, p) for c in range(len(classes))
             for p in random.sample(sorted(glob.glob(f"{classes[c]}/*")), PER_CLASS)]
    base = compute_embeddings([p for _, p in paths])
    with tempfile.TemporaryDirectory() as tmp:
        changed = {}
        for name, edit in PERTURBATIONS.items():
            edited = []
            for i, (_, p) in enumerate(paths):
                out = os.path.join(tmp, f"{name}_{i}.png")
                cv2.imwrite(out, edit(cv2.imread(p)))
                edited.append((p, out))
            emb = compute_embeddings([o for _, o in edited])
            changed[name] = {p: emb[o] for p, o in edited}
    return paths, base, changed


def rate(paths, base, changed, n_train, n_test, test_pool, perturbation=None):
    rng = random.Random(1)
    hits = 0
    for trial in range(TRIALS):
        everything = [p for _, p in paths]
        drawn = rng.sample(everything, n_train)
        rest = [p for p in test_pool(rng) if p not in drawn]
        test = rng.sample(rest, n_test)
        source = changed[perturbation] if perturbation else base
        result = check_shift({p: base[p] for p in drawn}, {p: source[p] for p in test}, seed=trial)
        hits += result["flagged"]
    return hits / TRIALS


def main():
    paths, base, changed = embed_pool()
    by_class = {}
    for c, p in paths:
        by_class.setdefault(c, []).append(p)
    everything = lambda rng: [p for _, p in paths]
    two_classes = lambda rng: [p for c in rng.sample(sorted(by_class), 2) for p in by_class[c]]

    print(f"{TRIALS} trials per cell; each cell is the share of trials where the check fired.\n")
    print(f"{'test set vs train set':<38}" + "".join(f"{f'{a} train / {b} test':>20}" for a, b in SIZES))
    rows = [("same photos, random split (no shift)", everything, None),
            ("test drawn from only 2 of 10 classes", two_classes, None)]
    rows += [(f"same photos, test made {name}", everything, name) for name in PERTURBATIONS]
    for label, pool, perturbation in rows:
        cells = []
        for n_train, n_test in SIZES:
            if label.startswith("test drawn") and n_test > 2 * PER_CLASS:
                cells.append("n/a")
                continue
            cells.append(f"{rate(paths, base, changed, n_train, n_test, pool, perturbation):.0%}")
        print(f"{label:<38}" + "".join(f"{c:>20}" for c in cells))


if __name__ == "__main__":
    main()
