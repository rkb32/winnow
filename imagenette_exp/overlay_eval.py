"""How well does the shared-overlay check work? False alarms on real photos that share nothing but
their shape, then detection of a watermark painted onto some or all of a group at varying strength.

Run from the repo root: python imagenette_exp/overlay_eval.py
"""
import collections
import functools
import glob
import os
import random
import sys
import tempfile

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "winnow"))
from overlay import _shape_key, find_overlays

ROOT = os.environ.get("IMAGENETTE", os.path.join(HERE, "imagenette2-160"))
TRIALS = 60
GROUP_SIZES = [8, 12, 20]


@functools.lru_cache(maxsize=None)
def photo(path):
    return cv2.imread(path)


def load_shapes():
    """Every train photo, bucketed by shape the way the detector buckets them."""
    buckets = collections.defaultdict(list)
    for path in sorted(glob.glob(f"{ROOT}/train/*/*")):
        image = cv2.imread(path)
        if image is not None:
            buckets[_shape_key(image.shape[1], image.shape[0])].append(path)
    return sorted(buckets.values(), key=len, reverse=True)


def text_mask(shape, tiled):
    h, w = shape[:2]
    mask = np.zeros((h, w), np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    if not tiled:
        scale = w / 300
        (tw, th), _ = cv2.getTextSize("STOCKPHOTO", font, scale, max(1, round(scale * 2)))
        cv2.putText(mask, "STOCKPHOTO", (w - tw - w // 30, h - h // 20), font, scale, 255, max(1, round(scale * 2)))
        return mask
    big = np.zeros((h * 2, w * 2), np.uint8)
    scale = w / 260
    for y in range(0, h * 2, max(1, round(h / 3))):
        for x in range(-w, w * 2, max(1, round(w * 0.55))):
            cv2.putText(big, "STOCK", (x, y), font, scale, 255, max(1, round(scale * 2)))
    rotate = cv2.getRotationMatrix2D((w, h), 30, 1)
    return cv2.warpAffine(big, rotate, (w * 2, h * 2))[h // 2: h // 2 + h, w // 2: w // 2 + w]


def stamp(image, tiled, alpha):
    weight = (text_mask(image.shape, tiled) / 255.0 * alpha)[..., None]
    return (image * (1 - weight) + 255 * weight).astype(np.uint8)


def false_alarm_rate(buckets, n, same_class):
    rng = random.Random(n)
    hits = trials = 0
    for _ in range(TRIALS):
        bucket = rng.choice([b for b in buckets if len(b) >= 4 * n] or buckets[:1])
        if same_class:
            by_class = collections.defaultdict(list)
            for p in bucket:
                by_class[p.split(os.sep)[-2]].append(p)
            pool = rng.choice([v for v in by_class.values() if len(v) >= n] or [bucket])
        else:
            pool = bucket
        if len(pool) < n:
            continue
        trials += 1
        hits += bool(find_overlays(rng.sample(pool, n)))
    return hits / max(1, trials)


def detection_rate(buckets, n, tiled, alpha, carrying):
    rng = random.Random(n)
    found = wrong = trials = 0
    with tempfile.TemporaryDirectory() as tmp:
        for t in range(TRIALS):
            bucket = rng.choice([b for b in buckets if len(b) >= 4 * n] or buckets[:1])
            if len(bucket) < n:
                continue
            trials += 1
            chosen = rng.sample(bucket, n)
            stamped = set(chosen[:round(n * carrying)])
            paths = []
            for i, p in enumerate(chosen):
                image = photo(p)
                if p in stamped:
                    image = stamp(image, tiled, alpha)
                out = os.path.join(tmp, f"{t}_{i}.png")
                cv2.imwrite(out, image)
                paths.append((out, p in stamped))
            groups = find_overlays([o for o, _ in paths])
            truth = {o for o, s in paths if s}
            if groups:
                found += 1
                wrong += bool(set(groups[0]["files"]) - truth)
    return found / max(1, trials), wrong / max(1, found)


def main():
    buckets = load_shapes()
    print(f"{len(buckets)} shape buckets; largest holds {len(buckets[0])} photos.\n")
    print("False alarms (no overlay anywhere):")
    for n in GROUP_SIZES:
        print(f"  {n:>2} photos: mixed classes {false_alarm_rate(buckets, n, False):.1%}, "
              f"one class {false_alarm_rate(buckets, n, True):.1%}")
    print("\nDetection (share of trials where a group is reported; in brackets, how often that group "
          "wrongly included an unstamped photo):")
    cases = [(False, 0.15, 1.0), (False, 0.3, 1.0), (False, 0.3, 0.8), (False, 0.5, 1.0), (False, 0.5, 0.8),
             (True, 0.3, 1.0), (True, 0.5, 1.0)]
    for tiled, alpha, carrying in cases:
        cells = []
        for n in GROUP_SIZES:
            d, w = detection_rate(buckets, n, tiled, alpha, carrying)
            cells.append(f"{d:>4.0%} ({w:.0%})")
        print(f"  {'tiled diagonal' if tiled else 'corner logo':<15} alpha {alpha:<4} "
              f"on {carrying:.0%} of photos: " + "  ".join(cells), flush=True)


if __name__ == "__main__":
    main()
