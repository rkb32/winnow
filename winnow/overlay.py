"""The same fixed pattern at the same spot in many photos: a watermark, a logo, a frame, or a
camera that never moved.

Natural content mostly puts edges of varying direction at varying places, so across enough photos
few pixels keep seeing an edge of the same sign. A pattern painted onto the photos does: its edges
agree in every photo that carries it, whatever is underneath. Photos that share a source or
template share the pattern, and a model can learn the pattern instead of the subject.
"""
import math
from collections import defaultdict

import cv2
import numpy as np

# Calibrated on Imagenette (imagenette_exp/overlay_eval.py). Natural photos share more structure
# than chance suggests (sky over ground, a centered subject), so a pattern has to appear in most
# of a group, and in an absolute minimum of photos, before it counts.
MIN_PHOTOS = 8           # photos of one shape needed before any comparison
MIN_CARRYING = 7         # photos that must show the pattern
AGREE_MIN = 0.7          # ...and this share of the group
WORK_SIDE = 192          # every photo of one shape is compared at this size (long side)
EDGE_LEVEL = 10          # Sobel response that counts as an edge at a pixel (blurred, 0-255 scale)
MIN_AREA = 0.002         # agreeing pixels needed, as a share of the frame
MEMBER_MIN = 0.7         # share of a pattern's pixels a photo must match to carry it
MAX_PER_GROUP = 200      # bounds memory and time on large folders; a pattern on most photos shows in a sample
MAX_GROUPS = 10
ASPECT_STEP = 0.05       # photos within ~5% of the same aspect ratio put a relative watermark in the same place


def _shape_key(width, height):
    return round(math.log(width / height) / ASPECT_STEP)


def _work_size(key):
    aspect = math.exp(key * ASPECT_STEP)
    return (WORK_SIDE, max(8, round(WORK_SIDE / aspect))) if aspect >= 1 else (max(8, round(WORK_SIDE * aspect)), WORK_SIDE)


def _edges(small):
    small = cv2.GaussianBlur(small, (0, 0), 0.8)
    return cv2.Sobel(small, cv2.CV_32F, 1, 0, ksize=3) / 4, cv2.Sobel(small, cv2.CV_32F, 0, 1, ksize=3) / 4


def _pattern_in(gx, gy):
    """The mask of pixels where the group keeps a same-signed edge, plus each pixel's strength,
    direction, and whether that direction is horizontal."""
    needed = max(MIN_CARRYING, math.ceil(AGREE_MIN * len(gx))) / len(gx)
    votes = []
    for g in (gx, gy):
        up, down = (g >= EDGE_LEVEL).mean(axis=0), (g <= -EDGE_LEVEL).mean(axis=0)
        votes.append((np.maximum(up, down), np.where(up >= down, 1.0, -1.0)))
    (agree_x, sign_x), (agree_y, sign_y) = votes
    use_x = agree_x >= agree_y
    agree = np.where(use_x, agree_x, agree_y)
    sign = np.where(use_x, sign_x, sign_y)
    return agree >= needed - 1e-9, agree, sign, use_x


def find_overlays(paths):
    """[{files, box, strength}]: photos sharing a fixed pattern, and where it sits ([x1, y1, x2, y2]
    as fractions of the frame). Only photos of about the same shape can be compared."""
    by_shape = defaultdict(list)
    for path in sorted(paths):
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if image is not None:
            key = _shape_key(image.shape[1], image.shape[0])
            by_shape[key].append((path, cv2.resize(image, _work_size(key), interpolation=cv2.INTER_AREA)))

    found = []
    for photos in by_shape.values():
        if len(photos) < MIN_PHOTOS:
            continue
        if len(photos) > MAX_PER_GROUP:
            photos = [photos[i] for i in np.linspace(0, len(photos) - 1, MAX_PER_GROUP).astype(int)]
        edges = [_edges(small) for _, small in photos]
        gx, gy = (np.stack([e[i] for e in edges]) for i in (0, 1))

        mask, agree, sign, use_x = _pattern_in(gx, gy)
        frame = mask.size
        if mask.sum() < MIN_AREA * frame:
            continue
        # Letter strokes are thin and separate; merge them so a logo counts as one pattern.
        _, labels, stats, _ = cv2.connectedComponentsWithStats(
            cv2.dilate(mask.astype(np.uint8), np.ones((5, 5), np.uint8)))
        best, best_pixels = 0, 0
        for label in range(1, len(stats)):
            pixels = int((mask & (labels == label)).sum())
            if pixels > best_pixels:
                best, best_pixels = label, pixels
        if best_pixels < MIN_AREA * frame:
            continue

        pattern = mask & (labels == best)
        strongest = np.where(use_x[None], gx, gy) * sign[None]
        carrying = (strongest[:, pattern] >= EDGE_LEVEL / 2).mean(axis=1) >= MEMBER_MIN
        if carrying.sum() < MIN_CARRYING:
            continue
        height, width = mask.shape
        ys, xs = np.nonzero(pattern)
        found.append({
            "files": [path for (path, _), yes in zip(photos, carrying) if yes],
            "box": [round(float(xs.min()) / width, 3), round(float(ys.min()) / height, 3),
                    round(float(xs.max() + 1) / width, 3), round(float(ys.max() + 1) / height, 3)],
            "strength": round(float(agree[pattern].mean()), 2),
        })
    return sorted(found, key=lambda g: -len(g["files"]))[:MAX_GROUPS]
