"""Finds edited copies (crops, mirrors, recolors) the perceptual hash misses, in two stages.

1. Retrieval: MobileNetV2 embeddings via cv2.dnn nominate pairs that look alike.
2. Verification: SIFT keypoints + RANSAC confirm the pair shares one geometric transform.
   Embeddings alone can't tell "same photo" from "same kind of photo" (two different
   garbage trucks score ~0.9); only a real copy has hundreds of points that line up.
"""
import os
import random
from functools import lru_cache

import cv2
import numpy as np

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "embedder.onnx")
# Both calibrated on Imagenette in imagenette_exp/calibrate_verify.py.
CANDIDATE_THRESHOLD = 0.80
MIN_INLIERS = 25
_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
_STD = np.array([0.229, 0.224, 0.225], np.float32)
_MAX_SIDE = 640
_net = None
_sift = cv2.SIFT_create(nfeatures=1000)
_matcher = cv2.BFMatcher(cv2.NORM_L2)


def _embed(net, image):
    rgb = cv2.cvtColor(cv2.resize(image, (224, 224)), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    net.setInput(cv2.dnn.blobFromImage((rgb - _MEAN) / _STD))
    vector = net.forward().flatten()
    return vector / np.linalg.norm(vector)


def compute_embedding(image_path):
    global _net
    if _net is None:
        _net = cv2.dnn.readNet(MODEL_PATH)
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    # CNN features aren't mirror-symmetric, so average in the flipped view to catch mirrored copies.
    both = _embed(_net, image) + _embed(_net, cv2.flip(image, 1))
    return both / np.linalg.norm(both)


def compute_embeddings(paths):
    embeddings = {}
    for path in paths:
        try:
            embeddings[path] = compute_embedding(path)
        except Exception:
            continue
    return embeddings


def _keypoints(image):
    scale = _MAX_SIDE / max(image.shape[:2])
    if scale < 1:
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    keypoints, descriptors = _sift.detectAndCompute(image, None)
    return keypoints, descriptors, image.shape[:2]


# A pair-comparison workload compares the same photo against many others (a burst of
# near-identical shots, or one photo checked against an entire other split), so SIFT features
# get requested for the same path repeatedly. Keyed by path since content is fixed for a scan.
@lru_cache(maxsize=512)
def _keypoints_for(path, flipped=False):
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if flipped:
        image = cv2.flip(image, 1)
    return _keypoints(image)


_NO_MATCH = (np.empty((0, 2), np.float32), np.empty((0, 2), np.float32), np.empty((0, 128), np.float32))


def _ransac_full(features_a, features_b):
    """The matched keypoints consistent with one homography: (points in a, points in b, a's descriptors)."""
    (kp_a, des_a, _), (kp_b, des_b, _) = features_a, features_b
    if des_a is None or des_b is None or len(kp_a) < 2 or len(kp_b) < 2:
        return _NO_MATCH
    good = [p[0] for p in _matcher.knnMatch(des_a, des_b, k=2)
            if len(p) == 2 and p[0].distance < 0.75 * p[1].distance]
    if len(good) < 8:
        return _NO_MATCH
    src = np.float32([kp_a[m.queryIdx].pt for m in good])
    dst = np.float32([kp_b[m.trainIdx].pt for m in good])
    _, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    if mask is None:
        return _NO_MATCH
    keep = mask.ravel().astype(bool)
    return src[keep], dst[keep], des_a[np.array([m.queryIdx for m in good])[keep]]


def _ransac(features_a, features_b):
    """The matched keypoints consistent with one homography, as (points in a, points in b)."""
    return _ransac_full(features_a, features_b)[:2]


def _best_match_full(path_a, path_b):
    """Inliers between the photos, trying path_a mirrored too; path_a's points come back un-mirrored.
    Returns (points in a, points in b, shape a, shape b, a's descriptors for those keypoints)."""
    features_b = _keypoints_for(path_b)
    features_a = _keypoints_for(path_a)
    straight = _ransac_full(features_a, features_b)
    mirrored = _ransac_full(_keypoints_for(path_a, flipped=True), features_b)
    if len(mirrored[0]) > len(straight[0]):
        points_a = mirrored[0].copy()
        points_a[:, 0] = features_a[2][1] - 1 - points_a[:, 0]
        return points_a, mirrored[1], features_a[2], features_b[2], mirrored[2]
    return straight[0], straight[1], features_a[2], features_b[2], straight[2]


def _best_match(path_a, path_b):
    return _best_match_full(path_a, path_b)[:4]


# A fixed camera repeats its background pixel for pixel in every frame, so whole-frame SIFT
# finds hundreds of inliers (and the embedder sees a match) even when the subject changed.
# When two photos line up with no shift at all, only the region that differs between them
# can say whether they show the same thing.
CHANGE_LEVEL = 25          # grayscale difference that counts as changed, after a light blur
# Below 0.1% changed, the same picture: JPEG noise leaves 0 even at quality 10, and a ticking
# seconds counter stays under it. The cost is that a subject under ~40px in a 1000px frame still
# reads as a copy; a false alarm someone can clear beats a near-duplicate frame nobody sees.
MIN_CHANGED_AREA = 0.001
MAX_CHANGED_AREA = 0.8     # more than this: nothing stayed put (a recolor), so no fixed background


def _same_framing(points_a, points_b, shape_a, shape_b):
    """True when the matched points sit at the same pixels in both photos (no crop, zoom or
    shift), which is what every pair of frames from a fixed camera looks like."""
    return (shape_a == shape_b and len(points_a) >= 8
            and float(np.median(np.abs(points_a - points_b).max(axis=1))) <= 2.0)


def _gray(path, shape):
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return cv2.resize(image, (shape[1], shape[0]), interpolation=cv2.INTER_AREA)


def _changed_region(path_a, path_b, shape):
    """Where two same-framed photos differ, as a mask at SIFT's working size, or None when
    nearly nothing or nearly everything changed (neither has a fixed background)."""
    a, b = (cv2.GaussianBlur(_gray(path, shape), (5, 5), 0) for path in (path_a, path_b))
    changed = (cv2.absdiff(a, b) > CHANGE_LEVEL).astype(np.uint8)
    changed = cv2.morphologyEx(changed, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    if not MIN_CHANGED_AREA <= changed.mean() <= MAX_CHANGED_AREA:
        return None
    # Keypoints sit on edges, so grow the region to take in the ones on the subject's outline.
    return cv2.dilate(changed, np.ones((15, 15), np.uint8))


def _inside(features, mask):
    keypoints, descriptors, shape = features
    keep = [i for i, k in enumerate(keypoints)
            if mask[min(int(k.pt[1]), shape[0] - 1), min(int(k.pt[0]), shape[1] - 1)]]
    return [keypoints[i] for i in keep], (descriptors[keep] if keep else None), shape


def _content_match_full(path_a, path_b):
    """_best_match, minus any background the photos only share because the camera didn't move.
    The fifth value says whether that background was set aside; the sixth is path_a's descriptors
    for the keypoints that count."""
    points_a, points_b, shape_a, shape_b, descriptors = _best_match_full(path_a, path_b)
    if _same_framing(points_a, points_b, shape_a, shape_b):
        changed = _changed_region(path_a, path_b, shape_a)
        if changed is not None:
            points_a, points_b, descriptors = _ransac_full(_inside(_keypoints_for(path_a), changed),
                                                            _inside(_keypoints_for(path_b), changed))
            return points_a, points_b, shape_a, shape_b, True, descriptors
    return points_a, points_b, shape_a, shape_b, False, descriptors


def _content_match(path_a, path_b):
    return _content_match_full(path_a, path_b)[:5]


def keypoint_inliers(path_a, path_b):
    """Keypoint matches consistent with one geometric transform, trying path_a mirrored too.
    For frames from a fixed camera, only matches on what changed between them count."""
    return len(_content_match(path_a, path_b)[0])


def _box(points, shape):
    height, width = shape
    (x1, y1), (x2, y2) = points.min(axis=0), points.max(axis=0)
    return [round(float(x1) / width, 3), round(float(y1) / height, 3),
            round(float(x2) / width, 3), round(float(y2) / height, 3)]


def match_regions(path_a, path_b):
    """Where the matched keypoints sit in each photo, as [x1, y1, x2, y2] fractions of its size."""
    points_a, points_b, shape_a, shape_b, _ = _content_match(path_a, path_b)
    if not len(points_a):
        return None
    return _box(points_a, shape_a), _box(points_b, shape_b)


def _verified_pairs(left, right, skip, same_set):
    pairs = []
    left_paths, right_paths = list(left), list(right)
    if not left_paths or not right_paths:
        return pairs
    similarity = np.stack([left[p] for p in left_paths]) @ np.stack([right[p] for p in right_paths]).T
    for i, j in zip(*np.nonzero(similarity >= CANDIDATE_THRESHOLD)):
        a, b = left_paths[i], right_paths[j]
        if (same_set and j <= i) or frozenset((a, b)) in skip:
            continue
        inliers = keypoint_inliers(a, b)
        if inliers >= MIN_INLIERS:
            pairs.append((a, b, round(float(similarity[i, j]), 3), inliers))
    return pairs


# Perceptual hashes collide on low-texture photos (products on white, silhouettes on sky): across
# Imagenette's 37M train/val pairs, all 13 raw hash matches were different photos, with at most
# 6 shared keypoints and similarity at most 0.80. Real copies caught by the hash shared 9+ keypoints
# or looked near-identical to the embedder (imagenette_exp/full_audit.py and README).
HASH_CONFIRM_INLIERS = 8
HASH_CONFIRM_SIMILARITY = 0.85


def confirm_hash_pairs(pairs, embeddings):
    confirmed = []
    for a, b, distance in pairs:
        points, _, _, _, fixed_camera = _content_match(a, b)
        # A shared background fools the embedder as much as the hash, so it can't vouch for
        # frames from a fixed camera; only keypoints on what changed can.
        similar = (not fixed_camera and a in embeddings and b in embeddings
                   and float(embeddings[a] @ embeddings[b]) >= HASH_CONFIRM_SIMILARITY)
        if similar or len(points) >= HASH_CONFIRM_INLIERS:
            confirmed.append((a, b, distance))
    return confirmed


# A printed template (a card frame, a form header, a stock-photo banner) repeats across many
# different photos, so its keypoints match "everywhere" and can make two different photos look like
# a copy. A keypoint that also matches in several unrelated photos is a template, not evidence, and a
# pair is judged on the keypoints that are left. Calibrated on Jesse Diaz's binder benchmark
# (binder_eval/) and checked against the hand-verified Imagenette audit (imagenette_exp/).
TEMPLATE_REFERENCES = 40         # unrelated photos each pair's keypoints are checked against
TEMPLATE_MATCHES = 3             # a keypoint that matches in this many of them is a template
MIN_DISTINCT_INLIERS = 20        # keypoints left, template ones set aside, for a pair to stay a copy
MIN_PHOTOS_FOR_TEMPLATES = 100   # smaller sets have too few unrelated photos to tell a template


def distinct_inliers(path_a, path_b, reference_paths):
    """How many of the pair's matching keypoints do not also match in the reference photos."""
    descriptors = _content_match_full(path_a, path_b)[5]
    if not len(descriptors):
        return 0
    seen_elsewhere = np.zeros(len(descriptors), int)
    for reference in reference_paths:
        des_ref = _keypoints_for(reference)[1]
        if des_ref is None or len(des_ref) < 2:
            continue
        seen_elsewhere += np.array([len(p) == 2 and p[0].distance < 0.75 * p[1].distance
                                    for p in _matcher.knnMatch(descriptors, des_ref, k=2)], int)
    return int((seen_elsewhere < TEMPLATE_MATCHES).sum())


def set_aside_templates(pairs, all_paths):
    """Splits edited-copy findings (a, b, similarity, inliers) into (kept, look_alikes).

    A look-alike is a pair whose matches are mostly a template shared with other photos. It is
    returned with its count of distinct keypoints (a, b, similarity, inliers, distinct) instead of
    being dropped, so it can still be shown. Each pair is checked against a fixed random sample of
    the other photos, so a rerun gives the same answer. Sets too small to sample from are left alone.
    """
    if len(all_paths) < MIN_PHOTOS_FOR_TEMPLATES:
        return list(pairs), []
    others = sorted(all_paths)
    kept, look_alikes = [], []
    for pair in pairs:
        a, b = pair[:2]
        pool = [p for p in others if p != a and p != b]
        references = random.Random(f"{a}|{b}").sample(pool, min(TEMPLATE_REFERENCES, len(pool)))
        distinct = distinct_inliers(a, b, references)
        if distinct >= MIN_DISTINCT_INLIERS:
            kept.append(pair)
        else:
            look_alikes.append((*pair, distinct))
    return kept, look_alikes


def find_semantic_pairs(embeddings, already_found):
    skip = {frozenset(p[:2]) for p in already_found}
    return _verified_pairs(embeddings, embeddings, skip, same_set=True)


def find_semantic_leaks(train_embeddings, test_embeddings, already_found):
    skip = {frozenset(p[:2]) for p in already_found}
    return _verified_pairs(train_embeddings, test_embeddings, skip, same_set=False)
