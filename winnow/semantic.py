"""Finds edited copies (crops, mirrors, recolors) the perceptual hash misses, in two stages.

1. Retrieval: MobileNetV2 embeddings via cv2.dnn nominate pairs that look alike.
2. Verification: SIFT keypoints + RANSAC confirm the pair shares one geometric transform.
   Embeddings alone can't tell "same photo" from "same kind of photo" (two different
   garbage trucks score ~0.9); only a real copy has hundreds of points that line up.
"""
import os
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
    return _sift.detectAndCompute(image, None)


# A pair-comparison workload compares the same photo against many others (a burst of
# near-identical shots, or one photo checked against an entire other split), so SIFT features
# get requested for the same path repeatedly. Keyed by path since content is fixed for a scan.
@lru_cache(maxsize=512)
def _keypoints_for(path, flipped=False):
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if flipped:
        image = cv2.flip(image, 1)
    return _keypoints(image)


def _ransac_inliers(features_a, features_b):
    (kp_a, des_a), (kp_b, des_b) = features_a, features_b
    if des_a is None or des_b is None or len(kp_a) < 2 or len(kp_b) < 2:
        return 0
    good = [p[0] for p in _matcher.knnMatch(des_a, des_b, k=2)
            if len(p) == 2 and p[0].distance < 0.75 * p[1].distance]
    if len(good) < 8:
        return 0
    src = np.float32([kp_a[m.queryIdx].pt for m in good])
    dst = np.float32([kp_b[m.trainIdx].pt for m in good])
    _, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    return int(mask.sum()) if mask is not None else 0


def keypoint_inliers(path_a, path_b):
    """Keypoint matches consistent with one geometric transform, trying path_a mirrored too."""
    features_b = _keypoints_for(path_b)
    return max(_ransac_inliers(_keypoints_for(path_a), features_b),
               _ransac_inliers(_keypoints_for(path_a, flipped=True), features_b))


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
        similar = a in embeddings and b in embeddings and float(embeddings[a] @ embeddings[b]) >= HASH_CONFIRM_SIMILARITY
        if similar or keypoint_inliers(a, b) >= HASH_CONFIRM_INLIERS:
            confirmed.append((a, b, distance))
    return confirmed


def find_semantic_pairs(embeddings, already_found):
    skip = {frozenset(p[:2]) for p in already_found}
    return _verified_pairs(embeddings, embeddings, skip, same_set=True)


def find_semantic_leaks(train_embeddings, test_embeddings, already_found):
    skip = {frozenset(p[:2]) for p in already_found}
    return _verified_pairs(train_embeddings, test_embeddings, skip, same_set=False)
