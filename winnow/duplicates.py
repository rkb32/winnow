import cv2
from cv2 import img_hash

_hasher = img_hash.BlockMeanHash_create()


def compute_hash(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    return _hasher.compute(image)


def hash_distance(hash_a, hash_b):
    return _hasher.compare(hash_a, hash_b)


def is_duplicate(distance):
    if distance <= 5:
        return True
    else:
        return False


def _safe_hashes(paths):
    hashes = {}
    for path in paths:
        try:
            hashes[path] = compute_hash(path)
        except Exception:
            continue
    return hashes


def find_leaked_pairs(train_paths, test_paths):
    train_hashes = _safe_hashes(train_paths)
    test_hashes = _safe_hashes(test_paths)
    leaks = []
    for train_path, train_hash in train_hashes.items():
        for test_path, test_hash in test_hashes.items():
            distance = hash_distance(train_hash, test_hash)
            if is_duplicate(distance):
                leaks.append((train_path, test_path, distance))
    return leaks


def find_duplicate_pairs(image_paths):
    hashes = _safe_hashes(image_paths)
    paths = list(hashes.keys())
    pairs = []
    for i in range(len(paths)):
        for j in range(i + 1, len(paths)):
            distance = hash_distance(hashes[paths[i]], hashes[paths[j]])
            if is_duplicate(distance):
                pairs.append((paths[i], paths[j], distance))
    return pairs
