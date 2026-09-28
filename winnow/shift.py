"""Do the test photos come from the same kind of pictures as the training photos?

A leak check compares photos one by one. This compares the two sets: a model can score well on
a test set that looks nothing like what it will meet, and then the score says little.

The statistic is the squared distance between the two sets' mean embeddings. Its p-value comes
from shuffling which photos count as test, so it needs no assumption about the embedding
distribution and stays valid at the small sizes the website allows.
"""
import numpy as np

MIN_PER_SET = 5
MAX_PER_SET = 200        # bounds the shuffling cost on large folders
PERMUTATIONS = 2000
ALPHA = 0.05
OUTLIERS_SHOWN = 3


def _sample(embeddings, rng):
    paths = sorted(embeddings)
    if len(paths) > MAX_PER_SET:
        paths = sorted(str(p) for p in rng.choice(paths, MAX_PER_SET, replace=False))
    return paths, np.stack([embeddings[p] for p in paths]).astype(np.float64)


def check_shift(train_embeddings, test_embeddings, seed=0):
    counts = {"train_count": len(train_embeddings), "test_count": len(test_embeddings)}
    if min(counts.values()) < MIN_PER_SET:
        return {"checked": False, **counts}

    rng = np.random.default_rng(seed)
    train_paths, train = _sample(train_embeddings, rng)
    test_paths, test = _sample(test_embeddings, rng)
    pooled = np.vstack([train, test])
    n, m = len(pooled), len(test)
    total = pooled.sum(axis=0)

    def gap(test_sums):
        return ((test_sums / m - (total - test_sums) / (n - m)) ** 2).sum(axis=-1)

    observed = float(gap(test.sum(axis=0)))
    chosen = np.argsort(rng.random((PERMUTATIONS, n)), axis=1)[:, :m]
    membership = np.zeros((PERMUTATIONS, n))
    np.put_along_axis(membership, chosen, 1.0, axis=1)
    as_extreme = int((gap(membership @ pooled) >= observed).sum())
    p_value = (1 + as_extreme) / (1 + PERMUTATIONS)

    flagged = p_value <= ALPHA
    outliers = []
    if flagged:
        closest = (test @ train.T).max(axis=1)
        outliers = [[test_paths[i], round(float(closest[i]), 3)] for i in np.argsort(closest)[:OUTLIERS_SHOWN]]
    return {"checked": True, **counts, "distance": round(observed, 4), "p_value": round(p_value, 4),
            "flagged": flagged, "outliers": outliers}
