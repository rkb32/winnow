"""Which keypoint-verified flags does the detector get wrong, and can its own numbers tell?

Uses the full audit's AI-verified pairs (full_audit/verified_pairs.csv). For every flag that came
from the embedding + keypoint stage, compares real copies with false matches on two signals: the
inlier count, and how much of each photo the matched keypoints cover. This sets which pairs the
live agent looks at as photos (agent.py, REVIEW_MAX_INLIERS).
"""
import csv
import statistics
import sys

sys.path.insert(0, "/app")
from semantic import match_regions

ROOT = "/data/imagenette2-160"


def area(box):
    return (box[2] - box[0]) * (box[3] - box[1])


rows = [r for r in csv.DictReader(open("/data/full_audit/verified_pairs.csv", encoding="utf-8"))
        if r["method"] == "keypoints"]
pairs = []
for r in rows:
    regions = match_regions(f"{ROOT}/{r['train_file']}", f"{ROOT}/{r['val_file']}")
    coverage = min(area(regions[0]), area(regions[1])) if regions else 0.0
    pairs.append({"real": r["verdict"] != "different", "inliers": int(r["score"]), "coverage": coverage,
                  "evidence": r["evidence"]})

real = [p for p in pairs if p["real"]]
false = [p for p in pairs if not p["real"]]
print(f"{len(pairs)} keypoint-stage flags: {len(real)} real copies, {len(false)} false matches\n")

for label, group in (("real copies", real), ("false matches", false)):
    cov = sorted(p["coverage"] for p in group)
    print(f"{label:14} coverage median {statistics.median(cov):.2f}, "
          f"quartiles {cov[len(cov) // 4]:.2f} / {cov[3 * len(cov) // 4]:.2f}")

print("\nfalse matches, lowest coverage first:")
for p in sorted(false, key=lambda p: p["coverage"]):
    print(f"  {p['inliers']:3d} inliers, coverage {p['coverage']:.2f}  {p['evidence'][:90]}")

print("\ngate                                   false caught   real sent too")
for max_inliers in (35, 50, 100):
    for max_coverage in (None, 0.25, 0.4):
        name = f"inliers < {max_inliers}" + (f" or coverage < {max_coverage}" if max_coverage else "")
        either = [p for p in pairs if p["inliers"] < max_inliers or
                  (max_coverage is not None and p["coverage"] < max_coverage)]
        caught = sum(not p["real"] for p in either)
        sent = sum(p["real"] for p in either)
        print(f"  {name:36} {caught:3d}/{len(false):<3d}        {sent:3d}/{len(real)}")
