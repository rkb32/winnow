"""Does setting aside shared-template matches keep the real Imagenette leaks?

Takes the 95 keypoint-stage flags of the full audit (full_audit.py), each hand-verified as the same
photo, the same scene, or a different photo, and runs winnow/semantic.py's set_aside_templates() over
them with all 13,394 Imagenette photos as the pool of other photos. A real leak set aside would be
a regression; a look-alike ("different") set aside is the point.

Run from the repo root: python imagenette_exp/template_eval.py
"""
import csv
import glob
import json
import os
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "winnow"))
from semantic import keypoint_inliers, set_aside_templates

ROOT = os.environ.get("IMAGENETTE", os.path.join(HERE, "imagenette2-160"))
local = lambda path: path.replace("/data/imagenette2-160", ROOT)    # the audit ran in Docker at /data

verdicts = {int(r["id"]): r["verdict"] for r in csv.DictReader(open(f"{HERE}/full_audit/verified_pairs.csv"))}
flagged = [f for f in json.load(open(f"{HERE}/full_audit/flagged.json")) if f["method"] == "keypoints"]
everything = sorted(glob.glob(f"{ROOT}/train/*/*.JPEG")) + sorted(glob.glob(f"{ROOT}/val/*/*.JPEG"))
print(f"{len(flagged)} keypoint-stage flags; {len(everything)} photos to sample other photos from")

pairs, verdict_of = [], {}
for f in flagged:
    a, b = local(f["train"]), local(f["val"])
    pairs.append((a, b, 0.0, keypoint_inliers(a, b)))
    verdict_of[a, b] = verdicts[f["id"]]

kept, aside = set_aside_templates(pairs, everything)
kept_count = Counter(verdict_of[p[0], p[1]] for p in kept)
aside_count = Counter(verdict_of[p[0], p[1]] for p in aside)
print(f"\n{'verdict':<12}{'still flagged':>15}{'set aside':>12}")
for verdict in ("same_photo", "same_scene", "different"):
    print(f"{verdict:<12}{kept_count[verdict]:>15}{aside_count[verdict]:>12}")
real = ("same_photo", "same_scene")
print(f"\nreal leaks kept: {sum(kept_count[v] for v in real)} of {sum(kept_count[v] + aside_count[v] for v in real)}; "
      f"look-alikes set aside: {aside_count['different']} of {kept_count['different'] + aside_count['different']}")
