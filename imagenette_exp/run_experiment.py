import os
import random
import shutil
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "winnow"))
from duplicates import find_duplicate_pairs

random.seed(0)

CLASSES = {"n01440764": "tench", "n02102040": "springer"}
PER_CLASS = 30
NUM_PLANTED_LEAKS = 5

BASE = os.path.dirname(__file__)
SRC = os.path.join(BASE, "imagenette2-160")
COMBINED = os.path.join(BASE, "combined")

shutil.rmtree(COMBINED, ignore_errors=True)
os.makedirs(COMBINED)

planted_leaks = []

for code, name in CLASSES.items():
    train_files = sorted(os.listdir(os.path.join(SRC, "train", code)))[:PER_CLASS]
    val_files = sorted(os.listdir(os.path.join(SRC, "val", code)))[:PER_CLASS]

    for f in train_files:
        shutil.copy(os.path.join(SRC, "train", code, f), os.path.join(COMBINED, f"train__{name}__{f}"))
    for f in val_files:
        shutil.copy(os.path.join(SRC, "val", code, f), os.path.join(COMBINED, f"val__{name}__{f}"))

    # Plant real leaks: copy a few TRAIN images again, disguised as VAL images.
    leak_source = random.sample(train_files, min(NUM_PLANTED_LEAKS, len(train_files)))
    for f in leak_source:
        leaked_name = f"val__{name}__LEAK_{f}"
        shutil.copy(os.path.join(SRC, "train", code, f), os.path.join(COMBINED, leaked_name))
        planted_leaks.append((f"train__{name}__{f}", leaked_name))

print(f"Combined folder built: {len(os.listdir(COMBINED))} images")
print(f"Planted leaks: {len(planted_leaks)}")

image_paths = [os.path.join(COMBINED, f) for f in os.listdir(COMBINED)]
duplicate_pairs = find_duplicate_pairs(image_paths)

print(f"\nTotal duplicate pairs found: {len(duplicate_pairs)}")

cross_split_pairs = [
    (a, b, d) for a, b, d in duplicate_pairs
    if ("train__" in os.path.basename(a)) != ("train__" in os.path.basename(b))
]
print(f"Cross-split (train<->val) pairs found: {len(cross_split_pairs)}")

planted_basenames = {(os.path.basename(a), os.path.basename(b)) for a, b in planted_leaks}
found_basenames = {(os.path.basename(a), os.path.basename(b)) for a, b, _ in cross_split_pairs}
found_basenames |= {(b, a) for a, b in found_basenames}

caught = sum(1 for a, b in planted_basenames if (a, b) in found_basenames or (b, a) in found_basenames)
print(f"\nPlanted leaks caught: {caught} / {len(planted_leaks)}")
print(f"False positive cross-split flags (not planted leaks): {len(cross_split_pairs) - caught}")

print("\nPlanted leaks detail:")
for orig, leak in planted_leaks:
    hit = "CAUGHT" if (orig, leak) in found_basenames or (leak, orig) in found_basenames else "MISSED"
    print(f"  [{hit}] {orig} <-> {leak}")
