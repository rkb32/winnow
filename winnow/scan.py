import os
import sys

from blur import compute_sharpness, is_too_blurry
from duplicates import find_duplicate_pairs, find_leaked_pairs
from exif_conflict import get_orientation, has_risky_orientation, swaps_dimensions


def list_images(folder_path):
    return [
        os.path.join(folder_path, name)
        for name in os.listdir(folder_path)
        if name.lower().endswith((".png", ".jpg", ".jpeg"))
    ]


def scan_folder(folder_path, test_folder_path=None):
    image_paths = list_images(folder_path)

    blurry_images = []
    risky_orientation_images = []
    unreadable_images = []
    readable_paths = []
    for path in image_paths:
        try:
            sharpness = compute_sharpness(path)
            orientation = get_orientation(path)
        except Exception as e:
            unreadable_images.append((path, str(e)))
            continue

        readable_paths.append(path)
        if is_too_blurry(sharpness):
            blurry_images.append((path, sharpness))
        if has_risky_orientation(orientation):
            risky_orientation_images.append((path, orientation, swaps_dimensions(orientation)))

    result = {
        "duplicate_pairs": find_duplicate_pairs(readable_paths),
        "blurry_images": blurry_images,
        "risky_orientation_images": risky_orientation_images,
        "unreadable_images": unreadable_images,
    }

    if test_folder_path:
        try:
            test_paths = list_images(test_folder_path)
        except OSError as e:
            unreadable_images.append((test_folder_path, str(e)))
            test_paths = []
        result["leaked_pairs"] = find_leaked_pairs(readable_paths, test_paths)

    return result


if __name__ == "__main__":
    result = scan_folder(sys.argv[1])
    print("Duplicate pairs:          ", result["duplicate_pairs"])
    print("Blurry images:            ", result["blurry_images"])
    print("Risky EXIF orientation:   ", result["risky_orientation_images"])
    print("Unreadable images:        ", result["unreadable_images"])
