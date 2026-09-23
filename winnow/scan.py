import os
import sys

from blur import compute_sharpness, is_too_blurry
from duplicates import find_duplicate_pairs
from exif_conflict import get_orientation, has_risky_orientation


def list_images(folder_path):
    return [
        os.path.join(folder_path, name)
        for name in os.listdir(folder_path)
        if name.lower().endswith((".png", ".jpg", ".jpeg"))
    ]


def scan_folder(folder_path):
    image_paths = list_images(folder_path)

    blurry_images = []
    risky_orientation_images = []
    for path in image_paths:
        sharpness = compute_sharpness(path)
        if is_too_blurry(sharpness):
            blurry_images.append((path, sharpness))

        orientation = get_orientation(path)
        if has_risky_orientation(orientation):
            risky_orientation_images.append((path, orientation))

    return {
        "duplicate_pairs": find_duplicate_pairs(image_paths),
        "blurry_images": blurry_images,
        "risky_orientation_images": risky_orientation_images,
    }


if __name__ == "__main__":
    result = scan_folder(sys.argv[1])
    print("Duplicate pairs:          ", result["duplicate_pairs"])
    print("Blurry images:            ", result["blurry_images"])
    print("Risky EXIF orientation:   ", result["risky_orientation_images"])
