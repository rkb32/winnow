import os
import sys

from blur import compute_sharpness, is_too_blurry
from bursts import find_bursts
from duplicates import find_duplicate_pairs, find_leaked_pairs
from exif_conflict import get_orientation, has_risky_orientation, swaps_dimensions
from labels import check_labels
from overlay import find_overlays
from provenance import find_provenance_flags
from semantic import (compute_embeddings, confirm_hash_pairs, find_semantic_leaks, find_semantic_pairs,
                      set_aside_templates)
from shift import check_shift

LABEL_EXTENSIONS = (".txt", ".xml", ".json")


def _advisory(check, fallback, *args):
    """Runs one of the report-only checks. They read visitors' metadata and label files, so if one
    fails, the scan still returns everything else."""
    try:
        return check(*args)
    except Exception as e:
        print(f"{check.__name__} failed: {type(e).__name__}: {e}")
        return fallback


def list_images(folder_path):
    return [
        os.path.join(folder_path, name)
        for name in os.listdir(folder_path)
        if name.lower().endswith((".png", ".jpg", ".jpeg"))
    ]


def list_label_files(folder_path):
    return [
        os.path.join(folder_path, name)
        for name in os.listdir(folder_path)
        if name.lower().endswith(LABEL_EXTENSIONS)
    ]


def scan_folder(folder_path, test_folder_path=None, label_folder_path=None):
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

    embeddings = compute_embeddings(readable_paths)
    duplicate_pairs = confirm_hash_pairs(find_duplicate_pairs(readable_paths), embeddings)
    result = {
        "duplicate_pairs": duplicate_pairs,
        "similar_pairs": find_semantic_pairs(embeddings, duplicate_pairs),
        "blurry_images": blurry_images,
        "risky_orientation_images": risky_orientation_images,
        "unreadable_images": unreadable_images,
    }

    test_paths, test_embeddings = [], {}
    if test_folder_path:
        try:
            test_paths = list_images(test_folder_path)
        except OSError as e:
            unreadable_images.append((test_folder_path, str(e)))
        test_embeddings = compute_embeddings(test_paths)
        result["leaked_pairs"] = confirm_hash_pairs(
            find_leaked_pairs(readable_paths, test_paths), {**embeddings, **test_embeddings})
        result["similar_leaks"] = find_semantic_leaks(embeddings, test_embeddings, result["leaked_pairs"])
        result["shift"] = _advisory(check_shift, None, embeddings, test_embeddings)

    # Matches made mostly of a printed template shared with other photos are set aside, not dropped.
    every_photo = readable_paths + list(test_embeddings)
    look_alikes = []
    for key, kind in (("similar_pairs", "duplicate"), ("similar_leaks", "leak")):
        if key in result:
            result[key], aside = _advisory(set_aside_templates, (result[key], []), result[key], every_photo)
            look_alikes += [(*pair, kind) for pair in aside]
    result["template_matches"] = look_alikes

    # The checks below only report; none of them removes anything.
    matched =[p for key in ("duplicate_pairs", "similar_pairs", "leaked_pairs", "similar_leaks") for p in result.get(key, [])]
    result["bursts"] = _advisory(find_bursts, [], readable_paths, test_paths, matched)
    result["overlays"] = _advisory(find_overlays, [], every_photo)
    result["ai_generated_images"], result["stock_images"] = _advisory(find_provenance_flags, ([], []), every_photo)
    if label_folder_path:
        result["label_problems"] = _advisory(
            lambda: check_labels(every_photo, list_label_files(label_folder_path)),
            [(label_folder_path, "unreadable", "the label check failed on these files")])

    return result


if __name__ == "__main__":
    result = scan_folder(sys.argv[1])
    print("Duplicate pairs:          ", result["duplicate_pairs"])
    print("Blurry images:            ", result["blurry_images"])
    print("Risky EXIF orientation:   ", result["risky_orientation_images"])
    print("Unreadable images:        ", result["unreadable_images"])
    print("Shared-template matches:  ", result["template_matches"])
    print("Same-moment groups:       ", result["bursts"])
    print("Shared overlays:          ", result["overlays"])
    print("AI-generated markers:     ", result["ai_generated_images"])
    print("Stock-agency markers:     ", result["stock_images"])
