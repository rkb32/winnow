import os

import cv2

from conftest import blur, jpeg_with_orientation, names, product_shot, recompress, scene
from duplicates import find_duplicate_pairs
from scan import scan_folder


def test_scan_folder_reports_every_problem_type(save, tmp_path):
    image = scene(20)
    save("photo.png", image, "train")
    save("photo_copy.png", image, "train")
    save("photo_mirrored.png", cv2.flip(image, 1), "train")
    save("soft.png", blur(scene(21)), "train")
    jpeg_with_orientation(str(tmp_path / "train" / "rotated.jpg"), 6, seed=22)
    (tmp_path / "train" / "broken.jpg").write_bytes(b"junk")
    save("leaked.png", image, "test")

    result = scan_folder(str(tmp_path / "train"), test_folder_path=str(tmp_path / "test"))

    assert names(result["duplicate_pairs"]) == {frozenset({"photo.png", "photo_copy.png"})}
    assert all("photo_mirrored.png" in pair for pair in names(result["similar_pairs"]))
    assert result["similar_pairs"]
    assert [os.path.basename(p) for p, _ in result["blurry_images"]] == ["soft.png"]
    assert [(os.path.basename(p), o, s) for p, o, s in result["risky_orientation_images"]] == [("rotated.jpg", 6, True)]
    assert [os.path.basename(p) for p, _ in result["unreadable_images"]] == ["broken.jpg"]
    assert {frozenset({"photo.png", "leaked.png"}), frozenset({"photo_copy.png", "leaked.png"})} <= names(result["leaked_pairs"])


def test_hash_collision_between_different_photos_is_not_reported(save, tmp_path):
    paths = [save("checkered.png", product_shot(True), "train"), save("plain.png", product_shot(False), "train")]
    assert names(find_duplicate_pairs(paths)), "fixture should fool the raw hash"
    assert scan_folder(str(tmp_path / "train"))["duplicate_pairs"] == []


def test_recompressed_copy_still_counts_after_confirmation(save, tmp_path):
    image = scene(24)
    save("original.png", image, "train")
    save("resaved.jpg", recompress(image), "train")
    assert names(scan_folder(str(tmp_path / "train"))["duplicate_pairs"]) == {frozenset({"original.png", "resaved.jpg"})}


def test_scan_without_test_folder_has_no_leak_keys(save, tmp_path):
    save("a.png", scene(23), "train")
    result = scan_folder(str(tmp_path / "train"))
    assert "leaked_pairs" not in result and "similar_leaks" not in result
