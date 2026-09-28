import os

import cv2

from conftest import IPTC_AI, blur, jpeg_with_capture, jpeg_with_orientation, names, photo_with_metadata, product_shot, recompress, scene
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


def test_scan_folder_reports_the_advisory_checks(tmp_path):
    for folder in ("train", "test", "labels"):
        (tmp_path / folder).mkdir()
    jpeg_with_capture(str(tmp_path / "train" / "a.jpg"), "Canon R5", "2024:05:01 10:00:00", seed=30)
    jpeg_with_capture(str(tmp_path / "test" / "b.jpg"), "Canon R5", "2024:05:01 10:00:02", seed=31)
    photo_with_metadata(str(tmp_path / "train" / "generated.jpg"), seed=32, xmp=IPTC_AI)
    (tmp_path / "labels" / "a.txt").write_text("0 0.5 0.5 0.4 0.4\n0 0.95 0.5 0.4 0.4\n")

    result = scan_folder(str(tmp_path / "train"), str(tmp_path / "test"), str(tmp_path / "labels"))

    assert [os.path.basename(f) for f in result["bursts"][0]["files"]] == ["a.jpg", "b.jpg"]
    assert [os.path.basename(f) for f in result["bursts"][0]["test_files"]] == ["b.jpg"]
    assert [os.path.basename(p) for p, _ in result["ai_generated_images"]] == ["generated.jpg"]
    assert result["stock_images"] == [] and result["overlays"] == []
    assert result["shift"] == {"checked": False, "train_count": 2, "test_count": 1}
    kinds = {(os.path.basename(p), k) for p, k, _ in result["label_problems"]}
    assert ("a.jpg", "outside") in kinds and ("generated.jpg", "missing") in kinds


def test_scan_without_labels_or_test_folder_has_neither_key(save, tmp_path):
    save("a.png", scene(23), "train")
    result = scan_folder(str(tmp_path / "train"))
    assert "label_problems" not in result and "shift" not in result
    assert result["bursts"] == [] and result["ai_generated_images"] == []


def test_a_failing_advisory_check_does_not_lose_the_rest_of_the_scan(save, tmp_path, monkeypatch):
    import scan

    def broken(*args):
        raise RuntimeError("boom")

    for name in ("find_bursts", "find_overlays", "find_provenance_flags"):
        monkeypatch.setattr(scan, name, broken)
    save("a.png", scene(23), "train")
    result = scan.scan_folder(str(tmp_path / "train"))
    assert result["bursts"] == [] and result["overlays"] == []
    assert result["ai_generated_images"] == [] and result["stock_images"] == []
    assert result["blurry_images"] == [] and "duplicate_pairs" in result
