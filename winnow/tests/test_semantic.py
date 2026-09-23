import cv2

from conftest import crop, names, scene
from semantic import (MIN_INLIERS, compute_embeddings, find_semantic_leaks, find_semantic_pairs,
                      keypoint_inliers, match_regions)


def test_mirrored_copy_is_found_and_different_photo_is_not(save):
    image = scene(10)
    paths = [save("a.png", image), save("mirror.png", cv2.flip(image, 1)), save("other.png", scene(11))]
    pairs = find_semantic_pairs(compute_embeddings(paths), already_found=[])
    assert names(pairs) == {frozenset({"a.png", "mirror.png"})}


def test_cropped_test_photo_is_a_leak(save):
    image = scene(12)
    train = [save("a.png", image, "train"), save("b.png", scene(13), "train")]
    test = [save("zoomed.png", crop(image), "test")]
    leaks = find_semantic_leaks(compute_embeddings(train), compute_embeddings(test), already_found=[])
    assert names(leaks) == {frozenset({"a.png", "zoomed.png"})}
    assert leaks[0][3] >= MIN_INLIERS


def test_pairs_the_hash_already_found_are_not_repeated(save):
    image = scene(14)
    a, b = save("a.png", image), save("b.png", image)
    assert find_semantic_pairs(compute_embeddings([a, b]), already_found=[(a, b, 0)]) == []


def test_different_photos_fail_the_keypoint_check(save):
    assert keypoint_inliers(save("a.png", scene(15)), save("b.png", scene(16))) < MIN_INLIERS


def test_match_regions_find_a_crop_inside_its_original(save):
    image = scene(18)
    box_a, box_b = match_regions(save("a.png", image), save("zoomed.png", crop(image)))
    # crop() keeps the middle 3/4 of the photo (1/8 off each side) and scales it back up.
    assert 0.1 <= box_a[0] and box_a[2] <= 0.9 and 0.1 <= box_a[1] and box_a[3] <= 0.9
    assert box_b[2] - box_b[0] > 0.6 and box_b[3] - box_b[1] > 0.6


def test_match_regions_undo_the_mirror(save):
    image = scene(19)
    left_half_mirrored = cv2.flip(image[:, :240], 1)
    box_a, _ = match_regions(save("a.png", image), save("b.png", left_half_mirrored))
    assert box_a[2] <= 0.55


def test_unreadable_files_are_skipped(save, tmp_path):
    bad = tmp_path / "bad.png"
    bad.write_bytes(b"junk")
    assert list(compute_embeddings([str(bad), save("ok.png", scene(17))])) == [save("ok.png", scene(17))]
