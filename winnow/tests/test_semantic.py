import cv2
import numpy as np

from conftest import crop, fixed_camera, names, scene
from semantic import (MIN_DISTINCT_INLIERS, MIN_INLIERS, MIN_PHOTOS_FOR_TEMPLATES, compute_embeddings,
                      confirm_hash_pairs, find_semantic_leaks, find_semantic_pairs, keypoint_inliers,
                      match_regions, set_aside_templates)


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


def test_fixed_camera_frames_with_different_subjects_are_not_copies(save):
    # The shared background alone gives 300+ inliers and near-identical embeddings.
    a, b = save("a.png", fixed_camera(50)), save("b.png", fixed_camera(51))
    embeddings = compute_embeddings([a, b])
    assert find_semantic_pairs(embeddings, already_found=[]) == []
    assert confirm_hash_pairs([(a, b, 0)], embeddings) == []


def test_fixed_camera_frames_of_the_same_subject_still_match(save):
    a, b = save("a.png", fixed_camera(50)), save("b.png", fixed_camera(50, shift=20))
    assert names(find_semantic_pairs(compute_embeddings([a, b]), already_found=[])) == {frozenset({"a.png", "b.png"})}


def test_same_size_recompressed_copy_is_still_a_copy(save):
    image = scene(20)
    jpeg = cv2.imdecode(cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 30])[1], cv2.IMREAD_COLOR)
    assert keypoint_inliers(save("a.png", image), save("b.png", jpeg)) >= MIN_INLIERS


def test_recolored_copy_is_still_a_copy(save):
    # Every pixel changed, so there's no fixed background to set aside.
    image = scene(21)
    brighter = cv2.add(image, np.full_like(image, 40))
    assert keypoint_inliers(save("a.png", image), save("b.png", brighter)) >= MIN_INLIERS


def test_unreadable_files_are_skipped(save, tmp_path):
    bad = tmp_path / "bad.png"
    bad.write_bytes(b"junk")
    assert list(compute_embeddings([str(bad), save("ok.png", scene(17))])) == [save("ok.png", scene(17))]


# --- shared printed templates ---

TEMPLATE = scene(999, 200, 150)      # a printed frame every photo carries, like the frame of a trading card


def templated_photo(seed):
    """A photo of its own, with the shared template pasted somewhere different each time."""
    image = scene(seed)
    x, y = 20 + (seed * 37) % 300, 20 + (seed * 53) % 130
    image[y:y + 200, x:x + 150] = TEMPLATE
    return image


def templated_set(save, count=MIN_PHOTOS_FOR_TEMPLATES + 5):
    return [save(f"p{i}.png", templated_photo(i + 1)) for i in range(count)]


def test_photos_that_only_share_a_template_are_set_aside(save):
    paths = templated_set(save)
    n = keypoint_inliers(paths[0], paths[1])
    assert n >= MIN_INLIERS      # the template alone is enough to look like a copy
    kept, look_alikes = set_aside_templates([(paths[0], paths[1], 0.9, n)], paths)
    assert kept == []
    (a, b, similarity, inliers, distinct), = look_alikes
    assert (a, b, similarity, inliers) == (paths[0], paths[1], 0.9, n)
    assert distinct < MIN_DISTINCT_INLIERS


def test_a_real_copy_that_carries_the_template_is_kept(save):
    paths = templated_set(save)
    copy = save("copy.png", crop(templated_photo(1)))
    n = keypoint_inliers(paths[0], copy)
    kept, look_alikes = set_aside_templates([(paths[0], copy, 0.95, n)], paths + [copy])
    assert kept == [(paths[0], copy, 0.95, n)] and look_alikes == []


def test_small_sets_are_left_alone(save):
    paths = templated_set(save, count=MIN_PHOTOS_FOR_TEMPLATES - 1)
    pair = (paths[0], paths[1], 0.9, keypoint_inliers(paths[0], paths[1]))
    assert set_aside_templates([pair], paths) == ([pair], [])


def test_the_same_pair_gets_the_same_answer_every_time(save):
    paths = templated_set(save)
    pair = (paths[2], paths[3], 0.9, keypoint_inliers(paths[2], paths[3]))
    assert set_aside_templates([pair], paths) == set_aside_templates([pair], list(reversed(paths)))
