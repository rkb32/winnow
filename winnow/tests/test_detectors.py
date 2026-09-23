import pytest

from blur import compute_sharpness, is_too_blurry
from conftest import blur, jpeg_with_orientation, names, recompress, scene
from duplicates import compute_hash, find_duplicate_pairs, find_leaked_pairs, hash_distance
from exif_conflict import get_orientation, has_risky_orientation, swaps_dimensions


def test_sharp_photo_is_not_blurry(save):
    assert not is_too_blurry(compute_sharpness(save("sharp.png", scene(1))))


def test_blurred_photo_is_blurry(save):
    assert is_too_blurry(compute_sharpness(save("soft.png", blur(scene(1)))))


def test_unreadable_file_raises(tmp_path):
    bad = tmp_path / "bad.jpg"
    bad.write_bytes(b"not an image")
    with pytest.raises(ValueError):
        compute_sharpness(str(bad))


def test_exact_copy_has_zero_hash_distance(save):
    image = scene(2)
    assert hash_distance(compute_hash(save("a.png", image)), compute_hash(save("b.png", image))) == 0


def test_resized_and_recompressed_copy_is_still_a_duplicate(save):
    image = scene(2)
    assert hash_distance(compute_hash(save("a.png", image)), compute_hash(save("b.jpg", recompress(image)))) <= 5


def test_different_photos_are_not_duplicates(save):
    assert find_duplicate_pairs([save("a.png", scene(3)), save("b.png", scene(4))]) == []


def test_duplicate_search_skips_unreadable_files(save, tmp_path):
    bad = tmp_path / "bad.png"
    bad.write_bytes(b"junk")
    image = scene(5)
    pairs = find_duplicate_pairs([save("a.png", image), str(bad), save("b.png", image)])
    assert names(pairs) == {frozenset({"a.png", "b.png"})}


def test_leak_search_only_pairs_train_with_test(save):
    image = scene(6)
    leaks = find_leaked_pairs(
        [save("train.png", image, "train")],
        [save("leaked.png", image, "test"), save("fine.png", scene(7), "test")],
    )
    assert names(leaks) == {frozenset({"train.png", "leaked.png"})}


@pytest.mark.parametrize("orientation, risky, swaps", [(1, False, False), (3, True, False), (6, True, True), (8, True, True)])
def test_exif_orientation(tmp_path, orientation, risky, swaps):
    path = jpeg_with_orientation(str(tmp_path / "photo.jpg"), orientation)
    assert get_orientation(path) == orientation
    assert has_risky_orientation(orientation) is risky
    assert swaps_dimensions(orientation) is swaps


def test_png_without_exif_counts_as_normal(save):
    assert get_orientation(save("plain.png", scene(9))) == 1
