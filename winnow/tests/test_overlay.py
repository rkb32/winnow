import os

import cv2

from conftest import scene, stamp
from overlay import find_overlays


def photos(save, count, stamped, height=360, width=480, first_seed=100):
    """`count` different scenes; the first `stamped` carry the same watermark."""
    paths = []
    for i in range(count):
        image = scene(first_seed + i, height, width)
        paths.append(save(f"p{i}.png", stamp(image) if i < stamped else image, "set"))
    return paths


def test_a_watermark_on_every_photo_is_found_where_it_sits(save):
    [group] = find_overlays(photos(save, 10, 10))
    x1, y1, x2, y2 = group["box"]
    assert len(group["files"]) == 10
    assert (x1 + x2) / 2 > 0.5 and (y1 + y2) / 2 > 0.75, "bottom-right"


def test_photos_with_nothing_in_common_report_nothing(save):
    assert find_overlays(photos(save, 10, 0)) == []


def test_a_watermark_on_most_photos_names_only_the_photos_that_carry_it(save):
    paths = photos(save, 10, 8)
    [group] = find_overlays(paths)
    assert sorted(os.path.basename(f) for f in group["files"]) == sorted(f"p{i}.png" for i in range(8))


def test_too_few_photos_of_one_shape_are_not_compared(save):
    assert find_overlays(photos(save, 7, 7)) == []


def test_a_small_group_needs_seven_carrying_photos_even_though_six_is_most_of_it(save):
    assert find_overlays(photos(save, 8, 6)) == []
    assert len(find_overlays(photos(save, 8, 8, first_seed=300))[0]["files"]) == 8


def test_photos_of_different_shapes_are_not_pooled(save):
    landscape = photos(save, 5, 5)
    portrait = [save(f"tall{i}.png", stamp(scene(200 + i, 480, 360)), "set") for i in range(5)]
    assert find_overlays(landscape + portrait) == []


def test_unreadable_files_are_skipped(save, tmp_path):
    paths = photos(save, 10, 10)
    (tmp_path / "set" / "junk.png").write_bytes(b"junk")
    assert len(find_overlays(paths + [str(tmp_path / "set" / "junk.png")])[0]["files"]) == 10
