import cv2
import numpy as np
import pytest

from zoom import MIN_BOX, THUMB_SIDE, ZOOM_SIDE, clamp_box, crop, thumbnail


def quadrants(height=400, width=600):
    image = np.zeros((height, width, 3), np.uint8)
    image[:height // 2, :width // 2] = (255, 0, 0)
    image[:height // 2, width // 2:] = (0, 255, 0)
    image[height // 2:, :width // 2] = (0, 0, 255)
    image[height // 2:, width // 2:] = (255, 255, 255)
    return image


def decode(data):
    return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)


def test_crop_returns_exactly_the_requested_region(save):
    data, size = crop(save("q.png", quadrants()), [0.5, 0.0, 1.0, 0.5])
    region = decode(data)
    assert size == (600, 400) and region.shape[:2] == (200, 300)
    assert np.allclose(region.reshape(-1, 3).mean(axis=0), (0, 255, 0), atol=8)


def test_big_photos_are_downscaled_for_the_model(save):
    path = save("big.png", quadrants(2000, 3000))
    assert max(decode(crop(path, [0, 0, 1, 1])[0]).shape[:2]) == ZOOM_SIDE
    assert max(decode(thumbnail(path)).shape[:2]) == THUMB_SIDE


def test_boxes_are_clamped_ordered_and_never_a_sliver():
    assert clamp_box([1.2, -0.5, 0.5, 0.6]) == [0.5, 0.0, 1.0, 0.6]
    x1, y1, x2, y2 = clamp_box([0.999, 0.5, 0.999, 0.5])
    assert x2 == 1.0 and x2 - x1 == pytest.approx(MIN_BOX) and y2 - y1 == pytest.approx(MIN_BOX)


@pytest.mark.parametrize("box", [[0, 0, 1], "0,0,1,1", [0, 0, "x", 1], None])
def test_malformed_boxes_are_rejected(box):
    with pytest.raises((ValueError, TypeError)):
        clamp_box(box)
