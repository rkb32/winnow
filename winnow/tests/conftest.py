import os
import sys

import cv2
import numpy as np
import pytest
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "api"))


def scene(seed, height=360, width=480):
    """A deterministic 'photo': gradient sky plus colored shapes, with real edges and corners."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1, height)[:, None, None]
    top, bottom = rng.integers(0, 256, 3), rng.integers(0, 256, 3)
    image = np.ascontiguousarray(np.broadcast_to(top * (1 - t) + bottom * t, (height, width, 3)).astype(np.uint8))
    for _ in range(120):
        color = tuple(int(c) for c in rng.integers(0, 256, 3))
        x, y, size = int(rng.integers(0, width)), int(rng.integers(0, height)), int(rng.integers(6, 45))
        if rng.random() < 0.5:
            cv2.rectangle(image, (x, y), (x + size, y + int(size * rng.uniform(0.4, 1.4))), color, -1)
        else:
            cv2.circle(image, (x, y), size // 2, color, -1)
    return image


def crop(image):
    h, w = image.shape[:2]
    return cv2.resize(image[h // 8: h - h // 8, w // 8: w - w // 8], (w, h))


def blur(image):
    return cv2.GaussianBlur(image, (0, 0), 6)


def product_shot(textured):
    """A dark product on white: textured and flat versions have the same block means, so the
    perceptual hash sees them as identical (distance 0), like the collisions found in Imagenette."""
    image = np.full((360, 480, 3), 255, np.uint8)
    if textured:
        yy, xx = np.mgrid[0:200, 0:240]
        image[80:280, 120:360] = np.where(((yy // 8 + xx // 8) % 2)[..., None] == 0, 0, 128).astype(np.uint8)
    else:
        image[80:280, 120:360] = 64
    return image


def recompress(image, scale=0.5, quality=60):
    small = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return cv2.imdecode(cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, quality])[1], cv2.IMREAD_COLOR)


def jpeg_with_orientation(path, orientation, seed=8):
    exif = Image.Exif()
    exif[274] = orientation
    Image.fromarray(scene(seed)[:, :, ::-1]).save(path, exif=exif)
    return path


@pytest.fixture
def save(tmp_path):
    def _save(name, image, folder=None):
        directory = tmp_path / folder if folder else tmp_path
        directory.mkdir(exist_ok=True)
        path = str(directory / name)
        cv2.imwrite(path, image)
        return path
    return _save


def names(pairs):
    return {frozenset(os.path.basename(p) for p in pair[:2]) for pair in pairs}
