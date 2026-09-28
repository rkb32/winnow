import os
import sys

import cv2
import numpy as np
import pytest
from PIL import Image, PngImagePlugin

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "api"))


IPTC_AI = (b'<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
           b'<rdf:Description xmlns:Iptc4xmpExt="http://iptc.org/std/Iptc4xmpExt/2008-02-29/" '
           b'Iptc4xmpExt:DigitalSourceType="http://cv.iptc.org/newscodes/digitalsourcetype/trainedAlgorithmicMedia"/>'
           b'</rdf:RDF></x:xmpmeta>')
GETTY = (b'<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
         b'<rdf:Description xmlns:photoshop="http://ns.adobe.com/photoshop/1.0/" photoshop:Credit="Getty Images"/>'
         b'</rdf:RDF></x:xmpmeta>')


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


def fixed_camera(subject_seed, shift=0, size=160):
    """A frame from a camera that never moves: the same busy background every time, with a
    subject in the middle (a different one per seed, nudged sideways by `shift` pixels)."""
    frame = scene(3, 720, 960)
    frame[280:280 + size, 400 + shift:400 + shift + size] = cv2.resize(scene(subject_seed), (size, size))
    return frame


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


def jpeg_with_capture(path, camera, taken, fraction="", seed=8):
    """A photo whose EXIF records the camera and when it was taken ("2024:05:01 10:00:07")."""
    exif = Image.Exif()
    exif[271], exif[272] = camera.split(" ", 1)
    exif.get_ifd(0x8769)[36867] = taken
    if fraction:
        exif.get_ifd(0x8769)[37521] = fraction
    Image.fromarray(scene(seed)[:, :, ::-1]).save(path, exif=exif)
    return path


def photo_with_metadata(path, seed=8, text=None, xmp=None):
    """A photo carrying a PNG text chunk (`text` = (key, value)) or a JPEG XMP packet."""
    image = Image.fromarray(scene(seed)[:, :, ::-1])
    if text:
        info = PngImagePlugin.PngInfo()
        info.add_text(*text)
        image.save(path, pnginfo=info)
    elif xmp:
        image.save(path, xmp=xmp)
    else:
        image.save(path)
    return path


def stamp(image, alpha=0.5):
    """A semi-transparent white label in the bottom-right corner, like a stock-photo watermark."""
    height, width = image.shape[:2]
    mask = np.zeros((height, width), np.uint8)
    scale = width / 300
    cv2.putText(mask, "STOCKPHOTO", (width - int(width * 0.42), height - height // 12),
                cv2.FONT_HERSHEY_SIMPLEX, scale, 255, max(1, round(scale * 2)))
    weight = (mask / 255.0 * alpha)[..., None]
    return (image * (1 - weight) + 255 * weight).astype(np.uint8)


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
