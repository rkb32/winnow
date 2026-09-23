"""The OpenCV tools the review agent looks through: a thumbnail of a whole photo, and a
full-resolution crop of any region it asks to zoom into.

Boxes are [x1, y1, x2, y2] fractions of the photo's width and height, so they mean the same
thing whatever size the model saw the photo at.
"""
import cv2

THUMB_SIDE = 384
ZOOM_SIDE = 512
MIN_BOX = 0.05


def _read(path):
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    return image


def _encode(image, max_side):
    scale = max_side / max(image.shape[:2])
    if scale < 1:
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes()


def _widen(lo, hi):
    if hi - lo >= MIN_BOX:
        return lo, hi
    lo = min(max((lo + hi - MIN_BOX) / 2, 0.0), 1.0 - MIN_BOX)
    return lo, lo + MIN_BOX


def clamp_box(box):
    """A box the model asked for, made valid: inside the photo, in order, and not a sliver."""
    if len(box) != 4:
        raise ValueError(f"box needs 4 numbers, got {len(box)}")
    x1, y1, x2, y2 = (min(max(float(v), 0.0), 1.0) for v in box)
    x1, x2 = _widen(*sorted((x1, x2)))
    y1, y2 = _widen(*sorted((y1, y2)))
    return [x1, y1, x2, y2]


def thumbnail(path):
    return _encode(_read(path), THUMB_SIDE)


def crop(path, box):
    """The region as JPEG bytes, plus the whole photo's (width, height)."""
    image = _read(path)
    height, width = image.shape[:2]
    x1, y1, x2, y2 = clamp_box(box)
    left, top = int(x1 * width), int(y1 * height)
    region = image[top:max(int(y2 * height), top + 1), left:max(int(x2 * width), left + 1)]
    return _encode(region, ZOOM_SIDE), (width, height)
