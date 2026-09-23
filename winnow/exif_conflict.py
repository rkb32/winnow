from PIL import Image

_ORIENTATION_TAG = 274  # the standard EXIF tag ID for "Orientation"
_ROTATES_90_DEGREES = {5, 6, 7, 8}


def get_orientation(image_path):
    image = Image.open(image_path)
    exif = image.getexif()
    return exif.get(_ORIENTATION_TAG, 1)


def has_risky_orientation(orientation):
    return orientation != 1


def swaps_dimensions(orientation):
    return orientation in _ROTATES_90_DEGREES
