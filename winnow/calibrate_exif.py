import numpy as np
from PIL import Image

from exif_conflict import get_orientation, has_risky_orientation, swaps_dimensions

img_array = np.zeros((100, 150, 3), dtype=np.uint8)
img_array[:, :, 0] = 255
pil_image = Image.fromarray(img_array)

pil_image.save("samples/normal.jpg")

exif = pil_image.getexif()
exif[274] = 6
pil_image.save("samples/rotated_tag.jpg", exif=exif)

for path in ["samples/normal.jpg", "samples/rotated_tag.jpg"]:
    orientation = get_orientation(path)
    print(
        path,
        "-> orientation:", orientation,
        "risky:", has_risky_orientation(orientation),
        "swaps dimensions:", swaps_dimensions(orientation),
    )
