import platform

print("Chip type inside this container:", platform.machine())

import cv2
print("OpenCV version:", cv2.__version__)

from cv2 import img_hash
hasher = img_hash.BlockMeanHash_create()
print("img_hash module loaded OK:", hasher is not None)

import numpy as np
fake_image = np.zeros((100, 100, 3), dtype=np.uint8)
result = hasher.compute(fake_image)
print("Hash computed OK, shape:", result.shape)

print("\nRESULT: PASS")
