import cv2
import numpy as np

from blur import compute_sharpness

checkerboard = np.zeros((200, 200), dtype=np.uint8)
checkerboard[::20, :] = 255
checkerboard[:, ::20] = 255
cv2.imwrite("samples/sharp_checkerboard.png", checkerboard)

blurry = cv2.GaussianBlur(checkerboard, (15, 15), 0)
cv2.imwrite("samples/blurry_checkerboard.png", blurry)

print("Sharp image score: ", compute_sharpness("samples/sharp_checkerboard.png"))
print("Blurry image score:", compute_sharpness("samples/blurry_checkerboard.png"))
