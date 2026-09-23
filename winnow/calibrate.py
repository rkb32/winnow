import os

import cv2
import numpy as np

from duplicates import compute_hash, hash_distance

os.makedirs("samples", exist_ok=True)

img_a = np.zeros((200, 200, 3), dtype=np.uint8)
cv2.rectangle(img_a, (50, 50), (150, 150), (0, 0, 255), -1)
cv2.imwrite("samples/a_original.png", img_a)

noise = np.random.randint(0, 10, img_a.shape, dtype=np.uint8)
img_b = cv2.add(img_a, noise)
cv2.imwrite("samples/b_near_duplicate.png", img_b)

img_c = np.zeros((200, 200, 3), dtype=np.uint8)
cv2.circle(img_c, (100, 100), 60, (255, 0, 0), -1)
cv2.imwrite("samples/c_different.png", img_c)

hash_a = compute_hash("samples/a_original.png")
hash_b = compute_hash("samples/b_near_duplicate.png")
hash_c = compute_hash("samples/c_different.png")

print("A vs B (near-duplicate, tiny noise added):", hash_distance(hash_a, hash_b))
print("A vs C (a completely different image):    ", hash_distance(hash_a, hash_c))
print("B vs C (a completely different image):    ", hash_distance(hash_b, hash_c))
