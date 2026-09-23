import cv2


def compute_sharpness(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    return cv2.Laplacian(image, cv2.CV_64F).var()


def is_too_blurry(sharpness):
    if sharpness <= 200:
        return True
    else:
        return False
