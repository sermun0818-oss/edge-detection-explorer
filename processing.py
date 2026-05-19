import cv2
import numpy as np


def apply_canny(image, threshold1, threshold2):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    edges = cv2.Canny(gray, threshold1, threshold2)

    return edges


def apply_sobel(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

    magnitude = np.sqrt(sobelx**2 + sobely**2)

    magnitude = np.uint8(
        255 * magnitude / np.max(magnitude)
    )

    return magnitude


def apply_laplacian(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    lap = cv2.Laplacian(gray, cv2.CV_64F)

    lap = np.uint8(
        255 * np.abs(lap) / np.max(np.abs(lap))
    )

    return lap