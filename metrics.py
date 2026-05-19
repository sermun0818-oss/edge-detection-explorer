import numpy as np


def edge_pixel_count(edge_image):
    return np.sum(edge_image > 0)