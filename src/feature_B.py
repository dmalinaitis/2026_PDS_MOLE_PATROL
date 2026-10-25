"""Feature B - border: compactness of the lesion mask (Polsby-Popper measure)."""
from math import pi

import numpy as np
from skimage import morphology


def get_compactness(mask):
    """Compactness 4*pi*area / perimeter^2, rounded to 4 decimals.

    Close to 1 for a smooth, round lesion and close to 0 for an irregular or spiky border.
    The perimeter is the band of pixels removed when the mask is eroded with a disk of radius 2.
    """
    mask = (mask > 0).astype(np.uint8)
    area = np.sum(mask)
    if area == 0:
        return 0.0

    perimeter = np.sum(mask - morphology.binary_erosion(mask, morphology.disk(2)))
    if perimeter == 0:
        return 0.0
    return round((4 * pi * area) / (perimeter ** 2), 4)
