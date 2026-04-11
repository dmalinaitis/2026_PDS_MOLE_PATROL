import numpy as np
from math import pi
from skimage import morphology
from scipy.spatial import ConvexHull

def get_compactness(mask):
    """
    Calculates the compactness based on the Polsby-Popper measure.
    A score closer to 1 indicates a perfect circle (smooth border).
    A score closer to 0 indicates a highly irregular or spiky border.
    """
    # 1. Ensure the mask is strictly binary (1s and 0s) to avoid math scaling errors
    mask_binary = (mask > 0).astype(np.uint8)
    
    A = np.sum(mask_binary)
    if A == 0:
        return 0.0 # Safety check for empty masks

    # 2. Use a "brush" (structural element) to erode the mask's edges
    struct_el = morphology.disk(2)
    mask_eroded = morphology.binary_erosion(mask_binary, struct_el)
    
    # 3. The perimeter is the difference between original and eroded masks
    perimeter_mask = mask_binary - mask_eroded
    l = np.sum(perimeter_mask)

    if l == 0:
        return 0.0 # Prevent division by zero

    # 4. Polsby-Popper Formula
    compactness = (4 * pi * A) / (l ** 2)
    
    return round(compactness, 4)