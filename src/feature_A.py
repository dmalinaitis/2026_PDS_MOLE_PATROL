"""Feature A - asymmetry: how much the lesion mask differs from its own mirror image."""
import numpy as np
from skimage.transform import rotate


def midpoint(mask):
    """x-coordinate of the vertical line that splits the lesion area in half."""
    column_sums = np.sum(mask, axis=0)
    half = np.sum(column_sums) / 2
    for x, area_so_far in enumerate(np.add.accumulate(column_sums)):
        if area_so_far > half:
            return x
    return 0  # empty mask


def crop(mask):
    """Crop the mask to the lesion, symmetrically around its midpoint so it can be mirrored."""
    mid = midpoint(mask)
    y_nonzero, x_nonzero = np.nonzero(mask)
    if len(y_nonzero) == 0:  # empty mask
        return mask

    y_lims = [np.min(y_nonzero), np.max(y_nonzero)]
    # Same distance to the left and right of the midpoint, clipped to the image borders
    x_dist = max(np.abs(np.array([np.min(x_nonzero), np.max(x_nonzero)]) - mid))
    x_lims = [max(0, int(mid - x_dist)), min(mask.shape[1], int(mid + x_dist))]
    return mask[y_lims[0]:y_lims[1], x_lims[0]:x_lims[1]]


def get_asymmetry(mask):
    """Asymmetry score from 0 (perfectly symmetric) to 1.

    The cropped mask is compared with its left-right mirror: XOR counts the pixels that do
    not overlap, normalised by twice the lesion area. This is repeated for 6 rotations of
    30 degrees and the scores are averaged.
    """
    mask = (mask > 0).astype(np.uint8)
    scores = []
    for _ in range(6):
        segment = crop(mask)
        area = np.sum(segment)
        if area == 0:
            scores.append(0)
        else:
            scores.append(np.sum(np.logical_xor(segment, np.fliplr(segment))) / (2 * area))
        # order=0 (nearest neighbour) keeps the rotated mask binary
        mask = rotate(mask, 30, order=0, preserve_range=True).astype(np.uint8)
    return sum(scores) / len(scores)
