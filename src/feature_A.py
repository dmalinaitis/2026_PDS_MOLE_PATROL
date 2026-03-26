import numpy as np
from skimage.transform import rotate

def midpointGroup4(mask):
    """
    Finds the vertical dividing line (x-coordinate) that splits the 
    mask's area exactly in half.
    """
    summed = np.sum(mask, axis=0)
    half_sum = np.sum(summed) / 2
    for i, n in enumerate(np.add.accumulate(summed)):
        if n > half_sum:
            return i
    return 0 # in case of an empty mask

def crop(mask):
    """
    Crops the mask symmetrically around the calculated midpoint.
    """
    mid = midpointGroup4(mask)
    y_nonzero, x_nonzero = np.nonzero(mask)
    
    # Check if mask is empty to avoid errors
    if len(y_nonzero) == 0 or len(x_nonzero) == 0:
        return mask 
        
    y_lims = [np.min(y_nonzero), np.max(y_nonzero)]
    x_lims = np.array([np.min(x_nonzero), np.max(x_nonzero)])
    
    # Force the x-limits to be perfectly symmetrical around the midpoint
    x_dist = max(np.abs(x_lims - mid))
    x_lims = [int(mid - x_dist), int(mid + x_dist)]
    
    # Handle edge cases where symmetric crop goes outside image bounds
    x_lims[0] = max(0, x_lims[0])
    x_lims[1] = min(mask.shape[1], x_lims[1])
    
    return mask[y_lims[0]:y_lims[1], x_lims[0]:x_lims[1]]

def get_asymmetry(mask):
    """
    Calculates the asymmetry score by comparing the mask to its flipped version
    across 6 different rotation angles.
    """
    scores = []
    
    # Loop 6 times, rotating 30 degrees each time (180 degrees total)
    for _ in range(6):
        segment = crop(mask)
        area = np.sum(segment)
        
        # Prevent division by zero if the cropped segment is empty
        if area == 0:
            scores.append(0)
        else:
            # Calculate XOR sum and divide by total area
            xor_sum = np.sum(np.logical_xor(segment, np.flip(segment)))
            scores.append(xor_sum / area)
            
        mask = rotate(mask, 30)
        
    return sum(scores) / len(scores)