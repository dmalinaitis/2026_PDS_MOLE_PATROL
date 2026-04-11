# Notes for myself
# Dataset has 6 types of skin lesions:
#  Malignant:
# - BCC: color - pearly, white, skin-colored, or pink; type - ?; edge - ?
# - SCC: color - brown, black, white, gray, skin-colored; type - ?; edge - ?
# - MEL: color - multiple shades of black, brown, or tan; type - uneven; edge - ?
#
#  Benign:
# - NEV: color - pink, tan, or flesh-toned to shades of brown or black; type - uniform; edge - distinct
# - ACK: color - pink and red to brown, tan, yellow, gray, skin-colored; type - ?; edge - ?
# - SEK: color - light tan, yellow, and grey to dark brown or black; type - ?; edge - ?
#
# Skin lesions have preferences for some regions of the body!
# Regions covered in dataset: face, scalp, nose, lips, ears, neck, chest, abdomen, back, arm, forearm, hand, thigh, shin, and foot.

import numpy as np
from skimage import io

path_mask = "../data/temp/masks/"
path_images = "../data/temp/imgs/"
image_id = 'PAT_20_30_44.png'
mask_id = image_id.replace('.png','_mask.png')

image_to_load = path_images + image_id
mask_to_load = path_mask + mask_id

image = io.imread(image_to_load) #ndarray
mask = io.imread(mask_to_load) # ndarray

def rgb_stats(image, mask):
    