import numpy as np
from skimage.color import rgb2hsv
from scipy.stats import skew

def get_color_data(image, mask = None):
    '''
    Return a dict of color features for an image. For each channel (Hue, Saturaion, Value) return its mean, S
    
    Input: 
        image - already loaded image
        mask - already loaded mask

    Output:
        dictionary of color features

    Notes:
        Clinical color rules or how to interpret the dictionary:
            TO-DO 
    '''
    # convert to RGB from RGBA, then to HSV
    if image.shape[-1] == 4:
        image = image[:, :, :3]
    image = rgb2hsv(image)

    if mask is not None:
        image = image[mask > 0]
    else:
        # reshape to 1D array for stats
        image = image.reshape(-1, 3)

    color_data = {}
    channel_names = ['hue', 'saturation', 'value']
    
    for ch_idx, ch_name in enumerate(channel_names):
        data = image[:, ch_idx]
        color_data[f'{ch_name}_mean'] = np.mean(data)
        color_data[f'{ch_name}_std'] = np.std(data)
        color_data[f'{ch_name}_skew'] = skew(data)
        color_data[f'{ch_name}_5p'] = np.percentile(data, 5)
        color_data[f'{ch_name}_50p'] = np.percentile(data, 50)
        color_data[f'{ch_name}_95p'] = np.percentile(data, 95)
    
    return color_data

def get_rgb_data(image, mask = None):
    '''
    Return a dict of RGB and greyscale features for an image. For each channel (Red, Green, Blue, Greyscale) return its mean, standard deviation, skew and 5, 50, 95 percentiles.

    Input:
        image - already loaded image
        mask - already loaded mask

    Output:
        dictionary of RGB and greyscale features
    '''
    if image.shape[-1] == 4:
        image = image[:, :, :3]

    if mask is not None:
        pixels = image[mask > 0]
    else:
        pixels = image.reshape(-1, 3)

    rgb_data = {}
    channel_names = ['red', 'green', 'blue']

    for ch_idx, ch_name in enumerate(channel_names):
        data = pixels[:, ch_idx].astype(float)
        rgb_data[f'{ch_name}_mean'] = np.mean(data)
        rgb_data[f'{ch_name}_std'] = np.std(data)
        rgb_data[f'{ch_name}_skew'] = skew(data)
        rgb_data[f'{ch_name}_5p'] = np.percentile(data, 5)
        rgb_data[f'{ch_name}_50p'] = np.percentile(data, 50)
        rgb_data[f'{ch_name}_95p'] = np.percentile(data, 95)

    grey = (0.2989 * pixels[:, 0] + 0.5870 * pixels[:, 1] + 0.1140 * pixels[:, 2]).astype(float)
    rgb_data['grey_mean'] = np.mean(grey)
    rgb_data['grey_std'] = np.std(grey)
    rgb_data['grey_skew'] = skew(grey)
    rgb_data['grey_5p'] = np.percentile(grey, 5)
    rgb_data['grey_50p'] = np.percentile(grey, 50)
    rgb_data['grey_95p'] = np.percentile(grey, 95)

    return rgb_data