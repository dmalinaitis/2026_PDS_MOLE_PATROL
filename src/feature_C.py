"""Feature C - colour: summary statistics of the pixels inside the lesion mask."""
import numpy as np
from scipy.stats import skew
from skimage.color import rgb2hsv


def channel_stats(name, values):
    """Mean, standard deviation, skew and 5th/50th/95th percentile of one colour channel."""
    return {
        f"{name}_mean": np.mean(values),
        f"{name}_std": np.std(values),
        f"{name}_skew": skew(values),
        f"{name}_5p": np.percentile(values, 5),
        f"{name}_50p": np.percentile(values, 50),
        f"{name}_95p": np.percentile(values, 95),
    }


def get_color_data(image, mask):
    """HSV features: statistics of the hue, saturation and value (all 0-1) of the lesion pixels."""
    pixels = rgb2hsv(image[:, :, :3])[mask > 0]  # drop an alpha channel if present
    features = {}
    for i, channel in enumerate(["hue", "saturation", "value"]):
        features.update(channel_stats(channel, pixels[:, i]))
    return features


def get_rgb_data(image, mask):
    """RGB and greyscale features (0-255) of the lesion pixels, used in the extended feature set."""
    pixels = image[:, :, :3][mask > 0]
    features = {}
    for i, channel in enumerate(["red", "green", "blue"]):
        features.update(channel_stats(channel, pixels[:, i].astype(float)))
    # Standard luminance weights for converting RGB to greyscale
    grey = (0.2989 * pixels[:, 0] + 0.5870 * pixels[:, 1] + 0.1140 * pixels[:, 2]).astype(float)
    features.update(channel_stats("grey", grey))
    return features
