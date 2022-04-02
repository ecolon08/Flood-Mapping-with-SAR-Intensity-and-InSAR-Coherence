"""
Visualization utilities
"""

# Import libraries
import numpy as np


def scale_img(matrix):
    """
    Reference: https://www.drivendata.co/blog/detect-floodwater-benchmark/

    Returns a scaled (H, W, D) image that is visually inspectable.
    Image is linearly scaled between min_ and max_value, by channel.

    Args:
        matrix (np.array): (H, W, D) image to be scaled

    Returns:
        np.array: Image (H, W, 3) ready for visualization
    """

    # Set min/max values
    min_values = np.array([-23, -23, -23])
    max_values = np.array([5, 5, 5])

    # Reshape matrix
    w, h, d = matrix.shape
    matrix = np.reshape(matrix, [w * h, d]).astype(np.float64)

    # Scale by min/max
    matrix = (matrix - min_values[None, :]) / (
        max_values[None, :] - min_values[None, :]
    )
    matrix = np.reshape(matrix, [w, h, d])

    # Limit values to 0/1 interval
    return matrix.clip(0, 1)


def sar_false_color(co_event_img, pre_event_img):
    """
    Function to create a false color composite given a flood co-event and pre-event Sentinel-1 raster pair
    :param co_event_img: ndarray with co-event SAR raster data
    :param pre_event_img: ndarray with pre-event SAR raster data
    :return: ndarray with false color composite
    """
    false_clr_int = np.dstack([co_event_img[:, :, 0],
                               pre_event_img[:, :, 0],
                               pre_event_img[:, :, 0]])

    # scale the false color composite
    false_clr_int = scale_img(false_clr_int)

    return false_clr_int
