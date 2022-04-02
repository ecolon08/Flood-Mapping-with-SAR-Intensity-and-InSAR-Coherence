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


def sar_temporal_false_color(co_event_img, pre_event_img):
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


def sar_false_color_composite(vv_img, vh_img):
    """
    Returns a S1 false color composite for visualization.

    Args:
        vv_img: ndarray with Sentinel-1 VV-band raster
        vh_img: ndarray with Sentinel-1 VH-band raster

    Returns:
        np.array: image (H, W, 3) ready for visualization
    """
    # Stack arrays along the last dimension
    s1_img = np.stack((vv_img, vh_img), axis=-1)

    # Create false color composite
    img = np.zeros((512, 512, 3), dtype=np.float32)
    img[:, :, :2] = s1_img.copy()
    img[:, :, 2] = s1_img[:, :, 0] / s1_img[:, :, 1]

    return scale_img(img)


def gen_lbl_overlap(y_true, y_pred, img_size=512):
    """
    Function to generate label overlap raster
    :param y_true: ndarray with ground truth labels
    :param y_pred: ndarray with predicted labels
    :param img_size: integer, raster dimension
    :return: ndarray with label overlap for: true positives, true negatives, false positives, and false negatives
    """
    combined = np.zeros((img_size, img_size))

    # true positives are labels that are predicted as water (1)
    tp =np.logical_and(np.where(y_pred == 1, 1, 0), np.where(y_true == 1, 1, 0))

    # true negatives
    tn = np.logical_and(np.where(y_pred == 0, 1, 0), np.where(y_true == 0, 1, 0))

    # false positives are labels that were labeled as 1 but that were 0 in reality
    fp = np.logical_and(np.where(y_pred == 1, 1, 0), np.where(y_true == 0, 1, 0))

    # false negatives are labels that were labeled as 0 but were 1 in reality
    fn = np.logical_and(np.where(y_pred == 0, 1, 0), np.where(y_true == 1, 1, 0))

    # combine the two
    combined[tp] = 1
    combined[tn] = 2
    combined[fp] = 3
    combined[fn] = 4

    return combined

