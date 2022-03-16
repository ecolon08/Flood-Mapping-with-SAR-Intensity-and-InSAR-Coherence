"""
Author: Ernesto Colon
Date: January 29th, 2022

The Cooper Union

Data generator utilities for flood mapping using synthetic aperture radar data
"""

# Import libraries
import numpy as np
from tqdm import tqdm
import rasterio
import tensorflow as tf
from tensorflow import keras
import pandas as pd


####################################
#      XGBoost Pipeline
####################################

def xgboost_load_ds_samples(train_df_pth, val_df_pth, test_df_pth):
    """
    Function to process training, validation, and test input image paths for the XGBoost models

    :param train_df_pth: dataframe with training data set input file paths
    :param val_df_pth: dataframe with validation data set input file paths
    :param test_df_pth: dataframe with test data set input file paths
    :return: lists for train, val, and test data file paths along with corresponding data set sizes
    """
    # Create lists to hold the sample paths
    test_samples = []
    train_samples = []
    val_samples = []

    # Load the csv files with the train, validation, and test splits
    train_fn_df = pd.read_csv(train_df_pth)
    val_fn_df = pd.read_csv(val_df_pth)
    test_fn_df = pd.read_csv(test_df_pth)

    # Loop through the CSV files and extract + organize the different chips
    for idx, row in test_fn_df.iterrows():
        test_samples.append((row['s1'], row['pre_event_grd'], row['pre_event_coh'], row['co_event_coh'], row['s2_lbl']))

    for idx, row in train_fn_df.iterrows():
        train_samples.append(
            (row['s1'], row['pre_event_grd'], row['pre_event_coh'], row['co_event_coh'], row['s2_lbl']))

    for idx, row in val_fn_df.iterrows():
        val_samples.append((row['s1'], row['pre_event_grd'], row['pre_event_coh'], row['co_event_coh'], row['s2_lbl']))

    # grab the dataset sizes
    train_size = len(train_samples)
    val_size = len(val_samples)
    test_size = len(test_samples)

    return train_samples, val_samples, test_samples, train_size, val_size, test_size


def rf_xgb_ds_generator(samples, coh_flag=False, int_flag=False):
    """
    Function to generate an XGBoost or Random Forest pixel-wise classification data set for training
    The function generates the data set depending on the input scenario requested. The coherence and intensity
    booleans determine the scenario.

    Co-event intensity: coh_flag = False, int_flag = True
    Bi-temporal intensity: coh_flag = False, int_flag = True
    Bi-temporal intensity and coherence: coh_flag = True, int_flag = True

    :param samples: list containing tuples with input file paths
    :param coh_flag: coherence band boolean flag
    :param int_flag: intensity band boolean flag
    :return: X_train, Y_train as ndarrays with training data set, number of samples and number of features
    """

    X_train = list()
    Y_train = list()

    for pth in tqdm(samples):

        if int_flag:
            # s1 co-event intensity
            with rasterio.open(pth[0]) as src:
                s1_co_event = np.squeeze(src.read())

                if np.any(np.isnan(s1_co_event)):
                    continue

                img_stack = np.dstack((s1_co_event))

        else:

            # s1 co-event intensity
            with rasterio.open(pth[0]) as src:
                s1_co_event = np.transpose(np.squeeze(src.read()), axes=(1, 2, 0))

                if np.any(np.isnan(s1_co_event)):
                    continue

            # s1 pre-event intensity
            with rasterio.open(pth[1]) as src:
                s1_pre_event = np.transpose(np.squeeze(src.read()), axes=(1, 2, 0))

                if np.any(np.isnan(s1_pre_event)):
                    continue

            if coh_flag:
                # pre-event coherence
                with rasterio.open(pth[2]) as src:
                    # coh_pre_event = np.transpose(np.squeeze(src.read()), axes=(1, 2, 0))
                    coh_pre_event = np.transpose(src.read(), axes=(1, 2, 0))

                    if np.any(np.isnan(coh_pre_event)):
                        continue

                # co-event coherence
                with rasterio.open(pth[3]) as src:
                    # coh_co_event = np.transpose(np.squeeze(src.read()), axes=(1, 2, 0))
                    coh_co_event = np.transpose(src.read(), axes=(1, 2, 0))

                    if np.any(np.isnan(coh_co_event)):
                        continue

                # stack into a single array
                img_stack = np.dstack((s1_co_event, s1_pre_event, coh_co_event, coh_pre_event))


            else:
                img_stack = np.dstack((s1_co_event, s1_pre_event))

        # append to list
        X_train.append(img_stack)

        # labels
        with rasterio.open(pth[4]) as src:
            lbl = np.squeeze(src.read())

            # take care of the not-valid pixels - just relabel them as not water for now
            lbl = np.where(lbl == -1, 0, lbl)

        Y_train.append(lbl)

    # Stack the lists into np arrays now

    X_train = np.stack(X_train)
    Y_train = np.stack(Y_train)

    num_samp = X_train.shape[0]
    num_feat = X_train.shape[-1]

    # Reshape
    X_train = X_train.reshape(-1, X_train.shape[3])
    Y_train = Y_train.reshape(-1)

    return X_train, Y_train, num_samp, num_feat


####################################
# U-Net Pipeline
####################################


def unet_load_ds_df(train_df_pth, val_df_pth, test_df_pth):
    """
    Function to process training, validation, and test input image paths for the U-Net models

    :param train_df_pth: dataframe with training data set input file paths
    :param val_df_pth: dataframe with validation data set input file paths
    :param test_df_pth: dataframe with test data set input file paths
    :return: dataframes for train, val, and test data file paths along with corresponding data set sizes
    """
    # Create lists to hold the sample paths
    test_samples = []
    train_samples = []
    val_samples = []

    # Load the csv files with the train, validation, and test splits
    train_fn_df = pd.read_csv(train_df_pth)
    val_fn_df = pd.read_csv(val_df_pth)
    test_fn_df = pd.read_csv(test_df_pth)

    # Loop through the CSV files and extract + organize the different chips
    for idx, row in test_fn_df.iterrows():
        test_samples.append((row['s1'], row['pre_event_grd'], row['pre_event_coh'], row['co_event_coh'], row['s2_lbl']))

    for idx, row in train_fn_df.iterrows():
        train_samples.append(
            (row['s1'], row['pre_event_grd'], row['pre_event_coh'], row['co_event_coh'], row['s2_lbl']))

    for idx, row in val_fn_df.iterrows():
        val_samples.append((row['s1'], row['pre_event_grd'], row['pre_event_coh'], row['co_event_coh'], row['s2_lbl']))

    # concatenate the train and validation dataframes
    train_val_fn_df = pd.concat([train_fn_df, val_fn_df])

    # grab the dataset sizes
    train_size = len(train_samples)
    val_size = len(val_samples)
    test_size = len(test_samples)

    return train_val_fn_df, test_fn_df, train_size, val_size, test_size


def create_samples_list(params):
    """
    Function that ingests the train + val and test dataframes containing the filepaths and returns a list of tuples with
    the filepaths. The list of tuples returned is assembled based on the particular multi-temporal scenario.

    :param train_val_df: dataframe with the train + val sample filepaths
    :param test_df: dataframe with the test sample filepaths
    :param scenario: string indicating multi-temporal scenario
    :return: test_samples and train_val_samples lists containing the filepaths for the data loader
    """

    # Pluck the arguments from the dictionary
    scenario = params['scenario']
    test_df = params['test_df']
    train_val_df = params['train_val_df']

    if scenario == 'co_event_coh_only':
        test_samples = []
        train_val_samples = []

        for idx, row in test_df.iterrows():
            # test_samples.append((row['co_event_coh'], row['s1_lbl']))
            test_samples.append((row['co_event_coh'], row['s2_lbl']))
            # test_samples.append((row['pre_event_coh'], row['s1_lbl']))

        for idx, row in train_val_df.iterrows():
            # train_val_samples.append((row['co_event_coh'], row['s1_lbl']))
            train_val_samples.append((row['co_event_coh'], row['s2_lbl']))
            # train_val_samples.append((row['pre_event_coh'], row['s1_lbl']))

    elif scenario == 'co_event_intensity_only':
        test_samples = []
        train_val_samples = []

        for idx, row in test_df.iterrows():
            # test_samples.append((row['s1'], row['s1_lbl']))
            test_samples.append((row['s1'], row['s2_lbl']))

        for idx, row in train_val_df.iterrows():
            # train_val_samples.append((row['s1'], row['s1_lbl']))
            train_val_samples.append((row['s1'], row['s2_lbl']))

    elif scenario == 'pre_co_event_coh':
        test_samples = []
        train_val_samples = []

        for idx, row in test_df.iterrows():
            # test_samples.append((row['pre_event_coh'], row['co_event_coh'], row['s1_lbl']))
            test_samples.append((row['pre_event_coh'], row['co_event_coh'], row['s2_lbl']))

        for idx, row in train_val_df.iterrows():
            # train_val_samples.append((row['pre_event_coh'], row['co_event_coh'], row['s1_lbl']))
            train_val_samples.append((row['pre_event_coh'], row['co_event_coh'], row['s2_lbl']))

    elif scenario == 'pre_co_event_intensity':
        test_samples = []
        train_val_samples = []

        for idx, row in test_df.iterrows():
            # test_samples.append((row['s1'], row['pre_event_grd'], row['s1_lbl']))
            test_samples.append((row['s1'], row['pre_event_grd'], row['s2_lbl']))

        for idx, row in train_val_df.iterrows():
            # train_val_samples.append((row['s1'], row['pre_event_grd'], row['s1_lbl']))
            train_val_samples.append((row['s1'], row['pre_event_grd'], row['s2_lbl']))

    elif scenario == 'pre_co_event_int_coh':
        test_samples = []
        train_val_samples = []

        for idx, row in test_df.iterrows():
            # test_samples.append((row['s1'], row['pre_event_grd'], row['pre_event_coh'], row['co_event_coh'], row['s1_lbl']))
            test_samples.append(
                (row['s1'], row['pre_event_grd'], row['pre_event_coh'], row['co_event_coh'], row['s2_lbl']))

        for idx, row in train_val_df.iterrows():
            # train_val_samples.append((row['s1'], row['pre_event_grd'], row['pre_event_coh'], row['co_event_coh'], row['s1_lbl']))
            train_val_samples.append(
                (row['s1'], row['pre_event_grd'], row['pre_event_coh'], row['co_event_coh'], row['s2_lbl']))

    return train_val_samples, test_samples


IMG_SIZE = 512
class_weight = {0: 1, 1: 1}

def read_sample(data_path: str) -> tuple:
    """
    Function to read the data path and return a tuple with input image, ground truth label, and pixel weights
    :param data_path: string with input file paths
    :return: tuple with input image, ground truth label, and pixel weights
    """

    path = data_path.numpy()

    # figure out the number of items in the path variable
    num_items = len(path)

    # the target path is always the last item
    target_path = path[-1].decode('utf-8')

    # loop through the rest of the paths and append to list
    pth_lst = list()
    for pth in path[:-1]:
        pth_lst.append(pth.decode('utf-8'))

    # work with the target separately
    with rasterio.open(target_path) as src:
        tgt = np.transpose(src.read(), axes=(1, 2, 0)).astype(np.int8)

        data_mask = np.where(tgt == -1, 0, 1)
        sample_weights = np.where(data_mask == 0, class_weight[0], class_weight[1])

        # Map the no valid pixels to not-water
        tgt = np.where(tgt == -1, 0, tgt)

    # Loop through the rest of the image stack, transpose and then stack
    stack = list()

    for pth in pth_lst:
        with rasterio.open(pth) as src:
            # transpose
            raster = np.transpose(src.read(), axes=(1, 2, 0))

            # check if we have multiple bands in the raster (e.g., sentinel 1 scene)
            if raster.shape[-1] > 1:
                num_bands = raster.shape[-1]
                for band in range(num_bands):
                    # squeeze and append to list
                    stack.append(np.squeeze(raster[:, :, band]))
            else:
                # squeeze and append to list
                stack.append(np.squeeze(raster[:, :, 0]))

    img = np.stack(stack)
    img = np.transpose(img, axes=(1, 2, 0))

    return (img, tgt, sample_weights)


@tf.function
def tf_read_sample(data_path: str) -> dict:
    """
    wrap custom dataloadeer into tensorflow
    """

    [image, target, sample_weight] = tf.py_function(read_sample, [data_path], [tf.float32, tf.int8, tf.uint8])

    # explicitly set tensor shapes
    num_bands = image.shape[-1]

    image.set_shape((IMG_SIZE, IMG_SIZE, num_bands))
    target.set_shape((IMG_SIZE, IMG_SIZE, 1))
    sample_weight.set_shape((IMG_SIZE, IMG_SIZE, 1))

    return {'image': image, 'target': target, 'sample_weight': sample_weight}


@tf.function
def load_sample(sample: dict) -> tuple:
    """
    Function to load samples using tensorflow data pipeline
    """
    # convert to tf image
    image = tf.image.resize(sample['image'], (IMG_SIZE, IMG_SIZE))
    target = tf.image.resize(sample['target'], (IMG_SIZE, IMG_SIZE))
    sample_weight = tf.image.resize(sample['sample_weight'], (IMG_SIZE, IMG_SIZE))

    # flip image at random to augment
    if tf.random.uniform(()) > 0.5:
        num_rot = tf.random.uniform(shape=(), minval=0, maxval=4, dtype=tf.int32)
        image = tf.image.rot90(tf.image.flip_left_right(image), k=num_rot)
        target = tf.image.rot90(tf.image.flip_left_right(target), k=num_rot)

    target = tf.cast(target, tf.uint8)

    return image, target, sample_weight


@tf.function
def load_test_sample(sample: dict) -> tuple:
    # convert to tf image
    image = tf.image.resize(sample['image'], (IMG_SIZE, IMG_SIZE))
    target = tf.image.resize(sample['target'], (IMG_SIZE, IMG_SIZE))
    sample_weight = tf.image.resize(sample['sample_weight'], (IMG_SIZE, IMG_SIZE))

    target = tf.cast(target, tf.uint8)

    return image, target, sample_weight


def unet_ds_creation(params):
    """
    Function to generate a tensorflow data set generator given train, validation, and test file paths as lists
    :param params: dictionary with train, val, and test file path lists
    :return: tensorflow data set generators for train, validation, and test
    """
    train_val_samples = params['train_val_list']
    test_samples = params['test_list']

    # create tensorflow dataset from file links
    train_val_ds = tf.data.Dataset.from_tensor_slices(train_val_samples)
    test_ds = tf.data.Dataset.from_tensor_slices(test_samples)

    # read in image/target pairs
    train_val_ds = train_val_ds.map(tf_read_sample, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    test_ds = test_ds.map(tf_read_sample, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # read in as tensors
    train_val_ds = train_val_ds.map(load_sample, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    test_ds = test_ds.map(load_test_sample, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    train_val_ds_size = len(train_val_samples)

    train_size = int(0.85 * train_val_ds_size)
    val_size = int(0.15 * train_val_ds_size)

    train_val_ds = train_val_ds.cache()
    train_val_ds = train_val_ds.shuffle(train_size)

    train_dataset = train_val_ds.take(train_size)
    val_dataset = train_val_ds.skip(train_size)

    test_dataset = test_ds.cache()

    return train_dataset, val_dataset, test_dataset


def create_samples_list_hand_lbl(params):
    """
    Function that ingests the train + val and test dataframes containing the filepaths and returns a list of tuples with
    the filepaths. The list of tuples returned is assembled based on the particular multi-temporal scenario.
    :param train_val_df: dataframe with the train + val sample filepaths
    :param test_df: dataframe with the test sample filepaths
    :param scenario: string indicating multi-temporal scenario
    :return: test_samples and train_val_samples lists containing the filepaths for the data loader
    """

    # Pluck the arguments from the dictionary
    scenario = params['scenario']
    test_df = params['test_df']

    if scenario == 'co_event_coh_only':
        test_samples = []

        for idx, row in test_df.iterrows():
            try:
                test_samples.append((row['pre_event_coh'], row['hand_lbl']))
            except:
                test_samples.append((row['pre_event_coh'], row['s2_lbl']))

    elif scenario == 'co_event_intensity_only':
        test_samples = []

        for idx, row in test_df.iterrows():
            try:
                test_samples.append((row['s1'], row['hand_lbl']))
            except:
                test_samples.append((row['s1'], row['s2_lbl']))


    elif scenario == 'pre_co_event_coh':
        test_samples = []

        for idx, row in test_df.iterrows():
            try:
                test_samples.append((row['pre_event_coh'], row['co_event_coh'], row['hand_lbl']))
            except:
                test_samples.append((row['pre_event_coh'], row['co_event_coh'], row['s2_lbl']))

    elif scenario == 'pre_co_event_intensity':
        test_samples = []

        for idx, row in test_df.iterrows():
            try:
                test_samples.append((row['s1'], row['pre_event_grd'], row['hand_lbl']))
            except:
                test_samples.append((row['s1'], row['pre_event_grd'], row['s2_lbl']))

    elif scenario == 'pre_co_event_int_coh':
        test_samples = []

        for idx, row in test_df.iterrows():
            try:
                test_samples.append(
                    (row['s1'], row['pre_event_grd'], row['pre_event_coh'], row['co_event_coh'], row['hand_lbl']))
            except:
                test_samples.append(
                    (row['s1'], row['pre_event_grd'], row['pre_event_coh'], row['co_event_coh'], row['s2_lbl']))

    return test_samples


def ds_creation_hand_lbl(params):
    """
    Function to create tensorflow data set generator for the hand-labeled data set
    :param params: dictionary with test data set list including input file paths
    :return: tensorflow data set generator for hand-labeled data set
    """
    test_samples = params['test_list']

    # create tensorflow dataset from file links
    test_ds = tf.data.Dataset.from_tensor_slices(test_samples)

    # read in image/target pairs
    test_ds = test_ds.map(tf_read_sample, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # read in as tensors
    test_ds = test_ds.map(load_test_sample, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    test_dataset = test_ds.cache()

    return test_dataset
