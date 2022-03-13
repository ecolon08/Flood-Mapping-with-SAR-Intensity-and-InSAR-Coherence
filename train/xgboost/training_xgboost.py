"""
Improving Semantic Water Segmentation by Fusing Sentinel-1 Intensity and Interferometric Synthetic Aperture Radar
(InSAR) Coherence Data

Author: Ernesto Colon
The Cooper Union for the Advancement of Science and Art
Spring 2022

XGBoost Model Training
"""

# Import libraries

import sys
sys.path.append('..')
from utils import dataset_gen
import xgboost as xgb
import time

"""
Define function to train the XGBoost models in two steps or batches. The data set is large (~28GB for scenario 3) and
does not fit in GPU memory. Depending on the GPU memory size, the training pipeline may require training in more than
two stages.
"""


def xgb_batch_train(X_train, Y_train, save_fname):
    """
    Function to serialize the XGBoost training for large data sets. This function only handles two batches
    since the data set we're using can be split in half and fit in the RTX 3090's memory.

    :param X_train: 2D-ndarray with shape (num_pix, num_feat) with input features
    :param Y_train: 2D-ndarray with shape (num_pix,) with labels
    :param save_fname: string with path and filename to save the final model
    :return: None
    """

    start_time = time.time()

    # create first model instance
    model_1 = xgb.XGBClassifier(use_label_encoder=False, tree_method='gpu_hist')

    # fit first model
    model_1.fit(X_train['batch_1'], Y_train['batch_1'])

    # create second model instance
    model_2 = xgb.XGBClassifier(use_label_encoder=False, tree_method='gpu_hist')

    # fit second model
    model_2.fit(X_train['batch_2'], Y_train['batch_2'], xgb_model=model_1)

    print("--- %s seconds ---" % (time.time() - start_time))

    # Save model
    model_2.save_model(save_fname)


if __name__ == "__main__":

    ###############################################################
    #         Load previously saved dataset splits
    ###############################################################

    # Define dictionary with filepaths
    base_dir = "base_dir_path"

    train_val_test_pths = {'train_fn_df': f"{base_dir}\\train_fn_df_fname",
                           'val_fn_df': f"{base_dir}\\val_fn_df_fname",
                           'test_fn_df': f"{base_dir}\\test_fn_df_fname"}

    train_samples, val_samples, test_samples, train_size, val_size, test_size = \
        dataset_gen.xgboost_load_ds_samples(train_val_test_pths['train_fn_df'],
                                            train_val_test_pths['val_fn_df'],
                                            train_val_test_pths['test_fn_df'])

    ###############################################################
    # Create dictionaries to store the training and test data sets
    ###############################################################

    batches = ['batch_1', 'batch_2']
    scenarios = ['scenario_1', 'scenario_2', 'scenario_3']

    X_train_dict = {scenario: {} for scenario in scenarios}
    Y_train_dict = {scenario: {} for scenario in scenarios}

    X_test_dict = {scenario: {} for scenario in scenarios}
    Y_test_dict = {scenario: {} for scenario in scenarios}

    ###############################################################
    # Split the training data set into batches for sequential training
    ###############################################################

    train_split_idx_low = [0, int(len(train_samples) / 2)]
    train_split_idx_high = [int(len(train_samples) / 2), len(train_samples)]

    test_split_idx_low = [0, int(len(test_samples) / 2)]
    test_split_idx_high = [int(len(test_samples) / 2), len(test_samples)]

    ###############################################################
    # Select the scenario to be trained
    train_scenario = 1
    ###############################################################

    # logic to determine whether the current scenario includes coherence data or not
    if train_scenario == 1:
        int_flag = True
    else:
        int_flag = False
    if train_scenario == 3:
        coh_flag = True
    else:
        coh_flag = False

    ###############################################################
    #                  Training Pipeline
    ###############################################################

    current_scenario = train_scenario

    # Generate data sets for the current scenario
    for idx, batch in enumerate(batches):
        X_train_dict[f"scenario_{current_scenario}"][batch],\
        Y_train_dict[f"scenario_{current_scenario}"][batch], _, _ =\
            dataset_gen.rf_xgb_ds_generator(train_samples[train_split_idx_low[idx]: train_split_idx_high[idx]],
                                            coh_flag=coh_flag,
                                            int_flag=int_flag)

        X_test_dict[f"scenario_{current_scenario}"], Y_test_dict[f"scenario_{current_scenario}"], _, _ =\
            dataset_gen.rf_xgb_ds_generator(test_samples[test_split_idx_low[idx]: test_split_idx_high[idx]],
                                            coh_flag=coh_flag,
                                            int_flag=int_flag)

    ###############################################################
    #                   XGBoost training
    ###############################################################

    # Define path and file name to save the trained model
    xgboost_model_pth = "xgboost_model_pth"
    fname = f"{xgboost_model_pth}\\scenario_{current_scenario}\\xgb_10m_raw_pix_feat_scen_{current_scenario}.model"

    # start the stage-wise training pipeline
    xgb_batch_train(X_train_dict[f"scenario_{current_scenario}"], Y_train_dict[f"scenario_{current_scenario}"], fname)