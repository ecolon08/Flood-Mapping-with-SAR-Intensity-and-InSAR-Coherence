"""
Improving Semantic Water Segmentation by Fusing Sentinel-1 Intensity and Interferometric Synthetic Aperture Radar
(InSAR) Coherence Data

Author: Ernesto Colon
The Cooper Union for the Advancement of Science and Art**
Spring 2022

XGBoost Model Inference
"""

# Import libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from utils import metrics_utils
from utils import dataset_gen
from utils import general_utils
import time
import xgboost as xgb


# Define helper functions

# Create function to generate the label overlap between ground truth and predictions
def gen_lbl_overlap(y_true, y_pred):
    """
    Function to return a semantic map with a label overlap given ground truth and predicted labels
    :param y_true: ndarray with ground truth labels
    :param y_pred: ndarray with predicted labels
    :return: combined, an ndarray with 4 classes (1: true positive, 2: true negatives, 3: false positives, 4: false neg)
    """

    # allocate space to store the label overlap
    combined = np.zeros(y_pred.shape)

    # true positives are labels that are predicted as water (1)
    tp = np.logical_and(np.where(y_pred == 1, 1, 0), np.where(y_true == 1, 1, 0))

    # true negatives
    tn = np.logical_and(np.where(y_pred == 0, 1, 0), np.where(y_true == 0, 1, 0))

    # false positives are labels that were labeled as 1 but that were 0 in reality
    fp = np.logical_and(np.where(y_pred == 1, 1, 0), np.where(y_true == 0, 1, 0))

    # false negatives are labels that were labeled as 0 but were 1 in reality
    fn = np.logical_and(np.where(y_pred == 0, 1, 0), np.where(y_true == 1, 1, 0))

    # combine all classes
    combined[tp] = 1
    combined[tn] = 2
    combined[fp] = 3
    combined[fn] = 4

    return combined


###############################################################
#       Create a function to plot the label overlap
###############################################################

# Generate color maps for the labels and label overlap

from utils import general_utils

wtr_cmap = general_utils.gen_cmap(['#f7f7f7', '#67a9cf'])
ovrlp_cmap = general_utils.gen_cmap(['#67a9cf', '#f7f7f7', '#ef8a62', '#999999'])

from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as fm

fontprops = fm.FontProperties(size=15)


def display_lbl_overlap(y_true, lbl_overlap, x_test, num_plot, region, indices=None):
    """
    Function to display the label overlap
    :param y_true: ndarray with ground truth labels
    :param lbl_overlap: ndarray with label overlap
    :param x_test: ndarray with Sentinel-1 co-event intensity (VH) raster
    :param num_plot: integer, number of scenes to display
    :param region: string with geographical region to display
    :param indices: list of integer with indices to plot from the entire data set
    :return: matplotlib figure handle
    """

    fontprops = fm.FontProperties(size=12)

    num_col = 5
    fig, ax = plt.subplots(num_plot + 1, num_col, figsize=(20, 5 * num_plot))
    ax = ax.ravel()

    if indices == None:
        indices = range(num_plot)

    for idx, raster in enumerate(indices):

        ax[num_col * idx].imshow(x_test[region]['scenario_1_hand_lbl'][raster, :, :, 0], cmap='gray')
        ax[num_col * idx].set_title(f'Co-event Intensity (VH)')

        # plot ground truth
        ax[num_col * idx + 1].imshow(y_true[region]['scenario_3_hand_lbl'][raster, :, :], cmap=wtr_cmap)
        # ax[num_col * idx + 1].set_title(f'Ground Truth Label, index: {raster}')
        ax[num_col * idx + 1].set_title(f'Ground Truth Label')

        # plot scenario 1
        ax[num_col * idx + 2].imshow(lbl_overlap[region]['scenario_1_hand_lbl'][raster, :, :], cmap=ovrlp_cmap)
        ax[num_col * idx + 2].set_title('Scenario 1 Label Overlap')

        # plot scenario 2
        ax[num_col * idx + 3].imshow(lbl_overlap[region]['scenario_2_hand_lbl'][raster, :, :], cmap=ovrlp_cmap)
        ax[num_col * idx + 3].set_title('Scenario 2 Label Overlap')

        # plot scenario 3
        ax[num_col * idx + 4].imshow(lbl_overlap[region]['scenario_3_hand_lbl'][raster, :, :], cmap=ovrlp_cmap)
        ax[num_col * idx + 4].set_title('Scenario 3 Label Overlap')

        for axis in ax[: num_col * num_plot]:
            scalebar = AnchoredSizeBar(
                axis.transData,
                100,
                '100m',
                'lower left',
                pad=0.1,
                color='black',
                frameon=False,
                size_vertical=1,
                fontproperties=fontprops)

            axis.add_artist(scalebar)
            axis.set_yticks([])
            axis.set_xticks([]);

    # Create legend
    checkerboard = np.zeros((512, 512))
    checkerboard[0:256, 0:256] = 1
    checkerboard[256:, 0:256] = 2
    checkerboard[0:256, 256:] = 3
    checkerboard[256:, 256:] = 4

    ax[num_col * idx + 4 + 3].imshow(checkerboard, cmap=ovrlp_cmap)
    ax[num_col * idx + 4 + 3].text(50, 128, "True Positives", fontsize=8.);
    ax[num_col * idx + 4 + 3].text(50, 384, "True Negatives", fontsize=8.);
    ax[num_col * idx + 4 + 3].text(290, 128, "False Positives", fontsize=8.);
    ax[num_col * idx + 4 + 3].text(290, 384, "False Negatives", fontsize=8.);
    ax[num_col * idx + 4 + 3].set_yticks([])
    ax[num_col * idx + 4 + 3].set_xticks([]);

    ind_to_del = [1, 2, 4, 5]
    for ind in ind_to_del:
        fig.delaxes(ax[num_col * idx + 4 + ind])

    return fig


if __name__ == "__main__":

    ###############################################################
    #           Load previously saved dataset splits
    ###############################################################

    # Define dictionary with filepaths
    base_dir = "base_dir"

    train_val_test_pths = {'train_fn_df': f"{base_dir}\\ds_train_split_10m.csv",
                           'val_fn_df': f"{base_dir}\\ds_val_split_10m.csv",
                           'test_fn_df': f"{base_dir}\\ds_test_split_10m.csv"}

    train_samples, val_samples, test_samples, train_size, val_size, test_size = \
        dataset_gen.xgboost_load_ds_samples(train_val_test_pths['train_fn_df'],
                                            train_val_test_pths['val_fn_df'],
                                            train_val_test_pths['test_fn_df'])

    ###############################################################
    # Define category names and a color mapping for semantic segmentation
    ###############################################################

    # Define category names
    tgt_cat_names = {
        0: 'Not water',
        1: 'Water'
    }

    # Define the colors per category
    wtr_clrs_hex = ['#f7f7f7', '#67a9cf']

    # Generate the labels colormap
    wtr_cmap = general_utils.gen_cmap(wtr_clrs_hex)

    # %% md
    ###############################################################
    # Generate data sets for inference
    ###############################################################
    """
    We generate datasets for the following scenarios:
    
    - Scenario 1: Co-event intensity data only
    - Scenario 2: Pre- and co-event intensity data only
    - Scenario 3: Pre- and co-event intensity and coherence data
    """

    # Define dictionaries to hold the datasets - the keys will be the different scenarios
    X_train_dict = {}
    Y_train_dict = {}

    X_test_dict = {}
    Y_test_dict = {}

    Y_pred_dict = {}

    scenarios = ['scenario_1', 'scenario_2', 'scenario_3']

    # Loop through each scenario and generate / load the data sets to memory
    for scenario in scenarios:
        # logic to determine whether the current scenario includes coherence data or not
        if scenario == 'scenario_1':
            int_flag = True
        else:
            int_flag = False
        if scenario == 'scenario_3':
            coh_flag = True
        else:
            coh_flag = False

        # generate data set
        X_test_dict[scenario], Y_test_dict[scenario], _, _ = \
            dataset_gen.rf_xgb_ds_generator(test_samples, coh_flag=coh_flag, int_flag=int_flag)

    ###############################################################
    # Gather dataset parameters we'll need later on
    ###############################################################

    num_train_samp = len(train_samples)
    img_size = 512

    num_feat_dict = {'scenario_1': 2,
                     'scenario_2': 4,
                     'scenario_3': 6,
                     'scenario_1_hand_lbl': 2,
                     'scenario_2_hand_lbl': 4,
                     'scenario_3_hand_lbl': 6}

    ###############################################################
    # Hand Labeled Dataset
    ###############################################################

    # load hand label dataset
    hand_lbl_ds_pth = "hand_lbl_ds_pth"
    hand_lbl_ds_fname = f"{hand_lbl_ds_pth}hand_lbl_ds_10m_res.csv"

    # load csv file to dataframe
    df_hand_lbl_samples = pd.read_csv(hand_lbl_ds_fname)

    # loop through df and append sample paths to a list
    hand_lbl_samples = list()

    for idx, row in df_hand_lbl_samples.iterrows():
        hand_lbl_samples.append((row['s1'],
                                 row['pre_event_grd'],
                                 row['pre_event_coh'],
                                 row['co_event_coh'],
                                 row['hand_lbl']))

    hand_lbl_scenarios = [f"{scenario}_hand_lbl" for scenario in scenarios]

    # Generate hand-labeled data set
    for scenario in hand_lbl_scenarios:
        # logic to determine whether the current scenario includes coherence data or not
        if scenario == 'scenario_1_hand_lbl':
            int_flag = True
        else:
            int_flag = False
        if scenario == 'scenario_3_hand_lbl':
            coh_flag = True
        else:
            coh_flag = False

        X_test_dict[scenario], Y_test_dict[scenario], _, _ = \
            dataset_gen.rf_xgb_ds_generator(hand_lbl_samples, coh_flag=coh_flag, int_flag=int_flag)

    ###############################################################
    #              Visualize some image-target pairs
    ###############################################################

    # Load a number of scenes
    scenes_list = list()

    num_scenes = 5

    for idx in range(num_scenes):
        temp_list = list()

        for j in range(len(train_samples[idx])):

            # Open rasters with rasterio
            with rasterio.open(train_samples[idx][j]) as src:
                src = src.read()
                if j == 4:  # account for labels and map the not-valid pixels to the not-water category
                    src = np.where(src == -1, 0, src)
                temp_list.append(src)

        scenes_list.append(temp_list)

    ###############################################################
    #                   Display the scenes
    ###############################################################

    num_col = 7

    fig, ax = plt.subplots(num_scenes, num_col, figsize=(80, num_scenes * 10))
    ax = ax.ravel()

    for i in range(len(scenes_list)):
        # s1 co-event
        ax[num_col * i].imshow(scenes_list[i][0][0, :, :], cmap='gray')
        ax[num_col * i].set_title('S1 co-event VH')

        ax[num_col * i + 1].imshow(scenes_list[i][0][1, :, :], cmap='gray')
        ax[num_col * i + 1].set_title('S1 co-event VV')

        # s1 pre-event
        ax[num_col * i + 2].imshow(scenes_list[i][1][0, :, :], cmap='gray')
        ax[num_col * i + 2].set_title('S1 pre-event VH')

        ax[num_col * i + 3].imshow(scenes_list[i][1][1, :, :], cmap='gray')
        ax[num_col * i + 3].set_title('S1 pre-event VV')

        # pre-event coh
        ax[num_col * i + 4].imshow(scenes_list[i][2][0, :, :], cmap='gray')
        ax[num_col * i + 4].set_title('Pre-event coherence')

        # co-event coh
        ax[num_col * i + 5].imshow(scenes_list[i][3][0, :, :], cmap='gray')
        ax[num_col * i + 5].set_title('Co-event coherence')

        # s2 label
        ax[num_col * i + 6].imshow(scenes_list[i][4][0, :, :], cmap=wtr_cmap)
        ax[num_col * i + 6].set_title('S2 Label')

    for ax in ax:
        ax.set_yticks([])
        ax.set_xticks([]);

    ###############################################################
    #                       XGBoost Models
    ###############################################################

    # Load previously trained models

    xgb_models_dir = {'scenario_1': "model_scen_1_pth",
                      'scenario_2': "model_scen_2_pth",
                      'scenario_3': "model_scen_3_pth"}

    xgb_classifier_models = {}

    for scenario in xgb_models_dir.keys():
        xgb_classifier_models[scenario] = xgb.XGBClassifier(use_label_encoder=False,
                                                            tree_method='gpu_hist')

        print(f"Loading model weights for scenario: {scenario}...")
        xgb_classifier_models[scenario].load_model(xgb_models_dir[scenario])

    ###############################################################
    #       Make predictions with the XGBoost models
    ###############################################################

    """
    Notes
    
    The held-out test set is comprised of Sentinel-2 weak labels from the Sen1Floods11 data set.
    
    The hand-labeled data set is also provided by the Sen1Floods11 data set, and provides an independent data set not
    used during training.
    """

    # predict on the held-out test dataset

    start_time = time.time()

    # loop through each scenario
    for scenario in xgb_classifier_models.keys():
        Y_pred_dict[scenario] = xgb_classifier_models[scenario].predict(X_test_dict[scenario])

    # Predict on the hand-labeled test dataset

    for scenario in scenarios:
        Y_pred_dict[f"{scenario}_hand_lbl"] = xgb_classifier_models[scenario].predict(
            X_test_dict[f"{scenario}_hand_lbl"])

    print(f"Inference took: {time.time() - start_time} seconds")

    ###############################################################
    #                   Compute Metrics
    ###############################################################

    """
    For metrics, we compute:
    
    - Overall accuracy
    - Mean intersection over union, mIoU
    - Jaccard score
    - Water precision
    - Water recall
    - Water f1-score
    - Not-Water precision
    - Not-Water recall
    - Not-Water f1-score

    """

    ###############################################################
    #                Held-out Test Dataset
    ###############################################################

    start_time = time.time()

    summary_df = metrics_utils.summary_report(Y_test_dict, Y_pred_dict)

    print(f"Process took: {time.time() - start_time} seconds")

    # save summary to csv file
    xgboost_summ_pth = "xgboost_summ_pth"
    fname = "xgboost_summary_stats.csv"
    summary_df.to_csv(f"{xgboost_summ_pth}\\{fname}")

    ###############################################################
    #    Computing IoU per class (i.e., water and not-water)
    ###############################################################

    miou_per_class = metrics_utils.miou_per_class(Y_test_dict, Y_pred_dict)

    # save to csv file
    mIou_fname = "xgboost_10m_mIoU_per_class_stats.csv"
    miou_per_class.to_csv(f"{xgboost_summ_pth}\\{mIou_fname}")

    ###############################################################
    # Testing Models Ability to Generalize
    ###############################################################

    # We use data over the Sri-Lanka region (both weakly labeled as well as hand-labeled) to
    # test the models' ability to generalize

    ###############################################################
    # Generate generalization dataset
    ###############################################################

    # create a list with all regions for both the held-out test set and the hand-labeled test set
    regions = ['USA', 'Mekong', 'Colombia', 'Paraguay', 'India', 'Bolivia']
    regions_w_hand_lbl = [region for region in regions if region != "Colombia"]

    generalization_ds_pth = "generalization_ds_pth"

    # Create empty list to store the samples' path
    gen_test_samples = []

    # Grab the number of samples in the data set
    gen_test_fn_df = pd.read_csv(generalization_ds_pth)

    for idx, row in gen_test_fn_df.iterrows():
        gen_test_samples.append(
            (row['s1'], row['pre_event_grd'], row['pre_event_coh'], row['co_event_coh'], row['s2_lbl']))

    num_gen_samp = len(gen_test_samples)

    # create dictionaries to store the data sets
    gener_X_test_dict = dict()
    gener_Y_test_dict = dict()

    # Loop through each scenario
    for scenario in scenarios:
        # logic to determine whether the current scenario includes coherence data or not
        if scenario == 'scenario_1':
            int_flag = True
        else:
            int_flag = False
        if scenario == 'scenario_3':
            coh_flag = True
        else:
            coh_flag = False

        gener_X_test_dict[scenario], gener_Y_test_dict[scenario], _, _ = dataset_gen.rf_xgb_ds_generator(
            gen_test_samples, coh_flag=coh_flag, int_flag=int_flag)

    ###############################################################
    #       Hand-Labeled Generalization Data Set
    ###############################################################

    # load hand label dataset
    gen_hand_lbl_ds_pth = "gen_hand_lbl_ds_pth"

    # read hand-labeled data set into dataframe
    gen_df_hand_lbl_samples = pd.read_csv(gen_hand_lbl_ds_pth)

    # create dict to store the data set
    gen_hand_samples_by_region_dict = {}

    # For now, we only have Sri-Lanka as the generalization region
    regions = ['Sri-Lanka']

    for region in regions:
        # temp list to store file paths
        pths = list()

        # pluck the test sample paths by region
        test_pth_region = gen_df_hand_lbl_samples[gen_df_hand_lbl_samples.s1.str.contains(region)]

        for idx, row in test_pth_region.iterrows():
            pths.append((row['s1'], row['pre_event_grd'], row['pre_event_coh'], row['co_event_coh'], row['hand_lbl']))

        gen_hand_samples_by_region_dict[region] = pths

    # Generate hand-labeled generalization test dataset

    for scenario in hand_lbl_scenarios:
        # logic to determine whether the current scenario includes coherence data or not
        if scenario == 'scenario_1_hand_lbl':
            int_flag = True
        else:
            int_flag = False
        if scenario == 'scenario_3_hand_lbl':
            coh_flag = True
        else:
            coh_flag = False

        gener_X_test_dict[scenario], gener_Y_test_dict[scenario], _, _ = dataset_gen.rf_xgb_ds_generator(
            gen_hand_samples_by_region_dict['Sri-Lanka'], coh_flag=coh_flag, int_flag=int_flag)

    ###############################################################
    # Make Predictions on the generalization data set
    ###############################################################

    start_time = time.time()

    gener_Y_pred_dict = dict()

    for scenario in xgb_classifier_models.keys():
        gener_Y_pred_dict[scenario] = xgb_classifier_models[scenario].predict(gener_X_test_dict[scenario])

    # Predict on the hand-labeled test dataset

    for scenario in scenarios:
        gener_Y_pred_dict[f"{scenario}_hand_lbl"] = xgb_classifier_models[scenario].predict(
            gener_X_test_dict[f"{scenario}_hand_lbl"])

    print(f"Inference took: {time.time() - start_time} seconds")

    ###############################################################
    #   Compute metrics for generalization data set
    ###############################################################

    start_time = time.time()

    gener_summary_df = metrics_utils.summary_report(gener_Y_test_dict, gener_Y_pred_dict)

    print(f"Process took: {time.time() - start_time} seconds")

    # save the metrics to a csv file for later recall
    gener_summ_fname = "xgboost_10m_generalization_stats.csv"
    gener_summary_df.to_csv(f"{xgboost_summ_pth}\\{gener_summ_fname}")

    ###############################################################
    #   Compute IoU per class for generalization dataset
    ###############################################################

    gener_miou_per_class = metrics_utils.miou_per_class(gener_Y_test_dict, gener_Y_pred_dict)

    # save to csv
    gener_miou_fname = "xgboost_10m_generalization_mIoU_stats.csv"
    gener_miou_per_class.to_csv(f"{xgboost_summ_pth}\\{gener_miou_fname}")

    ###############################################################
    # Making Inferences Aggregated by Geographical Region
    ###############################################################

    # Read csv file with the test filepaths
    test_fn_df = pd.read_csv(train_val_test_pths['test_fn_df'])

    test_samples_by_region_dict = {}
    regions = ['USA', 'Mekong', 'Colombia', 'Paraguay', 'India', 'Bolivia']

    for region in regions:
        pths = list()

        # pluck the test sample paths by region
        test_pth_region = test_fn_df[test_fn_df.s1.str.contains(region)]

        for idx, row in test_pth_region.iterrows():
            pths.append((row['s1'], row['pre_event_grd'], row['pre_event_coh'], row['co_event_coh'], row['s2_lbl']))

        test_samples_by_region_dict[region] = pths

    # Generate the data sets per region
    all_scenarios = scenarios + hand_lbl_scenarios

    # Create schemas for the data sets
    X_test_ds_region_dict = {region: {} for region in regions}
    Y_test_ds_region_dict = {region: {} for region in regions}

    for region in regions:
        # Scenario 1
        X_test_ds_region_dict[region]['scenario_1'], Y_test_ds_region_dict[region]['scenario_1'], _, _ = \
            dataset_gen.rf_xgb_ds_generator(test_samples_by_region_dict[region], coh_flag=False, int_flag=True)

        # Scenario 2
        X_test_ds_region_dict[region]['scenario_2'], Y_test_ds_region_dict[region]['scenario_2'], _, _ = \
            dataset_gen.rf_xgb_ds_generator(test_samples_by_region_dict[region], coh_flag=False, int_flag=False)

        # Scenario 3
        X_test_ds_region_dict[region]['scenario_3'], Y_test_ds_region_dict[region]['scenario_3'], _, _ = \
            dataset_gen.rf_xgb_ds_generator(test_samples_by_region_dict[region], coh_flag=True, int_flag=False)

    ###############################################################
    #                 Make inferences by region
    ###############################################################

    start_time = time.time()

    Y_pred_region_dict = {region: {} for region in regions}

    for scenario in scenarios:

        for region in regions:
            Y_pred_region_dict[region][scenario] = \
                xgb_classifier_models[scenario].predict(X_test_ds_region_dict[region][scenario])

    print(f"Inference took: {time.time() - start_time} seconds")

    ###############################################################
    # Compute predictions on hand-labeled dataset aggregated by region
    ###############################################################

    # Note: Colombia does not have hand-labeled chips**

    hand_lbl_samples_region_dict = {}

    # Colombia does not have any hand labels
    regions = ['USA', 'Mekong', 'Paraguay', 'India', 'Bolivia']

    for region in regions:
        pths = list()

        # pluck the test sample paths by region
        test_pth_region = df_hand_lbl_samples[df_hand_lbl_samples.s1.str.contains(region)]

        for idx, row in test_pth_region.iterrows():
            pths.append((row['s1'], row['pre_event_grd'], row['pre_event_coh'], row['co_event_coh'], row['hand_lbl']))

        hand_lbl_samples_region_dict[region] = pths

    for region in regions:
        # Scenario 2
        X_test_ds_region_dict[region]['scenario_1_hand_lbl'], Y_test_ds_region_dict[region][
            'scenario_1_hand_lbl'], _, _ = \
            dataset_gen.rf_xgb_ds_generator(hand_lbl_samples_region_dict[region], coh_flag=False, int_flag=True)

        # Scenario 4
        X_test_ds_region_dict[region]['scenario_2_hand_lbl'], Y_test_ds_region_dict[region][
            'scenario_2_hand_lbl'], _, _ = \
            dataset_gen.rf_xgb_ds_generator(hand_lbl_samples_region_dict[region], coh_flag=False, int_flag=False)

        # Scenario 5
        X_test_ds_region_dict[region]['scenario_3_hand_lbl'], Y_test_ds_region_dict[region][
            'scenario_3_hand_lbl'], _, _ = \
            dataset_gen.rf_xgb_ds_generator(hand_lbl_samples_region_dict[region], coh_flag=True, int_flag=False)

    ###############################################################
    # Make inferences with the hand-labeled dataset aggregated by region
    ###############################################################

    start_time = time.time()

    for scenario in scenarios:

        for region in regions:
            Y_pred_region_dict[region][f'{scenario}_hand_lbl'] = \
                xgb_classifier_models[scenario].predict(X_test_ds_region_dict[region][f'{scenario}_hand_lbl'])

    print(f"Inference took: {time.time() - start_time} seconds")

    ###############################################################
    #       Generate prediction summaries by region
    ###############################################################

    start_time = time.time()

    regions = ['USA', 'Mekong', 'Colombia', 'Paraguay', 'India', 'Bolivia']

    xgboost_summ_pth = "xgboost_summ_pth"

    summary_by_region = {}

    for region in regions:
        print(f"Region: {region}\n\n")
        summary_by_region[region] = metrics_utils.summary_report(Y_test_ds_region_dict[region],
                                                                 Y_pred_region_dict[region])
        print("\n\n")

        # save to csv
        summary_by_region[region].to_csv(f"{xgboost_summ_pth}\\{region}_summary_stats.csv")

    print(f"Process took: {time.time() - start_time} seconds")

    ###############################################################
    # Compute IoU per class aggregated by region
    ###############################################################

    # Create dict to store the IoU metrics by region
    regional_miou_per_class = {}

    for region in regions:
        print(f"Region: {region}\n\n")

        regional_miou_per_class[region] = metrics_utils.miou_per_class(Y_test_ds_region_dict[region],
                                                                       Y_pred_region_dict[region])

        print("\n\n")

        # save to csv
        regional_miou_per_class[region].to_csv(f"{xgboost_summ_pth}\\{region}_mIoU_stats.csv")

    ###############################################################
    #       Generate labels and label overlap by region
    ###############################################################

    # Merge the generalization data set with the rest of the data sets
    Y_pred_region_dict['Sri-Lanka'] = gener_Y_pred_dict

    Y_test_ds_region_dict['Sri-Lanka'] = gener_Y_test_dict

    X_test_ds_region_dict['Sri-Lanka'] = gener_X_test_dict

    ###############################################################
    #       Reshape predictions for visualization**
    ###############################################################

    all_regions = list(Y_pred_region_dict.keys())
    all_regions_hand_lbl = [region for region in all_regions if region != 'Colombia']

    # Create dicts to store the ground truth, predicted labels and the intensity rasters for visualization
    Y_pred_hand_lbl_by_region = {}
    Y_true_hand_lbl_by_region = {}
    X_test_hand_lbl_by_region = {}

    # scenarios to pluck
    scen_to_pluck = ['scenario_1_hand_lbl', 'scenario_2_hand_lbl', 'scenario_3_hand_lbl']

    # Create schema to store the predictions and test data
    Y_pred_hand_lbl_by_region = {region: {scen: [] for scen in scen_to_pluck} for region in all_regions_hand_lbl}
    Y_true_hand_lbl_by_region = {region: {scen: [] for scen in scen_to_pluck} for region in all_regions_hand_lbl}
    X_test_hand_lbl_by_region = {region: {scen: [] for scen in scen_to_pluck} for region in all_regions_hand_lbl}

    ###############################################################
    #       Copy the predictions and the ground truth labels
    ###############################################################

    for region in all_regions_hand_lbl:
        for scen in scen_to_pluck:
            try:
                Y_pred_hand_lbl_by_region[region][scen] = Y_pred_region_dict[region][scen].copy()

                Y_true_hand_lbl_by_region[region][scen] = Y_test_ds_region_dict[region][scen].copy()

                X_test_hand_lbl_by_region[region][scen] = X_test_ds_region_dict[region][scen].copy()
            except:
                continue

    ###############################################################
    #           Compute label overlap by region
    ###############################################################

    lbl_ovrlap_by_region = {region: {scen: [] for scen in scen_to_pluck} for region in all_regions_hand_lbl}

    for region in all_regions_hand_lbl:
        for scen in scen_to_pluck:
            lbl_ovrlap_by_region[region][scen] = \
                np.reshape(gen_lbl_overlap(
                    Y_true_hand_lbl_by_region[region][scen],
                    Y_pred_hand_lbl_by_region[region][scen]),
                    (-1, img_size, img_size))

    ###############################################################
    #     Reshape ground truth and display the label overlap
    ###############################################################

    for region in all_regions_hand_lbl:
        print(region)
        for scen in scen_to_pluck:
            Y_pred_hand_lbl_by_region[region][scen] = np.reshape(Y_pred_hand_lbl_by_region[region][scen],
                                                                 (-1, img_size, img_size))

            Y_true_hand_lbl_by_region[region][scen] = np.reshape(Y_true_hand_lbl_by_region[region][scen],
                                                                 (-1, img_size, img_size))

            X_test_hand_lbl_by_region[region][scen] = np.reshape(X_test_hand_lbl_by_region[region][scen],
                                                                 (-1, img_size, img_size, num_feat_dict[scen]))

    ###############################################################
    #               Label Overlap for Region: USA
    ###############################################################

    ovrlp_lbl_pth = "ovrlp_lbl_pth"
    region = "USA"

    idx_USA = [1, 3, 5, 8, 22]
    fig_USA = display_lbl_overlap(Y_true_hand_lbl_by_region,
                                  lbl_ovrlap_by_region,
                                  X_test_hand_lbl_by_region,
                                  num_plot=len(idx_USA),
                                  region='USA',
                                  indices=idx_USA)

    # save
    # fig_USA.savefig(fname)

    ###############################################################
    # Label Overlap for Region: Mekong
    ###############################################################

    region = "Mekong"
    idx_Mekong = [1, 2, 5, 7, 8]
    fig_Mekong = display_lbl_overlap(Y_true_hand_lbl_by_region,
                                     lbl_ovrlap_by_region,
                                     X_test_hand_lbl_by_region,
                                     num_plot=len(idx_Mekong),
                                     region=region,
                                     indices=idx_Mekong)

    # fname = f"{ovrlp_lbl_pth}\\{region}_lbl_ovrlp.pdf"
    # fig_Mekong.savefig(fname)

    ###############################################################
    # Label Overlap for Region: Bolivia
    ###############################################################

    idx_Bolivia = [1, 2, 3, 4, 5]
    region = "Bolivia"
    fig_Bol = display_lbl_overlap(Y_true_hand_lbl_by_region,
                                  lbl_ovrlap_by_region,
                                  X_test_hand_lbl_by_region,
                                  num_plot=len(idx_Bolivia),
                                  region=region,
                                  indices=idx_Bolivia)

    # fname = f"{ovrlp_lbl_pth}\\{region}_lbl_ovrlp.pdf"
    # fig_Bol.savefig(fname)

    ###############################################################
    # Label Overlap for Region: Paraguay
    ###############################################################

    region = 'Paraguay'
    idx_Paraguay = [0, 1, 2, 6, 7]
    fig_Par = display_lbl_overlap(Y_true_hand_lbl_by_region,
                                  lbl_ovrlap_by_region,
                                  X_test_hand_lbl_by_region,
                                  num_plot=len(idx_Paraguay),
                                  region=region,
                                  indices=idx_Paraguay)

    # fname = f"{ovrlp_lbl_pth}\\{region}_lbl_ovrlp.pdf"
    # fig_Par.savefig(fname)

    ###############################################################
    # Label Overlap for Region: India
    ###############################################################

    region = "India"
    idx_India = [0, 2, 4, 6, 23]
    fig_Ind = display_lbl_overlap(Y_true_hand_lbl_by_region,
                                  lbl_ovrlap_by_region,
                                  X_test_hand_lbl_by_region,
                                  num_plot=len(idx_India),
                                  region=region,
                                  indices=idx_India)

    # fname = f"{ovrlp_lbl_pth}\\{region}_lbl_ovrlp.pdf"
    # fig_Ind.savefig(fname)

    ###############################################################
    # Label Overlap for Region: Sri-Lanka
    ###############################################################

    region = "Sri-Lanka"
    idx_Sri_Lanka = [8, 9, 11, 16, 21]
    fig_Sri = display_lbl_overlap(Y_true_hand_lbl_by_region,
                                  lbl_ovrlap_by_region,
                                  X_test_hand_lbl_by_region,
                                  num_plot=len(idx_Sri_Lanka),
                                  region=region,
                                  indices=idx_Sri_Lanka)

    # fname = f"{ovrlp_lbl_pth}\\{region}_lbl_ovrlp.pdf"
    # fig_Sri.savefig(fname)
