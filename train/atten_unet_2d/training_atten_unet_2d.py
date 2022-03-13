"""

Improving Semantic Water Segmentation by Fusing Sentinel-1 Intensity and Interferometric Synthetic Aperture Radar
(InSAR) Coherence Data

**Author: Ernesto Colon**
**The Cooper Union for the Advancement of Science and Art**

#### Attention Unet-2D Model Training
"""

###############################################################
#                       Import libraries
###############################################################

import tensorflow as tf
import time
from utils import dataset_gen
import matplotlib.pyplot as plt
import sys
sys.path.append('..')

###############################################################
# Define function to plot train and validation loss
###############################################################

def plot_train_val_loss(model_history):
    """
    Function to plot training and validation loss
    """
    loss = model_history.history['loss']
    val_loss = model_history.history['val_loss']

    plt.figure()
    plt.plot(model_history.epoch, loss, 'r', label='Training loss')
    plt.plot(model_history.epoch, val_loss, 'bo', label='Validation loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss Value')
    plt.ylim([0, 1])
    plt.legend()
    plt.show()


if __name__ == "__main__":
    ###############################################################
    # check that a GPU is enabled
    ###############################################################

    device_name = tf.test.gpu_device_name()
    if device_name != '/device:GPU:0':
        raise SystemError('GPU device not found')
    print('Found GPU at: {}'.format(device_name))

    ###############################################################
    # Load the train, validation, and test dataframes
    ###############################################################

    # Define dictionary with filepaths
    base_dir = "base_dir_path"

    train_val_test_pths = {'train_fn_df': f"{base_dir}\\ds_train_split_10m.csv",
                           'val_fn_df': f"{base_dir}\\ds_val_split_10m.csv",
                           'test_fn_df': f"{base_dir}\\ds_test_split_10m.csv"}

    train_val_fn_df, test_fn_df, train_size, val_size, test_size = \
        dataset_gen.unet_load_ds_df(train_val_test_pths['train_fn_df'],
                                    train_val_test_pths['val_fn_df'],
                                    train_val_test_pths['test_fn_df'])

    ###############################################################
    #   We generate datasets for the following scenarios:

    #   - Scenario 1: Co-event intensity data only
    #   - Scenario 2: Pre- and co-event intensity data only
    #   - Scenario 3: Pre- and co-event intensity and coherence data
    ###############################################################

    # Define dictionaries to hold the datasets - the keys will be the different scenarios
    X_train_dict = {}
    Y_train_dict = {}

    X_val_dict = {}
    Y_val_dict = {}

    X_test_dict = {}
    Y_test_dict = {}

    Y_pred_dict = {}

    # Define scenario number to scenario name mapping
    scenario_dict = {1: 'co_event_intensity_only',
                     2: 'pre_co_event_intensity',
                     3: 'pre_co_event_int_coh'}

    scenario_num_bands = {1: 2,
                          2: 4,
                          3: 6}

    # Define the number of bands per scenario
    num_bands_dict = {'co_event_intensity_only': 2,
                      'pre_co_event_intensity': 4,
                      'pre_co_event_int_coh': 6}

    IMG_SIZE = 512

    # define dictionaries to hold the datasets
    train_val_samples_dict = {}
    test_samples_dict = {}

    # Loop through each scenario and create the tensorflow data loaders
    scenarios = [1, 2, 3]

    for scenario in scenarios:
        # Create the samples list given the dataframes with file paths as input
        train_val_samples_dict[f"scenario_{scenario}"], test_samples_dict[f"scenario_{scenario}"] = \
            dataset_gen.create_samples_list({'scenario': scenario_dict[scenario],
                                             'test_df': test_fn_df,
                                             'train_val_df': train_val_fn_df})

        # Create data sets dictionary
        X_train_dict[f"scenario_{scenario}"], X_val_dict[f"scenario_{scenario}"], X_test_dict[f"scenario_{scenario}"] = \
            dataset_gen.unet_ds_creation({'train_val_list': train_val_samples_dict[f"scenario_{scenario}"],
                                          'test_list': test_samples_dict[f"scenario_{scenario}"]})

        # Batch the tensorflow train, val, and test data set generators
        X_train_dict[f"scenario_{scenario}"] = \
            X_train_dict[f"scenario_{scenario}"].batch(10).prefetch(tf.data.experimental.AUTOTUNE)

        X_val_dict[f"scenario_{scenario}"] = \
            X_val_dict[f"scenario_{scenario}"].batch(10).prefetch(tf.data.experimental.AUTOTUNE)

        X_test_dict[f"scenario_{scenario}"] = X_test_dict[f"scenario_{scenario}"].batch(1)

    ###############################################################################################
    # Attention U-Net Models

    # For this study, we leverage the publicly available Keras UNet Collection linked below.

    # https://github.com/yingkaisha/keras-unet-collection
    ###############################################################################################

    from keras_unet_collection import models

    # create dictionary to hold the models by scenario
    attn_unet_2d_models = {}

    ###############################################################
    #       Loop through scenarios and generate the models
    ###############################################################

    for scenario in scenarios:
        # Create models for each scenario
        print("\n*******************************************\n")
        print(f"Generating model for scenario: {scenario}")

        attn_unet_2d_models[f"scenario_{scenario}"] = models.att_unet_2d(
            (IMG_SIZE, IMG_SIZE, scenario_num_bands[scenario]),
            filter_num=[64, 128, 256, 512, 1024],
            n_labels=2,
            stack_num_down=2,
            stack_num_up=2,
            activation='ReLU',
            atten_activation='ReLU',
            attention='add',
            output_activation='Sigmoid',
            batch_norm=True,
            pool=False,
            unpool=False,
            backbone='VGG16',
            weights=None,
            freeze_backbone=False,
            freeze_batch_norm=True,
            name='attunet')

        print("*******************************************")

    # unet_2d_models['scenario_3'].summary()

    # %%

    # Define a learning rate schedule
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(0.0001,
                                                                 decay_steps=200,
                                                                 decay_rate=0.96,
                                                                 staircase=True)

    # Define the optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule,
                                         beta_1=0.9,
                                         beta_2=0.999,
                                         epsilon=1e-07,
                                         amsgrad=False,
                                         name='Adam')

    # Create dictionary to store model training history
    attn_unet_2d_train_hist = {}

    ###############################################################
    # Scenario 1 Training- Co-event Intensity Model
    ###############################################################
    # %%

    # Compile the model
    current_scenario = 1
    attn_unet_2d_models[f"scenario_{current_scenario}"].compile(
        optimizer=optimizer,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=['accuracy'])

    # Start training routine
    train_start_time = time.time()

    EPOCHS = 30
    attn_unet_2d_train_hist[f"scenario_{current_scenario}"] = \
        attn_unet_2d_models[f"scenario_{current_scenario}"].fit(
            X_train_dict[f"scenario_{current_scenario}"],
            validation_data=X_val_dict[f"scenario_{current_scenario}"],
            epochs=EPOCHS)

    print("--- %s seconds ---" % (time.time() - train_start_time))

    ###############################################################
    #       Save the model weights for scenario 1
    ###############################################################

    attn_unet_2d_model_pth = "atten_unet_model_path"
    attn_unet_2d_models[f"scenario_{current_scenario}"].save_weights(
        f"{attn_unet_2d_model_pth}\\scenario_{current_scenario}"+"\\" + f"unet2d_10m_{scenario_dict[current_scenario]}")

    # Plot training and validation loss for scenario 1

    plot_train_val_loss(attn_unet_2d_train_hist[f"scenario_{current_scenario}"])

    ###############################################################
    # Scenario 2 Training - Pre-event and Co-event Intensity Model
    ###############################################################

    # Compile the model
    current_scenario = 2
    attn_unet_2d_models[f"scenario_{current_scenario}"].compile(
        optimizer=optimizer,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=['accuracy'])

    ###############################################################
    #               Start training routine
    ###############################################################
    train_start_time = time.time()

    EPOCHS = 30
    attn_unet_2d_train_hist[f"scenario_{current_scenario}"] = \
        attn_unet_2d_models[f"scenario_{current_scenario}"].fit(
            X_train_dict[f"scenario_{current_scenario}"],
            validation_data=X_val_dict[f"scenario_{current_scenario}"],
            epochs=EPOCHS)

    print("--- %s seconds ---" % (time.time() - train_start_time))

    ###############################################################
    #       Save the model weights for scenario 2
    ###############################################################

    attn_unet_2d_models[f"scenario_{current_scenario}"].save_weights(
        f"{attn_unet_2d_model_pth}\\scenario_{current_scenario}" + "\\" + f"unet2d_10m_{scenario_dict[current_scenario]}")

    # Plot training and validation loss for scenario 2

    plot_train_val_loss(attn_unet_2d_train_hist[f"scenario_{current_scenario}"])

    ###############################################################
    # Scenario 3 Training - Pre-event and Co-event Intensity Model
    ###############################################################

    # Compile the model
    current_scenario = 3
    attn_unet_2d_models[f"scenario_{current_scenario}"].compile(optimizer=optimizer,
                                                                loss=tf.keras.losses.SparseCategoricalCrossentropy(
                                                                    from_logits=False),
                                                                metrics=['accuracy'])

    # Start training routine
    train_start_time = time.time()

    EPOCHS = 1
    attn_unet_2d_train_hist[f"scenario_{current_scenario}"] = \
        attn_unet_2d_models[f"scenario_{current_scenario}"].fit(
            X_train_dict[f"scenario_{current_scenario}"],
            validation_data=X_val_dict[f"scenario_{current_scenario}"],
            epochs=EPOCHS)

    print("--- %s seconds ---" % (time.time() - train_start_time))

    ###############################################################
    #       Save the model weights for scenario 3
    ###############################################################

    attn_unet_2d_models[f"scenario_{current_scenario}"].save_weights(
        f"{attn_unet_2d_model_pth}\\scenario_{current_scenario}" + "\\" + f"unet2d_10m_{scenario_dict[current_scenario]}")

    # Plot training and validation loss for scenario 3
    plot_train_val_loss(attn_unet_2d_train_hist[f"scenario_{current_scenario}"])
