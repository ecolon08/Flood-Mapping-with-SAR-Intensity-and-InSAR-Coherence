"""
Author: Ernesto Colon
Date: January 29th, 2022

The Cooper Union

Data generator utilities for flood mapping using synthetic aperture radar data
"""

# Import libraries
import tensorflow as tf
import pandas as pd
from tabulate import tabulate
from sklearn.metrics import classification_report
from sklearn.metrics import jaccard_score
import numpy as np


def calc_metrics(y_true, y_pred):
    """
    Function used to calculate prediction metrics and report in tabular form
    :param y_true: ndarray with ground truth labels
    :param y_pred: ndarray with predicted labels
    :return: None
    """
    # Accuracy
    acc_fcn = tf.keras.metrics.Accuracy(name='Accuracy')
    acc_fcn.update_state(y_true=y_true,
                         y_pred=y_pred)
    accuracy = acc_fcn.result().numpy()

    # IoU
    mIoU_fcn = tf.keras.metrics.MeanIoU(num_classes=2, name='meanIoU')
    mIoU_fcn.update_state(y_true=y_true,
                          y_pred=y_pred)
    mIoU = mIoU_fcn.result().numpy()

    # True Positives
    tp_fcn = tf.keras.metrics.TruePositives(name='TruePositives')
    tp_fcn.update_state(y_true=y_true,
                        y_pred=y_pred)
    true_pos = tp_fcn.result().numpy()

    # True Negatives
    tn_fcn = tf.keras.metrics.TrueNegatives(name='TrueNegatives')
    tn_fcn.update_state(y_true=y_true,
                        y_pred=y_pred)
    true_neg = tn_fcn.result().numpy()

    # False Positives
    fp_fcn = tf.keras.metrics.FalsePositives(name='FalsePositives')
    fp_fcn.update_state(y_true=y_true,
                        y_pred=y_pred)
    false_pos = fp_fcn.result().numpy()

    # False Negatives
    fn_fcn = tf.keras.metrics.FalseNegatives(name='FalseNegatives')
    fn_fcn.update_state(y_true=y_true,
                        y_pred=y_pred)
    false_neg = fn_fcn.result().numpy()

    # Store in dict for summarizing
    metrics_summ = {
        "Accuracy": [accuracy],
        "mIoU": [mIoU],
        "True Positives": [true_pos],
        "True Negatives": [true_neg],
        "False Positives": [false_pos],
        "False Negatives": [false_neg]
    }

    metrics_df = pd.DataFrame.from_dict(metrics_summ)

    print("Metrics Summary\n")
    print(tabulate(metrics_df, headers='keys', tablefmt='psql'))


# SUMMARY REPORTS

def tf_miou(y_true, y_pred):
    """
    Function to wrap the mean intersection over union calculation using tensorflow
    :param y_true: ndarray with ground truth labels
    :param y_pred: ndarray with predicted labels
    :return: float with mean IoU
    """

    # IoU
    mIoU_fcn = tf.keras.metrics.MeanIoU(num_classes=2, name='meanIoU')
    mIoU_fcn.update_state(y_true=y_true,
                          y_pred=y_pred)
    mIoU = mIoU_fcn.result().numpy()

    return mIoU


def summary_report(y_true_dict, y_pred_dict):
    """
    Function to generate a summary report and return a dataframe object
    :param y_true_dict: dictionary with ground truth labels organized by scenario (e.g., co-event, etc.)
    :param y_pred_dict: dictionary with predicted labels organized by scenario
    :return: dataframe with metrics
    """
    target_names = ['Not Water', 'Water']

    report_cols = y_true_dict.keys()

    report_rows = ['Overall Accuracy', 'Mean IoU', 'Jaccard Score',
                   'Water Precision', 'Water Recall', 'Water f1-score',
                   'Not Water Precision', 'Not Water Recall', 'Not Water f1-score']

    # create dataframe
    report_df = pd.DataFrame(index=report_rows, columns=report_cols)

    # Now loop through the classification report and populate the dataframe

    scenarios = y_true_dict.keys()

    for scenario in scenarios:
        # compute the classification report
        scen_report = classification_report(y_true_dict[scenario], y_pred_dict[scenario], target_names=target_names,
                                            output_dict=True)

        report_df.loc['Overall Accuracy', scenario] = scen_report['accuracy']
        report_df.loc['Mean IoU', scenario] = tf_miou(y_true_dict[scenario], y_pred_dict[scenario])
        report_df.loc['Jaccard Score', scenario] = jaccard_score(y_true_dict[scenario], y_pred_dict[scenario])
        report_df.loc['Water Precision', scenario] = scen_report['Water']['precision']
        report_df.loc['Water Recall', scenario] = scen_report['Water']['recall']
        report_df.loc['Water f1-score', scenario] = scen_report['Water']['f1-score']
        report_df.loc['Not Water Precision', scenario] = scen_report['Not Water']['precision']
        report_df.loc['Not Water Recall', scenario] = scen_report['Not Water']['recall']
        report_df.loc['Not Water f1-score', scenario] = scen_report['Not Water']['f1-score']

    print(report_df)

    return report_df


def miou_per_class(y_true_dict, y_pred_dict):
    """
    Function to compute the intersection over union per class (i.e., water and not-water)
    :param y_true_dict: dictionary with ground truth labels organized by scenario (e.g., co-event, etc.)
    :param y_pred_dict: dictionary with predicted labels organized by scenario
    :return: dataframe with intersection over union metrics (floats)
    """
    # Using built-in keras function
    from keras.metrics import MeanIoU

    report_cols = y_true_dict.keys()

    report_rows = ['Total mIoU', 'Not Water mIoU', 'Water mIoU']

    # create dataframe
    miou_report = pd.DataFrame(index=report_rows, columns=report_cols)

    num_classes = 2
    IOU_keras = MeanIoU(num_classes=num_classes)

    for scenario in y_true_dict.keys():
        IOU_keras.update_state(y_true_dict[scenario], y_pred_dict[scenario])
        miou_report.loc['Total mIoU', scenario] = IOU_keras.result().numpy()

        # To calculate I0U for each class...
        values = np.array(IOU_keras.get_weights()).reshape(num_classes, num_classes)

        miou_report.loc['Not Water mIoU', scenario] = values[0, 0] / (values[0, 0] + values[0, 1] + values[1, 0])
        miou_report.loc['Water mIoU', scenario] = values[1, 1] / (values[1, 1] + values[1, 0] + values[0, 1])

        IOU_keras.reset_state()

    print(miou_report)

    return miou_report


def create_mask(pred_mask):
    """
    Function to create semantic mask given an ndarray of logits
    :param pred_mask: ndarray of logits
    :return: ndarray with shape (IMG_SIZE, IMG_SIZE) with predicted labels
    """
    pred_mask = np.argmax(pred_mask, axis=-1)
    return pred_mask