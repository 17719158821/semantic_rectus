import os
import sys

from skimage import filters
from sklearn.metrics import auc
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve
from skimage import measure, exposure
from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np



def print_metrics(itr, **kargs):
    print("*** Round {}  ====> ".format(itr))
    for name, value in kargs.items():
        print ("{} : {}, ".format(name, value)),
    print("")
    sys.stdout.flush()



def AUC_ROC(true_vessel_arr, pred_vessel_arr):
    """
    Area under the ROC curve with x axis flipped
    """
    """
    fpr, tpr, _ = roc_curve(true_vessel_arr, pred_vessel_arr)
    try:
        AUC_ROC = roc_auc_score(true_vessel_arr.flatten(), pred_vessel_arr.flatten())
    except:
        AUC_ROC = 0.
    return AUC_ROC, fpr, tpr
    """
    try:
        AUC_ROC = roc_auc_score(true_vessel_arr.flatten(), pred_vessel_arr.flatten())
    except:
        AUC_ROC = 0.
    return AUC_ROC


def AUC_PR(true_vessel_img, pred_vessel_img):
    """
    Precision-recall curve
    """
    """
    precision, recall, _ = precision_recall_curve(true_vessel_img.flatten(), pred_vessel_img.flatten(),  pos_label=1)
    try:
        AUC_prec_rec = auc(recall, precision)
    except:
        AUC_prec_rec = 0.
    return AUC_prec_rec, precision, recall
    """
    try:
        precision, recall, _ = precision_recall_curve(true_vessel_img.flatten(), pred_vessel_img.flatten(), pos_label=1)
        AUC_prec_rec = auc(recall, precision)
    except:
        AUC_prec_rec = 0.
    return AUC_prec_rec


def best_f1_threshold(precision, recall, thresholds):
    best_f1 = -1
    for index in range(len(precision)):
        curr_f1 = 2. * precision[index] * recall[index] / (precision[index] + recall[index])
        if best_f1 < curr_f1:
            best_f1 = curr_f1
            best_threshold = thresholds[index]

    return best_f1, best_threshold


def threshold_by_otsu_local(pred_vessesl, flatten=True, window=128, stride=32):
    assert len(pred_vessesl.shape)==2
    binary_vessel = np.zeros_like(pred_vessesl, dtype=np.uint8)
    for sw_x in range(0, pred_vessesl.shape[0]-window+1, stride):
        for sw_y in range(0, pred_vessesl.shape[1]-window+1, stride):
            local_image = pred_vessesl[sw_x: sw_x + window, sw_y: sw_y + window]
            if np.max(local_image) != np.min(local_image):
                threshold = filters.threshold_otsu(local_image)
                local_bin = np.zeros(shape=[window, window], dtype=np.uint8)
                local_bin[local_image > threshold] = 1
                binary_vessel[sw_x: sw_x+window, sw_y: sw_y+window] += local_bin

    binary_vessel = np.clip(binary_vessel, 0, 1)

    if flatten:
        return binary_vessel.flatten()
    else:
        return binary_vessel


def threshold_by_otsu(pred_vessels, flatten=True):
    # cut by otsu threshold
    threshold = filters.threshold_otsu(pred_vessels)
    pred_vessels_bin = np.zeros(pred_vessels.shape)
    pred_vessels_bin[pred_vessels >= threshold] = 1

    if flatten:
        return pred_vessels_bin.flatten()
    else:
        return pred_vessels_bin


def threshold_by_f1(true_vessels, generated, masks, flatten=True, f1_score=False):
    vessels_in_mask, generated_in_mask = pixel_values_in_mask(true_vessels, generated, masks)
    precision, recall, thresholds = precision_recall_curve(vessels_in_mask.flatten(), generated_in_mask.flatten(),
                                                           pos_label=1)
    best_f1, best_threshold = best_f1_threshold(precision, recall, thresholds)

    pred_vessels_bin = np.zeros(generated.shape)
    pred_vessels_bin[generated >= best_threshold] = 1

    if flatten:
        if f1_score:
            return pred_vessels_bin[masks == 1].flatten(), best_f1
        else:
            return pred_vessels_bin[masks == 1].flatten()
    else:
        if f1_score:
            return pred_vessels_bin, best_f1
        else:
            return pred_vessels_bin


def misc_measures(true_vessels, pred_vessels, masks):
    thresholded_vessel_arr, f1_score = threshold_by_f1(true_vessels, pred_vessels, masks, f1_score=True)
    true_vessel_arr = true_vessels[masks == 1].flatten()

    cm = confusion_matrix(true_vessel_arr, thresholded_vessel_arr)
    acc = 1. * (cm[0, 0] + cm[1, 1]) / np.sum(cm)
    sensitivity = 1. * cm[1, 1] / (cm[1, 0] + cm[1, 1])
    specificity = 1. * cm[0, 0] / (cm[0, 1] + cm[0, 0])
    return f1_score, acc, sensitivity, specificity


def misc_measures_in_train(true_vessel_arr, pred_vessel_arr):
    true_vessel_arr = true_vessel_arr.astype(np.bool)
    pred_vessel_arr = pred_vessel_arr.astype(np.bool)

    cm = confusion_matrix(true_vessel_arr, pred_vessel_arr)
    try:
        acc = 1. * (cm[0, 0] + cm[1, 1]) / np.sum(cm)
    except:
        acc = 0.

    try:
        sensitivity = 1. * cm[1, 1] / (cm[1, 0] + cm[1, 1])
    except:
        sensitivity = 0.

    try:
        specificity = 1. * cm[0, 0] / (cm[0, 1] + cm[0, 0])
    except:
        specificity = 0.
    return acc, sensitivity, specificity


def dice_coefficient(true_vessels, pred_vessels, masks):
    thresholded_vessels = threshold_by_f1(true_vessels, pred_vessels, masks, flatten=False)

    true_vessels = true_vessels.astype(np.bool)
    thresholded_vessels = thresholded_vessels.astype(np.bool)

    intersection = np.count_nonzero(true_vessels & thresholded_vessels)

    size1 = np.count_nonzero(true_vessels)
    size2 = np.count_nonzero(thresholded_vessels)

    try:
        dc = 2. * intersection / float(size1 + size2)
    except ZeroDivisionError:
        dc = 0.0

    return dc


def img_dice(pred_vessel, true_vessel):
    threshold = filters.threshold_otsu(pred_vessel)
    pred_vessels_bin = np.zeros(pred_vessel.shape)
    pred_vessels_bin[pred_vessel >= threshold] = 1
    dice_coeff = dice_coefficient_in_train(true_vessel.flatten(), pred_vessels_bin.flatten())
    return dice_coeff


def vessel_similarity(segmented_vessel_0, segmented_vessel_1):
    try:
        threshold_0 = filters.threshold_otsu(segmented_vessel_0)
        threshold_1 = filters.threshold_otsu(segmented_vessel_1)
        segmented_vessel_0_bin = np.zeros(segmented_vessel_0.shape)
        segmented_vessel_1_bin = np.zeros(segmented_vessel_1.shape)
        segmented_vessel_0_bin[segmented_vessel_0 > threshold_0] = 1
        segmented_vessel_1_bin[segmented_vessel_1 > threshold_1] = 1
        dice_coeff = dice_coefficient_in_train(segmented_vessel_0_bin.flatten(), segmented_vessel_1_bin.flatten())
        return dice_coeff
    except:
        return 0.


def dice_coefficient_in_train(true_vessel_arr, pred_vessel_arr):
    true_vessel_arr = true_vessel_arr.astype(np.bool)
    pred_vessel_arr = pred_vessel_arr.astype(np.bool)

    intersection = np.count_nonzero(true_vessel_arr & pred_vessel_arr)

    size1 = np.count_nonzero(true_vessel_arr)
    size2 = np.count_nonzero(pred_vessel_arr)

    try:
        dc = 2. * intersection / float(size1 + size2)
    except ZeroDivisionError:
        dc = 0.0

    return dc


def operating_pts_human_experts(gt_vessels, pred_vessels, masks):
    gt_vessels_in_mask, pred_vessels_in_mask = pixel_values_in_mask(gt_vessels, pred_vessels, masks, split_by_img=True)

    n = gt_vessels_in_mask.shape[0]
    op_pts_roc, op_pts_pr = [], []
    for i in range(n):
        cm = confusion_matrix(gt_vessels_in_mask[i], pred_vessels_in_mask[i])
        fpr = 1 - 1. * cm[0, 0] / (cm[0, 1] + cm[0, 0])
        tpr = 1. * cm[1, 1] / (cm[1, 0] + cm[1, 1])
        prec = 1. * cm[1, 1] / (cm[0, 1] + cm[1, 1])
        recall = tpr
        op_pts_roc.append((fpr, tpr))
        op_pts_pr.append((recall, prec))

    return op_pts_roc, op_pts_pr


def pixel_values_in_mask(true_vessels, pred_vessels, masks, split_by_img=False):
    assert np.max(pred_vessels) <= 1.0 and np.min(pred_vessels) >= 0.0
    assert np.max(true_vessels) == 1.0 and np.min(true_vessels) == 0.0
    assert np.max(masks) == 1.0 and np.min(masks) == 0.0
    assert pred_vessels.shape[0] == true_vessels.shape[0] and masks.shape[0] == true_vessels.shape[0]
    assert pred_vessels.shape[1] == true_vessels.shape[1] and masks.shape[1] == true_vessels.shape[1]
    assert pred_vessels.shape[2] == true_vessels.shape[2] and masks.shape[2] == true_vessels.shape[2]

    if split_by_img:
        n = pred_vessels.shape[0]
        return np.array([true_vessels[i, ...][masks[i, ...] == 1].flatten() for i in range(n)]), np.array(
            [pred_vessels[i, ...][masks[i, ...] == 1].flatten() for i in range(n)])
    else:
        return true_vessels[masks == 1].flatten(), pred_vessels[masks == 1].flatten()


def remain_in_mask(imgs, masks):
    imgs[masks == 0] = 0
    return imgs


def crop_to_original(imgs, ori_shape):
    pred_shape = imgs.shape
    assert len(pred_shape) < 4

    if ori_shape == pred_shape:
        return imgs
    else:
        if len(imgs.shape) > 2:
            ori_h, ori_w = ori_shape[1], ori_shape[2]
            pred_h, pred_w = pred_shape[1], pred_shape[2]
            return imgs[:, (pred_h - ori_h) // 2:(pred_h - ori_h) // 2 + ori_h,
                   (pred_w - ori_w) // 2:(pred_w - ori_w) // 2 + ori_w]
        else:
            ori_h, ori_w = ori_shape[0], ori_shape[1]
            pred_h, pred_w = pred_shape[0], pred_shape[1]
            return imgs[(pred_h - ori_h) // 2:(pred_h - ori_h) // 2 + ori_h,
                   (pred_w - ori_w) // 2:(pred_w - ori_w) // 2 + ori_w]


def difference_map(ori_vessel, pred_vessel, mask):
    # ori_vessel : an RGB image

    thresholded_vessel = threshold_by_f1(np.expand_dims(ori_vessel, axis=0), np.expand_dims(pred_vessel, axis=0),
                                         np.expand_dims(mask, axis=0), flatten=False)

    thresholded_vessel = np.squeeze(thresholded_vessel, axis=0)
    diff_map = np.zeros((ori_vessel.shape[0], ori_vessel.shape[1], 3))
    diff_map[(ori_vessel == 1) & (thresholded_vessel == 1)] = (0, 255, 0)  # Green (overlapping)
    diff_map[(ori_vessel == 1) & (thresholded_vessel != 1)] = (255, 0, 0)  # Red (false negative, missing in pred)
    diff_map[(ori_vessel != 1) & (thresholded_vessel == 1)] = (0, 0, 255)  # Blue (false positive)

    # compute dice coefficient for a given image
    overlap = len(diff_map[(ori_vessel == 1) & (thresholded_vessel == 1)])
    fn = len(diff_map[(ori_vessel == 1) & (thresholded_vessel != 1)])
    fp = len(diff_map[(ori_vessel != 1) & (thresholded_vessel == 1)])

    return diff_map, 2. * overlap / (2 * overlap + fn + fp)


"""
def metric_all_value(pred_vessels, true_vessels):
    assert len(pred_vessels.shape) == 3
    assert len(true_vessels.shape) == 3

    pred_vessels_vec = pred_vessels.flatten()
    true_vessels_vec = true_vessels.flatten()

    auc_roc, fpr, tpr = AUC_ROC(true_vessels_vec, pred_vessels_vec)
    auc_pr, precision, recall = AUC_PR(true_vessels_vec, pred_vessels_vec)

    binary_vessels = threshold_by_otsu(pred_vessels, flatten=False)
    binary_vessels_vec = binary_vessels.flatten()

    dice_coeff = dice_coefficient_in_train(true_vessels_vec, binary_vessels_vec)
    acc, sensitivity, specificity = misc_measures_in_train(true_vessels_vec, binary_vessels_vec)

    return binary_vessels, auc_roc, fpr, tpr, auc_pr, precision, recall, dice_coeff, acc, sensitivity, specificity
"""


def metric_single_img(pred_vessel, true_vessel):
    assert len(pred_vessel.shape) == 3
    assert len(true_vessel.shape) == 3

    pred_vessel_vec = pred_vessel.flatten()
    true_vessel_vec = true_vessel.flatten()

    """
    auc_roc, fpr, tpr = AUC_ROC(true_vessel_vec, pred_vessel_vec)
    auc_pr, precision, recall = AUC_PR(true_vessel_vec, pred_vessel_vec)
    """

    auc_roc = AUC_ROC(true_vessel_vec, pred_vessel_vec)
    auc_pr = AUC_PR(true_vessel_vec, pred_vessel_vec)

    # pred_vessel = exposure.equalize_hist(pred_vessel)
    #binary_vessels = threshold_by_otsu_local(pred_vessel, flatten=False, window=256, stride=64)
    binary_vessels = threshold_by_otsu(pred_vessel, flatten=False)
    binary_vessels_vec = binary_vessels.flatten()

    dice_coeff = dice_coefficient_in_train(true_vessel_vec, binary_vessels_vec)
    acc, sensitivity, specificity = misc_measures_in_train(true_vessel_vec, binary_vessels_vec)

    return binary_vessels, auc_roc, auc_pr, dice_coeff, acc, sensitivity, specificity


def evaluate_single_image(pred_image, true_image):
    assert len(pred_image.shape) == 4
    assert len(true_image.shape) == 4

    pred_image_vec = pred_image.flatten()
    true_image_vec = true_image.flatten()

    auc_roc = AUC_ROC(true_image_vec, pred_image_vec)
    auc_pr = AUC_PR(true_image_vec, pred_image_vec)

    try:
        binary_image = threshold_by_otsu(pred_image, flatten=False)
    except:
        binary_image = np.zeros_like(true_image)
    binary_image_vec = binary_image.flatten()

    dice_coeff = dice_coefficient_in_train(true_image_vec, binary_image_vec)
    acc, sensitivity, specificity = misc_measures_in_train(true_image_vec, binary_image_vec)

    return binary_image, auc_roc, auc_pr, dice_coeff, acc, sensitivity, specificity
