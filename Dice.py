import numpy as np

import keras.backend as K


def DiceLoss(y_true, y_pred):
    """输入为tensor"""
    # if K.max(y_true) > 1:
    #     y_true_temp = y_true / 255.
    # if K.max(y_pred) > 1:
    #     y_pred_temp = y_pred / 255.
    y_true_temp = y_true
    y_pred_temp = y_pred
    return 2 * K.sum(y_true_temp * y_pred_temp) / (K.sum(y_pred_temp) + K.sum(y_true_temp))


def Dice(y_true, y_pred):
    """输入为array"""
    if np.max(y_true) > 1:
        y_true = y_true / 255.
    if np.max(y_pred) > 1:
        y_pred = y_pred / 255.
    return 2 * np.sum(y_true * y_pred) / (np.sum(y_pred) + np.sum(y_true))

