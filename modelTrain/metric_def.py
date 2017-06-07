# 定义回调函数

import keras.backend as K
import tensorflow as tf


def precision(y_true, y_pred):
    """Precision metric.
    Only computes a batch-wise average of precision.
    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def recall(y_true, y_pred):
    """Recall metric.
    Only computes a batch-wise average of recall.
    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def fbeta_score(y_true, y_pred, beta=1):
    """Computes the F score.
    The F score is the weighted harmonic mean of precision and recall.
    Here it is only computed as a batch-wise average, not globally.
    This is useful for multi-label classification, where input samples can be
    classified as sets of labels. By only using accuracy (precision) a model
    would achieve a perfect score by simply assigning every class to every
    input. In order to avoid this, a metric should penalize incorrect class
    assignments as well (recall). The F-beta score (ranged from 0.0 to 1.0)
    computes this, as a weighted mean of the proportion of correct class
    assignments vs. the proportion of incorrect class assignments.
    With beta = 1, this is equivalent to a F-measure. With beta < 1, assigning
    correct classes becomes more important, and with beta > 1 the metric is
    instead weighted towards penalizing incorrect class assignments.
    """
    if beta < 0:
        raise ValueError('The lowest choosable beta is zero (only precision).')

    # If there are no true positives, fix the F score at 0 like sklearn.
    if K.sum(K.round(K.clip(y_true, 0, 1))) == 0:
        return 0

    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    bb = beta ** 2
    fbeta_score = (1 + bb) * (p * r) / (bb * p + r + K.epsilon())
    return fbeta_score

def fmeasure(y_true, y_pred):
    """Computes the f-measure, the harmonic mean of precision and recall.
    Here it is only computed as a batch-wise average, not globally.
    """
    return fbeta_score(y_true, y_pred, beta=1)

def f1_score(y_true, y_pred):

    # Count positive samples.
    c1 = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    c2 = K.sum(K.round(K.clip(y_pred, 0, 1)))
    c3 = K.sum(K.round(K.clip(y_true, 0, 1)))

    # # If there are no true samples, fix the F1 score at 0.
    # if c3 == 0:
    #     return 0

    # How many selected items are relevant?
    # tf.div(c1, c2)
    precision = tf.div(c1, c2)

    # How many relevant items are selected?
    # tf.div(c1, c3)
    recall = tf.div(c1, c3)

    # Calculate f1_score
    # f1_score = 2 * (precision * recall) / (precision + recall)
    temp = tf.constant([2.0])
    mul1 = tf.multiply(temp, precision)
    mul = tf.multiply(mul1, recall)
    add = tf.add(precision, recall)

    f1_score = tf.div(mul, add)
    return f1_score



# def pre_handle(y_true, y_pred):
#     tp = 0.0
#     tn = 0.0
#     fp = 0.0
#     fn = 0.0
#     for i in range(K.ndim(y_true)):
#         yt = K.get_value(y_true[i,:][0])
#         yp = K.get_value(y_pred[i,:][0])
#         if yt == yp and yt == 1:
#             tp += 1
#         elif yt == yp and yt == 0:
#             tn += 1
#         elif yt != yp and yp == 1:
#             fp += 1
#         elif yt != yp and yp == 0:
#             fn += 1
#     precison = tp/(tp+fp+K.epsilon())        
#     recall = tp/(tp+fn+K.epsilon())
#     true_negative_rate = tn/(tn+fp+K.epsilon())
#     f1 = (2*precison*recall)/(precison+recall+K.epsilon())
#     return [precison, recall, true_negative_rate, f1]


# def precison(y_true, y_pred):
#     return K.variable(value=pre_handle(y_true, y_pred)[0], dtype='float64')

# def recall( y_true, y_pred):
#     return K.variable(value=pre_handle(y_true, y_pred)[1], dtype='float64')

# def true_negative_rate( y_true, y_pred):
#     return K.variable(value=pre_handle(y_true, y_pred)[2], dtype='float64')

# def f1( y_true, y_pred):
#     return K.variable(value=pre_handle(y_true, y_pred)[3], dtype='float64')


