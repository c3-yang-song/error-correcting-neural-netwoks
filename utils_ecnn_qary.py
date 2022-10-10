from __future__ import print_function
import numpy as np
from tensorflow.keras import backend as K
from model_qary import *
import os

FLAGS = tf.compat.v1.app.flags.FLAGS
tf.compat.v1.app.flags.DEFINE_float('indv_CE_lamda', 1.0, "lamda for sum of individual CE")
tf.compat.v1.app.flags.DEFINE_float('log_det_lamda', .5, "lamda for non-ME")
tf.compat.v1.app.flags.DEFINE_float('div_lamda', .0, "lamda for non-ME")
tf.compat.v1.app.flags.DEFINE_bool('augmentation', True, "whether use data augmentation")
tf.compat.v1.app.flags.DEFINE_integer('num_models', 30, "The num of models in the ensemble")
tf.compat.v1.app.flags.DEFINE_integer('epochs', 200, "number of epochs")

tf.compat.v1.app.flags.DEFINE_string('save_dir', '/scratch/users/ntu/songy3/codematrix_v2/saved_models/', "where to save .h5")
tf.compat.v1.app.flags.DEFINE_string('cm_dir', '/scratch/users/ntu/songy3/codematrix_v2/saved_codematrix/', "where to load codematrix")
tf.compat.v1.app.flags.DEFINE_integer('batch_size', 128, "")
tf.compat.v1.app.flags.DEFINE_string('dataset', 'cifar10', "mnist or cifar10 or cifar100")


def lr_schedule(epoch):
    """Learning Rate Schedule

    Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
    Called automatically every epoch as part of callbacks during training.

    # Arguments
        epoch (int): The number of epochs

    # Returns
        lr (float32): learning rate
    """
    lr = 1e-3
    if epoch > 180:
        lr *= 0.5e-3
    elif epoch > 160:
        lr *= 1e-3
    elif epoch > 120:
        lr *= 1e-2
    elif epoch > 80:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr


def Entropy(input):
    #input shape is batch_size X num_class
    return tf.reduce_sum(-tf.multiply(input, tf.math.log(input+1e-20)), axis=-1)


def ens_div(y_true, y_pred, n, q):
    div = 0
    y_pb = tf.split(y_pred, [q]*n, axis=-1)
    for i in range(n):
        P = tf.nn.softmax(y_pb[i], axis=-1)
        div += Entropy(P)/np.log(q)
    return div/n


#def ens_div(y_true, y_pred, n, q):
#    div = 0
#    y_pb = tf.split(y_pred, [1]*n*q, axis=-1)
#    for i in range(n*q):
#        Pa = tf.nn.sigmoid(y_pb[i])
#        Pb = 1 - Pa
#        P = tf.concat([Pa, Pb], 1)
#        div += Entropy(P)
#    return div/n/q


def ce3(y_true, y_pred, n, q):
    y_tb = tf.split(y_true, [1]*n, axis=-1)
    y_pb = tf.split(y_pred, [q]*n, axis=-1)
    ce = 0
    for i in range(n):
        y_tb_i = tf.one_hot(tf.cast(tf.keras.backend.flatten(y_tb[i]), dtype=tf.int32), q, on_value=1.0, off_value=0.0, dtype=tf.float32)
        ce += tf.keras.losses.categorical_crossentropy(y_tb_i, tf.nn.softmax(y_pb[i], axis=1))
    return ce/n


def hinge_loss(y_true, y_pred, n, q, cfd_level):
    y_tb = tf.split(y_true, [1]*n, axis=-1)
    y_pb = tf.split(y_pred, [q]*n, axis=-1)
    hinge = 0
    for i in range(n):
        y_tb_i = tf.one_hot(tf.cast(tf.keras.backend.flatten(y_tb[i]), dtype=tf.int32), q, on_value=1.0, off_value=0.0, dtype=tf.float32)

        correct_logit = tf.reduce_sum(y_tb_i * y_pb[i], axis=1)
        wrong_logits = (1-y_tb_i) * y_pb[i] - y_tb_i*1e4
        wrong_logit = tf.reduce_max(wrong_logits, axis=1)

        # hinge += -tf.nn.relu(correct_logit - wrong_logit + 50)
        hinge += tf.nn.relu(wrong_logit - correct_logit + cfd_level)
    return hinge/n


#def hinge_loss(y_true, y_pred, n, q):
#    y_t = tf.split(y_true, [q]*n, axis=-1)
#    y_p = tf.split(y_pred, [q]*n, axis=-1)
#    hinge = 0
#    for i in range(n):
#        y_tb = tf.split(y_t, [1]*q, axis=-1)
#        y_pb = tf.split(y_p, [1]*q, axis=-1)
#        for j in range(q):
#            hinge += tf.keras.losses.hinge(y_tb[j], y_pb[j])
#    return hinge/n/q



def custom_loss(n, q, cfd_level, type):
    def loss(y_true, y_pred):
        if type == 'ce':
           total_loss = FLAGS.indv_CE_lamda*ce3(y_true, y_pred, n, q) + 0.1*hinge_loss(y_true, y_pred, n, q, cfd_level) - FLAGS.log_det_lamda * ens_div(y_true, y_pred, n, q) 
        if type == 'hinge':
           total_loss = FLAGS.indv_CE_lamda*hinge_loss(y_true, y_pred, n, q, cfd_level) - FLAGS.log_det_lamda * ens_div(y_true, y_pred, n, q) 
        return total_loss
    return loss


def ens_div_metric(n, q):
    def ens_div_(y_true, y_pred):
        div = ens_div(y_true, y_pred, n, q)
        return div
    return ens_div_


def ce_metric(n, q, type):
    def ce_acc(y_true, y_pred):
        y_t = tf.split(y_true, [1]*n, axis=-1)
        y_p = tf.split(y_pred, [q]*n, axis=-1)
        acc = 0
        for i in range(n):
            y_t_i = tf.one_hot(tf.cast(tf.keras.backend.flatten(y_t[i]), dtype=tf.int32), q, on_value=1.0, off_value=0.0, dtype=tf.float32)
            acc += tf.keras.metrics.categorical_accuracy(y_t_i, tf.nn.softmax(y_p[i], axis=1))
        return acc / n
    return ce_acc


#def ce_metric(n, q, type):
#    def ce_acc(y_true, y_pred):
#        y_t = tf.split(y_true, [1]*n*q, axis=-1)
#        y_p = tf.split(y_pred, [1]*n*q, axis=-1)
#        acc = 0
#        for i in range(n*q):
#            acc += tf.keras.metrics.binary_accuracy((y_t[i]+1)/2, tf.nn.sigmoid(y_p[i]), threshold=0.5) 
#        return acc / n / q
#    return ce_acc
