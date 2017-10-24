""" ############################################################################
This code is released as part of a tutorial on semantic segmentation located:
    http://ronny.rest/tutorials/
    TODO: Add link to specific tutorial page

author    : Ronny Restrepo
copyright : Copyright 2017, Ronny Restrepo
license   : Apache License
version   : 2.0
################################################################################
"""
from __future__ import print_function, division
import pickle
import tensorflow as tf
import numpy as np

from model_base import SegmentationModel


# LAYER OPERATION SHORTCUTS
conv = tf.contrib.layers.conv2d
# convsep = tf.contrib.layers.separable_conv
deconv = tf.contrib.layers.conv2d_transpose
relu = tf.nn.relu
# maxpool = tf.contrib.layers.max_pool2d
dropout_layer = tf.layers.dropout
batchnorm = tf.contrib.layers.batch_norm
# winit = tf.contrib.layers.xavier_initializer()
# repeat = tf.contrib.layers.repeat
arg_scope = tf.contrib.framework.arg_scope
# l2_regularizer = tf.contrib.layers.l2_regularizer


# ==============================================================================
#                                                                     PICKLE2OBJ
# ==============================================================================
def pickle2obj(file):
    """ Loads the contents of a pickle as a python object. """
    with open(file, mode = "rb") as fileObj:
        obj = pickle.load(fileObj)
    return obj

# ==============================================================================
#                                                                       GET_DATA
# ==============================================================================
data = pickle2obj("data64_flat_grey.pickle")

print("DATA SHAPES")
print("- X_train: ", data["X_train"].shape)
print("- Y_train: ", data["Y_train"].shape)
print("- X_valid: ", data["X_valid"].shape)
print("- Y_valid: ", data["Y_valid"].shape)
print("- X_test : ", data["X_test"].shape)
print("- Y_test : ", data["Y_test"].shape)


# ==============================================================================
#                                                                   MODEL_LOGITS
# ==============================================================================
def model_logits(X, n_classes, alpha, dropout, is_training):
    with tf.name_scope("preprocess") as scope:
        x = tf.div(X, 255., name="rescaled_inputs")

    # DOWN CONVOLUTIONS
    with tf.contrib.framework.arg_scope(\
        [conv], \
        padding = "SAME",
        stride = 2,
        activation_fn = relu,
        normalizer_fn = batchnorm,
        normalizer_params = {"is_training": is_training},
        weights_initializer =tf.contrib.layers.xavier_initializer(),
        ):
        with tf.variable_scope("d1") as scope:
            d1 = conv(x, num_outputs=32, kernel_size=3, stride=1, scope="conv1")
            d1 = conv(d1, num_outputs=32, kernel_size=3, scope="conv2")
            d1 = dropout_layer(d1, rate=dropout, name="dropout")
            print("d1", d1.shape.as_list())
        with tf.variable_scope("d2") as scope:
            d2 = conv(d1, num_outputs=64, kernel_size=3, stride=1, scope="conv1")
            d2 = conv(d2, num_outputs=64, kernel_size=3, scope="conv2")
            d2 = dropout_layer(d2, rate=dropout, name="dropout")
            print("d2", d2.shape.as_list())
        with tf.variable_scope("d3") as scope:
            d3 = conv(d2, num_outputs=128, kernel_size=3, stride=1, scope="conv1")
            d3 = conv(d3, num_outputs=128, kernel_size=3, scope="conv2")
            d3 = dropout_layer(d3, rate=dropout, name="dropout")
            print("d3", d3.shape.as_list())
        with tf.variable_scope("d4") as scope:
            d4 = conv(d3, num_outputs=256, kernel_size=3, stride=1, scope="conv1")
            d4 = conv(d4, num_outputs=256, kernel_size=3, scope="conv2")
            d4 = dropout_layer(d4, rate=dropout, name="dropout")
            print("d4", d4.shape.as_list())

    # UP CONVOLUTIONS
    with tf.contrib.framework.arg_scope([deconv, conv], \
        padding = "SAME",
        activation_fn = None,
        normalizer_fn = tf.contrib.layers.batch_norm,
        normalizer_params = {"is_training": is_training},
        weights_initializer = tf.contrib.layers.xavier_initializer(),
        ):
        with tf.variable_scope('u3') as scope:
            u3 = deconv(d4, num_outputs=n_classes, kernel_size=4, stride=2)
            s3 = conv(d3, num_outputs=n_classes, kernel_size=1, stride=1, activation_fn=relu, scope="s")
            u3 = tf.add(u3, s3, name="up")
            print("u3", u3.shape.as_list())

        with tf.variable_scope('u2') as scope:
            u2 = deconv(u3, num_outputs=n_classes, kernel_size=4, stride=2)
            s2 = conv(d2, num_outputs=n_classes, kernel_size=1, stride=1, activation_fn=relu, scope="s")
            u2 = tf.add(u2, s2, name="up")
            print("u2", u2.shape.as_list())

        with tf.variable_scope('u1') as scope:
            u1 = deconv(u2, num_outputs=n_classes, kernel_size=4, stride=2)
            s1 = conv(d1, num_outputs=n_classes, kernel_size=1, stride=1, activation_fn=relu, scope="s")
            u1 = tf.add(u1, s1, name="up")
            print("u1", u1.shape.as_list())

        logits = deconv(u1, num_outputs=n_classes, kernel_size=4, stride=2, activation_fn=None, normalizer_fn=None, scope="logits")
    return logits

# Create model and train
model = SegmentationModel(img_shape=[64,64], n_channels=1, n_classes=4)
model.create_graph_from_logits_func(model_logits)
model.train(data, n_epochs=20, alpha=0.0001, batch_size=32, print_every=10)
