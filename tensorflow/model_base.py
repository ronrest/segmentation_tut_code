""" ############################################################################
This code is released as part of a tutorial on semantic segmentation located:
    http://ronny.rest/tutorials/
    TODO: Add link to specific tutorial page

author    : Ronny Restrepo
copyright : Copyright 2017, Ronny Restrepo
license   : Apache License
version   : 2.0

ABOUT:
    Contains the ClassifierModel class. Which contains all the
    boilerplate code necessary to Create a tensorlfow graph, and training
    operations.
################################################################################
"""
import tensorflow as tf
import numpy as np
import os
import shutil
import time
import pickle
from viz import batch_vizseg


# ##############################################################################
#                                                             SEGMENTATION MODEL
# ##############################################################################
class SegmentationModel(object):
    """
    Examples:
        # Creating a Model that inherits from this class:

        class MyModel(ImageClassificationModel):
            def __init__(self, name, img_shape, n_channels=3, n_classes=10, dynamic=False, l2=None, best_evals_metric="valid_acc"):
                super().__init__(name=name, img_shape=img_shape, n_channels=n_channels, n_classes=n_classes, dynamic=dynamic, l2=l2, best_evals_metric=best_evals_metric)

            def create_body_ops(self):
                ...
                self.logits = ...
    """
    def __init__(self, img_shape, n_channels=3, n_classes=10):
        """ Initializes a Segmentation Model Class """

        # MODEL SETTINGS
        self.batch_size = 4
        self.img_shape = img_shape
        self.img_width, self.img_height = img_shape
        self.n_channels = n_channels
        self.n_classes = n_classes

        # DIRECTORIES TO STORE OUTPUTS
        self.snapshot_file = os.path.join("snapshots", "snapshot.chk")
        self.tensorboard_dir = "tensorboard"

        if not os.path.exists(self.tensorboard_dir):
            os.makedirs(self.tensorboard_dir)
        if not os.path.exists("snapshots"):
            os.makedirs("snapshots")
        if not os.path.exists("samples"):
            os.makedirs("samples")

    def create_graph_from_logits_func(self, logits_func):
        """ Creates the graph for the model, given a logits function with
            the following API:

                Arguments: (X, n_classes, alpha, dropout, is_training)
                Returns  : logits [n_batch, img_height, img_width, n_classes]
        """
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.create_input_ops()
            self.logits = logits_func(X=self.X,
                                      n_classes=self.n_classes,
                                      alpha=self.alpha,
                                      dropout=self.dropout,
                                      is_training=self.is_training)

            with tf.name_scope("preds") as scope:
                self.preds = tf.to_int32(tf.argmax(self.logits, axis=-1), name=scope)

            self.create_loss_ops()
            self.create_optimization_ops()
            self.create_evaluation_metric_ops()
            self.create_saver_ops()
            self.create_tensorboard_ops()

    def create_input_ops(self):
        with tf.variable_scope("inputs"):
            X_shape = (None, self.img_height, self.img_width, self.n_channels)
            Y_shape = (None, self.img_height, self.img_width)
            self.X = tf.placeholder(tf.float32, shape=X_shape, name="X")
            self.Y = tf.placeholder(tf.int32, shape=Y_shape, name="Y")
            self.alpha = tf.placeholder_with_default(0.001, shape=None, name="alpha")
            self.dropout = tf.placeholder_with_default(0.0, shape=None, name="dropout")
            self.is_training = tf.placeholder_with_default(False, shape=(), name="is_training")

    def create_evaluation_metric_ops(self):
        # EVALUATION METRIC - IoU
        with tf.name_scope("evaluation") as scope:
            # Define the evaluation metric and update operations
            self.evaluation, self.update_evaluation_vars = tf.metrics.mean_iou(
                tf.reshape(self.Y, [-1]),
                tf.reshape(self.preds, [-1]),
                num_classes=self.n_classes,
                name=scope)
            # Isolate metric's running variables & create their initializer/reset op
            evaluation_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope=scope)
            self.reset_evaluation_vars = tf.variables_initializer(var_list=evaluation_vars)

    def create_loss_ops(self):
        # LOSS - Sums all losses even Regularization losses automatically
        with tf.variable_scope('loss') as scope:
            unrolled_logits = tf.reshape(self.logits, (-1, self.n_classes))
            unrolled_labels = tf.reshape(self.Y, (-1,))
            tf.losses.sparse_softmax_cross_entropy(labels=unrolled_labels, logits=unrolled_logits)
            self.loss = tf.losses.get_total_loss()

    def create_optimization_ops(self):
        # OPTIMIZATION - Also updates batchnorm operations automatically
        with tf.variable_scope('opt') as scope:
            self.optimizer = tf.train.AdamOptimizer(self.alpha, name="optimizer")
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.train_op = self.optimizer.minimize(self.loss, name="train_op")

    def create_saver_ops(self):
        with tf.device('/cpu:0'):
            self.saver = tf.train.Saver(name="saver")

    def initialize_vars(self, session):
        if tf.train.checkpoint_exists(self.snapshot_file):
            print("- Restoring parameters from saved snapshot")
            print("  -", self.snapshot_file)
            self.saver.restore(session, self.snapshot_file)
        else:
            print("Initializing weights to random values")
            session.run(tf.global_variables_initializer())

    def create_tensorboard_ops(self):
        with tf.variable_scope('tensorboard') as scope:
            self.summary_writer = tf.summary.FileWriter(self.tensorboard_dir, graph=self.graph)
            self.dummy_summary = tf.summary.scalar(name="dummy", tensor=1)

    def train(self, data, n_epochs, alpha=0.001, dropout=0.0, batch_size=32, print_every=10, viz_every=1):
        """Trains the model, for n_epochs given a dictionary of data"""
        n_samples = len(data["X_train"])               # Num training samples
        n_batches = int(np.ceil(n_samples/batch_size)) # Num batches per epoch

        with tf.Session(graph=self.graph) as sess:
            self.initialize_vars(sess)

            for epoch in range(1, n_epochs+1):
                print("\nEPOCH {}/{}".format(epoch, n_epochs))

                # Iterate through each mini-batch
                for i in range(n_batches):
                    X_batch = data["X_train"][batch_size*i: batch_size*(i+1)]
                    Y_batch = data["Y_train"][batch_size*i: batch_size*(i+1)]

                    # TRAIN
                    feed_dict = {self.X:X_batch, self.Y:Y_batch, self.alpha:alpha, self.is_training:True, self.dropout: dropout}
                    loss, _ = sess.run([self.loss, self.train_op], feed_dict=feed_dict)

                    # Print feedback every so often
                    if print_every is not None and (i+1)%print_every==0:
                        print("{: 5d} Batch_loss: {}".format(i, loss))

                # Save snapshot
                self.saver.save(sess, self.snapshot_file)

                # Evaluate on train and validation sets after each epoch
                train_iou, train_loss = self.evaluate(data["X_train"][:1024], data["Y_train"][:1024], sess, batch_size=batch_size)
                valid_iou, valid_loss = self.evaluate(data["X_valid"], data["Y_valid"], sess, batch_size=batch_size)

                # Print evaluations
                s = "TR IOU: {: 3.3f} VA IOU: {: 3.3f} TR LOSS: {: 3.5f} VA LOSS: {: 3.5f}\n"
                print(s.format(train_iou, valid_iou, train_loss, valid_loss))

                # VISUALIZE PREDICTIONS - once every so many epochs
                if epoch%viz_every==0:
                    self.visualise_semgmentations(data=data, i=epoch, session=sess)

    def predict(self, X, session, batch_size=32):
        """ Make predictions on data `X` within a running session"""
        # Dimensions
        n_samples = X.shape[0]
        n_batches = int(np.ceil(n_samples/batch_size))
        out_shape = [n_samples, self.img_height, self.img_width]
        preds = np.zeros(out_shape, dtype=np.uint8)

        # MAKE PREDICTIONS ON MINI BATCHES
        for i in range(n_batches):
            X_batch = X[batch_size*i: batch_size*(i+1)]
            feed_dict = {self.X:X_batch, self.is_training:False}
            batch_preds = session.run(self.preds, feed_dict=feed_dict)
            preds[batch_size*i: batch_size*(i+1)] = batch_preds.squeeze()

        return preds

    def evaluate(self, X, Y, session, batch_size=32):
        """Given input X, and Labels Y, evaluate the accuracy of the model"""
        total_loss = 0
        n_samples = len(Y)
        n_batches = int(np.ceil(n_samples/batch_size)) # Num batches needed

        # Reset the running variables for evaluation metric
        session.run(self.reset_evaluation_vars)

        for i in range(n_batches):
            X_batch = X[batch_size*i: batch_size*(i+1)]
            Y_batch = Y[batch_size*i: batch_size*(i+1)]
            feed_dict = {self.X:X_batch, self.Y:Y_batch, self.is_training:False}

            # Get loss, and update running variables for evaluation metric
            loss, preds, confusion_mtx = session.run([self.loss, self.preds, self.update_evaluation_vars], feed_dict=feed_dict)
            total_loss += loss

        # Get the updated score from the running metric
        score = session.run(self.evaluation)
        # Average the loss
        avg_loss = total_loss/float(n_batches)

        return score, avg_loss

    def visualise_semgmentations(self, data, session, i=0, batch_size=4, shape=[2,8]):
        viz_rows, viz_cols = shape
        n_viz = viz_rows * viz_cols
        viz_img_template = os.path.join("samples", "{}_epoch_{:07d}.jpg")

        # On train data
        preds = self.predict(data["X_train"][:n_viz], session=session, batch_size=batch_size)
        batch_vizseg(data["X_train"][:n_viz],
                     labels=data["Y_train"][:n_viz],
                     labels2=preds[:n_viz],
                     colormap=data.get("colormap", None),
                     gridsize=shape,
                     saveto=viz_img_template.format("train", i)
                     )

        # On validation Data
        preds = self.predict(data["X_valid"][:n_viz], session=session, batch_size=batch_size)
        batch_vizseg(data["X_valid"][:n_viz],
                     labels=data["Y_valid"][:n_viz],
                     labels2=preds[:n_viz],
                     colormap=data.get("colormap", None),
                     gridsize=shape,
                     saveto=viz_img_template.format("valid", i)
                     )
