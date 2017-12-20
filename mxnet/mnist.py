#!/usr/bin/python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import time
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.tensorboard.plugins import projector

STARTER_LEARNING_RATE = 1e-4
BATCH_SIZE = 10
NUM_CLASSES = 10
MAX_STEPS = 10000
IMAGE_SIZE = 28
IMAGE_PIXELS = IMAGE_SIZE ** 2

def WeightsVariable(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1, name='weights'))

def BiasVariable(shape):
    return tf.Variable(tf.constant(0.1, shape=shape, name='biases'))

def Conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def MaxPool2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

def network(images):
    # Reshape
    with tf.name_scope('reshape'):
        images_reshape = tf.reshape(images, [-1, 28, 28, 1])

    # Convolution 1
    with tf.name_scope('conv1'):
        weights = WeightsVariable([5,5,1,20])
        biases = BiasVariable([20])
        conv1 = tf.nn.relu(Conv2d(images_reshape, weights) + biases)
        pool1 = MaxPool2x2(conv1)

    # Convolution 2
    with tf.name_scope('conv2'):
        weights = WeightsVariable([5,5,20,50])
        biases = BiasVariable([50])
        conv2 = tf.nn.relu(Conv2d(pool1, weights) + biases)
        pool2 = MaxPool2x2(conv2)
        pool2_flat = tf.reshape(pool2, [-1, 800])

    # Fully Connected 1
    with tf.name_scope('fc1'):
        weights = WeightsVariable([800, 500])
        biases = BiasVariable([500])
        fc1 = tf.nn.relu(tf.matmul(pool2_flat, weights) + biases)

    # Fully Connected 2
    with tf.name_scope('fc2'):
        weights = WeightsVariable([500, 10])
        biases = BiasVariable([10])
        fc2 = tf.nn.relu(tf.matmul(fc1, weights) + biases)

    return fc2

def LossMetrics(logits, labels):
    labels = tf.to_int64(labels)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits, name='softmax')
    return tf.reduce_mean(cross_entropy, name='softmax_mean')

def training(loss):
    tf.summary.scalar('loss', loss)
    global_step = tf.Variable(0, name='global_step', trainable=False)
    learning_rate = tf.train.exponential_decay(STARTER_LEARNING_RATE, global_step, 100000, 0.75, staircase=True)
    tf.summary.scalar('learning rate', learning_rate)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op

def evaluation(logits, labels):
    correct = tf.nn.in_top_k(logits, labels, 1)
    return tf.reduce_sum(tf.cast(correct, tf.int32))

def PlaceholderInputs(batch_size):
    images_placeholder = tf.placeholder(tf.float32, shape=(batch_size, IMAGE_PIXELS))
    labels_placeholder = tf.placeholder(tf.int32, shape=(batch_size))
    return images_placeholder, labels_placeholder

def fill_feed_dict(data_set, images_pl, labels_pl):
    images_feed, labels_feed = data_set.next_batch(BATCH_SIZE)
    feed_dict = {
        images_pl: images_feed,
        labels_pl: labels_feed,
    }
    return feed_dict

def do_eval(sess,
            eval_correct,
            images_placeholder,
            labels_placeholder,
            data_set,
            summary):

    true_count = 0
    steps_per_epoch = data_set.num_examples // BATCH_SIZE
    num_examples = steps_per_epoch * BATCH_SIZE
    for step in range(steps_per_epoch):
        feed_dict = fill_feed_dict(data_set,
                                   images_placeholder,
                                   labels_placeholder)
        log, correctness = sess.run([summary, eval_correct], feed_dict=feed_dict)
        true_count += correctness
    precision = float(true_count) / num_examples
    tf.summary.scalar('prcecision'), tf.constanct(precision)
    print('Num examples %d, Num Correct: %d Precision @ 1: %0.04f' % (num_examples, true_count, precision))
    return log

def run_training():
    data_sets = input_data.read_data_sets('/tmp/tensorflow/mnist/input_data')
    with tf.Graph().as_default():
        images_placeholder, labels_placeholder = placeholder_inputs(BATCH_SIZE)
        logits = network(images_placeholder)
        loss = LossMetrics(logits, labels_placeholder)
        train_op = training(loss)
        eval_correct = evaluation(logits, labels_placeholder)
        summary = tf.summary.merge_all()
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        sess = tf.Session()
        summary_writer = tf.summary.FileWriter("/tmp/tensorflow/mnist/log", sess.log)
        test_writer = tf.summary.FileWriter("/tmp/tensorflow/mnist/log/validation")
        sess.run(init)
        for step in range(MAX_STEPS):
            start_time = time.time()
            feed_dict = fill_feed_dict(data_sets.train,
                                       images_placeholder,
                                       labels_placeholder)
            _, loss_value = sess.run([train_op, loss], feed_dict=feed_dict)
            duration = time.time() - start_time
            if step % 100 == 0:
                print('Step %d: loss = %.2f (%.3f sec)' % (step, loss_value, duration))
                summary_str = sess.run(summary, feed_dict=feed_dict)
                summary_writer.add_summary(summary_str, step)
                summary_writer.flush()
            if (step + 1) % 1000 == 0 or (step + 1) == MAX_STEPS:
                checkpoint_file = os.path.join("/tmp/tensorflow/mnist/log", "model.ckpt")
                saver.save(sess, checkpoint_file, global_step=step)
                print('Validation Data Eval:')
                log = do_eval(sess,
                              eval_correct,
                              images_placeholder,
                              labels_placeholder,
                              data_sets.validation,
                              summary)
                test_writer.add_summary(log, step)

if __name__ == "__main__":
    run_training()
