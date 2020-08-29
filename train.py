import tensorflow as tf
import argparse
import os.path
import numpy as np
import helper
from src.batch import gen_batch_function
import tensorflow.contrib.slim as slim
import time

data_path = './data'
runs_path = './runs'
model_path = "models/model.ckpt"
epochs_default = 40
batch_size_default = 3

# KITTI dataset size: 1242 x 375
# VGG16 input size: 244 x 244 x 3
image_shape = (224, 741)
num_classes = 2

argparser = argparse.ArgumentParser(description='Training')
argparser.add_argument('-d',
                       '--dataset',
                       default=data_path,
                       help='path to dataset')
argparser.add_argument('-r',
                       '--runs',
                       default=runs_path,
                       help='path to saved directory')
argparser.add_argument('-m',
                       '--model',
                       default=model_path,
                       help='path to save model')
argparser.add_argument('-e',
                       '--epochs',
                       default=epochs_default,
                       help='number of epochs')
argparser.add_argument('-b',
                       '--batch',
                       default=batch_size_default,
                       help='batch size')

def load_model_ckpt(sess, ckpt='ckpts/vgg_16.ckpt'):
    variables = slim.get_variables(scope='vgg_16', suffix="weights") + slim.get_variables(scope='vgg_16', suffix="biases")
    init_assign_op, init_feed_dict = slim.assign_from_checkpoint(ckpt, variables)
    sess.run(init_assign_op, init_feed_dict)

def train():
    args = argparser.parse_args()

    x_placeholder = tf.placeholder(tf.float32, [None, image_shape[0], image_shape[1], 3])
    y_placeholder = tf.placeholder(tf.float32, [None, image_shape[0], image_shape[1], num_classes])
    lr_placeholder = tf.placeholder(tf.float32)
    is_train_placeholder = tf.placeholder(tf.bool)
    fcn_model = FcnModel(x_placeholder, y_placeholder, is_train_placeholder, num_classes)