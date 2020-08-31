import argparse
import os
import time
from datetime import timedelta
from math import cos, pi

import cv2
import keras
import keras.backend as K
import numpy as np
import tensorflow as tf
from keras.callbacks import (LearningRateScheduler, ModelCheckpoint,
                             ReduceLROnPlateau, TensorBoard)
from keras.optimizers import Adam
from keras.utils import multi_gpu_model

import constants
from generator import segmentationGenerator
from loss import modelLoss
from resnet_model import create_Model

# DEFAULT_DATA_DIR = './data'
# DEFAULT_RUNS_DIR = './runs'
# DEFAULT_MODEL_PATH = "models/model.ckpt"
DEFAULT_EPOCHS = 20
DEFAULT_BATCH_SIZE = 12
resnet_type = 18

argparser = argparse.ArgumentParser(description='Training')
# argparser.add_argument('-d',
#                        '--dataset',
#                        default=DEFAULT_DATA_DIR,
#                        help='path to dataset')
# argparser.add_argument('-r',
#                        '--runs',
#                        default=DEFAULT_RUNS_DIR,
#                        help='path to saved directory')
# argparser.add_argument('-m',
#                        '--model',
#                        default=DEFAULT_MODEL_PATH,
#                        help='path to save model')
argparser.add_argument('-e',
                       '--epochs',
                       default=DEFAULT_EPOCHS,
                       help='number of epochs')
argparser.add_argument('-b',
                       '--batch',
                       default=DEFAULT_BATCH_SIZE,
                       help='batch size')
argparser.add_argument('-r',
                       '--resnet',
                       default=resnet_type,
                       help='resnet type')

# convert string arguments to appropriate type
args = argparser.parse_args()
args.epochs = int(args.epochs)
args.batch = int(args.batch)
args.resnet = int(args.resnet)

print("\nTensorFlow detected the following GPU(s):")
tf.test.gpu_device_name()

print("\n\nSetup start: {}\n".format(time.ctime()))
setup_start = time.time()

# model naming parameter
trainingRunTime = time.ctime().replace(':', '_')

if constants.use_unet:
    Notes = 'KITTI_Road_UNet'
else:
    Notes = 'KITTI_Road'

# build loss
lossClass = modelLoss(0.001,0.85,640,192,args.batch)
loss = lossClass.applyLoss 

# build data generators
train_generator = segmentationGenerator(constants.data_train_image_dir, constants.data_train_gt_dir, batch_size=args.batch, shuffle=True)
test_generator = segmentationGenerator(constants.data_train_image_dir, constants.data_train_gt_dir, batch_size=args.batch, shuffle=True, test=True)

# build model
model = create_Model(input_shape=(640,192,3), encoder_type=args.resnet)
model.compile(optimizer=Adam(lr=1e-3),loss=loss, metrics=[loss, 'accuracy'])

modelSavePath = 'models/' + Notes + '_' + trainingRunTime +  '_batchsize_' + str(args.batch) + '_resnet_' + str(args.resnet) + '/_weights_epoch{epoch:02d}_val_loss_{val_loss:.4f}_train_loss_{loss:.4f}.hdf5'

# callbacks
if not os.path.exists('models/' + Notes + '_' + trainingRunTime + '_batchsize_' + str(args.batch) + '_resnet_' + str(args.resnet) + '/'):
    os.makedirs('models/' + Notes + '_' + trainingRunTime + '_batchsize_' + str(args.batch) + '_resnet_' + str(args.resnet) + '/')
mc = ModelCheckpoint(modelSavePath, monitor='val_loss')
mc1 = ModelCheckpoint(modelSavePath, monitor='loss')
rl = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2, verbose=1) # not used
tb = TensorBoard(log_dir='logs/' + Notes + '_' + trainingRunTime + '_batchsize_' + str(args.batch) + '_resnet_' + str(args.resnet), histogram_freq=0, write_graph=True, write_images=True)

# Schedule Learning rate Callback
def lr_schedule(epoch):
    if epoch < 15:
        return 1e-3 
    else:
        return 1e-4

lr = LearningRateScheduler(schedule=lr_schedule,verbose=1)

print("Model saved to:")
print(modelSavePath)

print("\n\nTraining start: {}\n".format(time.ctime()))
training_start = time.time()

model.fit_generator(train_generator, epochs=args.epochs, validation_data=test_generator, callbacks=[mc,mc1,lr,tb], initial_epoch=0)

print("\n\nTraining end:   {}\n".format(time.ctime()))
print("Model saved to: {}".format(modelSavePath))
training_end = time.time()

setup_time = training_start - setup_start
training_time = training_end - training_start

print("Total setup time: {}".format(str(timedelta(seconds=setup_time))))
print("Total train time: {}".format(str(timedelta(seconds=training_time))))
