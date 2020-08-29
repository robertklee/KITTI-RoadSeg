import os
import tensorflow as tf
import keras 
import cv2
import numpy as np
from math import cos, pi
import keras.backend as K
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, TensorBoard,LearningRateScheduler
from keras.utils import multi_gpu_model

from resnet_model import create_Model
# from lossFunctions import monoDepthV2Loss
from generator import segmentationGenerator

# define these
batchSize = 12
trainingRunDate = '2020_08_20'
Notes = 'KITTI_Road'

# build data generators
train_generator = segmentationGenerator('data/data_road/training/image_2','data/data_road/training/gt_image_2', batch_size=batchSize, shuffle=True)
test_generator = segmentationGenerator('data/data_road/testing/image_2','data/data_road/testing/gt_image_2', batch_size=batchSize, shuffle=True)

# build model
model = create_Model(input_shape=(640,192,3), encoder_type=18)
model.compile(optimizer=Adam(lr=1e-3),loss='categorical_crossentropy', metrics=['accuracy'])

# callbacks
if not os.path.exists('models/' + Notes + '_' + trainingRunDate + '_batchsize_' + str(batchSize) + '/'):
    os.makedirs('models/' + Notes + '_' + trainingRunDate + '_batchsize_' + str(batchSize) + '/')
mc = ModelCheckpoint('models/' + Notes + '_' + trainingRunDate +  '_batchsize_' + str(batchSize) + '/_weights_epoch{epoch:02d}_val_loss_{val_loss:.4f}_train_loss_{loss:.4f}.hdf5', monitor='val_loss')
mc1 = ModelCheckpoint('models/' + Notes + '_' + trainingRunDate +  '_batchsize_' + str(batchSize) + '/_weights_epoch{epoch:02d}_val_loss_{val_loss:.4f}_train_loss_{loss:.4f}.hdf5', monitor='loss')
rl = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2, verbose=1) # not used
tb = TensorBoard(log_dir='logs/' + Notes + '_' + trainingRunDate + '_batchsize_' + str(batchSize), update_freq=250)

# Schedule Learning rate Callback
def lr_schedule(epoch):
    if epoch < 15:
        return 1e-3 
    else:
        return 1e-4

lr = LearningRateScheduler(schedule=lr_schedule,verbose=1)

model.fit_generator(train_generator, epochs = 20, validation_data=test_generator, callbacks=[mc,mc1,lr,tb], initial_epoch=0)
