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
import time
import argparse

from resnet_model import create_Model
from loss import modelLoss
from generator import segmentationGenerator

DEFAULT_DATA_DIR = './data'
DEFAULT_RUNS_DIR = './runs'
DEFAULT_MODEL_PATH = "models/model.ckpt"
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

print("**************************************\nTensorFlow detected the following GPU(s):")
tf.test.gpu_device_name()

print("\n\nSetup start: {}\n".format(time.ctime()))

# define these
trainingRunTime = time.ctime().replace(':', '_')

Notes = 'KITTI_Road'

# build loss
lossClass = modelLoss(0.001,0.85,640,192,args.batch)
loss = lossClass.applyLoss 

# build data generators
train_generator = segmentationGenerator('data/data_road/training/image_2','data/data_road/training/gt_image_2', batch_size=args.batch, shuffle=True)
test_generator = segmentationGenerator('data/data_road/training/image_2','data/data_road/training/gt_image_2', batch_size=args.batch, shuffle=True, test=True)

# build model
model = create_Model(input_shape=(640,192,3), encoder_type=args.resnet)
model.compile(optimizer=Adam(lr=1e-3),loss=loss, metrics=[loss, 'accuracy'])

modelSavePath = 'models/' + Notes + '_' + trainingRunTime +  '_batchsize_' + str(args.batch) + '_resnet_' + str(resnet_type) + '/_weights_epoch{epoch:02d}_val_loss_{val_loss:.4f}_train_loss_{loss:.4f}.hdf5'

# callbacks
if not os.path.exists('models/' + Notes + '_' + trainingRunTime + '_batchsize_' + str(args.batch) + '_resnet_' + str(resnet_type) + '/'):
    os.makedirs('models/' + Notes + '_' + trainingRunTime + '_batchsize_' + str(args.batch) + '_resnet_' + str(resnet_type) + '/')
mc = ModelCheckpoint(modelSavePath, monitor='val_loss')
mc1 = ModelCheckpoint(modelSavePath, monitor='loss')
rl = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2, verbose=1) # not used
tb = TensorBoard(log_dir='logs/' + Notes + '_' + trainingRunTime + '_batchsize_' + str(args.batch) + '_resnet_' + str(resnet_type), update_freq=250)

# Schedule Learning rate Callback
def lr_schedule(epoch):
    if epoch < 15:
        return 1e-3 
    else:
        return 1e-4

lr = LearningRateScheduler(schedule=lr_schedule,verbose=1)

print("\n\nTraining start: {}\n".format(time.ctime()))

model.fit_generator(train_generator, epochs=args.epochs, validation_data=test_generator, callbacks=[mc,mc1,lr,tb], initial_epoch=0)

print("\n\nTraining end: {}\n".format(time.ctime()))
print("Model saved to:")
print(modelSavePath)