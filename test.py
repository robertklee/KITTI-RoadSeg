import os
import tensorflow as tf
import keras 
import cv2
import random
import numpy as np
from math import cos, pi
import keras.backend as K
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, TensorBoard, LearningRateScheduler
from keras.utils import multi_gpu_model

from resnet_model import create_Model
from loss import modelLoss
from generator import segmentationGenerator
import constants
import argparse
from helpers import make_overlay
import re
import matplotlib.pyplot as plt


'''
TODO 
add command line arguments to test.py
re-evaluate loss function 
overlay image 
examine model output since the values are very large
quick way to refer to a specific epoch checkpoint file
evaluation metrics for generated output

linear histogram equilization
'''

argparser = argparse.ArgumentParser(description='Testing')

model_epoch_number = 20#16
resnet_type = 18
session_id = 'KITTI_Road_UNet_Sun Aug 30 19_52_21 2020_batchsize_25_resnet_18'
batchSize = 1
model_epoch_base = '_weights_epoch'
output_img_base_dir = 'output'
model_base_dir = 'models'
visualize = False

argparser.add_argument('-e',
                       '--epoch',
                       default=model_epoch_number,
                       help='model epoch number')
argparser.add_argument('-s',
                       '--session',
                       default=session_id,
                       help='session id number')
argparser.add_argument('-b',
                       '--batch',
                       default=batchSize,
                       help='batch size')
argparser.add_argument('-r',
                       '--resnet',
                       default=resnet_type,
                       help='resnet type')
argparser.add_argument('-v',
                       '--visualize',
                       default=visualize,
                       help='enable visualize')

args = argparser.parse_args()

# convert string arguments to appropriate type
if type(args.visualize) is not bool:
    args.visualize = args.visualize == '1' or args.visualize.lower() == 'true'
args.epochs = int(args.epoch)
args.batch = int(args.batch)
args.resnet = int(args.resnet)

output_img_path = os.path.join(output_img_base_dir, args.session, str(args.epoch))
model_path = os.path.join(model_base_dir, args.session)

def get_model_name_from_epoch(src, epoch):
    models = os.listdir(src)

    pattern = re.compile(model_epoch_base + "[a-zA-Z_]*[0]*" + str(args.epoch))

    for modelName in models:
        if pattern.search(modelName):
            return modelName
    
    return None


model_name = get_model_name_from_epoch(model_path, args.epoch)

if model_name is None:
    print("Cannot find model corresponding to model_path: '" + model_path + "' and epoch " + str(args.epoch))
    exit(1)

# build loss
lossClass = modelLoss(0.001,0.85,640,192,batchSize)
loss = lossClass.applyLoss 

# build model
model = create_Model(input_shape=(640,192,3), encoder_type=args.resnet)

model.compile(optimizer=Adam(lr=1e-3),loss=loss, metrics=['accuracy'])

def modelPredictWrapper(model, inputImg):
    modelOutput = model.predict(np.expand_dims(inputImg,0))# * 640 * 0.3

    if constants.use_unet:
        # change from (1, 640, 192, 2) to (1, 640, 192, 3)
        zeros = np.zeros([modelOutput.shape[0], modelOutput.shape[1], modelOutput.shape[2], 1])
        modelOutput = np.concatenate((modelOutput, zeros), axis=3)
    
    return modelOutput

def evaluateModel(model,batchSize, visualize):
    val_generator  = segmentationGenerator('data/data_road/training/image_2','data/data_road/training/gt_image_2', batch_size=args.batch, shuffle=False, augmentations=False)
    # scores = model.evaluate_generator(val_generator, verbose=1)
    # print("Total Loss")
    # print(scores)
    ARD = 0
    count = 0 
    ABS = 0 
    SQR = 0 
    # Random Qualitative Evaluation
    imageList = os.listdir('data/data_road/testing/image_2/')
    if constants.system_files in imageList:
        imageList.remove(constants.system_files)
    imageName = random.choice(imageList)
    inputImg = cv2.imread('data/data_road/testing/image_2/' + imageName)
    rawImage = cv2.resize(inputImg, (640,192))
    inputImgOrig  = cv2.resize(cv2.imread('data/data_road/testing/image_2/' + imageName), (640,192))
    inputImg  = np.transpose(inputImgOrig.astype('float32'), axes=[1,0,2])
    output = modelPredictWrapper(model, inputImg)
    def displayOutput(output, dim=0):
        output = np.squeeze(output)
        outputTransformed = np.transpose(  output,    axes=[1,0,2])
        # outputTransformed = outputTransformed - np.mean(outputTransformed)
        # outputTransformed = np.clip(outputTransformed, (np.mean(outputTransformed) - 2*np.std(outputTransformed)), (np.mean(outputTransformed) + 2*np.std(outputTransformed)))
        # outputTransformed = outputTransformed - np.min(outputTransformed)
        outputTransformed = np.clip(outputTransformed / np.max(outputTransformed) * 255, 0, 255).astype('uint8')
        return outputTransformed

    def generateHist(output, name, dim=0):
        outputSqueezed = output.squeeze()

        outputChannelSingle = outputSqueezed[:,:,dim]
        _ = plt.hist(outputChannelSingle, bins='auto')  # arguments are passed to np.histogram
        plt.title("Histogram with 'auto' bins")

        plt.show()
        plt.savefig(name)
        plt.close()

    outputDisplay = displayOutput(output, 2)



    overlayedImage = cv2.addWeighted(inputImgOrig, 0.8, outputDisplay, 0.2, 0)

    if visualize:
        cv2.imshow("Input Image", rawImage)
        cv2.imshow("Segmentation Prediction",  outputDisplay)
        cv2.imshow("Segmentation Overlay",  overlayedImage)


        #cv2.imwrite("../Images/InputImages.png",  rawImage )
        #cv2.imwrite("../Images/SegmentationPrediction.png",  outputDisplay )
        cv2.waitKey(-1)


    # actual Evaluation
    imgs = os.listdir('data/data_road/testing/image_2/')
    print("")
    for filename in os.listdir('data/data_road/testing/image_2/'):
        if filename == '.DS_Store':
            continue
        inputImgOrig    = cv2.resize(cv2.imread('data/data_road/testing/image_2/' + filename), (640,192))
        inputImg    = np.transpose(inputImgOrig.astype('float32'),      axes=[1,0,2])
        output = modelPredictWrapper(model, inputImg)
        count += 1
        outputTransformed = displayOutput(output)

        overlayedImage = cv2.addWeighted(inputImgOrig, 0.8, outputTransformed, 0.2, 0)

        cv2.imwrite(os.path.join(output_img_path, filename),  outputTransformed )
        cv2.imwrite(os.path.join(output_img_path, "overlay_" + filename),  overlayedImage )

        # generateHist(output, os.path.join(output_img_path, "overlay_" + filename[:-4] + "_hist_0" + filename[-4:]),0)
        # generateHist(output, os.path.join(output_img_path, "overlay_" + filename[:-4] + "_hist_1" + filename[-4:]),1)
        # generateHist(output, os.path.join(output_img_path, "overlay_" + filename[:-4] + "_hist_2" + filename[-4:]),2)

        print(count, " of ", len(imgs) , end='\r')
    print("Mean ARD: ", ARD / count)
    print("Mean SQR: ", SQR / count)
    print("")


print("Testing model: {}".format(model_name))
if not os.path.exists(output_img_path):
    os.makedirs(output_img_path)

model.load_weights(os.path.join(model_path, model_name))
evaluateModel(model,args.batch, args.visualize)

# print("Testing model trained on Colab")
# model.load_weights("models/_weights_epoch20_val_loss_0.5520_train_loss_0.5597.hdf5")
# evaluateModel(model,args.batch, args.visualize)
