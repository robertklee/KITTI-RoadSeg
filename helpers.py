import os

import matplotlib.cm as cm
import numpy as np


'''
The following functions are modified from KittiSeg repository
'''

def make_overlay(image, gt_prob):

    mycm = cm.get_cmap('bwr')

    overimage = mycm(gt_prob, bytes=True)
    output = 0.4*overimage[:,:,0:3] + 0.6*image

    return output

def overlayImageWithConfidence(in_image, conf, vis_channel = 1, threshold = 0.5):
    '''
    
    :param in_image:
    :param conf:
    :param vis_channel:
    :param threshold:
    '''
    if in_image.dtype == 'uint8':
        visImage = in_image.copy().astype('f4')/255
    else:
        visImage = in_image.copy()
    
    channelPart = visImage[:, :, vis_channel] * (conf > threshold) - conf
    channelPart[channelPart < 0] = 0
    visImage[:, :, vis_channel] = 0.5*visImage[:, :, vis_channel] + 255*conf
    return visImage
