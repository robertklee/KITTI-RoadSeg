import keras
import keras.backend as K
import tensorflow as tf

smooth = 1.

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

def intersection_over_union(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)

class modelLoss():
    def __init__(self, lambda_, alpha, width, height, batchsize):
        self.lambda_ = lambda_
        self.width = width
        self.height = height
        self.batchsize = batchsize
        self.alpha = alpha

    def test(self, y_true, y_pred):
        # rename and split values
        # [batch, width, height, channel]
        img = y_true[:,:,:,0:3]
        seg = y_true[:,:,:,3:6]

        disp0        = K.expand_dims(y_pred[:,:,:,0],-1)
        disp1        = K.expand_dims(y_pred[:,:,:,1],-1)
        disp2        = K.expand_dims(y_pred[:,:,:,2],-1)
        disp3        = K.expand_dims(y_pred[:,:,:,3],-1)

        return None

    def applyLoss(self, y_true, y_pred):
        return dice_coef_loss(y_true, y_pred)
        #return L_p
