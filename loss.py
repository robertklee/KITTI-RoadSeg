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
        
        return L_p

    def applyLoss(self, y_true, y_pred):
        return dice_coef_loss(y_true, y_pred)
        #return L_p


if __name__ == "__main__":
    from dataGen import depthDataGenerator
    import cv2
    import numpy as np

    batchSize = 8
    
    train_generator  = depthDataGenerator('../val/left/', '../val/right/',   batch_size=batchSize, shuffle=True, max_img_time_diff=700)

    inputImage, y_true = train_generator.__getitem__(1)

    left_raw        = y_true[0,:,:,0:3]
    right_raw        = y_true[0,:,:,6:9]

    cv2.imshow("test", left_raw.astype('uint8'))
    cv2.waitKey(-1)

    cv2.imshow("test", right_raw.astype('uint8'))
    cv2.waitKey(-1)

    y_true = tf.convert_to_tensor(y_true.astype('float32'))

    left        = y_true[:,:,:,0:3 ]
    right_minus = y_true[:,:,:,3:6 ]
    right       = y_true[:,:,:,6:9 ]
    right_plus  = y_true[:,:,:,9:12]

    rand      = np.random.rand(batchSize,640,192,1 ).astype('float32')
    scale      = np.ones_like(rand).astype('float32') * 10/640
    randImage_tensor       = tf.convert_to_tensor(rand)
    scale_tensor       = tf.convert_to_tensor(scale)

    L1Direct, L1Reproject, SSIM_right_reproject, SSIM_right_left = photoMetric(randImage_tensor, left, right, 640, 192, batchSize)
    print(1-K.eval(SSIM_right_reproject))
    print(1-K.eval(SSIM_right_left))
    L1Direct, L1Reproject, SSIM_right_reproject, SSIM_right_left = photoMetric(scale_tensor, left, right, 640, 192, batchSize)
    print(1-K.eval(SSIM_right_reproject))
    print(1-K.eval(SSIM_right_left))

def oldTest():

    #leftImage  = '../val/left/2018-07-16-15-37-46_2018-07-16-15-38-12-727.jpg'
    #dispImage  = '../val/disp/2018-07-16-15-37-46_2018-07-16-15-38-12-727.png' # actuall associated disparity
    #dispImage1  = '../val/disp/2018-07-16-15-37-46_2018-07-16-16-32-48-979.png' # bad disparity totally random
    #rightImage = '../val/right/2018-07-16-15-37-46_2018-07-16-15-38-12-727.jpg'

    leftImage  = '../test/left/2018-07-16-15-37-46_2018-07-16-15-38-12-727.jpg'
    dispImage  = '../test/disp/2018-07-16-15-37-46_2018-07-16-15-38-12-727.png' # actuall associated disparity
    dispImage1 = '../test/disp/2018-07-16-15-37-46_2018-07-16-16-32-48-979.png' # bad disparity totally random
    rightImage = '../test/right/2018-07-16-15-37-46_2018-07-16-15-38-12-727.jpg'


    import numpy as np
    import cv2

    left      = np.transpose(cv2.imread(leftImage),     axes=[1,0,2]).astype('float32')
    dispTrue  = np.transpose(cv2.imread(dispImage),     axes=[1,0,2]).astype('float32')[:,:,0] / 256.
    dispWrong = np.transpose(cv2.imread(dispImage1),    axes=[1,0,2]).astype('float32')[:,:,0] / 256.
    right     = np.transpose(cv2.imread(rightImage),    axes=[1,0,2]).astype('float32')
    rand      = np.random.rand(left.shape[0],left.shape[1],1 ).astype('float32')
    leftButScaled = left * 0.4
    
    width = left.shape[0]
    height = left.shape[1]
    realOffset = 3

    leftNew = np.zeros_like(left)
    leftNew[75:,:,:] = left[0:-75,:,:]
    left[-75:0,:,:] = 0 
    dispTrue = np.ones_like(dispTrue) / width * 75
    dispTrue[-75:,:] = 0
         
    leftNew_tensor          = tf.expand_dims(tf.convert_to_tensor(leftNew), 0)
    leftImage_tensor        = tf.expand_dims(tf.convert_to_tensor(left), 0)
    rightImage_tensor       = tf.expand_dims(tf.convert_to_tensor(right), 0)
    dispImage_tensor        = tf.expand_dims(tf.expand_dims(tf.convert_to_tensor(dispTrue), 0), -1)
    dispImage_tensor1       = tf.expand_dims(tf.expand_dims(tf.convert_to_tensor(dispWrong), 0), -1)
    randImage_tensor        = tf.expand_dims(tf.convert_to_tensor(rand), 0)
    leftScaledImage_tensor  = tf.expand_dims(tf.convert_to_tensor(leftButScaled), 0)

    leftNew_tensor       = K.concatenate([leftNew_tensor        ,leftNew_tensor        ,leftNew_tensor      ], axis=0) 
    leftImage_tensor       = K.concatenate([leftImage_tensor        ,leftImage_tensor        ,leftImage_tensor      ], axis=0) 
    rightImage_tensor      = K.concatenate([rightImage_tensor       ,rightImage_tensor       ,rightImage_tensor     ], axis=0) 
    dispImage_tensor       = K.concatenate([dispImage_tensor        ,dispImage_tensor        ,dispImage_tensor      ], axis=0) 
    dispImage_tensor1      = K.concatenate([dispImage_tensor1       ,dispImage_tensor1       ,dispImage_tensor1     ], axis=0) 
    randImage_tensor       = K.concatenate([randImage_tensor        ,randImage_tensor        ,randImage_tensor      ], axis=0) 
    leftScaledImage_tensor = K.concatenate([leftScaledImage_tensor  ,leftScaledImage_tensor  ,leftScaledImage_tensor], axis=0) 

    #L1Direct, L1Reproject, SSIM_right_reproject, SSIM_right_left  = photoMetric(dispImage_tensor,  leftImage_tensor, rightImage_tensor, width, height, 3)
    #print(K.eval(SSIM_right_reproject))
    #L1Direct, L1Reproject, SSIM_right_reproject, SSIM_right_left = photoMetric(randImage_tensor, leftImage_tensor, rightImage_tensor, width, height, 3)
    #print(K.eval(SSIM_right_reproject))
    L1Direct, L1Reproject, SSIM_right_reproject, SSIM_right_left = photoMetric(dispImage_tensor, leftNew_tensor, leftImage_tensor, width, height, 3)
    print(K.eval(SSIM_right_reproject))
    # print("good")
    # print(K.eval(Lp))
    # print("bad")
    # print(K.eval(Lp1))

    #disp1 = np.random.uniform(size=disp.shape).astype('float32')

    #left.reshape(1,  left.shape[0], left.shape[1], left.shape[2]  )
    #disp.reshape(1,  disp.shape[0], disp.shape[1], 1)
    ##dispO.reshape(1,  dispO.shape[0], dispO.shape[1], 1)
    #disp1.reshape(1,  disp1.shape[0], disp1.shape[1], 1)
    #right.reshape(1, right.shape[0], right.shape[1], right.shape[2]  )

    #leftImage_tensor  = tf.expand_dims(tf.convert_to_tensor(left), 0)
    #rightImage_tensor = tf.expand_dims(tf.convert_to_tensor(right), 0)
    #dispImage_tensor  = tf.expand_dims(tf.expand_dims(tf.convert_to_tensor(disp), 0), -1)
    #dispImage_tensor1 = tf.expand_dims(tf.expand_dims(tf.convert_to_tensor(disp1), 0), -1)
    #dispImage_tensorO = tf.expand_dims(tf.expand_dims(tf.convert_to_tensor(dispO), 0), -1)

    #Lp  = photoMetric(dispImage_tensor,  leftImage_tensor, rightImage_tensor, left.shape[1], left.shape[2], 1)
    #Lp1 = photoMetric(dispImage_tensor1, leftImage_tensor, rightImage_tensor, left.shape[1], left.shape[2], 1)
    ##LpO = photoMetric(dispImage_tensorO, leftImage_tensor, rightImage_tensor, left.shape[1], left.shape[2], 1)

    #print("good")
    #print(K.eval(Lp))
    #print("random")
    #print(K.eval(Lp1))
    #print("other")
    #print(K.eval(LpO))  

    print("smoothness good test")
    comparator = leftImage_tensor
    smoothness = smoothnessLoss(comparator,leftImage_tensor, 1)
    print(K.eval(smoothness))

    smoothness = smoothnessLoss(comparator,leftImage_tensor, 2)
    print(K.eval(smoothness))

    smoothness = smoothnessLoss(comparator,leftImage_tensor, 3)
    print(K.eval(smoothness))

    smoothness = smoothnessLoss(comparator,leftImage_tensor, 4)
    print(K.eval(smoothness))
    
    '''
        convs  | disp vs left| left vs left | random vs left | right vs left | leftScaled0.4 vs left
        1        0.027379034    0.3894189     14266.269        1.4842504        0.15576762
        2        0.01510952     0.38573       8023.563         0.8445017        0.15429199
        3        0.008012139    0.24224764    4243.597         0.45495307       0.096899055
        4        0.004178587    0.14077793    2208.4468        0.24049726       0.05631118
    '''
