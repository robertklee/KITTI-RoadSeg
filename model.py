import keras
from keras import Model
from keras.layers import (Activation, Add, AveragePooling2D,
                          BatchNormalization, Conv2D, Conv2DTranspose, Flatten,
                          Input, MaxPool2D, Reshape, UpSampling2D,
                          ZeroPadding2D, concatenate)
from keras.optimizers import Adam

import constants
from resnet_def import ResNet18, ResNet50
from unet_def import UNet
from weights import load_model_weights

'''

ref:
https://medium.com/@14prakash/understanding-and-implementing-architectures-of-resnet-and-resnext-for-state-of-the-art-image-cf51669e1624    
https://keras.io/applications/#resnet
https://towardsdatascience.com/understanding-and-visualizing-resnets-442284831be8
'''

def generateResNetEncoderLayers(inputLayer, resnetType=18):
    '''
    takes an input layer of type Input from keras   

    returns the output layer of a resent of the specifified type
    input layer must be of form: (batches, even#, even#, 3*numImages)
    '''
    assert resnetType in [50,18]
    if resnetType == 50:
        x = ResNetLayerInitialStage(inputLayer)

        x = ResNetLayerLaterState_50(x, 64, [64,64,256],1,1)
        x = ResNetLayerLaterState_50(x, 256, [64,64,256],1,2)
        x = ResNetLayerLaterState_50(x, 256, [64,64,256],1,3)
        
        x1 = ResNetLayerLaterState_50(x, 256, [128,128,512],2,4)
        x1 = ResNetLayerLaterState_50(x1, 512, [128,128,512],1,5)
        x1 = ResNetLayerLaterState_50(x1, 512, [128,128,512],1,6)
        x1 = ResNetLayerLaterState_50(x1, 512, [128,128,512],1,7)

        x2 = ResNetLayerLaterState_50(x1, 512, [256,256,1024],2,8)
        x2 = ResNetLayerLaterState_50(x2, 1024, [256,256,1024],1,9)
        x2 = ResNetLayerLaterState_50(x2, 1024, [256,256,1024],1,10)
        x2 = ResNetLayerLaterState_50(x2, 1024, [256,256,1024],1,11)
        x2 = ResNetLayerLaterState_50(x2, 1024, [256,256,1024],1,12)
        x2 = ResNetLayerLaterState_50(x2, 1024, [256,256,1024],1,13)

        x3 = ResNetLayerLaterState_50(x2, 1024, [512,512,2048],2,14)
        x3 = ResNetLayerLaterState_50(x3, 2048, [512,512,2048],1,15)
        x3 = ResNetLayerLaterState_50(x3, 2048, [512,512,2048],1,16)

        x3 = ResNetOuputStage(x3)
        return x3, x2, x1, x

    else: # is resnet 18
        x = ResNetLayerInitialStage(inputLayer)
        
        x = ResNetLayerLaterState_18(x, 64, [64,64],1,1)
        x = ResNetLayerLaterState_18(x, 64, [64,64],1,2)

        x1 = ResNetLayerLaterState_18(x, 64, [128,128],2,3)
        x1 = ResNetLayerLaterState_18(x1, 128, [128,128],1,4)

        x2 = ResNetLayerLaterState_18(x1, 128, [256,256],2,6)
        x2 = ResNetLayerLaterState_18(x2, 256, [256,256],1,7)
        
        x3 = ResNetLayerLaterState_18(x2, 256, [512,512],2,8)
        x3 = ResNetLayerLaterState_18(x3, 512, [512,512],1,9)

        x3 = ResNetOuputStage(x3)
        return x3, x2, x1, x

def ResNetLayerInitialStage(inputLayer):
    x = Conv2D(filters=64,kernel_size=7,strides=2,data_format='channels_last',activation='relu',padding='same', name="InitialConv")(inputLayer)
    x = BatchNormalization(axis=3)(x)
    x = MaxPool2D(pool_size=(3,3),strides=2, data_format='channels_last',padding='same', name="InitalPool")(x)
    return x

def ResNetLayerLaterState_50(inputLayer, inputChannels, channels, poolingStride, resNetBlockID):
    '''
    3 convolutional blocks

    1x1, channels[0], relu
    3x3, channels[1], relu
    1x1, channels[2], linear
    add input and output
    relu
    '''

    assert len(channels) == 3

    x = Conv2D(channels[0], kernel_size=poolingStride, strides=poolingStride,data_format='channels_last',activation='relu',padding='same', name="Conv1_" + str(inputChannels) + "__" + str(resNetBlockID))(inputLayer)
    x = BatchNormalization(axis=3)(x)
    x = Conv2D(channels[1], kernel_size=3,strides=1,data_format='channels_last',activation='relu',padding='same', name="Conv2_" + str(inputChannels) + "__" + str(resNetBlockID))(x)
    x = BatchNormalization(axis=3)(x)
    x = Conv2D(channels[2], kernel_size=1,strides=1,data_format='channels_last',activation='linear',padding='same', name="Conv3_" + str(inputChannels) + "__" + str(resNetBlockID))(x)
    x = BatchNormalization(axis=3)(x)

    if inputChannels != channels[2]:
        # this could be zero padding but instread were doing 1x1 stride 1 convolution to make the shapes the same, both are technically from paper acceptable
        inputLayer = Conv2D(channels[2], kernel_size=1, strides=1,data_format='channels_last',activation='linear',padding='same', name="ConvSkip_" + str(inputChannels) + "__" + str(resNetBlockID))(inputLayer)
        if poolingStride != 1:
            inputLayer = MaxPool2D(pool_size=2)(inputLayer)

    x = Add()([x,inputLayer])
    x = Activation('relu')(x)
    return x

def ResNetLayerLaterState_18(inputLayer, inputChannels, channels, poolingStride, resNetBlockID):
    '''
    two convolutional blocks 
    3x3, channels[0], relu
    3x3, channels[0], linear
    add input and output
    relu
    '''

    assert len(channels) == 2

    x = Conv2D(channels[0], kernel_size=3,strides=poolingStride,data_format='channels_last',activation='relu',padding='same', name="Conv1_" + str(inputChannels) + "__" + str(resNetBlockID))(inputLayer)
    x = BatchNormalization(axis=3)(x)
    x = Conv2D(channels[1], kernel_size=3,strides=1,data_format='channels_last',activation='linear',padding='same', name="Conv2_" + str(inputChannels) + "__" + str(resNetBlockID))(x)
    x = BatchNormalization(axis=3)(x)

    if inputChannels != channels[1]:
        # this should be zero padding but just simple one conv for now untill fixed
        inputLayer = Conv2D(channels[1], kernel_size=1, strides=1,data_format='channels_last',activation='linear',padding='same', name="ConvSkip_" + str(inputChannels) + "__" + str(resNetBlockID))(inputLayer)
        if poolingStride != 1:
            inputLayer = MaxPool2D(pool_size=2,padding='same', name="Pool_" + str(inputChannels) + "__" + str(resNetBlockID))(inputLayer)

    x = Add()([x,inputLayer])
    x = Activation('relu')(x)

    return x

def ResNetOuputStage(inputLayer, pools=1000):
    output = AveragePooling2D(pool_size=2,strides=1,padding='same')(inputLayer)
    return output



def buildDecoder(inputLayer, scale_1, scale_2, scale_3, outputChannels=1):
    x = Conv2D(1024, kernel_size=3, strides=1, data_format='channels_last',padding='same', name="DecoderConv_Block_1_1", activation='relu')(inputLayer)
    x = Conv2D(1024, kernel_size=3, strides=1, data_format='channels_last',padding='same', name="DecoderConv_Block_1_2", activation='relu')(x)

    x = Conv2DTranspose(512, (2, 2), strides=(2, 2), data_format='channels_last', name="ConvTranspose1")(x)

    x = concatenate([x,scale_1],axis=3)

    x = Conv2D(512, kernel_size=3, strides=1, data_format='channels_last', padding='same', name="DecoderConv_Block_2_1", activation='relu')(x)
    x = Conv2D(512, kernel_size=3, strides=1, data_format='channels_last', padding='same', name="DecoderConv_Block_2_2", activation='relu')(x)

    x = Conv2DTranspose(256, (2, 2), strides=(2, 2), data_format='channels_last', name="ConvTranspose2")(x)

    x = concatenate([x,scale_2],axis=3)

    x = Conv2D(256, kernel_size=3, strides=1, data_format='channels_last', padding='same', name="DecoderConv_Block_3_1", activation='relu')(x)
    x = Conv2D(256, kernel_size=3, strides=1, data_format='channels_last', padding='same', name="DecoderConv_Block_3_2", activation='relu')(x)

    scale_3_out = Conv2D(32, kernel_size=3, strides=1, data_format='channels_last' ,padding='same', name="EndingConvBlock_Scale3")(x)
    scale_3_out = Conv2D(outputChannels, kernel_size=3, strides=1, data_format='channels_last' ,padding='same', name="OutputConvBlock_Scale3")(scale_3_out)
    scale_3_out = UpSampling2D(data_format='channels_last', name="upSampleScale3Out", size=(8,8), interpolation='bilinear')(scale_3_out)

    x = Conv2DTranspose(128, (2, 2), strides=(2, 2), data_format='channels_last', name="ConvTranspose3")(x)

    x = concatenate([x,scale_3],axis=3)

    x = Conv2D(128, kernel_size=3, strides=1, data_format='channels_last', padding='same', name="DecoderConv_Block_4_1", activation='relu')(x)
    x = Conv2D(128, kernel_size=3, strides=1, data_format='channels_last', padding='same', name="DecoderConv_Block_4_2", activation='relu')(x)

    scale_2_out = Conv2D(32, kernel_size=3, strides=1, data_format='channels_last' ,padding='same', name="EndingConvBlock_Scale2")(x)
    scale_2_out = Conv2D(outputChannels, kernel_size=3, strides=1, data_format='channels_last' ,padding='same', name="OutputConvBlock_Scale2")(scale_2_out)
    scale_2_out = UpSampling2D(data_format='channels_last', name="upSampleScale2Out", size=(4,4), interpolation='bilinear')(scale_2_out)

    x = Conv2DTranspose(64, (2, 2), strides=(2, 2), data_format='channels_last', name="ConvTranspose4")(x)

    x = Conv2D(64, kernel_size=3, strides=1, data_format='channels_last', padding='same', name="DecoderConv_Block_5_1", activation='relu')(x)
    x = Conv2D(64, kernel_size=3, strides=1, data_format='channels_last', padding='same', name="DecoderConv_Block_5_2", activation='relu')(x)
    
    scale_1_out = Conv2D(64, kernel_size=3, strides=1, data_format='channels_last' ,padding='same', name="EndingConvBlock_Scale1")(x)
    scale_1_out = Conv2D(outputChannels, kernel_size=3, strides=1, data_format='channels_last' ,padding='same', name="OutputConvBlock_Scale1")(scale_1_out)
    scale_1_out = UpSampling2D(data_format='channels_last', name="upSampleScale1Out", size=(2,2), interpolation='bilinear')(scale_1_out)
    
    x = Conv2DTranspose(64, (2, 2), strides=(2, 2), data_format='channels_last', name="ConvTranspose5")(x)

    x = Conv2D(64, kernel_size=3, strides=1, data_format='channels_last' ,padding='same', name="EndingConvBlock1", activation='relu')(x)
    x = Conv2D(64, kernel_size=3, strides=1, data_format='channels_last' ,padding='same', name="EndingConvBlock2", activation='relu')(x)
    x = Conv2D(outputChannels, kernel_size=3, strides=1, data_format='channels_last' ,padding='same', name="OutputConvBlock", activation='sigmoid')(x)
    
    return concatenate([x, scale_1_out, scale_2_out], axis=3), scale_1_out, scale_2_out


def create_Model(input_shape=constants.input_shape, encoder_type=constants.EncoderType.resnet50):
    model_include_top = constants.include_top
    
    if encoder_type == constants.EncoderType.resnet50:
        inputLayer, outputLayer, scaleLayers = ResNet50(input_shape=constants.input_shape,include_top=False, create_encoder=True)
        # modelPath = constants.resnet_50_model_path
        model_name = constants.EncoderType.resnet50.name
        
    elif encoder_type == constants.EncoderType.resnet18:
        inputLayer, outputLayer, scaleLayers = ResNet18(input_shape=constants.input_shape,include_top=False, create_encoder=True)
        # modelPath = constants.resnet_18_model_path
        model_name = constants.EncoderType.resnet18.name
    
    else:
        raise ValueError("Invalid encoder type. Encoder type must be within constants.EncoderType")

    if (constants.use_unet):
        networkOutput = UNet(outputLayer, scaleLayers[2], scaleLayers[1], scaleLayers[0], output_height=constants.input_shape[0], output_width=constants.input_shape[1])
    else:
        networkOutput, scale_1_out, scale_2_out = buildDecoder(outputLayer, scaleLayers[2], scaleLayers[1], scaleLayers[0], 1)
    model = Model(inputs=[inputLayer], outputs=[networkOutput])#, scale_1_out, scale_2_out, scale_3_out])

    # model.load_weights(modelPath, by_name=True)
    load_model_weights(model, model_name, constants.dataset, constants.classes, model_include_top)
    model.summary()
    return model

if __name__ == "__main__":
    print("Testing creating models")
    model = create_Model(input_shape=constants.input_shape, encoder_type=constants.EncoderType.resnet18)
    print("Done creating ResNet18 backbone model")
    model = create_Model(input_shape=constants.input_shape_full_size, encoder_type=constants.EncoderType.resnet50)
    print("Done creating ResNet50 backbone model")
