# Abstract
Lane recognition is one of the key technologies in the field of autonomous vehicles. It is widely used in assisted driving systems, lane departure warning systems and vehicle collision prevention system, and it has great significance in improving traffic safety. The lane detection is susceptible to interference from external environment such as illumination variations, road occlusions and markings and vehicles on the road. This requires larger dataset to obtain better results.

This document describes implementing a convolutional neural network (CNN) to perform semantic segmentation of road surfaces within a driving context. To improve the model performance, data augmentation was performed using noise addition, light adjustment, contrast adjustment, etc. We used the KITTI Road dataset which includes about 500 images in three different categories. We propose a network architecture using a UNet network structure, with a ResNet18 or ResNet50 decoder. We obtained good model performance and propose future work to improve results.

# Getting Started
Clone the repository and run `train.py` to train the model. The command line parameters that `train.py` are described in `argparser` help comments. The script will check for the pre-trained model weights and dataset, and automatically download them if they don't exist. The links are confirmed to be working as of Sept. 12th, 2020.

This project was tested with **Python 3.6.8**.

Requirements can be installed via the `requirements.txt` file.

Completed models, as well as the dataset and pre-trained encoder weights, are available on a personal [OneDrive link](https://1drv.ms/u/s!AnSUzPfRDFUagcFB0qdPnt9YKApvcw?e=Drg1qf).

## Downloading from source
The KITTI road dataset should be unzipped and placed in a subdirectory called `data`, so the final path to the images is consistent with the one in `constants.py`

The pre-trained ResNet weights should be placed in a subdirectory called `models`, so the final path is consistent with that in `constants.py`

When loading already trained weights, unzip and place in `models`, in a folder that matches the `Session ID`. Then, update the string in `test.py`. As it is configured currently, `test.py` will read the weights from epoch `20` from session ID `KITTI_Road_UNet_Sun Aug 30 19_52_21 2020_batchsize_25_resnet_18`. Thus, the file path should be `KITTI_Road_UNet_Sun Aug 30 19_52_21 2020_batchsize_25_resnet_18/_weights_epoch20_val_loss_-1.9577_train_loss_-1.9696.hdf5`

The output will be generated using `test.py` and will be placed in `output/{Session ID}/{epoch}/{train/test}/...` 

### Downloading dataset
Download the KITTI road dataset base kit from the [KITTI Vision Benchmark Suite website](http://www.cvlibs.net/datasets/kitti/eval_road.php). 

### Download pre-trained model files
Download the pre-trained ResNet encoder weights from [qubvel's GitHub repository](https://github.com/qubvel/classification_models)

# Network Architecture
We used a standard fully convolutional UNet [4] structure that receives an RGB colour image as input and generates a same-size semantic segmentation map as output (see Figure 1). The structure of the network is an encoder-decoder network with skip connections between various feature levels of the encoder to the decoder. This enables the network to combine information from both deep abstract features and local, high-resolution information to generate the final output. The encoder section uses a configurable ResNet18 or ResNet50 model [14]. Both models were tested.

The chosen UNet structure is a proven architecture for extracting both deep abstract features and local features. It consists of three sections: the contraction, bottleneck, and expansion. The downsampling contraction section, known as the encoder, extracts feature information from the input by decreasing in spatial dimensions but increasing in feature dimensions. The bottleneck is where we have a compact representation of the input. The upsampling expansion section, known as the decoder, reduces the image in feature dimension while increasing the spatial dimensions. It uses skip connections that allow it to tap into the same-sized output of the contraction section, which allows it to use higher-level locality features in its upsampling. This is a strong architecture for image segmentation, as we must assign each pixel a class. Thus, our output must use the high-resolution information from the input image to obtain a good prediction.

The encoder was chosen to be a ResNet model because it allows us to train a deep neural network and reduces the risk of the vanishing gradient problem. The key breakthrough is in the eponymous residual connections (see Figure 2), which allow gradient information to have an alternate path to flow through.

<!-- # Requirements
Download ResNet V2 50 and VGG 16 from [Tensorflow's GitHub link](https://github.com/tensorflow/models/tree/master/research/slim#pre-trained-models) -->

# TODO
- link to trained models
- train and test should be repeated and split evenly across the three types
- evaluation function
- fix decoder