# Description
A course project for ECE 470: Artificial Intelligence for lane detection using a CNN

## Getting Started
All files are available on a personal [OneDrive link](https://1drv.ms/u/s!AnSUzPfRDFUagcFB0qdPnt9YKApvcw?e=Drg1qf). Instructions to download from source are below.

The KITTI road dataset should be unzipped and placed in a subdirectory called `data`, so the final path to the images is consistent with the one in `constants.py`

The pre-trained ResNet weights should be placed in a subdirectory called `models`, so the final path is consistent with that in `constants.py`

When loading already trained weights, unzip and place in `models`, in a folder that matches the `Session ID`. Then, update the string in `test.py`. As it is configured currently, `test.py` will read the weights from epoch `20` from session ID `KITTI_Road_UNet_Sun Aug 30 19_52_21 2020_batchsize_25_resnet_18`. Thus, the file path should be `KITTI_Road_UNet_Sun Aug 30 19_52_21 2020_batchsize_25_resnet_18/_weights_epoch20_val_loss_-1.9577_train_loss_-1.9696.hdf5`

The output will be generated using `test.py` and will be placed in `output/{Session ID}/{epoch}/{train/test}/...` 

### Downloading dataset from source
Download the KITTI road dataset base kit from the [KITTI Vision Benchmark Suite website](http://www.cvlibs.net/datasets/kitti/eval_road.php). 

### Download pre-trained model files
TODO find source link for ImageNet trained ResNet weights

# Requirements
Download ResNet V2 50 and VGG 16 from [Tensorflow's GitHub link](https://github.com/tensorflow/models/tree/master/research/slim#pre-trained-models)

# TODO
- link to ResNet18, 50 initial training weights
- link to model
- link to trained outputs
- proper set up of repo to train/test