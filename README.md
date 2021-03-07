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

When loading already trained weights, unzip and place in `models`, in a folder that matches the `Session ID`. Then, update the string in `test.py`. As it is configured currently, `test.py` will read the weights from epoch `20` from session ID `KITTI_Road_UNet_v2_Conv2DTranspose_2021-02-03-20h-46m-06s_batchsize_12_resnet_18`. Thus, the file path should be `KITTI_Road_UNet_v2_Conv2DTranspose_2021-02-03-20h-46m-06s_batchsize_12_resnet_18/_weights_epoch20_val_loss_-1.9581_train_loss_-1.9632.hdf5`

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

# Results
The updated model results are not shown in the report, based on the previous model. The results of the previous model are available at: https://www.dropbox.com/s/hoxbbl20gql7f53/ECE%20470%20Final%20Report.pdf?dl=0
TODO: embed images into README

# Discussion
For the evaluation function, the F1 score in best case was 99.1% and accuracy is 98.6%; in worse case, the F1 score was 92.5% and accuracy is 91.3%. We also analyzed the two different encoders: ResNet18 and ResNet50. In the UU case, ResNet50 performed better than ResNet18. However, ResNet50 did not perform well in some complicated situations; thus, it sometimes performed worse. 

We note that the trained model is not overfitting, since the training and validation accuracy are both relatively monotonously increasing with respect to epochs. In addition, the training and validation loss function is also steadily decreasing with respect to epochs. This is seen in Figure 9 and Figure 10.

We observe that the model performs best when the road surface is clearly distinguished from surroundings. It is especially strong when there are distinct lane markings. This is likely because the model can extract the edges as an indicator of potential road areas. 

As shown in Figure 11, the model is robust in many different situations. As shown in Figure 12, the model is weak when there are misleading lines that are visually similar to road surfaces, such as the train tracks. It also has difficulty when there is a high contrast shadow region, or where the road is split by a median. The model seems to favour segmentations that are not split on the lower section of the image. It also prefers connected road regions. This is reasonable, since real-world roads will be connected. However, this results in road surfaces that are split from the very bottom of the image being incorrectly segmented, such as the bottom image in Figure 12. However, this may be tolerable, because the car cannot move into that lane anyway. 

In Table 1, we see the model has both the best and worst performance on urban unmarked images. This class of images has the greatest variation because urban environments have irregularly shaped roads, many pedestrians and parked vehicle occlusions, and highly varied surrounding obstacles, among others. It is reasonable that the model performs the worst in this class. 

Tighter spread in performance is seen on the urban multiple marked lines dataset. This is likely because the model has learned to use lane markings to determine road surfaces, and multiple parallel lines will strengthen that prediction. In addition, wider roads requiring multiple lane markings are often found in the outskirts of the city, where the roads are naturally straighter and have fewer random objects and occlusions. 

We note that the ResNet 50-based model had both better and worse performance. This is reasonable, since deeper networks with more trainable parameters may perform better if we can provide a larger dataset. Since the given dataset only contains hundreds of images, there may not be enough data to train the deeper model to achieve consistently better performance. Deeper models are also more likely to suffer from the gradient vanishing problem, which may affect the propagation of weight updates.

Multiple challenges were encountered while training the model. The biggest challenge was an incorrect concatenation in the decoder. The final layer was returned as a concatenation of multiple scale outputs, which likely jumbled the network results, causing it to output extremely large positive and negative values. Further, the final layer of that decoder had parts of the model that were visible, without a softmax or sigmoid function that can cap the output values. After fixing it with final output layer of softmax activation, the model results were between the range of [0,1] as expected. 

Another bug was encountered in the segmentation ground truth and was fixed by producing a binary one-hot segmentation for each pixel from the ground truth data.
Challenges were also encountered in implementing non-standard loss functions, such as birds’ eye view (BEV), in the TensorFlow and Keras language. This is because the available graph operations are limited, and workarounds must be found to implement some math operations. We decided to revert to a pixel-based Dice loss function.

The dataset also posed challenges. The ground truth data for the testing dataset was not available for users to download. Instead, they provide a server where results can be uploaded and evaluated. This is likely because they want to prevent cheating in the benchmark. However, this made it difficult to evaluate the performance of the testing dataset. This was solved by visually inspecting the test output, and calculating the model performance metrics on training data, knowing that this will overestimate the model performance. The test output was similar to the training output, so this did not greatly affect the results.

Another dataset issue was the lack of available driving corridor segmentation ground truth data. This was only available for a third of the total training data. Thus, we changed our plan from performing lane segmentation to road segmentation. If lane segmentation is still desired, we can use weights pre-trained on the road dataset to fine tune with the limited lane marking dataset.

Finally, issues were encountered with the various command line arguments that we programmed. This was solved by testing various combinations of parameters and inspecting and refactoring possible branches.


# Future Work
While the results of the model were strong, there remains work that can improve the results. The major item that will likely greatly improve results is implementing stronger loss functions based on the birds’ eye view metric discussed in Section IV.B: Performance Metric Types. This was not implemented in the first iteration of this model because the calculations require extensive debugging to translate it from normal Python into a language suitable for Keras Backend and TensorFlow. The operations that are available are limited, and not all calculations have a suitable mapping. By improving the loss function, the model can better tune the network parameters during backpropagation. A behaviour-based loss function also improves the results in a driving context.

[COMPLETED] The current output has a jagged appearance because Upsample2D blocks were used in the decoder. This is a simple scaling of the image, and while it is efficient, it results in artifacts in the output. This is especially noticeable in the last layer, where a Upsample2D of 4x is used, and produces the scaling pixilation artifacts seen in the output. This block can be extended to multiple Upsample2D of 2x separated by convolution layers, which will likely reduce pixilation. 

[COMPLETED] All of the Upsample2D blocks can also be changed to use a Conv2DTranspose. This also upsamples, but it uses a convolution kernel that is trainable, which can improve model performance.

[Partially Completed] The decoder is fairly shallow, especially compared to the ResNet50 encoder backbone depth. This may mean we are losing details in the upsampling. We can increase the depth of the decoder to increase the number of trainable parameters, which may improve results. However, this may also lead to a decrease in performance, so this should be compared to the current result.

Finally, we can explore using better Keras metrics to evaluate the training of the model. For example, Mean Intersection-Over-Union may be used. This computes the IOU for each segmentation class and averages them and is a better metric for segmentation problems.

# Updates
- UNet Decoder has been updated with improved upscaling by using trainable `Conv2DTranspose` instead of `Upsampling2D` 

# TODO
- train and test should be repeated and split evenly across the three categories of images
- data augmentation images should be repeated
- evaluation function needs to be updated to assess model performance in one numeric value
- ground truth label colours may not match those specified in `constants.py`
- other road surfaces are specified by pure black (0,0,0) which I didn't notice
- proper train/validation/test split using training set (since test labels are not provided)
- more epochs?
