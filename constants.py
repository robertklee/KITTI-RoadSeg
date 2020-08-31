# ************************************************ #
# Parameters
input_shape = (640,192,3)
input_shape_full_size = (1024,320,3)#(1242, 374, 3)

number_classes = 2

road_color =        [255,0,255]
background_color =  [255,0,0]

system_files = '.DS_Store'

use_unet = True

# ************************************************ #
# Dataset Locations

data_train_image_dir = 'data/data_road/training/image_2'
data_train_gt_dir    = 'data/data_road/training/gt_image_2'
data_test_image_dir  = 'data/data_road/testing/image_2/'

# ************************************************ #
# Model Pre-Trained Weight Locations

resnet_18_model_path = "models/resnet18_imagenet_1000_no_top.h5"
resnet_50_model_path = "models/resnet50_imagenet_1000_no_top.h5"