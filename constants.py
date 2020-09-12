from enum import Enum

# ************************************************ #
# Parameters
input_shape = (640,192,3)
input_shape_full_size = (1024,320,3)#(1242, 374, 3)

number_classes = 2

road_color =        [255,0,255]
background_color =  [255,0,0]

system_files = '.DS_Store'

use_unet = True

train_ratio = 0.8 # 80% train, 20% test
assert train_ratio > 0 and train_ratio <= 1

# ************************************************ #
# Dataset Locations

data_train_image_dir = 'data/data_road/training/image_2'
data_train_gt_dir    = 'data/data_road/training/gt_image_2'
data_test_image_dir  = 'data/data_road/testing/image_2/'
data_location        = 'data'

# ************************************************ #
# Model Pre-Trained Weight Locations

resnet_18_model_path = "models/resnet18_imagenet_1000_no_top.h5"
resnet_50_model_path = "models/resnet50_imagenet_1000_no_top.h5"

def get_model_path(resnet_type):
    return "models/resnet{}_imagenet_1000_no_top.h5".format(str(resnet_type))

# ************************************************ #
# Model Types

class EncoderType(Enum):
    resnet18  = 18
    resnet34  = 34
    resnet50  = 50
    resnet101 = 101
    resnet152 = 152

classes         = 1000
dataset         = 'imagenet'
include_top     = False
models_location = "models"