input_shape = (640,192,3)
input_shape_full_size = (1024,320,3)#(1242, 374, 3)

resnet_18_model_path = "models/resnet18_imagenet_1000_no_top.h5"
resnet_50_model_path = "models/resnet50_imagenet_1000_no_top.h5"

number_classes = 2

road_color =        [255,0,255]
background_color =  [255,0,0]

system_files = '.DS_Store'

use_unet = True