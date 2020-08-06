import argparse

# https://github.com/qinnzou/Robust-Lane-Detection/blob/master/LaneDetectionCode/config.py

# globel param
# dataset setting
img_width = 1242
img_height = 375
img_channel = 3
label_width = 1242
label_height = 375
label_channel = 1
data_loader_numworkers = 8
class_num = 2

# path
train_path = "./data/train_index.txt"
val_path = "./data/val_index.txt"
test_path = "./data/test_index_demo.txt"
save_path = "./save/result/"
pretrained_path='./pretrained/unetlstm.pth'

# weight
class_weight = [0.02, 1.02]
