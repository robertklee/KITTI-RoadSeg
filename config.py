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
# train_path = "./data/data_road/"
# val_path = "./data/val_index.txt"
# test_path = "./data/test_index_demo.txt"
# save_path = "./save/result/"
# pretrained_path='./pretrained/unetlstm.pth'

# weight
class_weight = [0.02, 1.02]

DEFAULT_DATA_DIR = './data'
DEFAULT_RUNS_DIR = './runs'
DEFAULT_MODEL_PATH = "models/model.ckpt"
DEFAULT_EPOCHS = 20
DEFAULT_BATCH_SIZE = 3

def args_setting():
    # Training settings
    argparser = argparse.ArgumentParser(description='Training')
    argparser.add_argument('-d',
                        '--dataset',
                        default=DEFAULT_DATA_DIR,
                        help='path to dataset')
    argparser.add_argument('-r',
                        '--runs',
                        default=DEFAULT_RUNS_DIR,
                        help='path to saved directory')
    argparser.add_argument('-m',
                        '--model',
                        default=DEFAULT_MODEL_PATH,
                        help='path to save model')
    argparser.add_argument('-e',
                        '--epochs',
                        default=DEFAULT_EPOCHS,
                        help='number of epochs')
    argparser.add_argument('-b',
                        '--batch',
                        default=DEFAULT_BATCH_SIZE,
                        help='batch size')
    args = argparser.parse_args()
    return args