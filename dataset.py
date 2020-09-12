import os
import zipfile

import requests

import constants

'''
Download KITTI Road base kit (http://www.cvlibs.net/datasets/kitti/eval_road.php) if it does not exist
'''
def verify_dataset():
    if not os.path.exists(constants.data_train_image_dir):
        print("Dataset was not found. Downloading...")
        if not os.path.exists(constants.data_location):
            print("\"{}\" directory was not found. Creating directory...".format(constants.data_location))
            os.makedirs(constants.data_location)
        
        dataset_zip = dataset_url.split('/')[-1]
        dataset_zip_location = os.path.join(constants.data_location, dataset_zip)

        if not os.path.exists(dataset_zip_location):
            print("Downloading dataset from: {}".format(dataset_url))
            r = requests.get(dataset_url, allow_redirects=True)

            open(dataset_zip_location, 'wb').write(r.content)

        with zipfile.ZipFile(dataset_zip_location,"r") as zip_ref:
            zip_ref.extractall(constants.data_location)
        
        print("Dataset extracted and placed in directory \"{}\"".format(constants.data_location))
        os.remove(dataset_zip_location)
        print("Dataset is set up. {} deleted".format(dataset_zip_location))
    
    assert os.path.exists(constants.data_train_image_dir) and os.path.exists(constants.data_train_gt_dir), \
        "Error creating training data directory. Aborting..."

dataset_url = 'https://s3.eu-central-1.amazonaws.com/avg-kitti/data_road.zip'

if __name__ == "__main__":
    verify_dataset()
