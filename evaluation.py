import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy.misc
import scipy.ndimage
import sklearn.metrics

umg=cv2.imread(r'C:\Users\xiuxi\Desktop\ECE470\data_road/uu_road_000084.png', cv2.IMREAD_UNCHANGED)
ump=cv2.imread(r'C:\Users\xiuxi\Desktop\ECE470\data_road/uu_000084.png', cv2.IMREAD_UNCHANGED)
img=umg[:,:,0]
img2=ump[:,:,0]
dim = (640, 192)
# resize image
groundtruth=cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
predicted=cv2.resize(img2, dim, interpolation = cv2.INTER_AREA)
cv2.imshow("groundtruth", groundtruth)
cv2.waitKey(0)
cv2.imshow("predicted", predicted)
cv2.waitKey(0)
groundtruth_list=groundtruth.flatten() 
predicted_list=predicted.flatten()
a=sklearn.metrics.f1_score(groundtruth_list,predicted_list, labels=None, pos_label=1, average='weighted', sample_weight=None, zero_division='warn')
b=sklearn.metrics.accuracy_score(groundtruth_list, predicted_list, normalize=True, sample_weight=None)
print('f1 score:'+str(a))
print('accuracy:'+str(b))
