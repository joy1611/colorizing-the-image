import os
import os.path as osp
import cv2
import numpy as np
from torch.utils.data import Dataset
import random

def get_cifar10_data(data_path):
    if not osp.exists(data_path):
        # download
Expand All
	@@ -27,19 +28,19 @@ def get_cifar10_data(data_path):
def unpickle_batch(file):
    import pickle
    with open(file, 'rb') as f:
        dict = pickle.load(f, encoding='bytes')
    return dict


# extract images from batches
def extract_cifar10_images(data_path):

    data_batches = {}
    data_batches["test"] =  [osp.join(data_path, "cifar-10-batches-py", "test_batch")]
    data_batches["train"] = [osp.join(data_path, "cifar-10-batches-py", f) for f in ["data_batch_1", "data_batch_2",
                                                                                     "data_batch_3", "data_batch_4",
                                                                                     "data_batch_5"]]
    data_dirs = {}
    data_dirs["test"] = osp.join(data_path, "cifar-10-images", "test")
    data_dirs["train"] = osp.join(data_path, "cifar-10-images", "train")

Expand All
	@@ -53,8 +54,8 @@ def extract_cifar10_images(data_path):

                for image_name, image_parts in zip(batch[b'filenames'], batch[b'data']):
                    r, g, b = image_parts[0:1024], image_parts[1024:2048], image_parts[2048:]
                    r, g, b = np.reshape(r, (32,-1)), np.reshape(g, (32,-1)), np.reshape(b, (32,-1))
                    img = np.stack((b,g,r), axis=2)

                    save_path = osp.join(data_dirs[phase], image_name.decode("utf-8"))
                    cv2.imwrite(save_path, img)
Expand All
	@@ -70,21 +71,21 @@ def preprocess(img_bgr):
    # transform to lab
    img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    # normalize
    img_lab[:,:,0] = img_lab[:,:,0]/50 - 1
    img_lab[:,:,1] = img_lab[:,:,1]/127
    img_lab[:,:,2] = img_lab[:,:,2]/127
    # transpose
    img_lab = img_lab.transpose((2,0,1))
    return img_lab  


def postprocess(img_lab):    
    # transpose back
    img_lab = img_lab.transpose((1,2,0))
    # transform back
    img_lab[:,:,0] = (img_lab[:,:,0] + 1)*50
    img_lab[:,:,1] = img_lab[:,:,1]*127
    img_lab[:,:,2] = img_lab[:,:,2]*127 
    # transform to bgr
    img_bgr = cv2.cvtColor(img_lab, cv2.COLOR_LAB2BGR)
    # to int8
Expand Down
	