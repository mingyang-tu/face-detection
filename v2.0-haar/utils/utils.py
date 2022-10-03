import numpy as np
import cv2
from argparse import ArgumentParser
from skimage import segmentation
import matplotlib.pyplot as plt
from matplotlib import patches

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--input', type=str)
    parser.add_argument('--resize', type=int, default=0)
    parser.add_argument('--detail', type=bool, default=False)
    args = parser.parse_args()
    return args

def ResizeImage(img, max_len):
    imsize = np.shape(img)
    if max_len == 0:
        return img
    elif (imsize[0] >= imsize[1]) and (imsize[0] > max_len):
        longside = max_len
        shortside = (imsize[1] * max_len) // imsize[0]
        img = cv2.resize(img, (shortside, longside))
    elif (imsize[0] < imsize[1]) and (imsize[1] > max_len):
        longside = max_len
        shortside = (imsize[0] * max_len) // imsize[1]
        img = cv2.resize(img, (longside, shortside))
    return img

def DisplayDetails(img, img_binary, faces=[]):
    # input: RGB img (np.uint8, shape: (row, col, channel))
    #        binary image (np.uint8, shape: (row, col))

    mask3 = np.array([img_binary, img_binary, img_binary], dtype=np.uint8).transpose(1, 2, 0)
    img_new = img * mask3
    img_new = segmentation.mark_boundaries(img_new, img_binary, 
                                           color=(1, 1, 0), 
                                           background_label=0)
    plt.figure()
    plt.imshow(img_new)
    ax = plt.gca()
    for (x, y, w, h) in faces:
        rect = patches.Rectangle((x, y), w, h, linewidth=1.5, edgecolor='r', fill=False)
        ax.add_patch(rect)
