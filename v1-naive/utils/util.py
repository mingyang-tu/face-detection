import numpy as np
import cv2
from argparse import ArgumentParser, Namespace
from typing import Tuple

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument('--dir', type=str)
    parser.add_argument('--input', type=str)
    parser.add_argument('--resize', type=int, default=512)
    parser.add_argument('--detail', type=bool, default=False)
    args = parser.parse_args()

    return args

def ResizeImage(img: np.uint8, max_len: int) -> np.float64:
    imsize = np.shape(img)
    if (imsize[0] >= imsize[1]) and (imsize[0] > max_len):
        longside = max_len
        shortside = (imsize[1] * max_len) // imsize[0]
        img = cv2.resize(img, (shortside, longside))
    elif (imsize[0] < imsize[1]) and (imsize[1] > max_len):
        longside = max_len
        shortside = (imsize[0] * max_len) // imsize[1]
        img = cv2.resize(img, (longside, shortside))
    return img

def RGB2YCbCr(img: np.float64) -> np.float64:
    imsize = np.shape(img)
    img_new = np.zeros(imsize)
    rgb2ycc = np.array([[0.299, 0.587, 0.114], 
                        [-0.169, -0.331, 0.500], 
                        [0.500, -0.419, -0.081]])
    for i in range(imsize[0]):
        for j in range(imsize[1]):
            img_new[i, j, :] = rgb2ycc.dot(img[i, j, :])
    return img_new

def get_boundary(img_region: np.uint8, num_region: int) -> Tuple[np.int64]:
    imsize = img_region.shape
    boundary = np.array([np.zeros(num_region, dtype=int), 
                         np.ones(num_region, dtype=int) * imsize[0], 
                         np.zeros(num_region, dtype=int), 
                         np.ones(num_region, dtype=int) * imsize[1]], dtype=int)
    # rM, rm, cM, cm
    for i in range(imsize[0]):
        for j in range(imsize[1]):
            if img_region[i, j] >= 1:
                index = img_region[i, j] - 1
                if i > boundary[0, index]:
                    boundary[0, index] = i
                if i < boundary[1, index]:
                    boundary[1, index] = i
                if j > boundary[2, index]:
                    boundary[2, index] = j
                if j < boundary[3, index]:
                    boundary[3, index] = j
    return boundary
