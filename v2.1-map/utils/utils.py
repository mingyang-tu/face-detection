import cv2
from argparse import ArgumentParser

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--input', type=str)
    parser.add_argument('--resize', type=int, default=0)
    args = parser.parse_args()
    return args

def ResizeImage(img, max_len):
    imsize = img.shape
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
