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
    parser.add_argument('--detail', type=int, default=0)
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

def get_boundary(binary):
    rows = np.any(binary, axis=1)
    cols = np.any(binary, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return rmin, rmax, cmin, cmax

def RotateImage(binary, angle):
    imsize = binary.shape
    center = (imsize[1] / 2, imsize[0] / 2)

    rotation_mat = cv2.getRotationMatrix2D(center, angle, 1.)

    abs_cos = abs(rotation_mat[0, 0]) 
    abs_sin = abs(rotation_mat[0, 1])

    newsize0 = int(imsize[0] * abs_sin + imsize[1] * abs_cos)
    newsize1 = int(imsize[0] * abs_cos + imsize[1] * abs_sin)

    rotation_mat[0, 2] += newsize0 / 2 - center[0]
    rotation_mat[1, 2] += newsize1 / 2 - center[1]

    output = cv2.warpAffine(binary, rotation_mat, (newsize0, newsize1))

    return output, rotation_mat

class Disjoint_Sets():
    def __init__(self, n): # sets: 0 ~ n-1
        self.size = n
        self.parent = list(range(n))
        self.rank = [0 for i in range(n)]
    def find_set(self, x):
        if x != self.parent[x]:
            self.parent[x] = self.find_set(self.parent[x])
        return self.parent[x]
    def link(self, x, y):
        if self.rank[x] > self.rank[y]:
            self.parent[y] = x
        else:
            self.parent[x] = y
            if self.rank[x] == self.rank[y]:
                self.rank[y] = self.rank[y] + 1     
    def union(self, x, y):
        self.link(self.find_set(x), self.find_set(y))

def DisplayDetails(img, img_binary, faces=[]):
    # input: RGB img (np.uint8, shape: (row, col, channel))
    #        binary image (np.uint8, shape: (row, col))

    mask3 = np.array([img_binary, img_binary, img_binary], dtype=np.uint8).transpose(1, 2, 0)
    img_new = img * mask3
    img_new = segmentation.mark_boundaries(img_new, img_binary, 
                                           color=(0, 1, 0), 
                                           background_label=0)
    plt.figure()
    plt.imshow(img_new)
    ax = plt.gca()
    for (x, y, w, h) in faces:
        rect = patches.Rectangle((x, y), w, h, linewidth=1.5, edgecolor='r', fill=False)
        ax.add_patch(rect)
