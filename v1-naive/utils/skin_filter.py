import numpy as np
from libsvm.svmutil import svm_load_model, svm_predict
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import Tuple, List
import cv2

def SkinSegmentation(img_YCbCr: np.float64) -> Tuple[np.uint8, int]:
    img_binary = SkinFilter(img_YCbCr)
    img_binary = Closing(img_binary)
    img_binary = Opening(img_binary)
    
    threshold = 200
    img_region, num_region = Connected_Components(img_binary, threshold)
    return img_region, num_region

def SkinFilter(img_YCbCr: np.float64) -> np.uint8:
    # load model
    model = svm_load_model("./model_new/model.ckpt")
    mean = np.load("./model_new/mean.npy") 
    std = np.load("./model_new/std.npy")

    imsize = img_YCbCr.shape

    # normalization
    img_YCbCr_norm = np.zeros(imsize)
    img_YCbCr_norm[:, :, 0] = (img_YCbCr[:, :, 0] - mean[0]) / std[0]
    img_YCbCr_norm[:, :, 1] = (img_YCbCr[:, :, 1] - mean[1]) / std[1]
    img_YCbCr_norm[:, :, 2] = (img_YCbCr[:, :, 2] - mean[2]) / std[2]

    img_YCbCr_norm = img_YCbCr_norm.reshape(imsize[0]*imsize[1], 3)
    # predict
    p_labels, _, _ = svm_predict([], img_YCbCr_norm, model, "-q")
    img_binary = np.array(p_labels, dtype=np.uint8).reshape(imsize[0], imsize[1])

    return img_binary

def Closing(img_binary: np.uint8) -> np.uint8:
    kernel = np.ones((5, 5), dtype=np.uint8)
    img_binary = cv2.dilate(img_binary, kernel, iterations=1)
    img_binary = cv2.erode(img_binary, kernel, iterations=1)
    return img_binary

def Opening(img_binary: np.uint8) -> np.uint8:
    kernel = np.ones((5, 5), dtype=np.uint8)
    img_binary = cv2.erode(img_binary, kernel, iterations=1)
    img_binary = cv2.dilate(img_binary, kernel, iterations=1)
    return img_binary

class Disjoint_Sets():
    def __init__(self, n: int): # sets: 0 ~ n-1
        self.size = n
        self.parent = list(range(n))
        self.rank = [0 for i in range(n)]
    
    def find_set(self, x: int) -> int:
        if x != self.parent[x]:
            self.parent[x] = self.find_set(self.parent[x])
        return self.parent[x]
    
    def link(self, x: int, y: int) -> None:
        if self.rank[x] > self.rank[y]:
            self.parent[y] = x
        else:
            self.parent[x] = y
            if self.rank[x] == self.rank[y]:
                self.rank[y] = self.rank[y] + 1
                
    def union(self, x: int, y: int) -> None:
        self.link(self.find_set(x), self.find_set(y))

def Connected_Components(img: np.uint8, 
                         threshold: float) -> Tuple[np.int64, int]: # input: 2D-array
    imsize = img.shape
    CC = np.zeros((imsize[0]+1, imsize[1]+1), dtype=int)
    label_lib = []
    num_CC = 0
    for i in range(1, imsize[0]+1):
        for j in range(1, imsize[1]+1):
            if img[i-1, j-1] == 1:
                up = CC[i-1, j]
                left = CC[i, j-1]
                if (up == 0) and (left == 0):
                    num_CC = num_CC + 1
                    CC[i, j] = num_CC
                elif up == 0:
                    CC[i, j] = left
                elif left == 0:
                    CC[i, j] = up
                else:
                    if up <= left:
                        CC[i, j] = up
                    else:
                        CC[i, j] = left
                    label_lib.append((left, up))
    num_CC += 1
    
    CC = CC[1:, 1:]
    label_set = Disjoint_Sets(num_CC)
    for x, y in label_lib:
        label_set.union(x, y)

    area_CC = np.zeros(num_CC, dtype=int)
    
    for i in range(imsize[0]):
        for j in range(imsize[1]):
            if CC[i, j] > 0:
                CC[i, j] = label_set.find_set(CC[i, j])
                area_CC[CC[i, j]] += 1

    map_new = np.zeros(num_CC, dtype=int)
    num_region = 0
    for i in range(1, num_CC):
        if area_CC[i] > threshold:
            num_region += 1
            map_new[i] = num_region

    
    CC = map_new[CC]

    return CC, num_region
    
def DisplaySkin(img_region: np.uint8, num_region: int, 
                centroid: np.float64, 
                boundary: np.int64) -> None:
    plt.figure()
    plt.imshow(img_region, cmap='gray', vmin=0, vmax=1)
    ax = plt.gca()
    for i in range(num_region):
        plt.plot(centroid[1, i], centroid[0, i], '+', color='r')
        rect = patches.Rectangle((boundary[3, i], boundary[1, i]), 
                                 boundary[2, i]-boundary[3, i], boundary[0, i]-boundary[1, i],
                                 linewidth=2, edgecolor='b', fill=False)
        ax.add_patch(rect)
    plt.title(f"Number of skin regions: {num_region}")

