import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage import segmentation, measure
from scipy import signal
from utils.utils import Disjoint_Sets

def SkinSegmentation(img, binary):
    CC = measure.label(binary, connectivity=1)
    _, big_CC, area = np.unique(CC, return_inverse=True, return_counts=True)
    big_CC[np.isin(big_CC, np.where(area < 400))] = 0
    num_CC = np.sum(area > 400)

    binary = (big_CC > 0).astype(np.uint8).reshape(binary.shape)
    n_segments = int(np.sum(binary) / 64 + num_CC)
    img_slic = segmentation.slic(img, 
                                 n_segments=n_segments, 
                                 compactness=10., 
                                 max_num_iter=10, 
                                 convert2lab=True,
                                 start_label=1, 
                                 mask=binary)

    cluster = Clustering(img, img_slic, binary)
    output = cluster.Clustering(threshold=40)

    _, img_region, area = np.unique(output, return_inverse=True, return_counts=True)
    img_region[np.isin(img_region, np.where(area < 400))] = 0
    new_label, img_region = np.unique(img_region, return_inverse=True)
    img_region = img_region.reshape(binary.shape)
    num_region = new_label.shape[0] - 1

    return img_region, num_region

def DisplaySkin(img, img_region, num_region):
    mask = (img_region > 0).astype(np.uint8)
    mask3 = np.array([mask, mask, mask], dtype=np.uint8).transpose(1, 2, 0)
    img_new = img * mask3
    img_new = segmentation.mark_boundaries(img_new, img_region, 
                                           color=(0, 1, 0), 
                                           background_label=0)
    plt.figure()
    plt.imshow(img_new)
    plt.title(f"Number of skin regions: {num_region}")

class Clustering():
    def __init__(self, img, img_slic, img_binary):
        self.img = img
        self.img_slic = img_slic
        self.mask = img_binary
        self.imsize = img_binary.shape
        self.img_Lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.float64)

        img_roi = self.img_slic * self.mask

        self.img_roi, _, _ = segmentation.relabel_sequential(img_roi, offset=1)
        self.n_segments = np.max(self.img_roi)

    def mean_color(self):
        colors = np.zeros((self.n_segments+1, 3))
        counts = np.zeros(self.n_segments+1)
        for i in range(self.imsize[0]):
            for j in range(self.imsize[1]):
                idx = self.img_roi[i, j]
                if idx != 0:
                    colors[idx, :] += self.img_Lab[i, j, :]
                    counts[idx] += 1
        for i in range(self.n_segments+1):
            if counts[i] != 0:
                colors[i, :] /= counts[i]
        return colors

    def Cal_Grad(self, gray, sigma=0.2, L=10):
        C = 1. / np.sum(np.exp(-sigma * (np.arange(1, L+1))))

        f = np.zeros(2*L+1)
        f[:L] = -C * np.exp(-sigma * (np.arange(L, 0, -1)))
        f[L+1:] = C * np.exp(-sigma * (np.arange(1, L+1)))
        
        fx = f.reshape((1, 2*L+1))
        fy = f.reshape((2*L+1, 1))
        Yx = np.abs(signal.convolve2d(gray, fx, boundary='symm', mode='same'))
        Yy = np.abs(signal.convolve2d(gray, fy, boundary='symm', mode='same'))

        return Yx, Yy

    def Grad_Map(self, gray, L=10):
        Y1x, Y1y = self.Cal_Grad(gray, sigma=1, L=L)
        Y2x, Y2y = self.Cal_Grad(gray, sigma=0.2, L=L)
        Y3x, Y3y = self.Cal_Grad(gray, sigma=0.05, L=L)
        output = np.max(np.array([Y1x, Y1y, Y2x, Y2y, Y3x, Y3y]), axis=0)
        return output

    def add_dict(self, dictionary, x, y, target):
        if x in dictionary[y]:
            dictionary[y][x] += target
        else:
            dictionary[y][x] = target
        if y in dictionary[x]:
            dictionary[x][y] += target
        else:
            dictionary[x][y] = target

    def avg_grad(self, grad):
        grad_edge = [dict() for i in range(self.n_segments+1)]
        counts = [dict() for i in range(self.n_segments+1)]
        for i in range(1, self.imsize[0]):
            for j in range(1, self.imsize[1]):
                present = self.img_roi[i, j]
                if present != 0:
                    top = self.img_roi[i-1, j]
                    if (top != 0) and (present != top):
                        gradient = (grad[i, j] + grad[i-1, j]) / 2
                        self.add_dict(grad_edge, present, top, gradient)
                        self.add_dict(counts, present, top, 1)
                    left = self.img_roi[i, j-1]
                    if (left != 0) and (present != left):
                        gradient = (grad[i, j] + grad[i-1, j]) / 2
                        self.add_dict(grad_edge, present, left, gradient)
                        self.add_dict(counts, present, left, 1)
        for i in range(self.n_segments+1):
            for j, key in enumerate(grad_edge[i].keys()):
                grad_edge[i][key] /= counts[i][key]
        return grad_edge

    def Clustering(self, threshold):
        colors = self.mean_color()
        colors[:, 0] /= 3
        grad_L = self.Grad_Map(self.img_Lab[:, :, 0])
        grad_edge_L = self.avg_grad(grad_L)

        linked_list = [dict() for i in range(self.n_segments+1)]
        for i in range(1, self.imsize[0]):
            for j in range(1, self.imsize[1]):
                present = self.img_roi[i, j]
                if present != 0:
                    top = self.img_roi[i-1, j]
                    if (top != 0) and (present != top):
                        linked_list[present][top] = 1
                    left = self.img_roi[i, j-1]
                    if (left != 0) and (present != left):
                        linked_list[present][left] = 1

        DS = Disjoint_Sets(self.n_segments+1)
        for i in range(1, self.n_segments+1):
            for _, key in enumerate(linked_list[i].keys()):
                color_diff = np.sum(np.abs(colors[i, :] - colors[key, :]))
                gL = grad_edge_L[i][key]
                criteria = color_diff + gL * 0.45
                if criteria <= threshold:
                    DS.union(i, key)

        img_roi = np.zeros(self.imsize, dtype=np.int64)
        for i in range(self.imsize[0]):
            for j in range(self.imsize[1]):
                img_roi[i, j] = DS.find_set(self.img_roi[i, j])

        return img_roi
