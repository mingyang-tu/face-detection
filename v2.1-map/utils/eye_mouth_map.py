import numpy as np
import cv2
from scipy import signal
import matplotlib.pyplot as plt

class Eye_Mouth_Map():
    def __init__(self, roi, img_binary):
        self.roi = roi
        self.imsize = roi.shape
        img_YCrCb = cv2.cvtColor(roi, cv2.COLOR_RGB2YCrCb).astype(np.float64)
        self.img_Y = img_YCrCb[:, :, 0]
        self.img_Cr = img_YCrCb[:, :, 1]
        self.img_Cb = img_YCrCb[:, :, 2]
        self.img_binary = img_binary.astype(np.float64)

    def FindEye(self):
        eyemapl = self.eyemapL(self.img_Y, lamb=5)
        eyemapc = self.eyemapC(self.img_Cr, self.img_Cb)
        eyemapt = self.eyemapT(self.img_Y, L=5)

        mask = self.image_mask(self.imsize[0], self.imsize[1], 0.8)
            
        eyemapl = self.normalization(eyemapl, mask)
        eyemapc = self.normalization(eyemapc, mask)
        eyemapt = self.normalization(eyemapt, mask)
            
        eyemap = eyemapl * 0.5 + eyemapc * 0.4 + eyemapt * 0.1

        radius = int(self.imsize[0] / 40 + 1)
        cir = self.sphere_mask(radius)
        Ceyemap = signal.convolve2d(eyemap, cir, boundary='symm', mode='same')

        min_value = np.min(Ceyemap)
        Ceyemap[mask == 0] = min_value - 1e-3
        Ceyemap[int(self.imsize[0] * 0.6):, :] = min_value - 1e-3

        threshold = Otsu_Threshold(Ceyemap, nrange=(min_value, np.max(Ceyemap))) + 0.75

        # self.Display(Ceyemap, threshold)

        rowrange = int(radius / 2 + 1)
        colrange = int(radius / 2 + 1)
        eyes = self.find_local_max_with_threshold(Ceyemap, 
                                                  threshold,
                                                  rowrange, colrange)
        return eyes

    def FindMouth(self):
        mask = self.image_mask(self.imsize[0], self.imsize[1], 0.9)
        mouthmap = self.mouthmap(self.img_Cb * mask, self.img_Cr * mask)
        mouthmap = self.normalization(mouthmap, mask)

        Rowaxis = int(self.imsize[0] // 25 + 1)
        Colaxis = int(self.imsize[1] // 8 + 1)
        elli = self.ellipse_mask(Rowaxis, Colaxis)
        Cmouthmap = signal.convolve2d(mouthmap, elli, boundary='symm', mode='same')

        min_value = np.min(Cmouthmap)
        Cmouthmap[mask == 0] = min_value - 1e-3
        Cmouthmap[:int(self.imsize[0] * 0.4), :] = min_value - 1e-3

        threshold = Otsu_Threshold(Cmouthmap, nrange=(min_value, np.max(Cmouthmap))) + 1.5

        # self.Display(Cmouthmap, threshold)

        rowrange = int(Rowaxis / 2 + 1)
        colrange = int(Colaxis / 2 + 1)
        mouths = self.find_local_max_with_threshold(Cmouthmap,
                                                    threshold, 
                                                    rowrange, colrange)
        return mouths

    def eyemapL(self, img_Y, lamb):
        imsize = img_Y.shape
        num_of_ero = int(min(imsize[0], imsize[1]) / 100)
        img_Y_ero = self.ErosionGray(img_Y)
        for i in range(num_of_ero):
            img_Y_ero = self.ErosionGray(img_Y_ero)

        img_Y1 = img_Y_ero / 255
        output = (1 - img_Y1) / (1 + lamb * img_Y1)
        return output

    def eyemapC(self, img_Cr, img_Cb):
        maxCr = np.max(img_Cr)
        img_Cr1 = 255. - img_Cr
        output = (img_Cb ** 2 + img_Cr1 ** 2 + img_Cb / (img_Cr+1e-5)) / 3
        return output

    def eyemapT(self, img_Y, L):
        Y1x, Y1y = self.edge_detection(img_Y, sigma=1, L=L)
        Y2x, Y2y = self.edge_detection(img_Y, sigma=0.2, L=L)
        Y3x, Y3y = self.edge_detection(img_Y, sigma=0.05, L=L)
        output = np.max(np.array([Y1x, Y1y, Y2x, Y2y, Y3x, Y3y]), axis=0)
        return output

    def mouthmap(self, img_Cb, img_Cr):
        Cr_sq = (img_Cr ** 2) * self.img_binary
        Cr_div_Cb = (img_Cr / (img_Cb+1e-5)) * self.img_binary
        eta = 0.85 * np.sum(Cr_sq) / (np.sum(Cr_div_Cb) + 1e-5)
        output = img_Cr ** 2 + (img_Cr ** 2 - eta * img_Cr / (img_Cb+1e-5)) ** 2 - (127.5)**2
        output[output < 0] = 0
        return output

    def Smap(self, orig_map):
        M = np.max(orig_map)
        p = 0.2 * M
        q = M
        m = (p + q) / 2
        smap = np.array(orig_map)
        imsize = orig_map.shape
        for i in range(imsize[0]):
            for j in range(imsize[1]):
                if orig_map[i, j] < p:
                    smap[i, j] = 0
                elif orig_map[i, j] < m:
                    smap[i, j] = 2 * ((orig_map[i, j] - p) / (q - p)) ** 2
                elif orig_map[i, j] < q:
                    smap[i, j] = 1 - 2 * ((orig_map[i, j] - q) / (q - p)) ** 2
                else:
                    smap[i, j] = 1
        return smap

    def normalization(self, img, binary):
        N = np.sum(binary)
        mean = np.sum(img * binary) / N
        std = np.sqrt(np.sum((img - mean)**2 * binary) / N)
        img_new = (img - mean) / std
        return img_new
    
    def ErosionGray(self, img):
        imsize = img.shape
        img_copy = np.ones((imsize[0]+2, imsize[1]+2)) * 255
        img_copy[1:imsize[0]+1, 1:imsize[1]+1] = img
        img_new = np.zeros(imsize)
        for i in range(imsize[0]):
            for j in range(imsize[1]):
                img_new[i, j] = min(img_copy[i+1, j+1], 
                                    img_copy[i, j+1],
                                    img_copy[i+1, j], 
                                    img_copy[i+1, j+2], 
                                    img_copy[i+2, j+1])
        return img_new

    def edge_detection(self, img_Y, sigma, L):
        C = 1. / np.sum(np.exp(-sigma * (np.arange(1, L+1))))

        f = np.zeros(2*L+1)
        f[:L] = -C * np.exp(-sigma * (np.arange(L, 0, -1)))
        f[L+1:] = C * np.exp(-sigma * (np.arange(1, L+1)))
        
        fx = f.reshape((1, 2*L+1))
        fy = f.reshape((2*L+1, 1))
        Yx = np.abs(signal.convolve2d(img_Y, fx, boundary='symm', mode='same'))
        Yy = np.abs(signal.convolve2d(img_Y, fy, boundary='symm', mode='same'))

        return Yx, Yy

    def sphere_mask(self, r):
        Mv, Nv = np.mgrid[-r: r+1, -r: r+1].astype(np.float64)
        SE = 1 - Mv ** 2 / r**2 - Nv ** 2 / r**2
        SE[SE <= 0] = 0
        SE = SE / np.sum(SE)
        return SE

    def ellipse_mask(self, r, c):
        Mv, Nv = np.mgrid[-r: r+1, -c: c+1].astype(np.float64)
        SE = Mv ** 2 / r**2 + Nv ** 2 / c**2
        SE[SE <= 1] = 1
        SE[SE > 1] = 0
        SE = SE / np.sum(SE)
        return SE

    def image_mask(self, r, c, ratio):
        a = -1 if r % 2 == 0 else 0
        b = -1 if c % 2 == 0 else 0
        x = int(r / 2)
        y = int(c / 2)
        rowaxis = x * ratio
        colaxis = y * ratio

        Mv, Nv = np.mgrid[-x: x+1+a, -y: y+1+b].astype(np.float64)
        SE = Mv ** 2 / rowaxis**2 + Nv ** 2 / colaxis**2
        SE[SE <= 1] = 1
        SE[SE > 1] = 0
        return SE

    def find_local_max_with_threshold(self, img, threshold, rowrange, colrange):
        result = []
        imsize = img.shape
        for i in range(1, imsize[0]-1):
            for j in range(1, imsize[1]-1):
                if img[i, j] > threshold:   # bigger than threshold
                    M = np.max(img[max(i-rowrange, 0):min(i+rowrange+1, imsize[0]), \
                                   max(j-colrange, 0):min(j+colrange+1, imsize[1])])
                    # local maximum
                    if img[i, j] == M:
                        result.append([j, i])
        return result

    def Display(self, map1, threshold):
        plt.figure()
        plt.subplot(131)
        plt.imshow(self.roi)
        plt.subplot(132)
        plt.imshow(map1, cmap="jet")
        plt.subplot(133)
        plt.imshow((map1 > threshold).astype(np.uint8), cmap="gray")
        plt.title(f"threshold = {threshold:.3f}")

def Otsu_Threshold(img, nbins=256, nrange=None):
    imsize = img.shape

    PDF, bin_edges = np.histogram(img, bins=nbins, range=nrange)
    PDF = PDF / (imsize[0] * imsize[1])
    bin_values = (bin_edges[:-1] + bin_edges[1:]) / 2

    q1 = np.zeros(nbins)
    for t in range(1, nbins):
        q1[t] = q1[t-1] + PDF[t-1]
    q2 = 1. - q1

    Exp = np.zeros(nbins)
    for i in range(1, nbins):
        Exp[i] = Exp[i-1] + bin_values[i-1] * PDF[i-1]
    Exp_all = Exp[nbins-1] + bin_values[nbins-1] * PDF[nbins-1]

    mu1 = np.zeros(nbins)
    mu2 = np.zeros(nbins)
    for t in range(1, nbins):
        if q1[t] != 0:
            mu1[t] = Exp[t] / q1[t]
        if q2[t] != 0:
            mu2[t] = (Exp_all - Exp[t]) / q2[t]

    sigma1_sq = np.zeros(nbins)
    sigma2_sq = np.zeros(nbins)
    for t in range(1, nbins):
        if q1[t] != 0:
            sigma1_sq[t] = np.sum(((bin_values[:t] - mu1[t]) ** 2) * PDF[:t]) / q1[t]
        if q2[t] != 0:
            sigma2_sq[t] = np.sum(((bin_values[t:] - mu2[t]) ** 2) * PDF[t:]) / q2[t]

    sigmaw_sq = q1 * sigma1_sq + q2 * sigma2_sq

    best_t = np.argmin(sigmaw_sq[1:])

    threshold = bin_values[best_t]
            
    return threshold

