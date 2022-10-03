import numpy as np
import math
from scipy import signal
import matplotlib.pyplot as plt
from typing import List, Tuple

class Eye_Mouth_Map():
    def __init__(self, img_YCbCr: np.float64, 
                       img_region: np.uint8, 
                       num_region: int, 
                       boundary: np.int64, 
                       centroid: np.float64, 
                       proj_axis: np.float64, 
                       axislength: np.float64):
        self.img_region = img_region
        self.imsize = img_region.shape
        self.num_region = num_region
        self.img_Y = img_YCbCr[:, :, 0]
        self.img_Cb = img_YCbCr[:, :, 1] + 127.5
        self.img_Cr = img_YCbCr[:, :, 2] + 127.5
        self.boundary = boundary
        self.centroid = centroid
        self.proj_axis = proj_axis
        self.axislength = axislength

    def FindEye(self) -> Tuple[List[List[np.float64]], List[List[float]]]:
        eye = [[] for i in range(self.num_region)]
        eye_value = [[] for i in range(self.num_region)]
        for i in range(self.num_region):
            top = self.boundary[1, i]
            bot = self.boundary[0, i]
            left = self.boundary[3, i]
            right = self.boundary[2, i]
            
            eyemapl = self.eyemapL(self.img_Y[top: bot, left: right], lamb=5)
            eyemapc = self.eyemapC(self.img_Cr[top: bot, left: right], self.img_Cb[top: bot, left: right])
            eyemapt = self.eyemapT(self.img_Y[top: bot, left: right], L=5)
            
            eyemapl = self.normalization(eyemapl)
            eyemapc = self.normalization(eyemapc)
            eyemapt = self.normalization(eyemapt)
            
            eyemap = eyemapl * 0.45 + eyemapc * 0.45 + eyemapt * 0.1

            radius = (self.boundary[0, i] - self.boundary[1, i]) // 40 + 1
            cir = self.circular_mask(radius)
            Ceyemap = signal.convolve2d(eyemap, cir, boundary='symm', mode='same')
                
            threshold = 0.5
            rowrange = radius
            colrange = radius
            EyeCan, EyeCan_value = self.find_local_max_with_threshold(Ceyemap, 
                                                                      threshold, 
                                                                      rowrange, colrange)

            shape_Can = EyeCan.shape
            num_Can = shape_Can[0]
            for j in range(num_Can):
                EyeCan[j, 0] = EyeCan[j, 0] + top
                EyeCan[j, 1] = EyeCan[j, 1] + left
            # ellipse matching
            EyeCan_mean = np.array(EyeCan)
            EyeCan_mean[:, 0] = EyeCan_mean[:, 0] - self.centroid[0, i]
            EyeCan_mean[:, 1] = EyeCan_mean[:, 1] - self.centroid[1, i]
            EyeCan_proj = np.dot(EyeCan_mean, self.proj_axis[i, :, :])
            for j in range(num_Can):
                temp = (EyeCan_proj[j, 0] / self.axislength[0, i]) ** 2 + (EyeCan_proj[j, 1] / self.axislength[1, i]) ** 2
                if temp < 0.8:
                    eye[i].append(np.array(EyeCan[j, :]))
                    eye_value[i].append(EyeCan_value[j])
            
            for j in range(len(eye_value[i])):
                if eye_value[i][j] >= 2.5:
                    eye_value[i][j] = 1
                elif eye_value[i][j] <= 0.5:
                    eye_value[i][j] = 0
                else:
                    eye_value[i][j] = (eye_value[i][j] - 0.5) / 2
        return eye, eye_value

    def FindMouth(self) -> Tuple[List[List[np.float64]], List[List[float]]]:
        mouth = [[] for i in range(self.num_region)]
        mouth_value = [[] for i in range(self.num_region)]
        for i in range(self.num_region):
            top = self.boundary[1, i]
            bot = self.boundary[0, i]
            left = self.boundary[3, i]
            right = self.boundary[2, i]
            mouthmap = self.mouthmap_cut(self.img_Cb[top: bot, left: right], 
                                         self.img_Cr[top: bot, left: right], 
                                         self.img_region[top: bot, left: right])

            Rowaxis = (self.boundary[0, i] - self.boundary[1, i]) // 25 + 1
            Colaxis = (self.boundary[2, i] - self.boundary[3, i]) // 8 + 1
            elli = self.ellipse_mask(Rowaxis, Colaxis)
            Cmouthmap = signal.convolve2d(mouthmap, elli, boundary='symm', mode='same')

            threshold = 7e6
            rowrange = int(Rowaxis)
            colrange = int(Colaxis)
            MouthCan, MouthCan_value = self.find_local_max_with_threshold(Cmouthmap, 
                                                                          threshold, 
                                                                          rowrange, colrange)
            
            shape_Can = MouthCan.shape
            num_Can = shape_Can[0]
            for j in range(num_Can):
                MouthCan[j, 0] = MouthCan[j, 0] + top
                MouthCan[j, 1] = MouthCan[j, 1] + left
            
            MouthCan_mean = np.array(MouthCan)
            MouthCan_mean[:, 0] = MouthCan_mean[:, 0] - self.centroid[0, i]
            MouthCan_mean[:, 1] = MouthCan_mean[:, 1] - self.centroid[1, i]
            MouthCan_proj = np.dot(MouthCan_mean, self.proj_axis[i, :, :])
            for j in range(num_Can):
                temp = (MouthCan_proj[j, 0] / self.axislength[0, i]) ** 2 + (MouthCan_proj[j, 1] / self.axislength[1, i]) ** 2
                if temp < 0.8:
                    mouth[i].append(np.array(MouthCan[j, :]))
                    mouth_value[i].append(MouthCan_value[j])
            
            for j in range(len(mouth_value[i])):
                if mouth_value[i][j] >= 3.2e7:
                    mouth_value[i][j] = 1
                elif mouth_value[i][j] <= 7e6:
                    mouth_value[i][j] = 0
                else:
                    mouth_value[i][j] = (mouth_value[i][j] - 5e6) / 2.5e7
        return mouth, mouth_value

    def DisplayEye(self, img: np.uint8, 
                         eye: Tuple[List[List[np.float64]]]) -> None:
        count_eye = 0
        plt.figure()
        plt.imshow(img)
        for i in range(self.num_region):
            l = len(eye[i])
            count_eye += l
            for e in range(l):
                plt.plot(eye[i][e][1], eye[i][e][0], 'o', c='r', markersize=3)
        plt.title(f"Number of eyes: {count_eye}")

    def DisplayMouth(self, img: np.uint8, 
                           mouth: Tuple[List[List[np.float64]]]) -> None:
        count_mouth = 0
        plt.figure()
        plt.imshow(img)
        for i in range(self.num_region):
            l = len(mouth[i])
            count_mouth += l
            for m in range(l):
                plt.plot(mouth[i][m][1], mouth[i][m][0], 'o', c='r', markersize=3)
        plt.title(f"Number of mouths: {count_mouth}")

    def eyemapL(self, img_Y: np.float64, 
                      lamb: float) -> np.float64:
        imsize = img_Y.shape
        img_Y_ero = self.ErosionGray(img_Y)
        img_Y_ero = self.ErosionGray(img_Y_ero)

        img_Y1 = img_Y_ero / 255
        eyemapl = np.zeros((imsize[0], imsize[1]))
        for i in range(imsize[0]):
            for j in range(imsize[1]):
                eyemapl[i, j] = (1 - img_Y1[i, j]) / (1 + lamb * img_Y1[i, j])
        return eyemapl

    def eyemapC(self, img_Cr: np.float64, 
                      img_Cb: np.float64) -> np.float64:
        imsize = img_Cr.shape
        maxCr = np.max(img_Cr)
        img_Cr1 = maxCr - img_Cr
        eyemapc = np.zeros((imsize[0], imsize[1]))
        for i in range(imsize[0]):
            for j in range(imsize[1]):
                eyemapc[i, j] = (img_Cb[i, j] ** 2 + img_Cr1[i, j] ** 2 + img_Cb[i, j] / img_Cr[i, j]) / 3
        return eyemapc

    def eyemapT(self, img_Y: np.float64, 
                      L: int) -> np.float64:
        imsize = img_Y.shape
        Y1x, Y1y = self.edge_detection(img_Y, sigma=1, L=L)
        Y2x, Y2y = self.edge_detection(img_Y, sigma=0.2, L=L)
        Y3x, Y3y = self.edge_detection(img_Y, sigma=0.05, L=L)

        eyemapt = np.zeros(imsize)
        for i in range(imsize[0]):
            for j in range(imsize[1]):
                eyemapt[i, j] = max(Y1x[i, j], Y1y[i, j], 
                                    Y2x[i, j], Y2y[i, j], 
                                    Y3x[i, j], Y3y[i, j])
        return eyemapt

    def mouthmap_cut(self, img_Cb: np.float64, 
                           img_Cr: np.float64, 
                           skin_region: np.uint8) -> np.float64:
        imsize = skin_region.shape
        Cr_sq = 0
        Cr_div_Cb = 1e-3
        for i in range(imsize[0]):
            for j in range(imsize[1]):
                if skin_region[i, j] >= 1:
                    Cr_sq = Cr_sq + img_Cr[i, j] ** 2
                    Cr_div_Cb = Cr_div_Cb + img_Cr[i, j] / img_Cb[i, j]
        eta = 0.95 * Cr_sq / Cr_div_Cb
        mouthmap = np.zeros(imsize)
        for i in range(imsize[0]):
            for j in range(imsize[1]):
                x = img_Cr[i, j] ** 2
                y = (img_Cr[i, j] ** 2 - eta * img_Cr[i, j] / img_Cb[i, j]) ** 2
                mouthmap[i, j] = max(x + y - (127.5)**2, 0)
        return mouthmap

    def normalization(self, img: np.float64) -> np.float64:
        mean = np.mean(img)
        std = np.std(img)
        img_new = (img - mean) / std
        return img_new
    
    def ErosionGray(self, img: np.float64) -> np.float64:
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

    def edge_detection(self, img_Y: np.float64, 
                             sigma: float, 
                             L: int) -> Tuple[np.float64, np.float64]:
        f = np.zeros(2*L+1)
        C = 0
        for i in range(L):
            C += np.exp(-sigma * (i+1))
        C = 1 / C
        for i in range(L):
            f[L+i+1] = C * np.exp(-sigma * (i+1))
            f[L-i-1] = -C * np.exp(-sigma * (i+1))

        fx = f.reshape((1, 2*L+1))
        fy = f.reshape((2*L+1, 1))
        Yx = signal.convolve2d(img_Y, fx, boundary='symm', mode='same')
        Yy = signal.convolve2d(img_Y, fy, boundary='symm', mode='same')
        Yx = np.abs(Yx)
        Yy = np.abs(Yy)
        return Yx, Yy

    def circular_mask(self, r: float) -> np.float64:
        r2 = r ** 2
        size = 2 * int(r) + 1
        SE = np.zeros((size, size))
        for i in range(size):
            for j in range(size):
                if ((i-r)**2 + (j-r)**2 <= r2):
                    SE[i, j] = 1 - (i-r)**2 / r2 - (j-r)**2 / r2
        SE = SE / np.sum(SE)
        return SE

    def ellipse_mask(self, r: float, c: float) -> np.float64:
        r2 = r ** 2
        c2 = c ** 2
        size_r = 2 * int(r) + 1
        size_c = 2 * int(c) + 1
        SE = np.ones((size_r, size_c))
        for i in range(size_r):
            for j in range(size_c):
                if ((i-r)**2 / r2 + (j-c)**2 / c2 > 1):
                    SE[i, j] = 0
        SE = SE / np.sum(SE)
        return SE

    def find_local_max_with_threshold(self, img: np.float64, 
                                            threshold: float, 
                                            rowrange: int, 
                                            colrange: int) -> Tuple[np.float64, np.float64]:
        result = []
        result_value = []
        imsize = img.shape
        for i in range(1, imsize[0]-1):
            for j in range(1, imsize[1]-1):
                if img[i, j] > threshold:   # bigger than threshold
                    M = np.max(img[max(i-rowrange, 0):min(i+rowrange+1, imsize[0]), \
                                   max(j-colrange, 0):min(j+colrange+1, imsize[1])])
                    # local maximum
                    if img[i, j] == M:
                        result.append([i, j])
                        result_value.append(img[i, j])
        num = len(result)
        result = np.array(result)
        result = np.reshape(result, (num, 2))
        result_value = np.array(result_value)
        return result, result_value

