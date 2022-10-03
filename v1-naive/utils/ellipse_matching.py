import numpy as np
import math
from typing import List

class Ellipse_Matching():
    def __init__(self, 
                 img_region: np.uint8, 
                 num_region: int):
        self.img_region = img_region
        self.num_region = num_region

        self.imsize = self.img_region.shape

        self.area = np.zeros(self.num_region)
        self.centroid = np.zeros((2, self.num_region))  # row, col
        for i in range(self.imsize[0]):
            for j in range(self.imsize[1]):
                if self.img_region[i, j] > 0:
                    index = self.img_region[i, j] - 1
                    self.area[index] += 1.
                    self.centroid[0, index] += i
                    self.centroid[1, index] += j
        for i in range(self.num_region):
            self.centroid[:, i] = self.centroid[:, i] / self.area[i]

    def Delete_Invalid_Region(self, 
                              valid: List[bool], 
                              length: int, 
                              target: np.float64) -> np.float64:
        output = np.zeros(length)
        count = 0
        for i in range(len(valid)):
            if valid[i]:
                output[count] = target[i]
                count += 1
        return output

    def PCA(self) -> None:
        # construct matrix Z
        matZ = []
        for i in range(self.num_region):
            matZ.append(np.zeros((int(self.area[i]), 2)))
        count = np.zeros(self.num_region, dtype=int)
        for i in range(self.imsize[0]):
            for j in range(self.imsize[1]):
                if self.img_region[i, j] > 0:
                    index = self.img_region[i, j] - 1
                    matZ[index][count[index], 0] = i - self.centroid[0, index]
                    matZ[index][count[index], 1] = j - self.centroid[1, index]
                    count[index] += 1

        # ellipse matching
        eigvector = []
        axis = []   # major, minor
        valid_skin = []
        for i in range(self.num_region):
            A = self.area[i]
            Z = np.array(matZ[i])
            # Z1 = Zt * Z
            Zsym = Z.T.dot(Z)
            # eigenvector
            eigval, eigvec = np.linalg.eig(Zsym)
            # put principle axis at column 0
            if eigval[1] > eigval[0]:
                eigvec[:, [0, 1]] = eigvec[:, [1, 0]]
            
            # projection
            ZE = Z.dot(eigvec)
            # calculate the major and minor radius
            fir_moment0 = np.sum(np.abs(ZE[:, 0])) / A
            fir_moment1 = np.sum(np.abs(ZE[:, 1])) / A
            sec_moment0 = np.sum(np.square(ZE[:, 0])) / A
            sec_moment1 = np.sum(np.square(ZE[:, 1])) / A
            
            a0 = 3 * math.pi * fir_moment0 / 4
            a1 = 2 * np.sqrt(sec_moment0)
            b0 = 3 * math.pi * fir_moment1 / 4
            b1 = 2 * np.sqrt(sec_moment1)
            
            mj = (a0 + a1) / 2
            mn = (b0 + b1) / 2
            
            # delete the wrong region
            if mj / mn > 3:
                valid_skin.append(False)
            else:
                valid_skin.append(True)
                eigvector.append(np.array(eigvec))
                axis.append([mj, mn])

        self.proj_axis = np.array(eigvector)
        self.angle = np.degrees(np.arctan(self.proj_axis[:, 0, 0] / (self.proj_axis[:, 1, 0] + 1e-4)))
        self.axislength = np.array(axis).T

        self.num_region = self.angle.shape[0]
        self.area = self.Delete_Invalid_Region(valid_skin, self.num_region, self.area)
        self.centroid[0, :self.num_region] = self.Delete_Invalid_Region(valid_skin, self.num_region, self.centroid[0, :])
        self.centroid[1, :self.num_region] = self.Delete_Invalid_Region(valid_skin, self.num_region, self.centroid[1, :])
        self.centroid = self.centroid[:, :self.num_region]

        map_new = dict()
        count = 1
        for i in range(len(valid_skin)):
            if valid_skin[i]:
                map_new[i+1] = count
                count += 1

        for i in range(self.imsize[0]):
            for j in range(self.imsize[1]):
                if self.img_region[i, j] > 0:
                    if not valid_skin[self.img_region[i, j]-1]:
                        self.img_region[i, j] = 0
                    else:
                        self.img_region[i, j] = map_new[self.img_region[i, j]]
