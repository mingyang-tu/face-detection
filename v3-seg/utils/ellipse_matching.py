import numpy as np
import math

def centroid(row_idx, col_idx):
    length = row_idx.shape[0]
    sum_x = np.sum(row_idx)
    sum_y = np.sum(col_idx)
    return sum_x / length, sum_y / length

def PCA(binary):
    area = np.sum(binary)
    row_idx, col_idx = np.where(binary == 1)
    row_idx, col_idx = row_idx.astype(np.float64), col_idx.astype(np.float64)
    c_row, c_col = centroid(row_idx, col_idx)

    # construct matrix Z
    matZ = np.array([row_idx-c_row, col_idx-c_col]).T

    Zsym = matZ.T.dot(matZ)

    # eigenvector
    eigval, eigvec = np.linalg.eig(Zsym)
    # put principle axis at column 0
    if eigval[1] > eigval[0]:
        eigvec[:, [0, 1]] = eigvec[:, [1, 0]]

    # projection
    ZE = matZ.dot(eigvec)
    # calculate the major and minor radius
    fir_moment0 = np.sum(np.abs(ZE[:, 0])) / area
    fir_moment1 = np.sum(np.abs(ZE[:, 1])) / area
    sec_moment0 = np.sum(np.square(ZE[:, 0])) / area
    sec_moment1 = np.sum(np.square(ZE[:, 1])) / area

    a0 = 3 * math.pi * fir_moment0 / 4
    a1 = 2 * np.sqrt(sec_moment0)
    b0 = 3 * math.pi * fir_moment1 / 4
    b1 = 2 * np.sqrt(sec_moment1)
            
    major = (a0 + a1) / 2
    minor = (b0 + b1) / 2

    in_ellipse = ((np.square(matZ[:, 0]) / major**2 + np.square(matZ[:, 1]) / minor**2) < 1)

    if (np.sum(in_ellipse.astype(np.float64)) > 0.7 * area) and (major / minor < 3):
        valid = True
    else:
        valid = False

    angle = np.degrees(np.arctan(eigvec[0, 0] / (eigvec[1, 0] + 1e-4)))

    return valid, angle

