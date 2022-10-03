import numpy as np
import math
from scipy import signal

def Blur(img, L=10, sigma=0.1):
    Mv, Nv = np.mgrid[-L: L+1, -L: L+1].astype(np.float32)
    filt = np.exp(-math.pi * sigma * (Mv ** 2 + Nv ** 2))
    filt = filt / np.sum(filt)
    img_blur = np.zeros(img.shape, dtype=np.float32)
    for i in range(3):
        img_blur[:, :, i] = signal.convolve2d(img[:, :, i].astype(np.float32), filt, boundary='symm', mode='same')
    img_blur = img_blur.astype(np.uint8)
    return img_blur
