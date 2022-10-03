import numpy as np
from libsvm.svmutil import svm_load_model, svm_predict
import cv2
from skimage import segmentation

def SkinDetection(img):
    # input: RGB img (np.uint8, shape: (row, col, channel))
    # output: binary image (np.uint8, shape: (row, col))
    
    img_slic, superpixels = SLIC(img)
    img_binary = SkinFilter(img_slic, superpixels)
    return img_binary

def SLIC(img):
    # input: RGB img (np.uint8, shape: (row, col, channel))
    # output: distribution of superpixels (np.int64, shape: (row, col))
    #         value of superpixels (np.uint8, shape: (n_segments, 1, 3))

    imsize = img.shape
    n_segments = int(imsize[0] * imsize[1] / 64)
    img_slic = segmentation.slic(img, 
                                 n_segments=n_segments, 
                                 compactness=10., 
                                 max_num_iter=10, 
                                 convert2lab=True,
                                 start_label=1)
    img_slic, _, _ = segmentation.relabel_sequential(img_slic, offset=1)
    n_segments = np.max(img_slic)

    sp_record = [[] for i in range(n_segments)]
    for i in range(imsize[0]):
        for j in range(imsize[1]):
            label = img_slic[i, j] - 1
            sp_record[label].append(img[i, j, :])
    superpixels = np.zeros((n_segments, 1, 3), dtype=np.uint8)
    for i in range(n_segments):
        superpixels[i, 0, :] = np.around(np.mean(np.array(sp_record[i]), axis=0), 
                                         decimals=0).astype(np.uint8)    
    return img_slic, superpixels

def SkinFilter(img_slic, superpixels):
    # input: distribution of superpixels (np.int64, shape: (row, col))
    #        value of superpixels (np.uint8, shape: (n_segments, 1, 3))
    # output: binary image (np.uint8, shape: (row, col))

    # load model
    model_brt = svm_load_model("./model_new/model_brt.ckpt")
    mean_brt = np.load("./model_new/mean_brt.npy") 
    std_brt = np.load("./model_new/std_brt.npy")

    model_2d = svm_load_model("./model_new/model_2d.ckpt")
    mean_2d = np.load("./model_new/mean_2d.npy") 
    std_2d = np.load("./model_new/std_2d.npy")

    length = superpixels.shape[0]
    img_Lab = cv2.cvtColor(superpixels, cv2.COLOR_RGB2LAB).astype(np.float64)
    img_HSV = cv2.cvtColor(superpixels, cv2.COLOR_RGB2HSV)

    # mask: bright 
    img_Lab_norm = np.zeros((length, 1, 3))
    for i in range(3):
        img_Lab_norm[:, :, i] = (img_Lab[:, :, i] - mean_brt[i]) / std_brt[i]
    p_labels, _, _ = svm_predict([], img_Lab_norm.squeeze(axis=1), model_brt, "-q")
    brt_mask = np.expand_dims(np.array(p_labels, dtype=np.uint8), axis=1)

    # mask: shadow
    img_Lab_norm = np.zeros((length, 1, 2))
    img_Lab_norm[:, :, 0] = (img_Lab[:, :, 1] - mean_2d[0]) / std_2d[0]
    img_Lab_norm[:, :, 1] = (img_Lab[:, :, 2] - mean_2d[1]) / std_2d[1]
    p_labels, _, _ = svm_predict([], img_Lab_norm.squeeze(axis=1), model_2d, "-q")
    dark_mask = np.expand_dims(np.array(p_labels, dtype=np.uint8), axis=1)
    
    result = cv2.bitwise_or(brt_mask, dark_mask)

    img_binary = result[img_slic - 1].squeeze(axis=2)
    return img_binary

def Closing(img_binary):
    imsize = img_binary.shape
    iterations = int(min(imsize[0], imsize[1]) / 100 + 0.5)
    kernel = np.array([[0, 1, 0], 
                       [1, 1, 1],
                       [0, 1, 0],], dtype=np.uint8)
    img_binary = cv2.dilate(img_binary, kernel, iterations=iterations)
    img_binary = cv2.erode(img_binary, kernel, iterations=iterations)
    return img_binary


