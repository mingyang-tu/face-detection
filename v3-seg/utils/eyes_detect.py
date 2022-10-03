import cv2
import numpy as np
from utils.utils import Disjoint_Sets

def Detect_Eyes(roi, lefteyeCascade, righteyeCascade): 
    # input: grayscale (np.uint8, shape: (row, col))
    #        left eye detector (cv2.CascadeClassifier)
    #        right eye detector (cv2.CascadeClassifier)
    # output: eyes positions (List[List[float]])

    # upsampling
    if max(roi.shape) <= 100:
        roi_copy = cv2.resize(roi, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        upsample = 2
    elif max(roi.shape) <= 50:
        roi_copy = cv2.resize(roi, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
        upsample = 1
    else:
        roi_copy = np.array(roi, dtype=np.uint8)
        upsample = 0

    imsize = roi_copy.shape
    maxSize = (int(imsize[0] / 2), int(imsize[1] / 2))
    eyes = []
    # left eyes
    eyes_l = lefteyeCascade.detectMultiScale(roi_copy,
                                             scaleFactor=1.02,
                                             minNeighbors=3, 
                                             maxSize=maxSize)
    eyes.extend(eyes_l)
    # right eyes
    eyes_r = righteyeCascade.detectMultiScale(roi_copy,
                                              scaleFactor=1.02,
                                              minNeighbors=3,
                                              maxSize=maxSize)
    eyes.extend(eyes_r)
    midpoints = Boxes_Midpoint(eyes)
    if upsample == 2:
        for i in range(len(midpoints)):
            midpoints[i][0] /= 2
            midpoints[i][1] /= 2
    elif upsample == 1:
        for i in range(len(midpoints)):
            midpoints[i][0] /= 4
            midpoints[i][1] /= 4

    DS = Disjoint_Sets(len(midpoints))
    for i in range(len(midpoints)):
        for j in range(i):
            a2 = (midpoints[i][0] - midpoints[j][0]) ** 2
            b2 = (midpoints[i][1] - midpoints[j][1]) ** 2
            distance = np.sqrt(a2 + b2)
            if distance < 5:
                DS.union(i, j)

    record = np.zeros((len(midpoints), 2))
    count = np.zeros(len(midpoints))
    for i in range(len(midpoints)):
        temp = DS.find_set(i)
        record[temp][0] += midpoints[i][0]
        record[temp][1] += midpoints[i][1]
        count[temp] += 1
    output = []
    for i in range(len(midpoints)):
        if count[i] != 0:
            output.append([record[i, 0] / count[i], record[i, 1] / count[i]])

    return output

def Boxes_Midpoint(boxes):
    # input: boundary boxes (List[List[int]])
    # output: middle positions (List[List[float]])

    length = len(boxes)
    midpoints = []
    for (x, y, w, h) in boxes:
        midpoints.append([x + w/2., y + h/2.])
    return midpoints
