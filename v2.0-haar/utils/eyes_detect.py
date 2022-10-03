import cv2
import numpy as np

def Detect_Eyes(roi, lefteyeCascade, righteyeCascade): 
    # input: grayscale (np.uint8, shape: (row, col))
    #        left eye detector (cv2.CascadeClassifier)
    #        right eye detector (cv2.CascadeClassifier)
    # output: eyes positions (List[List[float]])

    # upsampling
    if max(roi.shape) <= 50:
        roi_copy = cv2.resize(roi, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        upsample = True
    else:
        roi_copy = np.array(roi, dtype=np.uint8)
        upsample = False

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
    if upsample:
        for i in range(len(midpoints)):
            midpoints[i][0] /= 2
            midpoints[i][1] /= 2

    return midpoints

def Boxes_Midpoint(boxes):
    # input: boundary boxes (List[List[int]])
    # output: middle positions (List[List[float]])

    length = len(boxes)
    midpoints = []
    for (x, y, w, h) in boxes:
        midpoints.append([x + w/2., y + h/2.])
    return midpoints
