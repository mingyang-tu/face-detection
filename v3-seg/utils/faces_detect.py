import cv2
import numpy as np
from utils.eyes_detect import Detect_Eyes
import matplotlib.pyplot as plt

def Detect_Faces(img_gray, faceCascade):
    # input: grayscale (np.uint8, shape: (row, col))
    #        face detector (cv2.CascadeClassifier)
    # output: boundary boxes of faces (List[List[int]])

    faces = faceCascade.detectMultiScale(img_gray,
                                         scaleFactor=1.05,
                                         minNeighbors=5,
                                         minSize=(20, 20))
    return faces

def Eye_Validation(img_gray, face_boxes, lefteyeCascade, righteyeCascade):
    # input: binary image (np.uint8, shape: (row, col))
    #        boundary boxes of faces (List[List[int]])
    #        left eye detector (cv2.CascadeClassifier)
    #        right eye detector (cv2.CascadeClassifier)
    # output: boundary boxes of faces (List[List[int]])

    new_face_boxes = []
    for face in face_boxes:
        (x, y, w, h) = face
        roi = img_gray[y:y+h, x:x+w]
        eye_points = Detect_Eyes(roi, lefteyeCascade, righteyeCascade)

        count = 0
        for eye in eye_points:
            if eye[1] < h * 0.5:
                count += 1
        if count >= 1:
            new_face_boxes.append(face)
    return new_face_boxes

def Skin_Validation(img_binary, face_boxes, threshold):
    # input: binary image (np.uint8, shape: (row, col))
    #        boundary boxes of faces (List[List[int]])
    #        minimum ratio of skin region (int)
    # output: boundary boxes of faces (List[List[int]])

    new_face_boxes = []
    for face in face_boxes:
        (x, y, w, h) = face
        roi = img_binary[y:y+h, x:x+w]
        if np.sum(roi) > threshold * w*h:
            new_face_boxes.append(face)
    return new_face_boxes

def Delete_Face_Area(img_binary, face_boxes):
    # input: binary image (np.uint8, shape: (row, col))
    #        boundary boxes of faces (List[List[int]])
    # output: binary image (np.uint8, shape: (row, col))

    for (x, y, w, h) in face_boxes: 
        img_binary[y:y+h, x:x+w] = 0

def Next_Stage_Condition(binary, face_total, threshold=400):
    # input: binary image (np.uint8, shape: (row, col))
    #        boundary boxes of faces (List[List[int]])
    #        minimum size of face (int)
    # output: continue to the next stage or not (bool)

    area_face = np.sum(binary)
    area_box = 0
    count_box = 0
    for (x, y, w, h) in face_total: 
        area_box += w * h
        count_box += 1
    if count_box != 0:
        area_box = area_box / count_box

    if (area_face < threshold) or (area_face < area_box):
        return False
    else:
        return True

def Non_Maximum_Suppression(face_total, imsize):
    length = len(face_total)
    
    new_face_total = []
    for i in range(length):
        valid = True

        x1, y1, w1, h1 = face_total[i]
        area1 = w1 * h1
        face1 = np.zeros(imsize)
        face1[y1:y1+h1, x1:x1+w1] = 1

        for j in range(length):
            x2, y2, w2, h2 = face_total[j]
            area2 = w2 * h2
            face2 = np.zeros(imsize)
            face2[y2:y2+h2, x2:x2+w2] = 1

            if area1 < area2:
                overlap = np.sum(cv2.bitwise_and(face1, face2))
                if (overlap > area1 * 0.5):
                    valid = False

        if valid:
            new_face_total.append(face_total[i])
            
    return new_face_total
