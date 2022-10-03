import cv2
import numpy as np

def Detect_Faces(img_gray, faceCascade):
    faces = faceCascade.detectMultiScale(img_gray,
                                         scaleFactor=1.05,
                                         minNeighbors=5,
                                         minSize=(20, 20))
    return faces

def Skin_Validation(img_binary, face_boxes, threshold):
    new_face_boxes = []
    for face in face_boxes:
        (x, y, w, h) = face
        roi = img_binary[y:y+h, x:x+w]
        if np.sum(roi) > threshold * w*h:
            new_face_boxes.append(face)
    return new_face_boxes

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
