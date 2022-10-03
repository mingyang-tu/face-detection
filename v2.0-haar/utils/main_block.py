import numpy as np
import cv2
import utils.utils as UT
import utils.faces_detect as FD

def MainBlock(img, 
              img_gray, 
              img_binary, 
              binary, 
              faceCascade, 
              lefteyeCascade, 
              righteyeCascade, 
              face_total, 
              Stage=0, 
              show=False):
    if Stage == 1:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray = clahe.apply(img_gray)
    else:
        gray = img_gray
    roi = gray * binary
    face_boxes = FD.Detect_Faces(roi, faceCascade)
    # Validation
    face_boxes = FD.Skin_Validation(img_binary, face_boxes, threshold=0.5)
    face_boxes = FD.Eye_Validation(gray, face_boxes, lefteyeCascade, righteyeCascade)

    if show:
        UT.DisplayDetails(img, binary, face_boxes)
        
    # Delete Face Region
    binary = FD.Delete_Face_Area(binary, face_boxes)

    face_total.extend(face_boxes)

    NS = FD.Next_Stage_Condition(binary, face_total)

    return face_total, binary, NS

