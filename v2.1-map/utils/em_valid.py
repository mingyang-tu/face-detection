from utils.eye_mouth_map import Eye_Mouth_Map
import numpy as np
import matplotlib.pyplot as plt

def EyeMouth_Validation(img, face_boxes, img_binary):
    new_face_boxes = []
    for face in face_boxes:
        (x, y, w, h) = face
        roi = img[y:y+h, x:x+w, :]
        binary = img_binary[y:y+h, x:x+w]

        EMM = Eye_Mouth_Map(roi, binary)
        eye_points = EMM.FindEye()
        mouth_points = EMM.FindMouth()

        # ShowEyesMouths(roi, eye_points, mouth_points)

        if vaild_face(eye_points, mouth_points):
            new_face_boxes.append(face)
    return new_face_boxes

def vaild_face(elist, mlist):
    len_mouth = len(mlist)
    len_eye = len(elist)
    for m in range(len_mouth):
        for e1 in range(1, len_eye):
            for e2 in range(e1):
                valid = False
                # rule 1
                if (elist[e1][1] < mlist[m][1]) and (elist[e2][1] < mlist[m][1]):
                    valid = True

                # rule 2
                D = ((elist[e1][0] + elist[e2][0]) / 2, (elist[e1][1] + elist[e2][1]) / 2)

                AB = (elist[e1][0] - elist[e2][0], elist[e1][1] - elist[e2][1])
                AB_len = np.sqrt(AB[0]**2 + AB[1]**2) + 1e-4
                AB_norm = (- AB[1] / AB_len, AB[0] / AB_len)

                CD = (mlist[m][0] - D[0], mlist[m][1] - D[1])
                CD_len = np.sqrt(CD[0]**2 + CD[1]**2) + 1e-4
                CD_norm = (CD[0] / CD_len, CD[1] / CD_len)

                temp = AB_norm[0] * CD_norm[0] + AB_norm[1] * CD_norm[1]
                if abs(temp) < 0.85:    # smaller than 30 degree
                    valid = False

                # rule 3
                if abs(CD_norm[0] / (CD_norm[1]+1e-4)) > 0.6:    # smaller than 30 degree
                    valid = False

                # rule 4
                if (AB_len / CD_len > 2.5) or (AB_len / CD_len < 0.4):
                    valid = False

                if valid:
                    return True
    return False
                

def ShowEyesMouths(roi, eye_points, mouth_points):
    plt.figure()
    plt.imshow(roi)
    for eye in eye_points:
        plt.plot(eye[0], eye[1], "o", c="b")
    for mouth in mouth_points:
        plt.plot(mouth[0], mouth[1], "o", c="r")