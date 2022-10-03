import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Polygon
from typing import List, Dict

def Find_Face(eye: List[List[np.float64]], 
              eye_value: List[List[float]], 
              mouth: List[List[np.float64]], 
              mouth_value: List[List[float]], 
              num_region: int, 
              proj_axis: np.float64, 
              axislength: np.float64) -> List[Dict]:
    face = []
    for i in range(num_region):
        mlist = mouth[i]
        mvalue = mouth_value[i]
        elist = eye[i]
        evalue = eye_value[i]
        len_mouth = len(mlist)
        len_eye = len(elist)
        valid = np.zeros((len_mouth, len_eye, len_eye), dtype=np.uint8)
        diff = np.ones((len_mouth, len_eye, len_eye)) * 100
        for m in range(len_mouth):
            for e1 in range(1, len_eye):
                for e2 in range(e1):
                    # rule 1: the eyes are on the top of the mouth
                    if (elist[e1][0] < mlist[m][0]) and (elist[e2][0] < mlist[m][0]):
                        valid[m, e1, e2] = 1

                    # rule 2
                    eye_mid = (elist[e1] + elist[e2]) / 2       # midpoint of eyes
                    eye_vector = elist[e1] - elist[e2]          # vector from eye to eye (line AB)
                    eye_vector_len = np.sqrt(eye_vector[0]**2 + eye_vector[1]**2) + 0.0001
                    # vector perpendicular to line AB
                    eye_vector_norm = eye_vector / eye_vector_len
                    eye_vector_norm[0], eye_vector_norm[1] = -eye_vector_norm[1], eye_vector_norm[0]
                    # vector from mouth to the midpoint of eyes (line CD)
                    moutheye_vector = mlist[m] - eye_mid
                    moutheye_vector_len = np.sqrt(moutheye_vector[0]**2 + moutheye_vector[1]**2) + 0.0001
                    moutheye_vector_norm = moutheye_vector / moutheye_vector_len
                    # inner product
                    temp = np.dot(eye_vector_norm, moutheye_vector_norm)
                    # objective function part 1
                    diff[m, e1, e2] = math.acos(abs(temp))
                    if abs(temp) < 0.75:    # smaller than 41 degree
                        valid[m, e1, e2] = 0
                        
                    # rule 3
                    # inner product
                    temp = np.dot(proj_axis[i, :, 0], moutheye_vector_norm)
                    # objective function part 2
                    diff[m, e1, e2] += math.acos(abs(temp))
                    if abs(temp) < 0.75:     # smaller than 41 degree
                        valid[m, e1, e2] = 0

                    # rule 4
                    # 28 ~ 62 degree
                    if (eye_vector_len / moutheye_vector_len > 1.2) or (eye_vector_len / moutheye_vector_len < 0.5):
                        valid[m, e1, e2] = 0

                    # rule 5: eye-mouth confidence
                    diff[m, e1, e2] += 0.25 * (1 - mvalue[m])
                    diff[m, e1, e2] += 0.25 * (2 - evalue[e1]- evalue[e2])

                    # rule 6: the triangle area created by eyes and mouth must bigger than 
                    #         1/25 times the ellipse area
                    moutheye1 = elist[e1] - mlist[m]
                    moutheye2 = elist[e2] - mlist[m]
                    tri_area = abs(moutheye1[0] * moutheye2[1] - moutheye1[1] * moutheye2[0]) / 2
                    elli_area = math.pi * axislength[0, i] * axislength[1, i]
                    if tri_area < elli_area / 25:
                        valid[m, e1, e2] = 0

        mee = dict()
        choose = 100
        for m in range(len_mouth):
            for e1 in range(1, len_eye):
                for e2 in range(e1):
                    if (valid[m, e1, e2] == 1):
                        if diff[m, e1, e2] < choose:
                            choose = diff[m, e1, e2]
                            mee["mouth"] = mlist[m]
                            mee["eye1"] = elist[e1]
                            mee["eye2"] = elist[e2]
        if len(mee) != 0:
            mee["skin"] = i
            face.append(mee)
    return face

def DisplayFace(img: np.uint8, 
                face: List[Dict], 
                centroid: List[np.float64], 
                axislength: np.float64, 
                angle: np.float64) -> None:
    imsize = img.shape
    plt.figure()
    plt.imshow(img)
    ax = plt.gca()
    for d in face:
        plt.plot(d["eye1"][1], d["eye1"][0], 'o', c='r', markersize=3)
        plt.plot(d["eye2"][1], d["eye2"][0], 'o', c='r', markersize=3)
        plt.plot(d["mouth"][1], d["mouth"][0], 'o', c='cyan', markersize=3)
        triangle = Polygon((d["eye1"][[1,0]], d["eye2"][[1,0]], d["mouth"][[1,0]]), 
                           fill=False, linewidth=0.5, edgecolor='r')
        ax.add_artist(triangle)
        i = d["skin"]
        ellipse = Ellipse(xy=(centroid[1][i], centroid[0][i]), 
                          width=axislength[0, i]*2, height=axislength[1, i]*2, angle=angle[i], 
                          linewidth=0.8, edgecolor='b', facecolor='None')
        ax.add_artist(ellipse)
    plt.xlim(0, imsize[1]-1)
    plt.ylim(imsize[0]-1, 0)
    plt.title(f"Number of faces: {len(face)}")
