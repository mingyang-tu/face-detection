import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
import cv2
import time
import utils.utils as UT
import utils.skin_detect as SD
import utils.faces_detect as FD
from utils.main_block import MainBlock

def main(img, resize=0, detail=False):
    start = time.time()

    img = UT.ResizeImage(img, resize)
    
    img_binary = SD.SkinDetection(img)

    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    path_face = "./haarcascade/haarcascade_frontalface_alt2.xml"
    faceCascade = cv2.CascadeClassifier(path_face)
    path_profile = "./haarcascade/haarcascade_profileface.xml"
    profileCascade = cv2.CascadeClassifier(path_profile)
    path_lefteye = "./haarcascade/haarcascade_mcs_lefteye.xml"
    lefteyeCascade = cv2.CascadeClassifier(path_lefteye)
    path_righteye = "./haarcascade/haarcascade_mcs_righteye.xml"
    righteyeCascade = cv2.CascadeClassifier(path_righteye)

    binary = SD.Closing(img_binary)

    face_total = []
    NS = FD.Next_Stage_Condition(binary, face_total)
# ===== first stage =====
    if NS:
        face_total, binary, NS = MainBlock(img, 
                                           img_gray, 
                                           img_binary, 
                                           binary, 
                                           faceCascade, 
                                           lefteyeCascade, 
                                           righteyeCascade, 
                                           face_total, 
                                           Stage=0, 
                                           show=detail)

# ===== second stage =====
    if NS:
        face_total, binary, NS = MainBlock(img, 
                                           img_gray, 
                                           img_binary, 
                                           binary, 
                                           profileCascade,      # profile
                                           lefteyeCascade, 
                                           righteyeCascade, 
                                           face_total, 
                                           Stage=0, 
                                           show=detail)

# ===== Third stage =====
    if NS:
        face_total, binary, NS = MainBlock(img, 
                                           img_gray, 
                                           img_binary, 
                                           binary, 
                                           faceCascade, 
                                           lefteyeCascade, 
                                           righteyeCascade, 
                                           face_total, 
                                           Stage=1,     # clahe
                                           show=detail)

# ===== End stage =====
    face_total = FD.Non_Maximum_Suppression(face_total, img_binary.shape)

    end = time.time()
    
    plt.figure()
    plt.imshow(img)
    ax = plt.gca()
    for (x, y, w, h) in face_total:
        rect = patches.Rectangle((x, y), w, h, linewidth=1.5, edgecolor='r', fill=False)
        ax.add_patch(rect)
    plt.title(f"Number of faces: {len(face_total)}")

    plt.show()

    return len(face_total), end-start

if __name__ == '__main__':
    args = UT.parse_args()
    # input image
    img = cv2.imread(args.input)
    img = img[:, :, [2,1,0]]
    num_of_faces, elapsed_time = main(img, args.resize, args.detail)
    print(f"Number of faces: {num_of_faces}")
    print(f"Elapsed time: {elapsed_time:.3f} seconds")
