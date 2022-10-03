import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
import cv2
import time
from warnings import filterwarnings
from utils import utils, faces_detect
from utils.skin_detect import SkinDetection, Closing
from utils.em_valid import EyeMouth_Validation

def main(img, resize=0):
    filterwarnings("ignore", "One of the clusters is empty. ")

    start = time.time()

    img = utils.ResizeImage(img, resize)
    
    img_binary = SkinDetection(img)

    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    path_face = "./haarcascade/haarcascade_frontalface_alt2.xml"
    faceCascade = cv2.CascadeClassifier(path_face)

    imsize = img_binary.shape
    binary = Closing(img_binary, imsize)

    roi = img_gray * binary
    face_boxes = faces_detect.Detect_Faces(roi, faceCascade)
    # Validation
    face_boxes = faces_detect.Skin_Validation(img_binary, face_boxes, threshold=0.5)
    face_boxes = EyeMouth_Validation(img, face_boxes, img_binary)

    face_boxes = faces_detect.Non_Maximum_Suppression(face_boxes, imsize)

    end = time.time()

    num_of_faces = len(face_boxes)
    
    plt.figure()
    plt.imshow(img)
    ax = plt.gca()
    for (x, y, w, h) in face_boxes:
        rect = patches.Rectangle((x, y), w, h, linewidth=1.5, edgecolor='r', fill=False)
        ax.add_patch(rect)
    plt.title(f"Number of faces: {num_of_faces}")
    plt.show()

    return num_of_faces, end-start

if __name__ == '__main__':
    args = utils.parse_args()
    # input image
    img = cv2.imread(args.input)
    img = img[:, :, [2,1,0]]

    num_of_faces, elapsed_time = main(img, args.resize)
    print(f"Number of faces: {num_of_faces}")
    print(f"Elapsed time: {elapsed_time:.3f} seconds")
