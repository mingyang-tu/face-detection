import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
import cv2
import time
import gc
from warnings import filterwarnings
from utils import utils, faces_detect
from utils.skin_detect import SkinDetection, Closing
from utils.skin_segmentation import SkinSegmentation, DisplaySkin
from utils.ellipse_matching import PCA

def main(img, resize=0, detail=False):
    filterwarnings("ignore", "One of the clusters is empty. ")

    start = time.time()

    img = utils.ResizeImage(img, resize)
    
    img_binary = SkinDetection(img)

    gc.collect()

    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    path_face = "./haarcascade/haarcascade_frontalface_alt2.xml"
    faceCascade = cv2.CascadeClassifier(path_face)
    path_lefteye = "./haarcascade/haarcascade_mcs_lefteye.xml"
    lefteyeCascade = cv2.CascadeClassifier(path_lefteye)
    path_righteye = "./haarcascade/haarcascade_mcs_righteye.xml"
    righteyeCascade = cv2.CascadeClassifier(path_righteye)

    imsize = img_binary.shape
    binary = Closing(img_binary, imsize)

    face_total = []
    NS = faces_detect.Next_Stage_Condition(binary, face_total)
# ===== first stage =====
    if NS:
        roi = img_gray * binary
        face_boxes = faces_detect.Detect_Faces(roi, faceCascade)
        # Validation
        face_boxes = faces_detect.Skin_Validation(img_binary, face_boxes, threshold=0.5)
        face_boxes = faces_detect.Eye_Validation(img_gray, face_boxes, lefteyeCascade, righteyeCascade)
        
        if detail:
            utils.DisplayDetails(img, binary, face_boxes)
            
        # Delete Face Region
        faces_detect.Delete_Face_Area(binary, face_boxes)

        face_total.extend(face_boxes)

        NS = faces_detect.Next_Stage_Condition(binary, face_total)

    gc.collect()
# ===== In-plane Face Detect =====
    face_rotated = []
    if NS:
        img_region, num_region = SkinSegmentation(img, binary)
        if detail:
            DisplaySkin(img, img_region, num_region)
        for i in range(1, num_region+1):
            roi_binary = (img_region == i).astype(np.uint8)
            roi_binary = Closing(roi_binary, (imsize[0]*3, imsize[1]*3))
            valid, angle = PCA(roi_binary)
            if not valid:
                continue
            rmin, rmax, cmin, cmax = utils.get_boundary(roi_binary)
            roi = img_gray * roi_binary
            roi = roi[rmin:rmax, cmin:cmax]
        # rotate
            angle = angle + 90
            rotated, rotation_mat = utils.RotateImage(roi, angle)
            rotated_size = rotated.shape

            face_boxes = faces_detect.Detect_Faces(rotated, faceCascade)
            # Validation
            face_boxes = faces_detect.Eye_Validation(rotated, face_boxes, lefteyeCascade, righteyeCascade)

        # rotate 180
            rotated_180 = np.flip(rotated)

            face_boxes_180 = faces_detect.Detect_Faces(rotated_180, faceCascade)
            # Validation
            face_boxes_180 = faces_detect.Eye_Validation(rotated_180, face_boxes_180, lefteyeCascade, righteyeCascade)

            for i in range(len(face_boxes_180)):
                face_boxes_180[i][0] = rotated_size[1] - face_boxes_180[i][0] - face_boxes_180[i][2]
                face_boxes_180[i][1] = rotated_size[0] - face_boxes_180[i][1] - face_boxes_180[i][3]
            face_boxes.extend(face_boxes_180)

            face_boxes = faces_detect.Non_Maximum_Suppression(face_boxes, rotated_size)

            for i in range(len(face_boxes)):
                px = face_boxes[i][0] - rotation_mat[0, 2]
                py = face_boxes[i][1] - rotation_mat[1, 2]
                face_boxes[i][0] = rotation_mat[0, 0] * px + rotation_mat[1, 0] * py + cmin
                face_boxes[i][1] = rotation_mat[0, 1] * px + rotation_mat[1, 1] * py + rmin

            face_rotated.append({"face": face_boxes, "angle": angle})
            gc.collect()

# ===== End stage =====
    face_total = faces_detect.Non_Maximum_Suppression(face_total, img_binary.shape)

    end = time.time()

    num_of_faces = len(face_total)
    
    plt.figure()
    plt.imshow(img)
    ax = plt.gca()
    for (x, y, w, h) in face_total:
        rect = patches.Rectangle((x, y), w, h, linewidth=1.5, edgecolor='r', fill=False)
        ax.add_patch(rect)
    for d in face_rotated:
        angle = d["angle"]
        num_of_faces += len(d["face"])
        for (x, y, w, h) in d["face"]:
            rect = patches.Rectangle((x, y), w, h, linewidth=1.5, edgecolor='r', fill=False, angle=angle)
            ax.add_patch(rect)
    plt.title(f"Number of faces: {num_of_faces}")
    plt.show()

    return num_of_faces, end-start

if __name__ == '__main__':
    args = utils.parse_args()
    # input image
    img = cv2.imread(args.input)
    img = img[:, :, [2,1,0]]

    num_of_faces, elapsed_time = main(img, args.resize, args.detail)
    print(f"Number of faces: {num_of_faces}")
    print(f"Elapsed time: {elapsed_time:.3f} seconds")
