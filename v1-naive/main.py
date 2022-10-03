import numpy as np
import matplotlib.pyplot as plt
import time
import utils.util as UT
import utils.skin_filter as SF
import utils.ellipse_matching as ellipse_matching
import utils.eye_mouth_map as EyeMouthMap
import utils.find_face as FF

def main(img, resize=512, detail=False):
    start = time.time()

    img = UT.ResizeImage(img, resize)

    img_YCbCr = UT.RGB2YCbCr(img)

    img_region, num_region = SF.SkinSegmentation(img_YCbCr)

    EM = ellipse_matching.Ellipse_Matching(img_region, num_region)
    EM.PCA()
    img_region = EM.img_region
    num_region = EM.num_region

    boundary = UT.get_boundary(img_region, num_region)

    EMMap = EyeMouthMap.Eye_Mouth_Map(img_YCbCr, img_region, num_region, 
                                      boundary, EM.centroid,
                                      EM.proj_axis, EM.axislength)
    eye, eye_value = EMMap.FindEye()
    mouth, mouth_value = EMMap.FindMouth()

    face = FF.Find_Face(eye, eye_value, mouth, mouth_value, num_region, 
                        EM.proj_axis, EM.axislength)

    end = time.time()
    print(f"Elapsed time: {end-start:.4f} seconds")

    if detail:
        SF.DisplaySkin(img_region, num_region, EM.centroid, boundary)
        EMMap.DisplayEye(img, eye)
        EMMap.DisplayMouth(img, mouth)
    FF.DisplayFace(img, face, EM.centroid, EM.axislength, EM.angle)
    plt.show()

if __name__ == '__main__':
    args = UT.parse_args()
    # input image
    img_dir = args.dir + '/' + args.input
    img = plt.imread(img_dir)
    main(img, args.resize, args.detail)

