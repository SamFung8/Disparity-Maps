import numpy as np
import cv2 as cv2
from matplotlib import pyplot as plt

test_imgs = ["Art", "Dolls", "Reindeer"]

for index in range(3):
    imgL = cv2.imread('../../img/'+test_imgs[index]+'/view1.png', cv2.IMREAD_GRAYSCALE)
    imgR = cv2.imread('../../img/'+test_imgs[index]+'/view5.png', cv2.IMREAD_GRAYSCALE)

    bSize = 3
    minDisp = 16
    maxDisp = 192
    nDisp = maxDisp - minDisp + 100
    disp12MaxDiff = 5
    uRatio = 1
    speckleWindowSize = 3
    speckleRange = 3

    stereoSGBM = cv2.StereoSGBM_create(
        minDisparity=minDisp,
        numDisparities=nDisp,
        blockSize=bSize,
        uniquenessRatio=uRatio,
        speckleWindowSize=speckleWindowSize,
        speckleRange=speckleRange,
        disp12MaxDiff=disp12MaxDiff,
        P1=1 * bSize * bSize,
        P2=8 * bSize * bSize,
    )


    dis1 = stereoSGBM.compute(imgL, imgR)

    matcher_img2 = cv2.ximgproc.createRightMatcher(stereoSGBM)
    dis2 = matcher_img2.compute(imgR, imgL)

    lmb = 10000
    sigma = 1.5

    wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=stereoSGBM)
    wls_filter.setLambda(lmb)
    wls_filter.setSigmaColor(sigma)
    filtered_disp = wls_filter.filter(dis1, imgL, disparity_map_right=dis2)
    filtered_disp = cv2.normalize(filtered_disp, filtered_disp, alpha=255, beta=0, norm_type=cv2.NORM_MINMAX)

    plt.imshow(filtered_disp,'gray')
    plt.show()
    cv2.imwrite('./pred/'+test_imgs[index]+'/disp1.png', filtered_disp)
