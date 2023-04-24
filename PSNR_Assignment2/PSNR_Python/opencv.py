import cv2 as cv2
from matplotlib import pyplot as plt

test_imgs = ["Art", "Dolls", "Reindeer"]
max_value = [240,255,224]
min_value = [32,24,64]

for index in range(3):
    imgL = cv2.imread('../../img/'+test_imgs[index]+'/view1.png', cv2.IMREAD_GRAYSCALE)
    imgR = cv2.imread('../../img/'+test_imgs[index]+'/view5.png', cv2.IMREAD_GRAYSCALE)

    bSize = 3
    minDisp = min_value[index]
    maxDisp = max_value[index]
    nDisp = maxDisp - minDisp
    disp12MaxDiff = 5
    uRatio = 1
    speckleWindowSize = 3
    speckleRange = 3

    stereoSGBM_L = cv2.StereoSGBM_create(
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


    disp_L = stereoSGBM_L.compute(imgL, imgR)

    stereoSGBM_R = cv2.ximgproc.createRightMatcher(stereoSGBM_L)
    disp_R = stereoSGBM_R.compute(imgR, imgL)

    lmb = 10000
    sigma = 1.5

    wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=stereoSGBM_L)
    wls_filter.setLambda(lmb)
    wls_filter.setSigmaColor(sigma)

    filtered_disp_L = wls_filter.filter(disp_L, imgL, disparity_map_right=disp_R)
    filtered_disp_L = cv2.normalize(filtered_disp_L, filtered_disp_L, alpha=255, beta=0, norm_type=cv2.NORM_MINMAX)

    plt.imshow(filtered_disp_L,'gray')
    plt.show()
    cv2.imwrite('./pred/'+test_imgs[index]+'/disp1.png', filtered_disp_L)
