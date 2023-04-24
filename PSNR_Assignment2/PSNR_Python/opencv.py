import cv2 as cv2
from matplotlib import pyplot as plt

test_imgs = ["Art", "Dolls", "Reindeer"]
max_values = [240,255,224]
min_values = [32,24,64]

for index in range(3):
    img_L = cv2.imread('../../img/'+test_imgs[index]+'/view1.png', cv2.IMREAD_GRAYSCALE)
    img_R = cv2.imread('../../img/'+test_imgs[index]+'/view5.png', cv2.IMREAD_GRAYSCALE)

    block_size = 3
    min_disp = min_values[index]
    max_disp = max_values[index]
    num_disp = max_disp - min_disp
    disp12_max_diff = 5
    unique_ratio = 1
    speckle_window_size = 5
    speckle_range = 5

    stereoSGBM_L = cv2.StereoSGBM_create(
        minDisparity=min_disp,
        numDisparities=num_disp,
        blockSize=block_size,
        uniquenessRatio=unique_ratio,
        speckleWindowSize=speckle_window_size,
        speckleRange=speckle_range,
        disp12MaxDiff=disp12_max_diff,
        P1=(1 * block_size) * block_size,
        P2=(8 * block_size) * block_size,
    )


    disp_L = stereoSGBM_L.compute(img_L, img_R)


    stereoSGBM_R = cv2.ximgproc.createRightMatcher(stereoSGBM_L)
    disp_R = stereoSGBM_R.compute(img_R, img_L)

    lmb = 10000
    sigma = 1.4

    wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=stereoSGBM_L)
    wls_filter.setLambda(lmb)
    wls_filter.setSigmaColor(sigma)

    filtered_disp_L = wls_filter.filter(disp_L, img_L, disparity_map_right=disp_R)
    filtered_disp_L = cv2.normalize(filtered_disp_L, filtered_disp_L, alpha=255, beta=0, norm_type=cv2.NORM_MINMAX)

    plt.imshow(filtered_disp_L)
    plt.show()
    plt.imshow(filtered_disp_L,'gray')
    plt.show()
    cv2.imwrite('./pred/'+test_imgs[index]+'/disp1.png', filtered_disp_L)
