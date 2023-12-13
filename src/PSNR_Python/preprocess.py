import numpy as np
import cv2 as cv2
import matplotlib.pyplot as plt


test_imgs = ["Art", "Dolls", "Reindeer"]


# ------------------------------------------------------------
# PREPROCESSING

# Compare unprocessed images
def testing_on_origin_img():
    for index in range(3):
        img1 = cv2.imread('../../img/' + test_imgs[index] + '/view1.png', cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread('../../img/' + test_imgs[index] + '/view5.png', cv2.IMREAD_GRAYSCALE)

        fig, axes = plt.subplots(1, 2, figsize=(15, 10))
        axes[0].imshow(img1, cmap="gray")
        axes[1].imshow(img2, cmap="gray")
        axes[0].axhline(250)
        axes[1].axhline(250)
        axes[0].axhline(450)
        axes[1].axhline(450)
        plt.show()


def testing_on_rectified_img():
    for index in range(3):
        img1 = cv2.imread('../../rectifiedImg/' + test_imgs[index] + '/rectified_L.png', cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread('../../rectifiedImg/' + test_imgs[index] + '/rectified_R.png', cv2.IMREAD_GRAYSCALE)

        fig, axes = plt.subplots(1, 2, figsize=(15, 10))
        axes[0].imshow(img1, cmap="gray")
        axes[1].imshow(img2, cmap="gray")
        axes[0].axhline(250)
        axes[1].axhline(250)
        axes[0].axhline(450)
        axes[1].axhline(450)
        plt.show()


# Stereo Rectification for Depth Maps
def reectification():
    for index in range(3):
        img1 = cv2.imread('../../img/' + test_imgs[index] + '/view1.png', cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread('../../img/' + test_imgs[index] + '/view5.png', cv2.IMREAD_GRAYSCALE)

        # Initiate SIFT detector
        sift = cv2.SIFT_create()
        # find the keypoints and descriptors with SIFT
        kp1, des1 = sift.detectAndCompute(img1, None)
        kp2, des2 = sift.detectAndCompute(img2, None)

        # Visualize keypoints
        imgSift = cv2.drawKeypoints(img1, kp1, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        '''plt.imshow(imgSift)
        plt.show()'''

        # Match keypoints in both images
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)  # or pass empty dictionary
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1, des2, k=2)

        # Keep good matches: calculate distinctive image features
        matchesMask = [[0, 0] for i in range(len(matches))]
        good = []
        pts1 = []
        pts2 = []

        for i, (m, n) in enumerate(matches):
            if m.distance < 0.75 * n.distance:
                # Keep this keypoint pair
                matchesMask[i] = [1, 0]
                good.append(m)
                pts2.append(kp2[m.trainIdx].pt)
                pts1.append(kp1[m.queryIdx].pt)

        # Draw the keypoint matches between both pictures
        draw_params = dict(matchColor=(0, 255, 0),
                           singlePointColor=(255, 0, 0),
                           matchesMask=matchesMask[300:500],
                           flags=cv2.DrawMatchesFlags_DEFAULT)

        keypoint_matches = cv2.drawMatchesKnn(img1, kp1, img2, kp2, matches[300:500], None, **draw_params)
        '''plt.imshow(keypoint_matches)
        plt.show()'''

        # STEREO RECTIFICATION
        # Calculate the fundamental matrix for the cameras
        pts1 = np.int32(pts1)
        pts2 = np.int32(pts2)
        fundamental_matrix, inliers = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC)

        # We select only inlier points
        pts1 = pts1[inliers.ravel() == 1]
        pts2 = pts2[inliers.ravel() == 1]

        # Visualize epilines
        def drawlines(img1src, img2src, lines, pts1src, pts2src):
            r, c = img1src.shape
            img1color = cv2.cvtColor(img1src, cv2.COLOR_GRAY2BGR)
            img2color = cv2.cvtColor(img2src, cv2.COLOR_GRAY2BGR)
            # Edit: use the same random seed so that two images are comparable!
            np.random.seed(0)
            for r, pt1, pt2 in zip(lines, pts1src, pts2src):
                color = tuple(np.random.randint(0, 255, 3).tolist())
                x0, y0 = map(int, [0, -r[2] / r[1]])
                x1, y1 = map(int, [c, -(r[2] + r[0] * c) / r[1]])
                img1color = cv2.line(img1color, (x0, y0), (x1, y1), color, 1)
                img1color = cv2.circle(img1color, tuple(pt1), 5, color, -1)
                img2color = cv2.circle(img2color, tuple(pt2), 5, color, -1)
            return img1color, img2color

        # Find epilines corresponding to points in right image (second image) and
        # drawing its lines on left image
        lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1, 1, 2), 2, fundamental_matrix)
        lines1 = lines1.reshape(-1, 3)
        img5, img6 = drawlines(img1, img2, lines1, pts1, pts2)

        # Find epilines corresponding to points in left image (first image) and
        # drawing its lines on right image
        lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1, 1, 2), 1, fundamental_matrix)
        lines2 = lines2.reshape(-1, 3)
        img3, img4 = drawlines(img2, img1, lines2, pts2, pts1)

        '''plt.subplot(121), plt.imshow(img5)
        plt.subplot(122), plt.imshow(img3)
        plt.suptitle("Epilines in both images")
        plt.show()'''

        # Stereo rectification (uncalibrated variant)
        h1, w1 = img1.shape
        h2, w2 = img2.shape
        _, H1, H2 = cv2.stereoRectifyUncalibrated(
            np.float32(pts1), np.float32(pts2), fundamental_matrix, imgSize=(w1, h1)
        )

        # Undistort (rectify) the images and save them
        img1_rectified = cv2.warpPerspective(img1, H1, (w1, h1))
        img2_rectified = cv2.warpPerspective(img2, H2, (w2, h2))
        cv2.imwrite('../../rectifiedImg/' + test_imgs[index] + '/rectified_L.png', img1_rectified)
        cv2.imwrite('../../rectifiedImg/' + test_imgs[index] + '/rectified_R.png', img2_rectified)


if __name__ == '__main__':
    testing_on_origin_img()
    reectification()
    testing_on_rectified_img()