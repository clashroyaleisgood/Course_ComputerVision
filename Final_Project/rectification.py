import numpy as np
import cv2
from matplotlib import pyplot as plt
import os

from datasets import getDataset, getUnrectDataset
from HW2_helper import img_to_gray, kNN, RANSAC

def rectify(image_l, image_r):
    '''
    input un-rectified images (tilted images)
    return rectified images

    steps:
    - find cooresponding point [pairs]
    - find Fundamental [matrix] by [pairs]
    - find H1, H2: Homography matrices to wrap images
      from [pairs] and [matrix]
    - wrapPerspective(image_l, H1), ...
    https://www.andreasjakl.com/understand-and-apply-stereo-rectification-for-depth-maps-part-2/
    '''
    SIFT = cv2.SIFT_create()

    image_gray1 = img_to_gray(image_l)
    image_gray2 = img_to_gray(image_r)
    kp1, f1 = SIFT.detectAndCompute(image_gray1, None)
    kp2, f2 = SIFT.detectAndCompute(image_gray2, None)

    # -------------------------------
    # kNN fail
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)   # or pass empty dictionary
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(f1, f2, k=2)

    # matches = kNN(f1, f2)  # low test embeded in kNN func(filter below)
    print(f'kNN find {len(matches)} matches')

    # -------------------------------
    # filter better matches
    points_l = []
    points_r = []

    for i, (m, n) in enumerate(matches):
        if m.distance < 0.7*n.distance:
            # Keep this keypoint pair
            points_l.append(kp1[m.queryIdx].pt)
            points_r.append(kp2[m.trainIdx].pt)
    
    # good_matches = RANSAC(matches, kp1, kp2)
    # # [[p1, p1'], [p2, p2'], ...]
    # print(f'RANSAC select {len(good_matches)} matches')

    # points_l = []
    # points_r = []
    # for pt_l, pt_r in good_matches:
    #     points_l += [pt_l]
    #     points_r += [pt_r]

    # -------------------------------
    points_l = np.array(points_l, dtype=np.int32)
    points_r = np.array(points_r, dtype=np.int32)

    fundamental_matrix, inliers = cv2.findFundamentalMat(points_l, points_r, cv2.FM_RANSAC)

    # We select only inlier points
    points_l = points_l[inliers.ravel() == 1]
    points_r = points_r[inliers.ravel() == 1]

    # -------------------------------
    h1, w1 = image_gray1.shape
    h2, w2 = image_gray2.shape
    _, H1, H2 = cv2.stereoRectifyUncalibrated(
        np.float32(points_l), np.float32(points_r), fundamental_matrix, imgSize=(w1, h1)
    )

    # -------------------------------
    # Undistort (rectify) the images and save them
    # Adapted from: https://stackoverflow.com/a/62607343
    image_l_rectified = cv2.warpPerspective(image_l, H1, (w1, h1))
    image_r_rectified = cv2.warpPerspective(image_r, H2, (w2, h2))

    return image_l_rectified, image_r_rectified

def getFundamentalMatrix():
    pass

def saveRectifiedImages(folder_name, image_l, image_r):
    folderpath = f'Final_Project\\Dataset\\{folder_name}\\stereo'
    os.makedirs(folderpath, exist_ok=True)
    cv2.imwrite(os.path.join(folderpath, f'{folder_name}_1.jpg'), image_l)
    cv2.imwrite(os.path.join(folderpath, f'{folder_name}_2.jpg'), image_r)

if __name__ == '__main__':
    folder_name = 'self_laptop'

    dataset = getUnrectDataset(folder_name)

    image_l = dataset[0]
    image_r = dataset[1]
    image_l_rect, image_r_rect = rectify(image_l, image_r)

    saveRectifiedImages('self_laptop', image_l_rect, image_r_rect)
    # plt.figure()
    # plt.imshow(image_l_rectified)
    # plt.title('left')

    # plt.figure()
    # plt.imshow(image_r_rectified)
    # plt.title('right')

    # plt.show()
