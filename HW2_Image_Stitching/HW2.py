import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
import math
import sys

# read the image file & output the color & gray image
def read_img(path):
    # opencv read image in BGR color space
    img = cv2.imread(path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img, img_gray

# the dtype of img must be "uint8" to avoid the error of SIFT detector
def img_to_gray(img):
    if img.dtype != "uint8":
        print("The input image dtype is not uint8 , image type is : ", img.dtype)
        return
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img_gray

# create a window to show the image
# It will show all the windows after you call im_show()
# Remember to call im_show() in the end of main
def create_im_window(window_name, img):
    cv2.imshow(window_name, img)

# show the all window you call before im_show()
# and press any key to close all windows
def im_show():
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def get_PicturePath(prefix: str, i: int) -> str:
    return f'{prefix}test/m{i}.jpg'

def dist(feature1, featureImage2):
    '''
    return distance from f1 to each featureImage2
    '''
    distance1to2 = feature1 - featureImage2
    return np.linalg.norm(distance1to2, axis=1)

def kNN(featureImage1, featureImage2, threshold=1.33):
    '''
    calc 2-NN from f1 to f2
    do Lowe's Ratio test
        to eliminate bad pairs
    dist1 < 0.75 * dist2 -> good
    if dist2 < 1.33 * dist1 -> bad
    
    feature1: [
        [point1 feature],
        [point2 feature],...
    ]
    feature2: same as feature1
    return good matches [[p1 in f1, p2 in f2], []]
    '''
    good_matches = []
    for idx1, feat1 in enumerate(featureImage1):
        min_dist = 100000000
        min_idx2 = None
        distances = dist(feat1, featureImage2)
        for idx2, distance in enumerate(distances):
            if distance < min_dist:
                min_dist = distance
                min_idx2 = idx2
        # already find min distance

        # dist1 < 0.75 * dist2 -> good
        # if dist2 < 1.33 * dist1 -> bad
        find_close_dist2 = False
        thresold_dist2 = threshold * min_dist
        for idx2, distance in enumerate(distances):
            if distance != min_dist and \
               distance < thresold_dist2:
                find_close_dist2 = True
                break
                
        # for idx2, feat2 in enumerate(featureImage2):
        #     distance = dist(feat1, feat2)
        #     if distance < min_dist:
        #         min_dist = distance
        #         min_idx2 = idx2
        # already find min distance

        # # dist1 < 0.75 * dist2 -> good
        # # if dist2 < 1.33 * dist1 -> bad
        # find_close_dist2 = False
        # thresold_dist2 = threshold * min_dist
        # for idx2, feat2 in enumerate(featureImage2):
        #     distance = dist(feat1, feat2)
        #     if distance != min_dist and \
        #        distance < thresold_dist2:
        #         find_close_dist2 = True
        #         break

        if not find_close_dist2:
            good_matches += [[idx1, min_idx2]]
            print(f'add{idx1, min_idx2}')
        if len(good_matches) > 200:
            break
    return good_matches

def get_HomographyMatrix(matches4):
    '''
    M: [m1, m2, m3, m4]
    mi_1~4: [[x1, y1], [x1h, y1h]]
    return homography matrix
    '''

    mat_A = []
    for match in matches4:
        (x1, y1), (x1_p, y1_p) = match
        mat_A += [
            [x1, y1,  1,  0,  0,  0, -x1_p * x1, -x1_p * y1, -x1_p],
            [ 0,  0,  0, x1, y1,  1, -y1_p * x1, -y1_p * y1, -y1_p]
        ]

    u, s, vh = np.linalg.svd(mat_A)

    H = vh[8]  # The "rows" of vh are the eigenvectors of ATA
    H = H / H[8]  # normalize A33 to 1
    H = H.reshape((3, 3))
    return H
    

def RANSAC(matches, kp1, f1, kp2, f2):
    '''
    random choose 4 from matches
    do PrespectiveTransform(all matches)
        and counting supports
    return best Transform Mat
    '''
    rand4number = np.random.choice(len(matches), 4)
    match4 = []
    for i in range(4):
        idx = rand4number[i]
        m = [
            kp1[matches[idx][0]].pt,
            kp2[matches[idx][1]].pt
        ]  # [[x1, y1], [x1h, y1h]]
        match4 += [m]

    H = get_HomographyMatrix(match4)

    p1 = np.array([e for e in kp1[matches[rand4number[3]][0]].pt] + [1])
    p2 = np.array([e for e in kp2[matches[rand4number[3]][1]].pt] + [1])
    print(p1)
    print(p2)
    print(H @ p1)
    # for idx1, idx2 in matches:
    #     # H * kp1[idx1].pt -> kp2[idx2].pt
    #     pass


def combine(image1, image2):
    '''
    transform(image1) + image2
    transform from image1 to image2 <- knn from image1 to image2
    '''
    kp1, f1 = SIFT.detectAndCompute(image1, None)
    # 3196, (3196, 128)
    kp2, f2 = SIFT.detectAndCompute(image2, None)

    matches = kNN(f1, f2)  # about 610
    
    id1, id2 = matches[100]
    # print(f'pos1: {kp1[id1].pt}')
    # print(f'pos2: {kp2[id2].pt}')
    kp1s = [kp1[match[0]] for match in matches[100: 120]]
    kp2s = [kp2[match[1]] for match in matches[100: 120]]
    image1 = cv2.drawKeypoints(image1, kp1s, image1, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    image2 = cv2.drawKeypoints(image2, kp2s, image2, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    create_im_window('im1', image1)
    create_im_window('im2', image2)
    im_show()

    # RANSAC(matches, kp1=kp1, f1=f1, kp2=kp2, f2=f2)

    # bestMat = RANSAC(matches, kp1=kp1, f1=f1, kp2=kp2, f2=f2)

    # pos = kp1[0].pt  # (x, y)

if __name__ == '__main__':
    prefix = 'HW2_Image_Stitching/' if True else ''
    SIFT = cv2.SIFT_create()

    img_name = get_PicturePath(prefix, 1)
    image1, image1_gray = read_img(img_name)
    img_name = get_PicturePath(prefix, 2)
    image2, image2_gray = read_img(img_name)

    combine(image1_gray, image2_gray)

    # create_im_window('rgb', image)
    # create_im_window('gray', image1_gray)
    # im_show()

    # you can use this function to store the result
    # cv2.imwrite("result.jpg", img)
