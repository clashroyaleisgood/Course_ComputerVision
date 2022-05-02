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

def kNN(featureImage1, featureImage2, threshold=2):
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


def RANSAC(matches, kp1, kp2, threshold=3, repeat=200):
    '''
    random choose 4 from matches
    do PrespectiveTransform(all matches)
        and counting supports
    return best Transform Mat
    '''
    max_support = 0
    max_H = None
    
    for r in range(repeat):
        rand4number = np.random.choice(len(matches), 4)
        # print(f'try {r}: rand select {rand4number}')
        match4 = []
        for i in range(4):
            idx = rand4number[i]
            m = [
                kp1[matches[idx][0]].pt,
                kp2[matches[idx][1]].pt
            ]  # [[x1, y1], [x1h, y1h]]
            match4 += [m]

        H = get_HomographyMatrix(match4)

        # correctness test
        # p1 = np.array([e for e in kp1[matches[rand4number[3]][0]].pt] + [1]).reshape((3, 1))
        # p2 = np.array([e for e in kp2[matches[rand4number[3]][1]].pt] + [1]).reshape((3, 1))
        # print(p1)
        # print(p2)
        # p3 = H @ p1
        # p3 /= p3[2]
        # print(p3)

        support = 0
        for idx1, idx2 in matches:
            p1 = np.array([*(kp1[idx1].pt), 1])  # [x, y, 1]
            p2_proj = H @ p1
            p2_proj = p2_proj[:2] / p2_proj[2]  # [wx, wy, w] -> [x, y]

            p2 = np.array(kp2[idx2].pt)

            distance = np.linalg.norm(p2 - p2_proj)

            if distance < threshold:
                support += 1
        # end calc suport
        if support > max_support:
            max_support = support
            max_H = H
        print(f'try {r}: support = {support}')
    print("support for H", max_support)
    return max_H

def project(H, p):
    '''
    H: (3, 3) Homography matrix
    p: (3,) or (2,) are OK
    return  p_proj (2,)
    '''
    if p.shape[0] == 2:
        p = np.array((*p, 1))
    p_proj = H @ p
    p_proj = p_proj[:2] / p_proj[2]
    return p_proj

def get_Mask(image, threshold=20):
    '''
    image: (H, W, c)
    return (H, W), value = 255 if gray[i][j] > threshold else 0
    '''
    gray = img_to_gray(image)
    mask = np.zeros(image.shape[:2], dtype=np.uint8)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if gray[i][j] > threshold:
                mask[i][j] = 255
            else:
                mask[i][j] = 0
    mask = cv2.GaussianBlur(mask, (101, 101), 0)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if gray[i][j] > threshold:
                pass
            else:
                mask[i][j] = 0
    return mask

def average_Masks(mask1, mask2):
    '''
    mask1, mask2: (H, W), uint8 0 | 255 mask
    return 0~1 smooth float mask
    '''
    total = np.zeros((mask1.shape), dtype=np.int32)

    Masks = [mask1, mask2]
    for mask in Masks:
        total += mask

    for i in range(2):
        Masks[i] = Masks[i] / total
        Masks[i] = Masks[i].astype(np.float) / 255.0
        Masks[i] = Masks[i].reshape((*Masks[i].shape, 1))

    return Masks[0], Masks[1]

def blending(warped1, warped2):
    mask1, mask2 = get_Mask(warped1), get_Mask(warped2)
    mask1, mask2 = average_Masks(mask1, mask2)
    result = mask1 * warped1 + mask2 * warped2
    result = (result * 255).astype(np.uint8)
    return result


def combine(image1, image2):
    '''
    transform(image1) + image2
    transform from image1 to image2 <- knn from image1 to image2
    '''
    image_gray1 = img_to_gray(image1)
    image_gray2 = img_to_gray(image2)
    kp1, f1 = SIFT.detectAndCompute(image_gray1, None)
    # 3196, (3196, 128)
    kp2, f2 = SIFT.detectAndCompute(image_gray2, None)

    matches = kNN(f1, f2)  # about 610

    # matches correctness test
    # kp1s = [kp1[match[0]] for match in matches[100: 120]]
    # kp2s = [kp2[match[1]] for match in matches[100: 120]]
    # image_gray1 = cv2.drawKeypoints(image_gray1, kp1s, image_gray1, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # image_gray2 = cv2.drawKeypoints(image_gray2, kp2s, image_gray2, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # create_im_window('im1', image_gray1)
    # create_im_window('im2', image_gray2)
    # im_show()

    H = RANSAC(matches, kp1=kp1, kp2=kp2)

    ori_size = image_gray1.shape  # (756, 1008, 3)
    # (0, 0), (size[1]-1, 0), (0, size[0]-1), (size[1]-1, size[0]-1)
    points4 = np.array([
        [            0,             0],
        [ori_size[1]-1,             0],
        [            0, ori_size[0]-1],
        [ori_size[1]-1, ori_size[0]-1]
    ])
    min_x = min_y = 0
    max_y, max_x = ori_size
    for p in points4:
        x, y = project(H, p)
        min_x = x if x < min_x else min_x
        min_y = y if y < min_y else min_y
        max_x = x if x > max_x else max_x
        max_y = y if y > max_y else max_y
    size_proj = (int(max_x - min_x), int(max_y - min_y))

    affine = np.array([  # x moves from min_x to 0, y form min_y to 0
        [1, 0, -min_x],
        [0, 1, -min_y],
        [0, 0, 1]
    ])

    warped_1 = cv2.warpPerspective(src=image1, M=affine @ H, dsize=size_proj)
    warped_2 = cv2.warpPerspective(src=image2, M=affine,     dsize=size_proj)
    # NOTICE that: dsize: (x_size, y_size)

    blended = blending(warped_1, warped_2)

    return blended


if __name__ == '__main__':
    prefix = 'HW2_Image_Stitching/' if True else ''
    SIFT = cv2.SIFT_create()

    images = []
    for i in range(1, 9):
        image_name = get_PicturePath(prefix, i)
        image, _ = read_img(image_name)
        images += [image]

    image_01 = combine(images[0], images[1])
    cv2.imwrite(f'{prefix}stitching_results/comb01.jpg', image_01)

    image_23 = combine(images[2], images[3])
    image_0123 = combine(image_01, image_23)
    cv2.imwrite(f'{prefix}stitching_results/comb0123.jpg', image_0123)

    image_45 = combine(images[4], images[5])
    image_67 = combine(images[6], images[7])
    image_4567 = combine(image_45, image_67)
    result = combine(image_0123, image_4567)

    # result = images[7]
    # for i in range(6, -1, -1):
    #     result = combine(images[i], result)

    cv2.imwrite(f'{prefix}stitching_results/comb01234567.jpg', result)
