import cv2
import numpy as np
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

def RANSAC(matches, kp1, f1, kp2, f2):
    '''
    random choose 4 from matches
    do PrespectiveTransform(all matches)
        and counting supports
    return best Transform Mat
    '''
    pass


def combine(image1, image2):
    '''
    transform(image1) + image2
    transform from image1 to image2 <- knn from image1 to image2
    '''
    kp1, f1 = SIFT.detectAndCompute(image1, None)
    # 3196, (3196, 128)
    kp2, f2 = SIFT.detectAndCompute(image2, None)

    matches = kNN(f1, f2)  # about 610
    
    bestMat = RANSAC(matches, kp1=kp1, f1=f1, kp2=kp2, f2=f2)

    pos = kp1[0].pt  # (x, y)

if __name__ == '__main__':
    prefix = 'HW2_Image_Stitching/' if True else ''
    SIFT = cv2.SIFT_create()

    img_name = get_PicturePath(prefix, 1)
    _, image1 = read_img(img_name)
    img_name = get_PicturePath(prefix, 2)
    _, image2 = read_img(img_name)

    combine(image1, image2)

    # create_im_window('rgb', image)
    create_im_window('gray', image1)
    im_show()

    # you can use this function to store the result
    # cv2.imwrite("result.jpg", img)
