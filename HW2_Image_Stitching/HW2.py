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

if __name__ == '__main__':
    prefix = 'HW2_Image_Stitching/' if True else ''
    img_name = get_PicturePath(prefix, 1)
    image, image_gray = read_img(img_name)
    
    create_im_window('rgb', image)
    create_im_window('gray', image_gray)
    im_show()

    # you can use this function to store the result
    # cv2.imwrite("result.jpg", img)
