import numpy as np
import cv2

def getDisparityMap(image_l, image_r, method='BlockSearch'):
    '''
    calculate disparity map from two
    rectified images(epipolar lines are well aligned)
    method in ['BlockSearch', 'DP']
    '''
    # some_block_search_implementation(image_l, image_r)
    pass

def getDepthMap(disparity, B=None, f=None):
    '''
    calculate depth map from disparity map
    by formula: Z = f * B / d
        Z: depth
        f: focal length
        B: two camera distance
        d: disparity map value
    '''
    pass
