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
    disparity[disparity == 0] = 1
    disparity = 255 / disparity

    disparity = disparity * 32  # changable?
    disparity[disparity > 255] = 255

    disparity = disparity.astype(np.uint8)

    return disparity

if __name__ == '__main__':
    disparity = cv2.imread(r'Final_Project\\Dataset\\tsukuba_new\\truedisp.row3.col3.jpg', cv2.IMREAD_GRAYSCALE)
    depth = getDepthMap(disparity)

    cv2.imshow('Depth', depth)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
