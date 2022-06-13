import numpy as np
import cv2

from datasets import getDataset

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
    '''
    pass

if __name__ == '__main__':
    pass
