import numpy as np
import cv2
from matplotlib import pyplot as plt
from datasets import getDataset

def getDisparityMap(image_l, image_r, method='BlockSearch'):
    '''
    calculate disparity map from two
    rectified images(epipolar lines are well aligned)
    method in ['BlockSearch', 'DP']
    '''
    # some_block_search_implementation(image_l, image_r)
    pass

def getDepthMap(disparity, mode, B=None, f=None, norm=None):
    '''
    calculate depth map from disparity map
    mode: ['Accurate', 'Related']
    - Accurate:
        by formula: Z = f * B / d
            Z: depth
            f: focal length
            B: two camera distance
            d: disparity map value
        param: B, f

    - Related:
        normalize(image)
        param: norm
    '''
    if mode == 'Accurate':
        disp_min = getDisparityMin(disparity)       # needed?
        disparity[disparity < disp_min] = disp_min  # needed?
        disparity = (1 / disparity) * B * f

    elif mode == 'Related':
        # showHistogram(disparity)

        disp_min = getDisparityMin(disparity)
        disparity[disparity < disp_min] = disp_min  # this process makes edge black the SAME value as background
        
        disparity = 1 / disparity
        disparity = norm(disparity)

        # showHistogram(disparity)

    return disparity

def normalizeImage(image):
    '''
    norm image to 0~255
    '''
    image = (image - image.min()) / image.ptp() * 255
    return image.astype(np.uint8)

def getDisparityMin(disparity):
    '''
    disparity image will sometimes have a thick edge of black color(value = 0)
    return the real min value in disparity map
    return 1 if there is no thick edge

    also means: return the 2nd min value in {high freq color: > sum / 10}
    出現頻率高的顏色中，選數值第二低( 真實照片的低 pixel 移動 )的數字回傳
    '''
    hist = cv2.calcHist([disparity], [0], None, [256], [0, 256])  # ←計算直方圖資訊
    high_freq_threshold = np.prod(disparity.shape) / 20
    high_freq_values = []

    for i, e in enumerate(hist):
        if e > high_freq_threshold:
            high_freq_values += [i]

    high_freq_values = sorted(high_freq_values)

    if high_freq_values[0] < 5:
        return high_freq_values[1]
    else:
        return high_freq_values[0]

def showHistogram(image):
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])  # 計算直方圖資訊
    # 使用 MatPlot 繪出 histogram
    plt.figure()
    plt.title("Grayscale Histogram")
    plt.xlabel("Bins")
    plt.ylabel("# of Pixels")
    plt.plot(hist)
    plt.xlim([0, 256])
    plt.show()

def visualizeDepthMap(depth_map):
    '''
    HW1 tools
    '''
    plt.figure()
    plt.imshow(depth_map)
    plt.colorbar(label='Distance to Camera')
    plt.title('Depth map')
    plt.xlabel('X Pixel')
    plt.ylabel('Y Pixel')
    plt.show()

if __name__ == '__main__':
    dataset = getDataset('tsukuba')
    disparity = dataset.getDisparity()
    depth = getDepthMap(disparity, mode='Accurate', B = 1, f = 1)

    visualizeDepthMap(depth)
