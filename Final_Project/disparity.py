from typing import List
import numpy as np
from scipy import interpolate
import cv2
from matplotlib import pyplot as plt
from datasets import getDataset

def getDisparityMap(image_l, image_r, method='BlockSearch'):
    '''
    calculate disparity map from two
        rectified images(epipolar lines are well aligned)
        method in ['BlockSearch', 'DP']
    also stored at Final_Project\\Dataset\\disp_temp.jpg
    '''
    # some_block_search_implementation(image_l, image_r)
    if method == 'DP':
        disparity = disparityDPmethod(image_l, image_r)  # -13 ~ 67

    elif method == 'BlockSearch':
        pass

    disparity -= disparity.min()
    disparity = disparity.astype(np.uint8)
    # visualizeDepthMap(disparity)

    # disparity = cv2.imread('Final_Project\\report\\disp_temp.jpg', cv2.IMREAD_GRAYSCALE)

    return disparity

def disparityDPmethod(image_l, image_r):
    disparity = np.zeros((image_l.shape[:2]), dtype=np.int8)
    for i in range(image_l.shape[0]):
        if i % 10 == 0:
            print(f'line {i}')
        # i = 100
        relation = getRelation(image_l[i], image_r[i])
        # visualizeDepthMap(relation)
        disparity[i] = DPsolver(relation)

    print('end disp DP')
    return disparity

# DP solver
def DPsolver(relation, occlusionConstant=30):
    '''
    return line disparity
    '''
    n = relation.shape[0]
    _, direction_map = getDPmap_DirectionMap(relation, occlusionConstant)
    # direction_map, 1: right, 2: right down, 3: down

    path = getDPpath(direction_map)
    # visualizeDepthMap(direction_map)

    # Path to Disparity: p.left_pixel - p.right_pixel
    ### Three Strategy ###
    # path is longer than disp_line..., so
    # 1. disp_line = compress(path), 將長度縮小，中間取插植
    # 2. disp_line = path[a:b +n], 取中間 disp 長度的資料
    # 3. disp_line = {left_p} - right_p, 只記錄左邊有移動的 disp # choose this
    ################################

    disparity_line = getDPdisparityLine_Left(path, n)
    # disparity_line = getDPdisparityLine_All(path, n)

    return disparity_line

def getDPmap_DirectionMap(relation, occlusionConstant):
    n = relation.shape[0]
    DPmap = np.zeros((relation.shape))
    direction_map = np.zeros((relation.shape), dtype=np.uint8)
    # 1: right, 2: right down, 3: down
    for i in range(n):
        DPmap[i][0] = relation[i][0]    # fill left edge
        direction_map[i][0] = 3     # direction down
        DPmap[0][i] = relation[0][i]    # fill right edge
        direction_map[0][i] = 1     # direction right

    for i in range(1, n):
        for j in range(1, n):
            val2 = DPmap[i-1][j-1] + relation[i][j]     # No occlusion
            val3 = DPmap[i-1][j  ] + occlusionConstant  # Occluded from left
                                    # right can see, left cannot see
            val1 = DPmap[i  ][j-1] + occlusionConstant  # Occluded from right

            DPmap[i][j] = min(val1, val2, val3)
            # Direction map
            if DPmap[i][j] == val2:
                direction_map[i][j] = 2
            elif DPmap[i][j] == val3:
                direction_map[i][j] = 3
            elif DPmap[i][j] == val1:
                direction_map[i][j] = 1

    return DPmap, direction_map

def getDPpath(direction_map: np.ndarray) -> List[int]:
    '''
    return list of directions
        from top left to button right
    * Note that: this will also highlight the PATH on direction_map
        direction_map[path_pixel] = 10
    '''
    n = direction_map.shape[0]
    path = []

    i_trace = n-1
    j_trace = n-1
    while i_trace != 0 or j_trace != 0:
        direction = direction_map[i_trace][j_trace]
        direction_map[i_trace][j_trace] = 10  # highlight the path

        path += [direction]
        if direction == 1:
            j_trace -= 1
        elif direction == 2:
            i_trace -= 1
            j_trace -= 1
        elif direction == 3:
            i_trace -= 1
    
    path = path[::-1]  # inverse path, from start to end
    return path

def getDPdisparityLine_Left(path: List[int], n):
    '''
    return line disparity of shape(n)

    Strategy 3. disp_line = {left_p} - right_p, 只記錄左邊有移動的 disp # choose this
    '''
    disparity_line = np.zeros(n, dtype=np.int8)
    i_trace = 0
    j_trace = 0

    for direction in path:
        if direction == 2:  # matched
            disp = j_trace - i_trace  # left - right
            disparity_line[j_trace] = disp
            j_trace += 1
            i_trace += 1
        elif direction == 1:
            # right occ: left can see, right cannot see
            j_trace += 1
            # disparity_line[j_trace] = righter value
        elif direction == 3:
            # left occ: right can see, left cannot see
            i_trace += 1

    # filled in empty values
    # filled right occ places, left can see, right cannot see
    righter_value = 0
    for i in range(n-1, -1, -1):
        if disparity_line[i] != 0:
            righter_value = disparity_line[i]
        else:
            disparity_line[i] = righter_value

    return disparity_line

def getDPdisparityLine_All(path: List[int], n):
    '''
    Strategy 1
    '''
    disp_all = []
    disp = 0
    for dir in path:
        if dir == 2:    # Matched
            pass
        elif dir == 1:  # right: pixel in left  image moves right
            disp += 1
        elif dir == 3:  # left:  pixel in right image moves right
            disp -= 1
        disp_all += [disp]

    x = np.arange(len(path))  # 0 ~ 382, len = 383
    disp_all = np.array(disp_all, dtype=np.float)

    x_new = np.linspace(0, len(path) -1, n)  # fill in 0~383

    interp = interpolate.interp1d(x, disp_all)
    disp_new = interp(x_new)

    return disp_new

def getRelation(line_l, line_r):
    '''
    line_l, line_r: ndarray(width, 3)
    return relation
    #
    relation[i][j] = norm(line_r[i] - line_l[j])  # as text book
    #
    '''
    n = line_l.shape[0]
    left = np.copy(line_l).astype(np.float)
    right = np.copy(line_r).astype(np.float)
    relation = np.zeros((n, n))

    for i in range(n):
        # complete relation [i][...]
        # relation[i] = np.norm(line_r[i] - line_l)
        to_norm = right[i] - left
        relation[i] = np.linalg.norm(to_norm, axis=1)  # norm([r, g, b]), norm([r, g, b])

    return relation
# DP solver end

# Build Depth map from disparity
def getDepthMap(disparity, mode, B=None, f=None, norm=None):  # disparity map should > 0
    '''
    calculate depth map from disparity map
    0 <= disparity <= 255
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
    disparity[disparity == 0] = 1

    if mode == 'Accurate':
        # disp_min = getDisparityMin(disparity)       # needed?
        # disparity[disparity < disp_min] = disp_min  # needed?
        disparity = (1 / disparity) * B * f

    elif mode == 'Related':
        # showHistogram(disparity)

        # disp_min = getDisparityMin(disparity)
        # disparity[disparity < disp_min] = disp_min  # this process makes edge black the SAME value as background
        # disparity -= disparity.min()  # disparity map >= 0 already

        disparity = 1 / disparity
        disparity = norm(disparity)

        # showHistogram(disparity)

    return disparity

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

    if len(high_freq_values) < 3:
        if high_freq_values[0] < 5:
            return high_freq_values[1]
        else:
            return high_freq_values[0]
# Build Depth map end

def normalizeImage(image):
    '''
    norm image to 0~255
    '''
    image = (image - image.min()) / image.ptp() * 255
    return image.astype(np.uint8)

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

def visualizeDepthMap(depth_map, ground_truth=None):
    '''
    HW1 tools
    '''
    plt.figure()
    plt.imshow(depth_map)
    plt.colorbar(label='Distance to Camera')
    plt.title('Depth map')
    plt.xlabel('X Pixel')
    plt.ylabel('Y Pixel')

    if ground_truth is not None:
        plt.figure()
        plt.imshow(ground_truth)
        plt.colorbar(label='Distance to Camera')
        plt.title('Ground truth')
        plt.xlabel('X Pixel')
        plt.ylabel('Y Pixel')

    plt.show()

if __name__ == '__main__':
    dataset = getDataset('self_laptop')
    # disparity = dataset.getDisparity()
    disparity = getDisparityMap(dataset[0], dataset[1], method='DP')
    cv2.imwrite('Final_Project\\Dataset\\disp_temp.jpg', disparity)

    depth = getDepthMap(disparity, mode='Related', norm=normalizeImage)
    cv2.imwrite('Final_Project\\Dataset\\depth_temp.jpg', depth)

    # g_disparity = dataset.getDisparity()
    # g_depth = getDepthMap(g_disparity, mode='Related', norm=normalizeImage)

    visualizeDepthMap(depth)
