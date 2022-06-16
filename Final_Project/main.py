import numpy as np
import os
import cv2
from matplotlib import pyplot as plt

from datasets import getDataset, getUnrectDataset
from disparity import getDisparityMap, getDepthMap, visualizeDepthMap, showHistogram, getDisparityMin, normalizeImage
from rectification import rectify, saveRectifiedImages

def showImages(**images):
    '''
    images: Dictionary of {
        title name: image,
        title name: ...,
    }
    '''
    for title, img in images.items():
        img_RGB = img
        if img.shape[-1] == 3:
            img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        plt.figure()
        plt.imshow(img_RGB)
        plt.title(title)

    plt.show()

def TsukubaExp():
    '''
    read tsukuba image 0 and 4
        0: Final_Project\\Dataset\\tsukuba_new\\scene1.row3.col1.jpg
        1: Final_Project\\Dataset\\tsukuba_new\\scene1.row3.col5.jpg
    calculate disparity from two images and save
        will load image first if image exists
    calculate depth map from disparity and save
        will load image first if image exists
        histogram post-processing inside
    '''
    # Path to store images
    save_folder_path = os.path.join('Final_Project', 'Experiments', 'Tsukuba')
    os.makedirs(save_folder_path, exist_ok=True)
    save_path = {
        'disp': os.path.join(save_folder_path, 'disp_origin.jpg'),
        'disp_norm': os.path.join(save_folder_path, 'disp_norm.jpg'),
        'depth': os.path.join(save_folder_path, 'depth_result.jpg')
    }

    dataset = getDataset('tsukuba')
    image_l = dataset[0]
    image_r = dataset[4]
    showImages(
        left_image = image_l,
        right_image = image_r
    )

    # Compute or Read disparity
    disparity = None
    if not os.path.exists(save_path["disp"]):
        disparity = getDisparityMap(image_l, image_r, method='DP')
        print(f'save original value disparity map to: {save_path["disp"]}')
        print(f'save normalized value disparity map to: {save_path["disp_norm"]}')
        cv2.imwrite(save_path["disp"], disparity)
        cv2.imwrite(save_path["disp_norm"], normalizeImage(disparity))
    else:
        print(f'load original value disparity map from: {save_path["disp"]}')
        disparity = cv2.imread(save_path["disp"], cv2.IMREAD_GRAYSCALE)
    disparity_ground_truth = dataset.getDisparity()

    showImages(
        disparity = normalizeImage(disparity),
        ground_truth = disparity_ground_truth
    )

    # Compute or Read depth map
    if not os.path.exists(save_path["depth"]):
        depth = getDepthMap(disparity, mode='Related', norm=normalizeImage)
        print(f'save normalized value depth map to: {save_path["depth"]}')
        cv2.imwrite(save_path["depth"], depth)
    else:
        print(f'load normalized value depth map from: {save_path["depth"]}')
        depth = cv2.imread(save_path["depth"], cv2.IMREAD_GRAYSCALE)

    showImages(
        depth = depth
    )

def NYCUExp():
    '''
    read self_NYCU image 0 and 1
        0: Final_Project\\Dataset\\self_NYCU\\unrect\\self_NYCU_1.jpg
        1: Final_Project\\Dataset\\self_NYCU\\unrect\\self_NYCU_2.jpg
    rectify these 2 images and save to
           Final_Project\\Dataset\\self_NYCU\\stereo\\self_NYCU_1.jpg
           Final_Project\\Dataset\\self_NYCU\\stereo\\self_NYCU_2.jpg
    calculate disparity from two images and save
        will load image first if image exists
    calculate depth map from disparity and save
        will load image first if image exists
        histogram post-processing inside
    '''

    # Path to store images
    folder_name = 'self_NYCU'
    save_folder_path = os.path.join('Final_Project', 'Experiments', 'NYCU')
    os.makedirs(save_folder_path, exist_ok=True)
    save_path = {
        'disp': os.path.join(save_folder_path, 'disp_origin.jpg'),
        'disp_norm': os.path.join(save_folder_path, 'disp_norm.jpg'),
        'depth': os.path.join(save_folder_path, 'depth_result.jpg')
    }
    Udataset = getUnrectDataset(folder_name)
    image_l = Udataset[0]
    image_r = Udataset[1]
    showImages(
        left = image_l,
        right = image_r
    )

    # Rectify images
    image_l, image_r = rectify(image_l, image_r)
    saveRectifiedImages(folder_name, image_l, image_r, smaller_size=False)  # original large scale
    showImages(
        left_after_rectified = image_l,
        right_after_rectified = image_r
    )

    # Compute or Read disparity
    disparity = None
    if not os.path.exists(save_path["disp"]):
        disparity = getDisparityMap(image_l, image_r, method='DP')
        print(f'save original value disparity map to: {save_path["disp"]}')
        print(f'save normalized value disparity map to: {save_path["disp_norm"]}')
        cv2.imwrite(save_path["disp"], disparity)
        cv2.imwrite(save_path["disp_norm"], normalizeImage(disparity))
    else:
        print(f'load original value disparity map from: {save_path["disp"]}')
        disparity = cv2.imread(save_path["disp"], cv2.IMREAD_GRAYSCALE)

    showImages(
        disparity = normalizeImage(disparity)
    )

    # Compute or Read depth map
    if not os.path.exists(save_path["depth"]):
        depth = getDepthMap(disparity, mode='Related', norm=normalizeImage)
        print(f'save normalized value depth map to: {save_path["depth"]}')
        cv2.imwrite(save_path["depth"], depth)
    else:
        print(f'load normalized value depth map from: {save_path["depth"]}')
        depth = cv2.imread(save_path["depth"], cv2.IMREAD_GRAYSCALE)

    showImages(
        depth = depth
    )

if __name__ == '__main__':
    # implementation of all
    # TsukubaExp()
    NYCUExp()
