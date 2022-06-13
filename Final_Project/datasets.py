import numpy as np
import os
import cv2

class Dataset:
    def __init__(self, folder_name):
        '''
        folder_name: Dataset/{folder_name}/images
        '''
        self.stereo = []
        self.disparity = None
        self.length = 0
        self.folder_name = folder_name

        self.stereoFolderPath = None
        self.disparityImagePath = None

    def getPaths(self):
        # print('path1')
        datasetFolderPath = os.path.join(
            os.path.dirname(__file__), 'Dataset', self.folder_name)
        self.stereoFolderPath = os.path.join(datasetFolderPath, 'stereo')
        self.disparityImagePath = os.path.join(datasetFolderPath, f'{self.folder_name}_o_d.jpg')

    def loadImages(self):
        if self.stereoFolderPath == None:
            self.getPaths()

        if not os.path.exists(self.stereoFolderPath):
            raise Exception(f'stereo images not found in: {self.stereoFolderPath}')
        else:
            self.loadStereo()

        if not os.path.exists(self.disparityImagePath):
            print(f'disparity map not found in: {self.stereoFolderPath}')
        else:
            self.loadDisp()

    def loadStereo(self):
        for i in range(len(os.listdir(self.stereoFolderPath))):
            filename = f'{self.folder_name}_{i + 1}.jpg'

            fullpath = os.path.join(self.stereoFolderPath, filename)
            image = cv2.imread(fullpath)
            self.stereo += [image]
        self.length = len(self.stereo)

    def loadDisp(self):
        image = cv2.imread(self.disparityImagePath, cv2.IMREAD_GRAYSCALE)
        self.disparity = image

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, key: int) -> np.ndarray:
        if key >= self.length:
            raise KeyError('Dataset[{key}] not exists with len: {self.length}')
        return self.stereo[key]

    def getDisparity(self):
        return self.disparity  # ndarray or None

class Dataset_Tsukuba(Dataset):
    def __init__(self):
        super(Dataset_Tsukuba, self).__init__('tsukuba_new')
    
    def getPaths(self):
        datasetFolderPath = os.path.join(
            os.path.dirname(__file__), 'Dataset', self.folder_name)
        self.stereoFolderPath = datasetFolderPath
        self.disparityImagePath = os.path.join(datasetFolderPath, 'truedisp.row3.col3.jpg')
    
    def loadStereo(self):
        for i in range(5):
            filename = f'scene1.row3.col{i + 1}.jpg'

            fullpath = os.path.join(self.stereoFolderPath, filename)
            image = cv2.imread(fullpath)
            self.stereo += [image]
        self.length = len(self.stereo)

def getDataset(folder_name):
    if folder_name == 'tsukuba':
        dataset = Dataset_Tsukuba()
        # dataset.getPaths()
        dataset.loadImages()  # use child version getPaths()
        return dataset
    else:
        dataset = Dataset(folder_name)
        dataset.loadImages()
        return dataset

class Dataset_Unrect(Dataset):
    def __init__(self, folder_name):
        self.unrect = []
        self.length = 0
        self.folder_name = folder_name

        self.unrectFolderPath = None

    def getPaths(self):
        # print('path1')
        datasetFolderPath = os.path.join(
            os.path.dirname(__file__), 'Dataset', self.folder_name)
        self.unrectFolderPath = os.path.join(datasetFolderPath, 'unrect')

    def loadUnrectImages(self):
        if self.unrectFolderPath == None:
            self.getPaths()

        for i in range(len(os.listdir(self.unrectFolderPath))):
            filename = f'{self.folder_name}_{i + 1}.jpg'

            fullpath = os.path.join(self.unrectFolderPath, filename)
            image = cv2.imread(fullpath)
            print(image.shape)
            self.unrect += [image]
        self.length = len(self.unrect)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, key: int) -> np.ndarray:
        if key >= self.length:
            raise KeyError('Dataset[{key}] not exists with len: {self.length}')
        return self.unrect[key]

def getUnrectDataset(folder_name):
    dataset = Dataset_Unrect(folder_name)
    dataset.loadUnrectImages()
    return dataset

def show(image):
    cv2.imshow('Window', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # dataset = getDataset('tsukuba')
    # for i in range(len(dataset)):
    #     show(dataset[i])
    # show(dataset.getDisparity())
    dataset = getUnrectDataset('self_door')
    for i in range(len(dataset)):
        show(dataset[i])
